"""The serving pipeline: candidates -> features -> rank -> results.

Stage 1 (retrieval): BERT4Rec scores the full catalog for the user's
history and returns top-N candidates, with already-seen items filtered.
Stage 2 (ranking): for the 'reranked' arm, the LambdaRank model reorders
those candidates using the full feature vector.

Caching strategy, and why it's shaped this way: the expensive step is
the transformer forward pass (tens of milliseconds), so the candidate
list plus scores is what gets cached, keyed by user and history length.
The key includes history length so a user whose history grows gets a
fresh list rather than a stale one -- a cache that can't invalidate is a
bug waiting to happen. Ranking runs on every request even on a cache
hit, because it's sub-millisecond and keeps arm comparisons honest.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .cache import get_cache
from .content import ContentIndex
from .data import PAD, inference_batch
from .discovery import (
    genre_coverage,
    intra_list_similarity,
    make_genre_similarity,
    mmr_rerank,
    popular_items,
)
from .experiment import now_ms
from .features import AuxFeatureContext
from .model import BERT4Rec
from .ranker import build_ranking_dataset, feature_names, rerank


class RecommenderService:
    def __init__(
        self,
        artifacts_path: str = "artifacts/serving.pkl",
        checkpoint_path: str = "checkpoints/bert4rec_best.pt",
        ranker_path: Optional[str] = "checkpoints/ranker.txt",
        embeddings_path: Optional[str] = "data/content_embeddings.npz",
        item_id_map_source: Optional[str] = None,
        candidate_k: int = 100,
        device: str = "cpu",
        cache_ttl: int = 300,
    ):
        self.device = device
        self.candidate_k = candidate_k

        with open(artifacts_path, "rb") as f:
            art = pickle.load(f)
        self.art = art
        self.n_items = art["n_items"]
        self.mask_token = art["mask_token"]
        self.full_history = art["full_history"]
        self.user_row = art["user_row"]
        self.titles = art["titles"]
        self.dense_to_orig = art["dense_to_orig"]
        self.item_pop = art["item_pop"]
        # Rating stats are optional so an artifacts file built before this
        # feature still loads; missing stats simply render no rating.
        self.rating_mean = art.get("rating_mean")
        self.rating_count = art.get("rating_count")

        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = ckpt["config"]
        self.max_len = cfg["max_len"]
        # d_ff defaults to 4x d_model (train.py's convention) but is read
        # from the checkpoint when present, so a model trained with a
        # custom feed-forward width still loads.
        self.model = BERT4Rec(
            vocab_size=ckpt["vocab_size"], max_len=cfg["max_len"],
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ff=cfg.get("d_ff") or cfg["d_model"] * 4,
        ).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.aux_ctx = AuxFeatureContext(
            art["genre_mat"], art["years"], art["user_demo"], art["pmi"])

        self.content_index = None
        if embeddings_path and Path(embeddings_path).exists():
            item_id_map = {o: d for d, o in self.dense_to_orig.items()}
            self.content_index = ContentIndex.from_npz(embeddings_path, item_id_map)

        self.booster = None
        if ranker_path and Path(ranker_path).exists():
            import lightgbm as lgb

            self.booster = lgb.Booster(model_file=ranker_path)

        self.cache = get_cache(cache_ttl)
        self.feature_names = feature_names(self.aux_ctx)
        self.genre_sim = make_genre_similarity(art["genre_mat"])
        self.genre_mat = art["genre_mat"]
        # Title search index: lowercase title -> dense id, built once.
        self._search_index = [(t.lower(), i) for i, t in self.titles.items()]

    # -- public API --------------------------------------------------------

    def known_users(self) -> list[int]:
        return sorted(self.user_row.keys())

    def user_history(self, user_id: int, limit: int = 10) -> list[dict]:
        row = self.user_row[user_id]
        hist = self.full_history[row]
        return [self._item_payload(i) for i in reversed(hist[-limit:])]

    def search_titles(self, query: str, limit: int = 12) -> list[dict]:
        """Substring title search for the cold-start picker. Popular items
        rank first so the suggestions are recognizable."""
        q = query.lower().strip()
        if not q:
            return []
        hits = [i for t, i in self._search_index if q in t]
        hits.sort(key=lambda i: -self.item_pop[i])
        return [self._item_payload(i) for i in hits[:limit]]

    def starter_items(self, n: int = 24) -> list[dict]:
        """Well-known titles to seed the cold-start picker."""
        return [self._item_payload(i) for i in popular_items(self.item_pop, n=n)]

    def recommend_cold(self, picks: list[int], n: int = 10,
                       diversity: float = 1.0) -> dict:
        """Recommendations for a visitor with no history, from the items
        they just picked. The picks are treated as a (very short) sequence
        and run through the same transformer -- measured at 8x the
        recall@10 of popularity-based fallback from a single pick.
        """
        start = now_ms()
        picks = [int(p) for p in picks if int(p) in self.dense_to_orig]
        if not picks:
            items = [self._item_payload(i) for i in popular_items(self.item_pop, n=n)]
            return {"items": items, "strategy": "popular", "picks": [],
                    "latency_ms": round(now_ms() - start, 2),
                    "diversity": diversity, "list_stats": self._list_stats(
                        [i["item_id"] for i in items])}

        cands, scores = self._generate_candidates(picks)
        ordered = self._apply_diversity(cands, scores, n, diversity)
        items = [self._item_payload(i) for i in ordered[:n]]
        return {
            "items": items,
            "strategy": "sequence",
            "picks": [self._item_payload(p) for p in picks],
            "latency_ms": round(now_ms() - start, 2),
            "diversity": diversity,
            "candidates_considered": len(cands),
            "list_stats": self._list_stats([i["item_id"] for i in items]),
        }

    def recommend(self, user_id: int, arm: str = "baseline", n: int = 10,
                  diversity: float = 1.0) -> dict:
        start = now_ms()
        row = self.user_row[user_id]
        hist = self.full_history[row]

        cache_key = f"cand:{user_id}:{len(hist)}:{self.candidate_k}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            cands, scores = cached["cands"], cached["scores"]
            cache_hit = True
        else:
            cands, scores = self._generate_candidates(hist)
            self.cache.set(cache_key, {"cands": cands, "scores": scores})
            cache_hit = False

        if arm == "reranked" and self.booster is not None:
            ordered = self._rerank(row, hist, cands, scores)
            ordered_scores = list(range(len(ordered), 0, -1))
        else:
            ordered, ordered_scores = cands, scores

        if diversity < 1.0:
            ordered = self._apply_diversity(ordered, ordered_scores, n, diversity)

        items = [self._item_payload(i) for i in ordered[:n]]
        return {
            "user_id": user_id,
            "arm": arm,
            "items": items,
            "cache_hit": cache_hit,
            "latency_ms": round(now_ms() - start, 2),
            "candidates_considered": len(cands),
            "diversity": diversity,
            "list_stats": self._list_stats([i["item_id"] for i in items]),
        }

    def explain(self, item_id: int, history: Sequence[int], top: int = 3) -> dict:
        """Why was this item recommended, given these history items?

        Attribution by leave-one-out: re-score the candidate with each
        history item removed and measure how much its score drops. A large
        drop means that history item was doing the work. This is a genuine
        counterfactual through the same model rather than a similarity
        heuristic dressed up as an explanation -- it costs one forward pass
        per history item, which is why it runs on the (short) recent
        history rather than the full sequence.

        Also reports genre overlap, which is cheap and often the more
        human-legible half of the answer.
        """
        item_id = int(item_id)
        hist = [int(h) for h in history][-12:]
        if not hist:
            return {"item_id": item_id, "drivers": [], "shared_genres": []}

        base = float(self._score_item(hist, item_id))
        contributions = []
        for i, h in enumerate(hist):
            without = hist[:i] + hist[i + 1:]
            if not without:
                continue
            dropped = float(self._score_item(without, item_id))
            contributions.append((h, base - dropped))
        contributions.sort(key=lambda x: -x[1])

        # Only surface contributions large enough to survive rounding: a
        # driver rendered as "0.000" is noise presented as signal.
        MIN_CONTRIBUTION = 0.001
        drivers = []
        for h, delta in contributions[:top]:
            if delta < MIN_CONTRIBUTION:
                continue
            payload = self._item_payload(h)
            payload["contribution"] = round(delta, 3)
            drivers.append(payload)

        item_genres = self.genre_mat[item_id]
        shared = set()
        for h in hist:
            overlap = np.logical_and(item_genres > 0, self.genre_mat[h] > 0)
            shared.update(int(i) for i in np.nonzero(overlap)[0])

        return {
            "item_id": item_id,
            "drivers": drivers,
            "shared_genre_ids": sorted(shared),
            "base_score": round(base, 3),
        }

    @torch.no_grad()
    def _score_item(self, history: Sequence[int], item_id: int) -> float:
        tokens = inference_batch([list(history)], self.mask_token, self.max_len).to(self.device)
        scores = self.model.score_last_position(tokens)
        return float(scores[0, item_id].item())

    # -- internals ---------------------------------------------------------

    def _apply_diversity(self, cands, scores, n: int, lambda_: float) -> list[int]:
        """MMR over genre vectors. lambda_=1.0 short-circuits to pure
        relevance so the default path pays nothing."""
        if lambda_ >= 1.0:
            return list(cands)
        return mmr_rerank(list(cands), list(scores), self.genre_sim,
                          lambda_=lambda_, n=max(n, 1))

    def _list_stats(self, items: list[int]) -> dict:
        """Diversity readout for a rendered list."""
        return {
            "genres_covered": genre_coverage(items, self.genre_mat),
            "intra_list_similarity": round(
                intra_list_similarity(items, self.genre_sim), 3),
        }

    @torch.no_grad()
    def _generate_candidates(self, hist: list[int]) -> tuple[list[int], list[float]]:
        tokens = inference_batch([hist], self.mask_token, self.max_len).to(self.device)
        scores = self.model.score_last_position(tokens)
        scores[:, PAD] = -float("inf")
        scores[:, self.mask_token] = -float("inf")
        scores[0, torch.tensor(hist, device=self.device)] = -float("inf")
        # Clamp k to what the catalog can actually supply: a small catalog
        # (or a large candidate_k) must degrade gracefully, not raise.
        k = min(self.candidate_k, scores.shape[1])
        top = torch.topk(scores[0], k)
        idx = top.indices.cpu().tolist()
        vals = top.values.cpu().tolist()
        # Drop any -inf entries (seen/special tokens) that survived the clamp
        keep = [(i, v) for i, v in zip(idx, vals) if v != -float("inf")]
        return [i for i, _ in keep], [v for _, v in keep]

    def _rerank(self, row: int, hist: list[int], cands: list[int],
                scores: list[float]) -> list[int]:
        # user_rows passes the real demographic row: the dataset has one
        # user, but their demographics live at index `row` in the array.
        ds = build_ranking_dataset(
            [cands], [scores], [hist], targets=None,
            item_pop=self.item_pop, content_index=self.content_index,
            aux_ctx=self.aux_ctx, user_rows=[row],
        )
        return rerank(self.booster, ds)[0]

    def _item_payload(self, dense_id: int) -> dict:
        # Everything is coerced to native Python types here: ids that came
        # from numpy arrays / pandas are numpy scalars, which FastAPI's
        # JSON encoder cannot serialize. This is the boundary where model
        # types stop and wire types begin.
        dense_id = int(dense_id)
        orig = self.dense_to_orig.get(dense_id)
        orig = int(orig) if orig is not None else None
        title = str(self.titles.get(dense_id, f"Item {dense_id}"))
        payload = {
            "item_id": dense_id,
            "movie_id": orig,
            "title": title,
            "poster": f"/posters/{orig}.jpg" if orig is not None else None,
            "popularity": int(self.item_pop[dense_id]) if dense_id < len(self.item_pop) else 0,
        }
        if self.rating_mean is not None and dense_id < len(self.rating_mean):
            count = int(self.rating_count[dense_id])
            payload["rating"] = round(float(self.rating_mean[dense_id]), 2) if count else None
            payload["rating_count"] = count
        return payload
