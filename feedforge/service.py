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
from typing import Optional

import numpy as np
import torch

from .cache import get_cache
from .content import ContentIndex
from .data import PAD, inference_batch
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

    # -- public API --------------------------------------------------------

    def known_users(self) -> list[int]:
        return sorted(self.user_row.keys())

    def user_history(self, user_id: int, limit: int = 10) -> list[dict]:
        row = self.user_row[user_id]
        hist = self.full_history[row]
        return [self._item_payload(i) for i in reversed(hist[-limit:])]

    def recommend(self, user_id: int, arm: str = "baseline", n: int = 10) -> dict:
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
        else:
            ordered = cands

        items = [self._item_payload(i) for i in ordered[:n]]
        return {
            "user_id": user_id,
            "arm": arm,
            "items": items,
            "cache_hit": cache_hit,
            "latency_ms": round(now_ms() - start, 2),
            "candidates_considered": len(cands),
        }

    # -- internals ---------------------------------------------------------

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
        return {
            "item_id": dense_id,
            "movie_id": orig,
            "title": title,
            "poster": f"/posters/{orig}.jpg" if orig is not None else None,
            "popularity": int(self.item_pop[dense_id]) if dense_id < len(self.item_pop) else 0,
        }
