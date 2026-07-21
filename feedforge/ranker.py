"""Ranking stage: LightGBM LambdaRank over BERT4Rec candidates.

Design informed directly by the Phase 2 measurement: equal-weight RRF
fusion of content candidates into the retrieval stage cost ~10 points of
candidate recall@100, so candidate generation stays collaborative-only
(BERT4Rec top-K) and content similarity is demoted to a ranking FEATURE,
where the learned model assigns it whatever weight it actually earns.
This is the standard production resolution of the same tradeoff:
high-recall retrieval, feature-rich ranking.

LightGBM is gradient-boosted trees, not a neural network, and that is a
deliberate choice, not a compromise: GBDT rankers with listwise
LambdaRank objectives were the production standard for years precisely
because they handle heterogeneous tabular features (ranks, counts,
similarities, flags) without normalization gymnastics and train in
seconds. The interview-honest description is "GBDT LambdaRank reranker".

Feature vector per (user, candidate):
    collab_rank      position in BERT4Rec's top-K (the strongest feature
                     by construction; the ranker must add value beyond it)
    collab_score     BERT4Rec logit for the candidate
    content_sim      cosine(candidate embedding, user content profile);
                     0 when either side has no embedding
    content_sim_last cosine(candidate, user's most recent item)
    has_content      1 if the candidate has a poster embedding
    item_pop_log     log1p(train interaction count of the candidate)
    user_len_log     log1p(user history length)

Labels: 1 for the held-out target, 0 otherwise; one query group per
user. Training queries use the validation target (history = train);
evaluation uses the test target (history = train + valid), so the ranker
never trains on anything the final evaluation sees.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install lightgbm") from e

BASE_FEATURE_NAMES = [
    "collab_rank",
    "collab_score",
    "content_sim",
    "content_sim_last",
    "has_content",
    "item_pop_log",
    "user_len_log",
]
# Kept as an alias for backward compatibility with existing callers/tests.
FEATURE_NAMES = BASE_FEATURE_NAMES


def feature_names(aux_ctx=None) -> list[str]:
    if aux_ctx is None:
        return list(BASE_FEATURE_NAMES)
    from .features import AUX_FEATURE_NAMES
    return list(BASE_FEATURE_NAMES) + list(AUX_FEATURE_NAMES)


@dataclass
class RankingDataset:
    X: np.ndarray          # (n_rows, n_features)
    y: np.ndarray          # (n_rows,) 0/1
    groups: np.ndarray     # (n_users,) candidates per user, in row order
    candidate_ids: list[list[int]]  # per user, aligned with rows


def build_item_popularity(train: Sequence[Sequence[int]], n_items: int) -> np.ndarray:
    counts = np.zeros(n_items + 1, dtype=np.float64)
    c = Counter(item for seq in train for item in seq)
    for item, n in c.items():
        counts[item] = n
    return counts


def build_ranking_dataset(
    candidates: Sequence[Sequence[int]],   # per user, BERT4Rec top-K (ranked)
    collab_scores: Sequence[Sequence[float]],  # aligned logits
    histories: Sequence[Sequence[int]],
    targets: Optional[Sequence[int]],      # None at pure inference time
    item_pop: np.ndarray,
    content_index=None,                    # feedforge.content.ContentIndex or None
    aux_ctx=None,                          # feedforge.features.AuxFeatureContext or None
    user_rows=None,                        # explicit demographic row per user;
                                           # defaults to positional index
) -> RankingDataset:
    rows, labels, groups, cand_ids = [], [], [], []

    for u, (cands, scores, hist) in enumerate(zip(candidates, collab_scores, histories)):
        demo_row = user_rows[u] if user_rows is not None else u
        uctx = aux_ctx.user_context(demo_row, hist) if aux_ctx is not None else None
        # User content profile: mean of last-10 history embeddings
        profile = last_vec = None
        if content_index is not None:
            vecs = [v for item in hist[-10:]
                    if (v := content_index.vector_of(item)) is not None]
            if vecs:
                profile = np.mean(vecs, axis=0)
                profile /= np.linalg.norm(profile) + 1e-12
            last_vec = content_index.vector_of(hist[-1]) if hist else None

        user_len_log = math.log1p(len(hist))
        for rank, (item, score) in enumerate(zip(cands, scores)):
            sim = sim_last = 0.0
            has_content = 0.0
            if content_index is not None:
                v = content_index.vector_of(item)
                if v is not None:
                    has_content = 1.0
                    if profile is not None:
                        sim = float(np.dot(v, profile))
                    if last_vec is not None:
                        sim_last = float(np.dot(v, last_vec))
            row = [
                float(rank), float(score), sim, sim_last, has_content,
                math.log1p(item_pop[item]), user_len_log,
            ]
            if uctx is not None:
                row.extend(aux_ctx.features_for(uctx, item))
            rows.append(row)
            if targets is not None:
                labels.append(1 if item == targets[u] else 0)
        groups.append(len(cands))
        cand_ids.append(list(cands))

    return RankingDataset(
        X=np.asarray(rows, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int8) if targets is not None else np.zeros(0, np.int8),
        groups=np.asarray(groups, dtype=np.int32),
        candidate_ids=cand_ids,
    )


def train_ranker(
    train_ds: RankingDataset,
    valid_ds: Optional[RankingDataset] = None,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 30,
    params: Optional[dict] = None,
    names: Optional[list] = None,
    categorical: Optional[list] = None,
) -> "lgb.Booster":
    default_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "verbosity": -1,
        # Positions matter down the full candidate list, not just top-10:
        "lambdarank_truncation_level": 30,
    }
    if params:
        default_params.update(params)

    names = names or FEATURE_NAMES
    cat = categorical or "auto"
    dtrain = lgb.Dataset(train_ds.X, label=train_ds.y, group=train_ds.groups,
                         feature_name=names, categorical_feature=cat)
    valid_sets, callbacks = [dtrain], []
    if valid_ds is not None:
        valid_sets.append(lgb.Dataset(valid_ds.X, label=valid_ds.y,
                                      group=valid_ds.groups, reference=dtrain,
                                      feature_name=names, categorical_feature=cat))
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    return lgb.train(default_params, dtrain, num_boost_round=num_boost_round,
                     valid_sets=valid_sets, callbacks=callbacks)


def rerank(booster: "lgb.Booster", ds: RankingDataset) -> list[list[int]]:
    """Reorder each user's candidate list by ranker score."""
    preds = booster.predict(ds.X)
    out, offset = [], 0
    for u, g in enumerate(ds.groups):
        scores = preds[offset : offset + g]
        order = np.argsort(-scores, kind="stable")
        out.append([ds.candidate_ids[u][i] for i in order])
        offset += g
    return out


def ranked_metrics(ranked: Sequence[Sequence[int]], targets: Sequence[int],
                   ks: tuple[int, ...] = (10, 20)) -> dict:
    """End-to-end recall/NDCG of a ranked list against the target. Same
    definition as evaluate.full_ranking_metrics, restricted to the
    candidate set, so numbers are directly comparable to the candidate
    recall ceiling."""
    out = {}
    n = len(targets)
    for k in ks:
        hits = ndcg = 0.0
        for lst, t in zip(ranked, targets):
            if t in lst[:k]:
                hits += 1
                ndcg += 1.0 / math.log2(lst.index(t) + 2)
        out[f"recall@{k}"] = hits / n
        out[f"ndcg@{k}"] = ndcg / n
    return out
