"""Is the reranker's NDCG gain real? Paired bootstrap over users.

The aux-featured ranker beat the BERT4Rec baseline by +0.0013 absolute
NDCG@10 (+0.8% relative) under the clean protocol. On 6,040 users that
may be noise. This script computes per-user NDCG@10 for both orderings
and bootstrap-resamples users (paired, 10,000 resamples) to produce a
confidence interval on the mean delta. If the 95% CI includes zero, the
honest claim is "statistically indistinguishable", and the production
decision defaults to the simpler system (serve the retriever's order).

Usage:
    python scripts/significance_test.py \
        --data data/ml-1m/ratings.dat \
        --checkpoint checkpoints/bert4rec_best.pt \
        --ranker checkpoints/ranker.txt \
        --embeddings data/content_embeddings.npz \
        --movies data/ml-1m/movies.dat --users data/ml-1m/users.dat
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb  # noqa: E402

from feedforge.content import ContentIndex  # noqa: E402
from feedforge.data import build_sequences, load_movielens_1m  # noqa: E402
from feedforge.features import (  # noqa: E402
    AuxFeatureContext,
    build_cooccurrence_pmi,
    load_movie_features,
    load_user_features,
)
from feedforge.model import BERT4Rec  # noqa: E402
from feedforge.ranker import (  # noqa: E402
    build_item_popularity,
    build_ranking_dataset,
    rerank,
)
from scripts.train_ranker import candidates_with_scores  # noqa: E402


def per_user_ndcg(ranked: list[list[int]], targets: list[int], k: int = 10) -> np.ndarray:
    out = np.zeros(len(targets))
    for i, (lst, t) in enumerate(zip(ranked, targets)):
        if t in lst[:k]:
            out[i] = 1.0 / math.log2(lst.index(t) + 2)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ranker", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--movies", required=True)
    p.add_argument("--users", required=True)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    split = build_sequences(load_movielens_1m(args.data))
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    model = BERT4Rec(
        vocab_size=ckpt["vocab_size"], max_len=cfg["max_len"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], d_ff=cfg["d_model"] * 4,
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])

    te_hist = [tr + [v] for tr, v in zip(split.train, split.valid_target)]
    te_cands, te_scores = candidates_with_scores(
        model, te_hist, split.mask_token, cfg["max_len"], args.k, args.device)

    item_pop = build_item_popularity(split.train, split.n_items)
    cindex = ContentIndex.from_npz(args.embeddings, split.item_id_map)
    genre_mat, years = load_movie_features(args.movies, split.item_id_map, split.n_items)
    user_demo = load_user_features(args.users, split.user_ids)
    pmi = build_cooccurrence_pmi(split.train, split.n_items)
    aux_ctx = AuxFeatureContext(genre_mat, years, user_demo, pmi)

    test_ds = build_ranking_dataset(te_cands, te_scores, te_hist,
                                    split.test_target, item_pop, cindex, aux_ctx)
    booster = lgb.Booster(model_file=args.ranker)
    reranked = rerank(booster, test_ds)

    base = per_user_ndcg(te_cands, split.test_target)
    rank = per_user_ndcg(reranked, split.test_target)
    delta = rank - base
    n = len(delta)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(args.resamples, n))
    boot_means = delta[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    # One-sided p: fraction of resamples where the mean delta <= 0
    p_leq_zero = float((boot_means <= 0).mean())

    report = {
        "n_users": n,
        "mean_ndcg10_baseline": float(base.mean()),
        "mean_ndcg10_reranked": float(rank.mean()),
        "mean_delta": float(delta.mean()),
        "delta_95ci": [float(ci_low), float(ci_high)],
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        "bootstrap_p_delta_leq_0": p_leq_zero,
        "users_improved": int((delta > 0).sum()),
        "users_hurt": int((delta < 0).sum()),
        "users_unchanged": int((delta == 0).sum()),
    }
    print(json.dumps(report, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path("results/ranker_significance.json").write_text(json.dumps(report, indent=2))
    print("\nsaved results/ranker_significance.json")


if __name__ == "__main__":
    main()
