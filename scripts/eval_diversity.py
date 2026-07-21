"""Diversity/accuracy tradeoff curve and popularity-bias audit.

Sweeps the MMR lambda from 1.0 (pure relevance) down to 0.3 and measures,
at each point: NDCG@10 against the true held-out item, intra-list genre
similarity (lower = more diverse), distinct genres covered, and the share
of recommendation slots going to head/mid/tail items.

Two questions get answered:
  1. What does diversity cost? Producing the frontier turns "should we
     diversify" into an operating-point decision with a price attached.
  2. Does the model amplify popularity? Comparing exposure share against
     the catalog's own interaction share shows whether recommendations
     reflect existing popularity or concentrate it further -- the same
     kind of subgroup audit as a fairness analysis, applied to items.

    python scripts/eval_diversity.py --data data/ml-1m/ratings.dat \
        --movies data/ml-1m/movies.dat --checkpoint checkpoints/bert4rec_best.pt
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

from feedforge.data import PAD, build_sequences, inference_batch, load_movielens_1m  # noqa: E402
from feedforge.discovery import (  # noqa: E402
    catalog_distribution,
    exposure_distribution,
    genre_coverage,
    intra_list_similarity,
    make_genre_similarity,
    mmr_rerank,
    popularity_buckets_by_item,
)
from feedforge.features import load_movie_features  # noqa: E402
from feedforge.model import BERT4Rec  # noqa: E402
from feedforge.ranker import build_item_popularity  # noqa: E402


@torch.no_grad()
def candidates_with_scores(model, histories, mask_token, max_len, k, device, batch_size=256):
    cands, scores_out = [], []
    for start in range(0, len(histories), batch_size):
        batch = histories[start : start + batch_size]
        tokens = inference_batch(batch, mask_token, max_len).to(device)
        scores = model.score_last_position(tokens)
        scores[:, PAD] = -float("inf")
        scores[:, mask_token] = -float("inf")
        for i, hist in enumerate(batch):
            scores[i, torch.tensor(hist, device=device)] = -float("inf")
        top = torch.topk(scores, k, dim=1)
        cands.extend(top.indices.cpu().tolist())
        scores_out.extend(top.values.cpu().tolist())
    return cands, scores_out


def ndcg_at_k(lists, targets, k=10) -> float:
    total = 0.0
    for lst, t in zip(lists, targets):
        if t in lst[:k]:
            total += 1.0 / math.log2(lst.index(t) + 2)
    return total / len(targets)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--movies", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lambdas", default="1.0,0.9,0.8,0.7,0.6,0.5,0.3")
    p.add_argument("--candidate-k", type=int, default=50)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--sample-users", type=int, default=1500,
                   help="MMR is greedy per user; sampling keeps the sweep quick")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    split = build_sequences(load_movielens_1m(args.data))
    genre_mat, _ = load_movie_features(args.movies, split.item_id_map, split.n_items)
    item_pop = build_item_popularity(split.train, split.n_items)
    sim = make_genre_similarity(genre_mat)
    buckets = popularity_buckets_by_item(item_pop)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    model = BERT4Rec(
        vocab_size=ckpt["vocab_size"], max_len=cfg["max_len"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        d_ff=cfg.get("d_ff") or cfg["d_model"] * 4,
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = np.random.default_rng(0)
    n_users = len(split.train)
    idx = rng.choice(n_users, size=min(args.sample_users, n_users), replace=False)
    histories = [split.train[i] + [split.valid_target[i]] for i in idx]
    targets = [split.test_target[i] for i in idx]

    print(f"generating candidates for {len(histories)} users...")
    cands, scores = candidates_with_scores(
        model, histories, split.mask_token, cfg["max_len"],
        args.candidate_k, args.device)

    curve = []
    for lam in [float(x) for x in args.lambdas.split(",")]:
        lists = [mmr_rerank(c, s, sim, lambda_=lam, n=args.n)
                 for c, s in zip(cands, scores)]
        point = {
            "lambda": lam,
            "ndcg@10": round(ndcg_at_k(lists, targets), 5),
            "intra_list_similarity": round(
                float(np.mean([intra_list_similarity(l, sim) for l in lists])), 4),
            "mean_genres_covered": round(
                float(np.mean([genre_coverage(l, genre_mat) for l in lists])), 2),
            "exposure": exposure_distribution(lists, buckets),
        }
        curve.append(point)
        print(f"  lambda={lam:.1f}  ndcg@10={point['ndcg@10']:.4f}  "
              f"ILS={point['intra_list_similarity']:.3f}  "
              f"genres={point['mean_genres_covered']:.1f}")

    baseline = curve[0]
    report = {
        "n_users_sampled": len(histories),
        "candidate_pool": args.candidate_k,
        "list_size": args.n,
        "tradeoff_curve": curve,
        "catalog_popularity_distribution": catalog_distribution(item_pop, buckets),
        "note": ("exposure buckets are equal-mass in interactions: bucket2 is the "
                 "head, bucket0 the long tail. Comparing exposure against "
                 "catalog_popularity_distribution shows whether the recommender "
                 "amplifies or reflects existing popularity."),
    }
    # A useful summary line: the cheapest meaningful diversity gain
    for point in curve[1:]:
        ndcg_cost = (baseline["ndcg@10"] - point["ndcg@10"]) / baseline["ndcg@10"]
        ils_gain = (baseline["intra_list_similarity"] - point["intra_list_similarity"])
        if ils_gain > 0.05:
            report["suggested_operating_point"] = {
                "lambda": point["lambda"],
                "ndcg_relative_cost": round(ndcg_cost, 4),
                "intra_list_similarity_reduction": round(ils_gain, 4),
            }
            break

    Path("results").mkdir(exist_ok=True)
    Path("results/diversity_experiment.json").write_text(json.dumps(report, indent=2))
    print("\ncatalog popularity share:", report["catalog_popularity_distribution"])
    print("recommendation exposure at lambda=1.0:", baseline["exposure"])
    print("saved results/diversity_experiment.json")


if __name__ == "__main__":
    main()
