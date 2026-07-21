"""Cold-start experiment: how much history does personalization need?

Real users' histories are truncated to k items to simulate a visitor who
has just told us k things they like. Each strategy then predicts that
user's true held-out next item, so the comparison uses real ground truth
rather than a synthetic proxy.

Strategies:
    popular    most-watched overall (needs nothing; the floor to beat)
    sequence   BERT4Rec conditioned on just the k picks
    hybrid     BERT4Rec blended with a genre prior from the k picks

The question this answers is the one a recsys interviewer asks
immediately: at what history length does your model actually beat
recommending the most popular items? Reporting the k where the crossover
happens is more useful than a single accuracy number.

    python scripts/eval_coldstart.py --data data/ml-1m/ratings.dat \
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
    blend_scores,
    genre_affinity_scores,
    popular_items,
)
from feedforge.features import load_movie_features  # noqa: E402
from feedforge.model import BERT4Rec  # noqa: E402
from feedforge.ranker import build_item_popularity  # noqa: E402


@torch.no_grad()
def model_scores_for(model, histories, mask_token, max_len, device, batch_size=256):
    """Full-catalog scores per user, seen items masked out."""
    out = []
    for start in range(0, len(histories), batch_size):
        batch = histories[start : start + batch_size]
        tokens = inference_batch(batch, mask_token, max_len).to(device)
        scores = model.score_last_position(tokens)
        scores[:, PAD] = -float("inf")
        scores[:, mask_token] = -float("inf")
        for i, hist in enumerate(batch):
            scores[i, torch.tensor(hist, device=device)] = -float("inf")
        out.append(scores.cpu().numpy())
    return np.concatenate(out)


def topk_from_scores(scores: np.ndarray, k: int) -> list[int]:
    idx = np.argpartition(-scores, k)[:k]
    return [int(i) for i in idx[np.argsort(-scores[idx])]]


def metrics_at_k(pred_lists, targets, ks=(10, 20)) -> dict:
    out = {}
    n = len(targets)
    for k in ks:
        hits = ndcg = 0.0
        for lst, t in zip(pred_lists, targets):
            if t in lst[:k]:
                hits += 1
                ndcg += 1.0 / math.log2(lst.index(t) + 2)
        out[f"recall@{k}"] = hits / n
        out[f"ndcg@{k}"] = ndcg / n
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--movies", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--history-sizes", default="1,3,5,10,20")
    p.add_argument("--blend-weight", type=float, default=0.3)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    split = build_sequences(load_movielens_1m(args.data))
    genre_mat, _years = load_movie_features(args.movies, split.item_id_map, split.n_items)
    item_pop = build_item_popularity(split.train, split.n_items)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    model = BERT4Rec(
        vocab_size=ckpt["vocab_size"], max_len=cfg["max_len"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        d_ff=cfg.get("d_ff") or cfg["d_model"] * 4,
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    targets = split.test_target
    full_hist = [tr + [v] for tr, v in zip(split.train, split.valid_target)]
    results = {}

    for k in [int(x) for x in args.history_sizes.split(",")]:
        # A "new user" who has named their k most recent items
        truncated = [h[-k:] for h in full_hist]
        print(f"history size {k}...")

        pop_list = popular_items(item_pop, n=20)
        pop_preds = [[i for i in pop_list if i not in set(h)][:20] for h in truncated]

        scores = model_scores_for(model, truncated, split.mask_token,
                                  cfg["max_len"], args.device)
        seq_preds = [topk_from_scores(s, 20) for s in scores]

        hybrid_preds = []
        for s, hist in zip(scores, truncated):
            prior = genre_affinity_scores(hist, genre_mat)
            prior = np.pad(prior, (0, len(s) - len(prior)))[: len(s)]
            blended = blend_scores(s, prior, weight=args.blend_weight)
            blended[list(hist)] = -np.inf
            hybrid_preds.append(topk_from_scores(blended, 20))

        results[f"history_{k}"] = {
            "popular": metrics_at_k(pop_preds, targets),
            "sequence": metrics_at_k(seq_preds, targets),
            "hybrid": metrics_at_k(hybrid_preds, targets),
        }
        for name, m in results[f"history_{k}"].items():
            print(f"  {name:9s} recall@10={m['recall@10']:.4f} ndcg@10={m['ndcg@10']:.4f}")

    # Where does personalization overtake popularity?
    crossover = None
    for key in results:
        if results[key]["sequence"]["recall@10"] > results[key]["popular"]["recall@10"]:
            crossover = int(key.split("_")[1])
            break
    results["personalization_beats_popularity_from_history_size"] = crossover

    Path("results").mkdir(exist_ok=True)
    Path("results/coldstart_experiment.json").write_text(json.dumps(results, indent=2))
    print(f"\npersonalization overtakes popularity at history size: {crossover}")
    print("saved results/coldstart_experiment.json")


if __name__ == "__main__":
    main()
