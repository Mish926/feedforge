"""Final metrics report for a trained BERT4Rec checkpoint.

Reports, on the untouched TEST target (last item per user, conditioning
on train + validation history, standard leave-one-out):
  1. Full-ranking Recall@10/20 and NDCG@10/20 -- the headline numbers.
  2. Sampled-100-negatives Recall@10/NDCG@10 -- the protocol the
     BERT4Rec paper used, reported ONLY to quantify how much it inflates
     results relative to full ranking (Krichene & Rendle, KDD 2020).
  3. The popularity baseline under the same protocol.

Usage:
    python scripts/report_metrics.py \
        --data data/ml-1m/ratings.dat \
        --checkpoint checkpoints/bert4rec_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feedforge.data import build_sequences, load_movielens_1m  # noqa: E402
from feedforge.evaluate import (  # noqa: E402
    full_ranking_metrics,
    popularity_baseline,
    sampled_metrics,
)
from feedforge.model import BERT4Rec  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    split = build_sequences(load_movielens_1m(args.data))
    histories = [tr + [v] for tr, v in zip(split.train, split.valid_target)]
    targets = split.test_target

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    model = BERT4Rec(
        vocab_size=ckpt["vocab_size"], max_len=cfg["max_len"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], d_ff=cfg["d_model"] * 4,
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])

    full = full_ranking_metrics(
        model, histories, targets, split.n_items, split.mask_token,
        cfg["max_len"], device=args.device,
    )
    sampled = sampled_metrics(
        model, histories, targets, split.n_items, split.mask_token,
        cfg["max_len"], n_negatives=100, device=args.device,
    )
    pop = popularity_baseline(split.train, histories, targets, split.n_items)

    inflation = sampled["sampled100_recall@10"] / max(full["recall@10"], 1e-12)
    report = {
        "test_full_ranking": full,
        "test_sampled_100_negatives": sampled,
        "popularity_baseline_full_ranking": pop,
        "sampled_vs_full_recall10_inflation": round(inflation, 2),
        "checkpoint_valid_metrics": ckpt.get("valid_metrics"),
    }
    print(json.dumps(report, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path("results/final_metrics.json").write_text(json.dumps(report, indent=2))
    print("\nsaved results/final_metrics.json")


if __name__ == "__main__":
    main()
