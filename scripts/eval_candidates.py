"""Phase 2 experiment: does the content leg earn its place?

Compares candidate recall@K for three strategies on the test split:
  collaborative  BERT4Rec top-K
  content        ViT/FAISS top-K from the user's recent history
  hybrid         RRF fusion of both

Reported overall and split by target-popularity tercile, because the
hypothesis is that content helps most where collaborative signal is
thinnest.

Usage (after training + fetching posters + embedding):
    python scripts/eval_candidates.py \
        --data data/ml-1m/ratings.dat \
        --checkpoint checkpoints/bert4rec_best.pt \
        --embeddings data/content_embeddings.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feedforge.content import ContentIndex  # noqa: E402
from feedforge.data import PAD, build_sequences, inference_batch, load_movielens_1m  # noqa: E402
from feedforge.fusion import (  # noqa: E402
    candidate_recall,
    popularity_buckets,
    reciprocal_rank_fusion,
)
from feedforge.model import BERT4Rec  # noqa: E402


@torch.no_grad()
def bert4rec_candidates(model, histories, mask_token, max_len, k, device, batch_size=256):
    model.eval()
    all_cands = []
    for start in range(0, len(histories), batch_size):
        hist_b = histories[start : start + batch_size]
        tokens = inference_batch(hist_b, mask_token, max_len).to(device)
        scores = model.score_last_position(tokens)
        scores[:, PAD] = -float("inf")
        scores[:, mask_token] = -float("inf")
        for i, hist in enumerate(hist_b):
            scores[i, torch.tensor(hist, device=device)] = -float("inf")
        top = torch.topk(scores, k, dim=1).indices.cpu().tolist()
        all_cands.extend(top)
    return all_cands


def bucketed(metric_fn, lists, targets, buckets):
    out = {}
    for b in sorted(set(buckets)):
        idx = [i for i, bb in enumerate(buckets) if bb == b]
        out[f"bucket{b}"] = metric_fn([lists[i] for i in idx], [targets[i] for i in idx])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    split = build_sequences(load_movielens_1m(args.data))
    # Test-time history includes the validation item (standard leave-one-out)
    histories = [tr + [v] for tr, v in zip(split.train, split.valid_target)]
    targets = split.test_target

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    model = BERT4Rec(
        vocab_size=ckpt["vocab_size"], max_len=cfg["max_len"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], d_ff=cfg["d_model"] * 4,
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])

    print("generating collaborative candidates...")
    collab = bert4rec_candidates(
        model, histories, split.mask_token, cfg["max_len"], args.k, args.device
    )
    print("generating content candidates...")
    cindex = ContentIndex.from_npz(args.embeddings, split.item_id_map)
    print(f"  content coverage: {cindex.index.ntotal}/{split.n_items} items")
    content = [cindex.candidates_for_history(h, k=args.k) for h in histories]
    hybrid = [
        reciprocal_rank_fusion([c1, c2], top_k=args.k)
        for c1, c2 in zip(collab, content)
    ]

    buckets = popularity_buckets(split.train, targets)
    results = {}
    for name, lists in [("collaborative", collab), ("content", content), ("hybrid", hybrid)]:
        overall = candidate_recall(lists, targets)
        by_bucket = bucketed(candidate_recall, lists, targets, buckets)
        results[name] = {"overall": overall, "by_popularity": by_bucket}
        print(f"\n{name}: {overall}")
        for b, m in by_bucket.items():
            print(f"  {b} (0=coldest): {m}")

    Path("results").mkdir(exist_ok=True)
    Path("results/candidate_experiment.json").write_text(json.dumps(results, indent=2))
    print("\nsaved results/candidate_experiment.json")


if __name__ == "__main__":
    main()
