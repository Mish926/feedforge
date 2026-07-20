"""Phase 3: train and evaluate the LambdaRank reranker.

Trains on the validation task (history = train, target = valid item),
evaluates on the test task (history = train + valid, target = test
item). The baseline to beat is BERT4Rec's own candidate ordering -- the
reranker's strongest feature is that ordering, so any reported gain is
value added on top of the retriever, not a strawman comparison. The
candidate recall@100 from Phase 2 (0.669) is the ceiling: a target
outside the candidate set is unrankable by construction.

Usage:
    python scripts/train_ranker.py \
        --data data/ml-1m/ratings.dat \
        --checkpoint checkpoints/bert4rec_best.pt \
        --embeddings data/content_embeddings.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feedforge.content import ContentIndex  # noqa: E402
from feedforge.data import PAD, build_sequences, inference_batch, load_movielens_1m  # noqa: E402
from feedforge.model import BERT4Rec  # noqa: E402
from feedforge.ranker import (  # noqa: E402
    FEATURE_NAMES,
    build_item_popularity,
    build_ranking_dataset,
    ranked_metrics,
    rerank,
    train_ranker,
)


@torch.no_grad()
def candidates_with_scores(model, histories, mask_token, max_len, k, device, batch_size=256):
    model.eval()
    cands, cscores = [], []
    for start in range(0, len(histories), batch_size):
        hist_b = histories[start : start + batch_size]
        tokens = inference_batch(hist_b, mask_token, max_len).to(device)
        scores = model.score_last_position(tokens)
        scores[:, PAD] = -float("inf")
        scores[:, mask_token] = -float("inf")
        for i, hist in enumerate(hist_b):
            scores[i, torch.tensor(hist, device=device)] = -float("inf")
        top = torch.topk(scores, k, dim=1)
        cands.extend(top.indices.cpu().tolist())
        cscores.extend(top.values.cpu().tolist())
    return cands, cscores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--embeddings", default=None,
                   help="content_embeddings.npz; omit to ablate content features")
    p.add_argument("--k", type=int, default=100)
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

    item_pop = build_item_popularity(split.train, split.n_items)
    cindex = ContentIndex.from_npz(args.embeddings, split.item_id_map) if args.embeddings else None

    # Training task: history = train, target = valid. 10% of users are
    # held out for early stopping so the test set is never consulted
    # during model selection (early-stopping on test is leakage, and the
    # first version of this script did exactly that -- fixed and noted).
    print("candidates for training task...")
    tr_hist = split.train
    tr_cands, tr_scores = candidates_with_scores(
        model, tr_hist, split.mask_token, cfg["max_len"], args.k, args.device)

    rng = np.random.default_rng(42)
    n_users = len(tr_hist)
    es_idx = set(rng.choice(n_users, size=max(1, n_users // 10), replace=False).tolist())
    fit = [i for i in range(n_users) if i not in es_idx]
    es = sorted(es_idx)

    def subset(idx):
        return build_ranking_dataset(
            [tr_cands[i] for i in idx], [tr_scores[i] for i in idx],
            [tr_hist[i] for i in idx], [split.valid_target[i] for i in idx],
            item_pop, cindex)

    train_ds = subset(fit)
    earlystop_ds = subset(es)

    # Test task: history = train + valid, target = test
    print("candidates for test task...")
    te_hist = [tr + [v] for tr, v in zip(split.train, split.valid_target)]
    te_cands, te_scores = candidates_with_scores(
        model, te_hist, split.mask_token, cfg["max_len"], args.k, args.device)
    test_ds = build_ranking_dataset(te_cands, te_scores, te_hist,
                                    split.test_target, item_pop, cindex)

    print("training LambdaRank...")
    booster = train_ranker(train_ds, valid_ds=earlystop_ds)

    baseline = ranked_metrics(te_cands, split.test_target)
    reranked = rerank(booster, test_ds)
    ranked = ranked_metrics(reranked, split.test_target)

    importance = dict(zip(FEATURE_NAMES,
                          booster.feature_importance("gain").round(1).tolist()))
    cand_ceiling = float(np.mean([t in c for c, t in zip(te_cands, split.test_target)]))

    report = {
        "candidate_recall_at_k_ceiling": cand_ceiling,
        "baseline_bert4rec_order": baseline,
        "reranked_lambdarank": ranked,
        "feature_importance_gain": importance,
        "best_iteration": booster.best_iteration,
        "content_features_enabled": cindex is not None,
    }
    print(json.dumps(report, indent=2))
    Path("results").mkdir(exist_ok=True)
    out = "results/ranker_experiment.json" if cindex else "results/ranker_experiment_nocontent.json"
    Path(out).write_text(json.dumps(report, indent=2))
    booster.save_model("checkpoints/ranker.txt")
    print(f"\nsaved {out} and checkpoints/ranker.txt")


if __name__ == "__main__":
    main()
