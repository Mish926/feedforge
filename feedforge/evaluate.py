"""Evaluation for FeedForge candidate generation.

The core protocol is FULL RANKING: the held-out target item is ranked
against every item in the catalog (minus the user's already-seen items),
and Recall@K / NDCG@K are computed from that rank. This is the honest
number.

sampled_metrics() implements the common alternative -- rank the target
against 100 random negatives -- for one purpose: demonstrating side by
side how much that protocol inflates results (Krichene & Rendle, "On
Sampled Metrics for Item Recommendation", KDD 2020). On ML-1M the
sampled Recall@10 is roughly 3x the full-ranking number. FeedForge
reports full-ranking metrics everywhere; the sampled numbers exist to be
disbelieved, and to explain why our reported Recall@10 looks lower than
the BERT4Rec paper's.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from .data import PAD, inference_batch
from .model import BERT4Rec


@torch.no_grad()
def full_ranking_metrics(
    model: BERT4Rec,
    histories: list[list[int]],
    targets: list[int],
    n_items: int,
    mask_token: int,
    max_len: int,
    ks: tuple[int, ...] = (10, 20),
    batch_size: int = 256,
    device: str = "cpu",
) -> dict:
    """Rank each user's target against the full catalog.

    histories: the sequence the model may condition on (must NOT include
    the target). Items already in the history are excluded from ranking
    ("filter seen"), the standard protocol: recommending something the
    user already consumed isn't a win, and every published baseline
    filters it.
    """
    model.eval()
    recalls = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    n_users = len(histories)

    for start in range(0, n_users, batch_size):
        hist_b = histories[start : start + batch_size]
        targ_b = targets[start : start + batch_size]

        tokens = inference_batch(hist_b, mask_token, max_len).to(device)
        scores = model.score_last_position(tokens)  # (B, V)

        # Never recommend PAD or MASK
        scores[:, PAD] = -float("inf")
        scores[:, mask_token] = -float("inf")
        # Filter seen items per user
        for i, hist in enumerate(hist_b):
            scores[i, torch.tensor(hist, device=device)] = -float("inf")

        # Rank of the target: number of items scoring strictly higher
        target_idx = torch.tensor(targ_b, device=device)
        target_scores = scores.gather(1, target_idx.unsqueeze(1))
        ranks = (scores > target_scores).sum(dim=1)  # 0-based rank

        for k in ks:
            hit = ranks < k
            recalls[k] += int(hit.sum())
            ndcgs[k] += float((1.0 / torch.log2(ranks[hit].float() + 2)).sum())

    out = {}
    for k in ks:
        out[f"recall@{k}"] = recalls[k] / n_users
        out[f"ndcg@{k}"] = ndcgs[k] / n_users
    return out


@torch.no_grad()
def sampled_metrics(
    model: BERT4Rec,
    histories: list[list[int]],
    targets: list[int],
    n_items: int,
    mask_token: int,
    max_len: int,
    n_negatives: int = 100,
    k: int = 10,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """The protocol we DON'T trust: target vs n_negatives random unseen
    items. Provided only for the inflation comparison."""
    model.eval()
    rng = random.Random(seed)
    hits, ndcg_sum = 0, 0.0
    n_users = len(histories)

    for start in range(0, n_users, batch_size):
        hist_b = histories[start : start + batch_size]
        targ_b = targets[start : start + batch_size]
        tokens = inference_batch(hist_b, mask_token, max_len).to(device)
        scores = model.score_last_position(tokens)

        for i, (hist, targ) in enumerate(zip(hist_b, targ_b)):
            seen = set(hist) | {targ}
            negs = []
            while len(negs) < n_negatives:
                cand = rng.randint(1, n_items)
                if cand not in seen:
                    negs.append(cand)
            candidates = torch.tensor([targ] + negs, device=device)
            cand_scores = scores[i, candidates]
            rank = int((cand_scores > cand_scores[0]).sum())
            if rank < k:
                hits += 1
                ndcg_sum += 1.0 / math.log2(rank + 2)

    return {f"sampled{n_negatives}_recall@{k}": hits / n_users,
            f"sampled{n_negatives}_ndcg@{k}": ndcg_sum / n_users}


def popularity_baseline(
    train: list[list[int]],
    histories: list[list[int]],
    targets: list[int],
    n_items: int,
    ks: tuple[int, ...] = (10, 20),
) -> dict:
    """Most-popular-unseen baseline. Any model that can't clearly beat
    this is memorizing popularity, a known failure mode on ML-1M."""
    counts = np.zeros(n_items + 1)
    for seq in train:
        for item in seq:
            counts[item] += 1
    pop_order = np.argsort(-counts)

    recalls = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    for hist, targ in zip(histories, targets):
        seen = set(hist)
        rank = 0
        for item in pop_order:
            if item == 0 or item in seen:
                continue
            if item == targ:
                break
            rank += 1
        for k in ks:
            if rank < k:
                recalls[k] += 1
                ndcgs[k] += 1.0 / math.log2(rank + 2)
    n = len(targets)
    out = {}
    for k in ks:
        out[f"recall@{k}"] = recalls[k] / n
        out[f"ndcg@{k}"] = ndcgs[k] / n
    return out
