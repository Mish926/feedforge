"""Candidate fusion and the Phase 2 experiment.

Reciprocal Rank Fusion (RRF) merges the collaborative (BERT4Rec) and
content (ViT/FAISS) candidate lists: each item scores
sum(1 / (rrf_k + rank_in_list)) over the lists it appears in. RRF is
scale-free -- it never compares a transformer logit to a cosine
similarity, only ranks -- which is exactly why it's the standard first
choice for merging heterogeneous retrievers.

The experiment that justifies (or kills) the content leg is candidate
recall: does hybrid retrieval put the held-out item inside the top-K
candidate set more often than collaborative-only? Candidate recall@K for
K around 100 is the metric that matters for a candidate generator -- the
ranker can only rank what generation surfaces. The honest possible
outcome, documented rather than hidden if it happens: on ML-1M, content
may add little for warm items and help mainly when collaborative signal
is thin (short histories, unpopular targets), so results are also
reported split by target-item popularity bucket.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def reciprocal_rank_fusion(
    lists: Sequence[Sequence[int]],
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[int]:
    """Merge ranked lists of item ids. rrf_k=60 is the value from the
    original RRF paper (Cormack et al.) and is deliberately not tuned:
    tuning it on the test set would be leakage, and candidate generation
    is not sensitive to it in a way that survives honest validation."""
    scores: dict[int, float] = defaultdict(float)
    for lst in lists:
        for rank, item in enumerate(lst):
            scores[item] += 1.0 / (rrf_k + rank)
    fused = sorted(scores, key=lambda i: (-scores[i], i))
    return fused[:top_k] if top_k else fused


def candidate_recall(
    candidate_lists: Sequence[Sequence[int]],
    targets: Sequence[int],
    ks: tuple[int, ...] = (20, 50, 100),
) -> dict:
    """Fraction of users whose held-out target appears in their top-K
    candidates."""
    out = {}
    n = len(targets)
    for k in ks:
        hits = sum(1 for cands, t in zip(candidate_lists, targets) if t in cands[:k])
        out[f"cand_recall@{k}"] = hits / n
    return out


def popularity_buckets(
    train: Sequence[Sequence[int]],
    targets: Sequence[int],
    n_buckets: int = 3,
) -> list[int]:
    """Assign each target a popularity bucket (0 = coldest tercile) based
    on training interaction counts, for the split analysis."""
    from collections import Counter

    counts = Counter(item for seq in train for item in seq)
    target_counts = [counts.get(t, 0) for t in targets]
    sorted_counts = sorted(target_counts)
    cut = [sorted_counts[int(len(sorted_counts) * (i + 1) / n_buckets) - 1]
           for i in range(n_buckets)]
    buckets = []
    for c in target_counts:
        for b, edge in enumerate(cut):
            if c <= edge:
                buckets.append(b)
                break
    return buckets
