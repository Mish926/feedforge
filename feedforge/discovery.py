"""Cold start and diversity: the two things the base pipeline can't do.

COLD START
The trained BERT4Rec conditions on a user's history, so a brand-new user
with no history has nothing to condition on. Three strategies are
implemented, in increasing order of what they need from the visitor:

  popular    most-watched items overall. The honest floor: it needs
             nothing at all, and any personalization must beat it.
  sequence   treat the visitor's picks as a (very short) history and run
             them through BERT4Rec exactly as if they were a real user.
             The transformer was trained with masked positions and
             variable-length sequences, so a 3-item sequence is in
             distribution, just weakly informative.
  hybrid     blend the sequence scores with a genre-affinity prior built
             from the picks, which stabilizes very short histories where
             the transformer is least confident.

The interesting measurement (scripts/eval_coldstart.py) simulates cold
start by truncating real users' histories to k items and comparing the
strategies against those users' true held-out next item.

DIVERSITY
Pure relevance ranking returns near-duplicates: ten similar comedies from
the same era. Maximal Marginal Relevance re-ranks for
    score = lambda * relevance - (1 - lambda) * max_similarity_to_selected
where similarity is genre-vector cosine (content-based, always available)
optionally blended with poster-embedding cosine. lambda = 1 is pure
relevance; lower values trade accuracy for variety. The point isn't that
diversity is free -- it isn't -- but that the tradeoff is measurable, so
the operating point becomes a decision instead of an accident.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# -- cold start ------------------------------------------------------------

def popular_items(item_pop: np.ndarray, n: int, exclude: Sequence[int] = ()) -> list[int]:
    """Most-interacted items, excluding anything the visitor named."""
    order = np.argsort(-item_pop)
    seen = set(int(i) for i in exclude)
    out = []
    for item in order:
        item = int(item)
        if item == 0 or item in seen:
            continue
        out.append(item)
        if len(out) == n:
            break
    return out


def genre_affinity_scores(picks: Sequence[int], genre_mat: np.ndarray) -> np.ndarray:
    """Cosine similarity between every item's genre vector and the mean
    genre vector of the visitor's picks. Returns (n_items+1,) with 0 for
    items lacking genre data."""
    if len(picks) == 0:
        return np.zeros(genre_mat.shape[0], dtype=np.float32)
    profile = genre_mat[list(picks)].mean(axis=0)
    pnorm = np.linalg.norm(profile)
    if pnorm == 0:
        return np.zeros(genre_mat.shape[0], dtype=np.float32)
    norms = np.linalg.norm(genre_mat, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    return (genre_mat @ profile) / (safe * pnorm)


def blend_scores(
    model_scores: np.ndarray,
    prior_scores: np.ndarray,
    weight: float = 0.3,
) -> np.ndarray:
    """Combine transformer scores with a genre prior. Both are rank-
    normalized first: raw logits and cosine similarities live on
    incomparable scales, and normalizing by rank rather than value avoids
    one signal's outliers dominating the blend."""
    def rank_norm(x: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(-x))
        return 1.0 - order / max(len(x) - 1, 1)

    return (1 - weight) * rank_norm(model_scores) + weight * rank_norm(prior_scores)


# -- diversity -------------------------------------------------------------

def mmr_rerank(
    candidates: Sequence[int],
    relevance: Sequence[float],
    similarity_fn,
    lambda_: float = 0.7,
    n: int = 10,
) -> list[int]:
    """Maximal Marginal Relevance selection.

    similarity_fn(a, b) -> float in [0, 1]. Greedy: repeatedly pick the
    candidate maximizing lambda*rel - (1-lambda)*max_sim_to_already_picked.
    O(n * |candidates|) similarity calls, which at n=10 and 100 candidates
    is trivial, and is why greedy MMR is what production systems actually
    run rather than an exact diversification.
    """
    if not candidates:
        return []
    rel = np.asarray(relevance, dtype=np.float64)
    # Rank-normalize relevance to [0,1] so lambda has consistent meaning
    order = np.argsort(np.argsort(-rel))
    rel = 1.0 - order / max(len(rel) - 1, 1)

    remaining = list(range(len(candidates)))
    selected: list[int] = []
    while remaining and len(selected) < n:
        best_idx, best_score = None, -np.inf
        for idx in remaining:
            if selected:
                max_sim = max(similarity_fn(candidates[idx], candidates[s])
                              for s in selected)
            else:
                max_sim = 0.0
            score = lambda_ * rel[idx] - (1 - lambda_) * max_sim
            if score > best_score:
                best_score, best_idx = score, idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidates[i] for i in selected]


def make_genre_similarity(genre_mat: np.ndarray):
    """Cosine similarity over genre vectors, in [0,1]."""
    norms = np.linalg.norm(genre_mat, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    normed = genre_mat / safe[:, None]

    def sim(a: int, b: int) -> float:
        return float(np.dot(normed[a], normed[b]))

    return sim


def intra_list_similarity(items: Sequence[int], similarity_fn) -> float:
    """Mean pairwise similarity within a recommendation list: the standard
    diversity metric (lower = more diverse)."""
    if len(items) < 2:
        return 0.0
    sims = [similarity_fn(items[i], items[j])
            for i in range(len(items)) for j in range(i + 1, len(items))]
    return float(np.mean(sims))


def genre_coverage(items: Sequence[int], genre_mat: np.ndarray) -> int:
    """How many distinct genres a list touches."""
    if not items:
        return 0
    return int((genre_mat[list(items)].sum(axis=0) > 0).sum())


# -- popularity bias -------------------------------------------------------

def popularity_buckets_by_item(item_pop: np.ndarray, n_buckets: int = 3) -> np.ndarray:
    """Assign every item a popularity bucket, 0 = long tail. Buckets are
    equal-mass in interactions, not equal-count in items, which is the
    standard construction: the head is a handful of items carrying a third
    of all traffic."""
    order = np.argsort(-item_pop)
    total = item_pop.sum()
    buckets = np.zeros(len(item_pop), dtype=np.int8)
    cum, edge = 0.0, 0
    for item in order:
        if item_pop[item] == 0:
            buckets[item] = 0
            continue
        cum += item_pop[item]
        buckets[item] = n_buckets - 1 - edge
        if cum >= total * (edge + 1) / n_buckets and edge < n_buckets - 1:
            edge += 1
    return buckets


def exposure_distribution(
    recommendation_lists: Sequence[Sequence[int]],
    buckets: np.ndarray,
    n_buckets: int = 3,
) -> dict:
    """What share of recommended slots goes to each popularity bucket.
    Compared against the catalog's own bucket shares, this quantifies
    popularity bias: a model that recommends the head 90% of the time
    while the head is 33% of interactions is amplifying, not reflecting,
    existing popularity."""
    counts = np.zeros(n_buckets)
    for lst in recommendation_lists:
        for item in lst:
            counts[buckets[item]] += 1
    total = counts.sum()
    return {f"bucket{b}": float(counts[b] / total) if total else 0.0
            for b in range(n_buckets)}


def catalog_distribution(item_pop: np.ndarray, buckets: np.ndarray,
                         n_buckets: int = 3) -> dict:
    """Interaction share per bucket -- the reference the exposure
    distribution is judged against."""
    counts = np.zeros(n_buckets)
    for item in range(len(item_pop)):
        counts[buckets[item]] += item_pop[item]
    total = counts.sum()
    return {f"bucket{b}": float(counts[b] / total) if total else 0.0
            for b in range(n_buckets)}
