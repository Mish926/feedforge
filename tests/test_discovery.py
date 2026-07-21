"""Tests for cold start, diversity, and popularity-bias measurement.

Synthetic worlds with known structure so each property is checked
against a ground truth rather than a snapshot.
"""

import numpy as np
import pytest

from feedforge.discovery import (
    blend_scores,
    catalog_distribution,
    exposure_distribution,
    genre_affinity_scores,
    genre_coverage,
    intra_list_similarity,
    make_genre_similarity,
    mmr_rerank,
    popular_items,
    popularity_buckets_by_item,
)

N_ITEMS = 30
N_GENRES = 6


@pytest.fixture
def genre_mat():
    """Items 1-10 genre A, 11-20 genre B, 21-30 genre C."""
    g = np.zeros((N_ITEMS + 1, N_GENRES), dtype=np.float32)
    g[1:11, 0] = 1.0
    g[11:21, 1] = 1.0
    g[21:31, 2] = 1.0
    return g


@pytest.fixture
def item_pop():
    pop = np.zeros(N_ITEMS + 1)
    pop[1:11] = 100.0   # head
    pop[11:21] = 10.0   # middle
    pop[21:31] = 1.0    # tail
    return pop


# -- cold start ------------------------------------------------------------

def test_popular_items_orders_and_excludes(item_pop):
    top = popular_items(item_pop, n=5, exclude=[1, 2])
    assert len(top) == 5
    assert 1 not in top and 2 not in top
    assert all(item_pop[i] == 100.0 for i in top)


def test_genre_affinity_prefers_matching_genre(genre_mat):
    scores = genre_affinity_scores([1, 2, 3], genre_mat)  # genre A picks
    assert scores[5] > 0.9      # another genre A item
    assert scores[15] < 0.1     # genre B item
    assert scores[25] < 0.1     # genre C item


def test_genre_affinity_empty_picks(genre_mat):
    assert not genre_affinity_scores([], genre_mat).any()


def test_blend_moves_toward_prior():
    model = np.array([5.0, 4.0, 3.0, 2.0, 1.0])   # prefers index 0
    prior = np.array([1.0, 2.0, 3.0, 4.0, 5.0])   # prefers index 4
    pure = blend_scores(model, prior, weight=0.0)
    mixed = blend_scores(model, prior, weight=0.5)
    assert np.argmax(pure) == 0
    # With equal weight the two disagree symmetrically; the model's top
    # item must lose ground relative to the pure-model blend.
    assert mixed[0] - mixed[4] < pure[0] - pure[4]


def test_blend_is_scale_invariant():
    """Rank normalization means multiplying one signal by 1000 changes
    nothing -- the reason the blend uses ranks, not raw values."""
    model = np.array([5.0, 4.0, 3.0])
    prior = np.array([0.1, 0.2, 0.3])
    a = blend_scores(model, prior, 0.5)
    b = blend_scores(model * 1000, prior, 0.5)
    assert np.allclose(a, b)


# -- diversity -------------------------------------------------------------

def test_mmr_lambda_one_preserves_relevance_order(genre_mat):
    sim = make_genre_similarity(genre_mat)
    cands = [1, 2, 3, 11, 21]
    rel = [5.0, 4.0, 3.0, 2.0, 1.0]
    out = mmr_rerank(cands, rel, sim, lambda_=1.0, n=5)
    assert out == cands


def test_mmr_diversifies_when_lambda_low(genre_mat):
    """Top-3 by relevance are all genre A; MMR must reach for other
    genres once diversity is weighted."""
    sim = make_genre_similarity(genre_mat)
    cands = [1, 2, 3, 11, 21]
    rel = [5.0, 4.9, 4.8, 1.0, 0.5]
    greedy = mmr_rerank(cands, rel, sim, lambda_=1.0, n=3)
    diverse = mmr_rerank(cands, rel, sim, lambda_=0.4, n=3)
    assert genre_coverage(greedy, genre_mat) == 1
    assert genre_coverage(diverse, genre_mat) >= 2
    assert intra_list_similarity(diverse, sim) < intra_list_similarity(greedy, sim)


def test_mmr_returns_permutation_subset(genre_mat):
    sim = make_genre_similarity(genre_mat)
    cands = [1, 5, 11, 15, 21, 25]
    out = mmr_rerank(cands, [6, 5, 4, 3, 2, 1], sim, lambda_=0.6, n=4)
    assert len(out) == 4
    assert len(set(out)) == 4
    assert set(out) <= set(cands)


def test_intra_list_similarity_extremes(genre_mat):
    sim = make_genre_similarity(genre_mat)
    assert intra_list_similarity([1, 2, 3], sim) == pytest.approx(1.0)   # same genre
    assert intra_list_similarity([1, 11, 21], sim) == pytest.approx(0.0) # disjoint
    assert intra_list_similarity([1], sim) == 0.0


# -- popularity bias -------------------------------------------------------

def test_buckets_are_equal_mass_not_equal_count(item_pop):
    b = popularity_buckets_by_item(item_pop)
    # Highest-popularity item is in the top bucket, lowest in the bottom
    assert b[1] == 2
    assert b[30] == 0
    # Equal-MASS construction: the head bucket holds far fewer ITEMS than
    # the tail bucket while carrying a comparable share of interactions.
    head_items = [i for i in range(1, N_ITEMS + 1) if b[i] == 2]
    tail_items = [i for i in range(1, N_ITEMS + 1) if b[i] == 0]
    assert len(head_items) < len(tail_items)
    head_mass = item_pop[head_items].sum()
    tail_mass = item_pop[tail_items].sum()
    assert head_mass == pytest.approx(tail_mass, rel=0.5)


def test_exposure_vs_catalog_detects_amplification(item_pop):
    b = popularity_buckets_by_item(item_pop)
    # Recommend only the single most popular item: maximal amplification
    head_only = [[1]] * 20
    exposure = exposure_distribution(head_only, b)
    catalog = catalog_distribution(item_pop, b)
    assert exposure["bucket2"] == 1.0
    assert exposure["bucket2"] > catalog["bucket2"]
    assert exposure["bucket0"] == 0.0

    # A tail-only recommender shows the opposite bias
    tail_only = [[30]] * 20
    tail_exposure = exposure_distribution(tail_only, b)
    assert tail_exposure["bucket0"] == 1.0
    assert tail_exposure["bucket0"] > catalog["bucket0"]


def test_exposure_sums_to_one(item_pop):
    b = popularity_buckets_by_item(item_pop)
    lists = [[1, 11, 21], [2, 12, 22]]
    dist = exposure_distribution(lists, b)
    assert sum(dist.values()) == pytest.approx(1.0)
