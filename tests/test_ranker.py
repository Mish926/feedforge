"""Ranker tests on synthetic candidates.

The learnability test constructs a world where BERT4Rec's ordering is
deliberately mediocre (target buried mid-list) but a secondary feature
(popularity) identifies the target perfectly. A working LambdaRank must
learn to promote the target well above the baseline ordering; a broken
grouping, labeling, or rerank alignment cannot.
"""

import numpy as np
import pytest

from feedforge.ranker import (
    build_item_popularity,
    build_ranking_dataset,
    ranked_metrics,
    rerank,
    train_ranker,
)

N_USERS = 400
K = 20
N_ITEMS = 500
RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def synthetic_world():
    """Targets always sit at candidate rank 10-15 (bad collab order) but
    have distinctive high popularity (the learnable signal)."""
    train_seqs = []
    candidates, scores, histories, targets = [], [], [], []
    # Popular "target pool" items get big train counts
    target_pool = list(range(1, 51))
    for item in target_pool:
        train_seqs.append([item] * 40)
    for _ in range(N_USERS):
        target = int(RNG.choice(target_pool))
        others = RNG.choice(np.arange(51, N_ITEMS), size=K - 1, replace=False).tolist()
        pos = int(RNG.integers(10, 16))
        cands = others[:pos] + [target] + others[pos:]
        candidates.append(cands)
        scores.append(list(np.linspace(5.0, 0.0, K)))  # monotone fake logits
        histories.append(RNG.choice(np.arange(51, N_ITEMS), size=8, replace=False).tolist())
        targets.append(target)
    item_pop = build_item_popularity(train_seqs, N_ITEMS)
    return candidates, scores, histories, targets, item_pop


def test_dataset_shapes_and_labels(synthetic_world):
    cands, scores, hists, targets, pop = synthetic_world
    ds = build_ranking_dataset(cands, scores, hists, targets, pop)
    assert ds.X.shape == (N_USERS * K, 7)
    assert ds.groups.sum() == N_USERS * K
    assert ds.y.sum() == N_USERS  # exactly one positive per user


def test_no_content_index_means_zero_content_features(synthetic_world):
    cands, scores, hists, targets, pop = synthetic_world
    ds = build_ranking_dataset(cands, scores, hists, targets, pop, content_index=None)
    # content_sim, content_sim_last, has_content columns all zero
    assert not ds.X[:, 2:5].any()


def test_ranker_learns_secondary_signal(synthetic_world):
    cands, scores, hists, targets, pop = synthetic_world
    half = N_USERS // 2
    train_ds = build_ranking_dataset(cands[:half], scores[:half], hists[:half],
                                     targets[:half], pop)
    test_ds = build_ranking_dataset(cands[half:], scores[half:], hists[half:],
                                    targets[half:], pop)
    booster = train_ranker(train_ds, num_boost_round=100,
                           params={"min_data_in_leaf": 10})
    baseline = ranked_metrics(cands[half:], targets[half:], ks=(10,))
    reranked = rerank(booster, test_ds)
    after = ranked_metrics(reranked, targets[half:], ks=(10,))
    # Baseline: target at rank 10-15 -> recall@10 near 0. Ranker: pop
    # feature identifies it -> near 1.
    assert baseline["recall@10"] < 0.2
    assert after["recall@10"] > 0.8, f"ranker failed to learn: {after}"


def test_rerank_preserves_candidate_sets(synthetic_world):
    cands, scores, hists, targets, pop = synthetic_world
    ds = build_ranking_dataset(cands, scores, hists, targets, pop)
    booster = train_ranker(ds, num_boost_round=10,
                           params={"min_data_in_leaf": 10})
    reranked = rerank(booster, ds)
    for orig, new in zip(cands, reranked):
        assert sorted(orig) == sorted(new)  # a permutation, nothing added/lost


def test_ranked_metrics_definition():
    ranked = [[5, 3, 9], [1, 2, 3]]
    targets = [9, 7]
    m = ranked_metrics(ranked, targets, ks=(2, 3))
    assert m["recall@2"] == 0.0
    assert m["recall@3"] == 0.5
    assert m["ndcg@3"] == pytest.approx(0.5 / np.log2(4))
