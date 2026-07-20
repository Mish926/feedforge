"""Phase 2 tests: content index, RRF fusion, graceful degradation.

Synthetic embeddings with known cluster structure stand in for ViT
output, so these run offline with no model download and validate the
FAISS plumbing, id mapping, seen-filtering, and fusion math exactly.
"""

import numpy as np
import pytest

from feedforge.content import ContentIndex
from feedforge.fusion import (
    candidate_recall,
    popularity_buckets,
    reciprocal_rank_fusion,
)


@pytest.fixture
def clustered_index():
    """20 items in 2 tight clusters: dense ids 1-10 near e1, 11-20 near e2."""
    rng = np.random.default_rng(0)
    dim = 8
    vecs = []
    for i in range(20):
        base = np.zeros(dim)
        base[0 if i < 10 else 1] = 1.0
        v = base + rng.normal(0, 0.05, dim)
        vecs.append(v / np.linalg.norm(v))
    return ContentIndex(np.arange(1, 21), np.array(vecs, dtype=np.float32))


def test_content_candidates_respect_clusters(clustered_index):
    # History in cluster 1 -> candidates should come from cluster 1
    cands = clustered_index.candidates_for_history([1, 2, 3], k=5)
    assert len(cands) == 5
    assert all(c <= 10 for c in cands), f"cross-cluster leak: {cands}"
    # Seen items are filtered
    assert not {1, 2, 3} & set(cands)


def test_content_graceful_degradation(clustered_index):
    # History entirely of items with no embedding -> empty, not a crash
    assert clustered_index.candidates_for_history([99, 100], k=5) == []


def test_rrf_rewards_agreement():
    # Item 7 is mid-rank in both lists; 1 and 2 top only one list each.
    a = [1, 7, 3, 4]
    b = [2, 7, 5, 6]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0] == 7, f"consensus item should win: {fused}"


def test_rrf_empty_list_is_identity():
    a = [3, 1, 2]
    assert reciprocal_rank_fusion([a, []]) == a


def test_rrf_deterministic_tiebreak():
    assert reciprocal_rank_fusion([[5], [9]]) == reciprocal_rank_fusion([[9], [5]])


def test_candidate_recall_counts_topk_only():
    lists = [[1, 2, 3], [4, 5, 6]]
    targets = [3, 9]
    m = candidate_recall(lists, targets, ks=(2, 3))
    assert m["cand_recall@2"] == 0.0
    assert m["cand_recall@3"] == 0.5


def test_popularity_buckets_ordering():
    train = [[1] * 50 + [2] * 5 + [3]]
    buckets = popularity_buckets(train, targets=[3, 2, 1], n_buckets=3)
    assert buckets[0] <= buckets[1] <= buckets[2]
