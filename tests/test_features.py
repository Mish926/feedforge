"""Phase 3b tests: orthogonal features.

The learnability test builds a world where the collaborative order is
random noise and ONLY genre affinity identifies the target (targets
share the user's history genre; distractors don't). A ranker with aux
features must find them; the base 7-feature ranker cannot. This proves
the feature plumbing end to end before spending real training time.
"""

import numpy as np
import pytest

from feedforge.features import (
    GENRES,
    AuxFeatureContext,
    build_cooccurrence_pmi,
)
from feedforge.ranker import (
    build_item_popularity,
    build_ranking_dataset,
    feature_names,
    ranked_metrics,
    rerank,
    train_ranker,
)

N_ITEMS = 200
K = 20
N_USERS = 400
RNG = np.random.default_rng(3)


def _genre_world():
    """Items 1-100 are genre A, 101-200 genre B. Each user's history and
    target share one genre; distractor candidates are the other genre."""
    genre_mat = np.zeros((N_ITEMS + 1, len(GENRES)), dtype=np.float32)
    genre_mat[1:101, 0] = 1.0    # Action
    genre_mat[101:, 4] = 1.0     # Comedy
    years = np.full(N_ITEMS + 1, 1995.0, dtype=np.float32)
    user_demo = np.zeros((N_USERS, 3), dtype=np.float32)

    candidates, scores, histories, targets = [], [], [], []
    for u in range(N_USERS):
        action_user = u % 2 == 0
        pool = np.arange(1, 101) if action_user else np.arange(101, 201)
        other = np.arange(101, 201) if action_user else np.arange(1, 101)
        hist = RNG.choice(pool, size=8, replace=False).tolist()
        target = int(RNG.choice([i for i in pool if i not in hist]))
        distractors = RNG.choice(other, size=K - 1, replace=False).tolist()
        pos = int(RNG.integers(5, K))
        cands = distractors[:pos] + [target] + distractors[pos:]
        candidates.append(cands)
        scores.append(RNG.normal(0, 1, K).tolist())  # collab order = noise
        histories.append(hist)
        targets.append(target)

    ctx = AuxFeatureContext(genre_mat, years, user_demo, pmi={})
    return candidates, scores, histories, targets, ctx


def test_feature_names_extend():
    assert len(feature_names(None)) == 7
    cands, scores, hists, targets, ctx = _genre_world()
    assert len(feature_names(ctx)) == 14


def test_aux_dataset_width():
    cands, scores, hists, targets, ctx = _genre_world()
    pop = build_item_popularity(hists, N_ITEMS)
    ds = build_ranking_dataset(cands[:5], scores[:5], hists[:5], targets[:5],
                               pop, aux_ctx=ctx)
    assert ds.X.shape[1] == 14


def test_genre_similarity_separates():
    cands, scores, hists, targets, ctx = _genre_world()
    uctx = ctx.user_context(0, hists[0])
    target_feats = ctx.features_for(uctx, targets[0])
    distractor = next(c for c in cands[0] if c != targets[0])
    distractor_feats = ctx.features_for(uctx, distractor)
    assert target_feats[0] > 0.9        # genre_sim_profile high for target
    assert distractor_feats[0] < 0.1    # and near zero for cross-genre


def test_ranker_uses_genre_signal():
    cands, scores, hists, targets, ctx = _genre_world()
    pop = build_item_popularity(hists, N_ITEMS)
    half = N_USERS // 2
    tr = build_ranking_dataset(cands[:half], scores[:half], hists[:half],
                               targets[:half], pop, aux_ctx=ctx)
    te = build_ranking_dataset(cands[half:], scores[half:], hists[half:],
                               targets[half:], pop, aux_ctx=ctx)
    booster = train_ranker(tr, num_boost_round=80,
                           params={"min_data_in_leaf": 10},
                           names=feature_names(ctx))
    baseline = ranked_metrics(cands[half:], targets[half:], ks=(5,))
    after = ranked_metrics(rerank(booster, te), targets[half:], ks=(5,))
    assert baseline["recall@5"] < 0.4          # collab order is noise
    assert after["recall@5"] > 0.9, f"genre signal not learned: {after}"


def test_cooccurrence_pmi_positive_for_pairs():
    seqs = [[1, 2, 3, 1, 2], [1, 2, 4], [5, 6]] * 10
    pmi = build_cooccurrence_pmi(seqs, n_items=6, window=2)
    assert pmi.get((1, 2), 0) > 0              # frequent pair
    assert (3, 5) not in pmi                    # never co-occur
