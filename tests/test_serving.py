"""Serving tests: cache, A/B assignment, experiment log, and the API.

A tiny synthetic artifacts file + an untrained BERT4Rec checkpoint are
built in a temp dir, so the whole API is exercised end to end without
the real 300MB of assets. What's verified: deterministic bucketing,
traffic split, cache hit behaviour, arm forcing, that compare() runs
both arms without polluting the experiment log, and that recommendations
never include items the user has already seen.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from feedforge.cache import InMemoryCache
from feedforge.experiment import ARMS, ExperimentLog, RequestRecord, assign_arm
from feedforge.features import GENRES
from feedforge.model import BERT4Rec

N_ITEMS = 40
N_USERS = 12


@pytest.fixture(scope="module")
def artifacts_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("serving")
    rng = np.random.default_rng(0)

    history = [sorted(rng.choice(np.arange(1, N_ITEMS + 1), size=8, replace=False).tolist())
               for _ in range(N_USERS)]
    user_ids = list(range(1, N_USERS + 1))
    art = {
        "n_items": N_ITEMS,
        "mask_token": N_ITEMS + 1,
        "vocab_size": N_ITEMS + 2,
        "full_history": history,
        "user_row": {uid: i for i, uid in enumerate(user_ids)},
        "titles": {i: f"Movie {i} (199{i % 10})" for i in range(1, N_ITEMS + 1)},
        # numpy scalars on purpose: mirrors the real artifacts, where ids
        # come from numpy/pandas and must be coerced before JSON encoding.
        "dense_to_orig": {i: np.int64(1000 + i) for i in range(1, N_ITEMS + 1)},
        "genre_mat": np.eye(N_ITEMS + 1, len(GENRES), dtype=np.float32),
        "years": np.full(N_ITEMS + 1, 1995.0, dtype=np.float32),
        "user_demo": np.zeros((N_USERS, 3), dtype=np.float32),
        "pmi": {},
        "item_pop": np.arange(N_ITEMS + 1, dtype=np.float64),
    }
    with open(d / "serving.pkl", "wb") as f:
        pickle.dump(art, f)

    model = BERT4Rec(vocab_size=N_ITEMS + 2, max_len=16, d_model=16,
                     n_heads=2, n_layers=1, d_ff=32)
    torch.save({"model_state": model.state_dict(),
                "vocab_size": N_ITEMS + 2,
                "n_items": N_ITEMS,
                "config": {"max_len": 16, "d_model": 16, "n_heads": 2,
                           "n_layers": 1, "d_ff": 32}},
               d / "model.pt")
    return d


@pytest.fixture(scope="module")
def client(artifacts_dir, monkeypatch_module=None):
    import os

    os.environ["FF_ARTIFACTS"] = str(artifacts_dir / "serving.pkl")
    os.environ["FF_CHECKPOINT"] = str(artifacts_dir / "model.pt")
    os.environ["FF_RANKER"] = str(artifacts_dir / "missing_ranker.txt")
    os.environ["FF_EMBEDDINGS"] = str(artifacts_dir / "missing_emb.npz")
    os.environ["FF_EXPLOG"] = str(artifacts_dir / "exp.db")
    os.environ["FF_POSTERS"] = str(artifacts_dir / "no_posters")

    from api.app import app

    with TestClient(app) as c:
        yield c


# -- unit level ------------------------------------------------------------

def test_assignment_is_deterministic_and_splits():
    assert all(assign_arm(u) == assign_arm(u) for u in range(200))
    arms = [assign_arm(u) for u in range(2000)]
    share = arms.count("baseline") / len(arms)
    assert 0.4 < share < 0.6, f"split badly skewed: {share}"
    assert set(arms) == set(ARMS)


def test_salt_changes_assignment():
    users = range(500)
    a = [assign_arm(u, salt="v1") for u in users]
    b = [assign_arm(u, salt="v2") for u in users]
    assert a != b


def test_cache_hit_miss_and_expiry():
    c = InMemoryCache(ttl_seconds=100)
    assert c.get("k") is None
    c.set("k", {"v": 1})
    assert c.get("k") == {"v": 1}
    assert c.stats()["hits"] == 1 and c.stats()["misses"] == 1
    expired = InMemoryCache(ttl_seconds=-1)
    expired.set("k", 1)
    assert expired.get("k") is None


def test_experiment_log_percentiles(tmp_path):
    log = ExperimentLog(str(tmp_path / "e.db"))
    for i in range(100):
        log.record(RequestRecord(user_id=i, arm="baseline", latency_ms=float(i),
                                 n_results=10, cache_hit=i % 2 == 0, ts=0.0))
    m = log.metrics()
    assert m["total_requests"] == 100
    assert m["arms"]["baseline"]["latency_ms"]["p50"] == pytest.approx(50, abs=2)
    assert m["arms"]["baseline"]["latency_ms"]["p95"] == pytest.approx(95, abs=2)
    assert m["cache_hit_rate"] == pytest.approx(0.5)


# -- api level -------------------------------------------------------------

def test_health_and_users(client):
    assert client.get("/healthz").json()["users"] == N_USERS
    users = client.get("/api/users").json()
    assert users["total"] == N_USERS
    assert all(u["arm"] in ARMS for u in users["users"])


def test_recommend_excludes_seen_items(client):
    art_hist = client.get("/api/user/1").json()["history"]
    seen = {h["item_id"] for h in art_hist}
    recs = client.get("/api/recommend/1?n=10").json()
    assert len(recs["items"]) == 10
    assert not seen & {i["item_id"] for i in recs["items"]}


def test_cache_hit_on_second_call(client):
    first = client.get("/api/recommend/2").json()
    second = client.get("/api/recommend/2").json()
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert [i["item_id"] for i in first["items"]] == [i["item_id"] for i in second["items"]]


def test_forced_arm_overrides_assignment(client):
    forced = client.get("/api/recommend/3?arm=reranked").json()
    assert forced["arm"] == "reranked"


def test_compare_returns_both_arms_without_logging(client):
    before = client.get("/api/metrics").json()["total_requests"]
    cmp = client.get("/api/compare/4?n=5").json()
    after = client.get("/api/metrics").json()["total_requests"]
    assert after == before, "compare must not pollute the experiment log"
    assert len(cmp["baseline"]["items"]) == 5
    assert len(cmp["reranked"]["items"]) == 5
    # No ranker in this fixture, so both arms fall back to retrieval order
    assert cmp["identical_order"] is True


def test_unknown_user_404(client):
    assert client.get("/api/recommend/99999").status_code == 404


def test_payloads_are_json_native(client):
    """Numpy scalars must never reach the JSON encoder."""
    import json

    recs = client.get("/api/recommend/6").json()
    json.dumps(recs)  # raises if anything is a numpy type
    for item in recs["items"]:
        assert type(item["item_id"]) is int
        assert type(item["movie_id"]) is int
        assert type(item["title"]) is str


def test_metrics_shape(client):
    client.get("/api/recommend/5")
    m = client.get("/api/metrics").json()
    assert m["total_requests"] > 0
    assert "cache" in m and "note" in m
    for arm in ARMS:
        assert "latency_ms" in m["arms"][arm]
