"""Load test for the FeedForge serving API.

    locust -f loadtest/locustfile.py --host http://localhost:8000

Headless, for a reproducible number to put in the README:

    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 60s --csv results/loadtest

Two user classes model the traffic mix a recommender actually sees:
BrowsingUser hits /api/recommend for a random user (the hot path, mostly
cache hits after warmup), and ComparingUser hits /api/compare (the
expensive path: two full pipeline passes, no logging). The 4:1 weighting
reflects a real deployment where the inspector view is rare.

Note when reading results: cache hit rate rises with load because the
user pool is fixed, which is realistic for a recommender (popular users
are re-requested) but means p50 drops as the run progresses. The honest
headline number is p95 under sustained load, reported alongside the
cache hit rate rather than in isolation.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, events, task

USER_POOL: list[int] = []


@events.test_start.add_listener
def fetch_user_pool(environment, **kwargs):
    """Pull real user ids from the API so the test hits valid users."""
    import requests

    host = environment.host or "http://localhost:8000"
    try:
        data = requests.get(f"{host}/api/users?limit=200", timeout=10).json()
        USER_POOL.extend(u["user_id"] for u in data["users"])
        print(f"[loadtest] user pool: {len(USER_POOL)} ids")
    except Exception as e:  # noqa: BLE001
        print(f"[loadtest] could not fetch users ({e}); falling back to 1..500")
        USER_POOL.extend(range(1, 501))


class BrowsingUser(HttpUser):
    """The hot path: one recommendation request per view."""

    weight = 4
    wait_time = between(0.1, 0.5)

    @task(10)
    def recommend(self):
        uid = random.choice(USER_POOL)
        with self.client.get(f"/api/recommend/{uid}?n=10",
                             name="/api/recommend/[id]",
                             catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}")
            elif r.elapsed.total_seconds() * 1000 > 100:
                # Not a failure, but surfaces SLO breaches in the report
                r.failure("exceeded 100ms target")

    @task(1)
    def history(self):
        uid = random.choice(USER_POOL)
        self.client.get(f"/api/user/{uid}?limit=12", name="/api/user/[id]")


class ComparingUser(HttpUser):
    """The inspector path: both arms in one request."""

    weight = 1
    wait_time = between(0.5, 2.0)

    @task
    def compare(self):
        uid = random.choice(USER_POOL)
        self.client.get(f"/api/compare/{uid}?n=10", name="/api/compare/[id]")

    @task
    def metrics(self):
        self.client.get("/api/metrics")
