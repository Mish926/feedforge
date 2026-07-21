"""A/B infrastructure: deterministic assignment + per-arm request logging.

Honesty note that belongs in the README: this system has no real users,
so there are no real clicks. What is demonstrated here is the A/B
*infrastructure* -- deterministic bucketing, per-arm request logging,
and latency percentiles per arm -- not a click-through-rate result. The
quality comparison between arms comes from the offline evaluation
(results/ranker_significance.json), where the difference was measured
and found statistically indistinguishable. Reporting simulated CTR as if
it were a finding would be fabrication, so this module reports only what
it can actually observe: traffic split and latency.

Assignment is md5(salt + user_id) mod 100, so a user always lands in the
same arm across requests and restarts (no session state), and changing
the salt reshuffles the population for a fresh experiment.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ARMS = ("baseline", "reranked")


def assign_arm(user_id: int, salt: str = "feedforge-v1", split: int = 50) -> str:
    """Deterministic bucketing. split = percent of traffic to 'baseline'."""
    digest = hashlib.md5(f"{salt}:{user_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return ARMS[0] if bucket < split else ARMS[1]


@dataclass
class RequestRecord:
    user_id: int
    arm: str
    latency_ms: float
    n_results: int
    cache_hit: bool
    ts: float


class ExperimentLog:
    """SQLite-backed request log. Small, boring, and inspectable, which
    is what makes the /metrics numbers auditable rather than asserted."""

    def __init__(self, db_path: str = "artifacts/experiment.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                arm TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                n_results INTEGER NOT NULL,
                cache_hit INTEGER NOT NULL,
                ts REAL NOT NULL
            )""")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_arm ON requests(arm)")
        self._conn.commit()

    def record(self, rec: RequestRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO requests (user_id, arm, latency_ms, n_results, cache_hit, ts)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (rec.user_id, rec.arm, rec.latency_ms, rec.n_results,
                 int(rec.cache_hit), rec.ts),
            )
            self._conn.commit()

    def metrics(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT arm, latency_ms, cache_hit FROM requests").fetchall()
        by_arm: dict[str, list[float]] = {a: [] for a in ARMS}
        cache_hits = 0
        for arm, latency, hit in rows:
            by_arm.setdefault(arm, []).append(latency)
            cache_hits += hit

        def pct(values: list[float], q: float) -> float:
            if not values:
                return 0.0
            s = sorted(values)
            idx = min(int(q * len(s)), len(s) - 1)
            return round(s[idx], 2)

        out = {"total_requests": len(rows),
               "cache_hit_rate": round(cache_hits / len(rows), 3) if rows else 0.0,
               "arms": {}}
        for arm, lats in by_arm.items():
            out["arms"][arm] = {
                "requests": len(lats),
                "traffic_share": round(len(lats) / len(rows), 3) if rows else 0.0,
                "latency_ms": {
                    "p50": pct(lats, 0.50),
                    "p95": pct(lats, 0.95),
                    "p99": pct(lats, 0.99),
                    "mean": round(sum(lats) / len(lats), 2) if lats else 0.0,
                },
            }
        out["note"] = ("Traffic split and latency are measured. Recommendation "
                       "quality per arm is NOT measured here (no real users, "
                       "therefore no real clicks); see "
                       "results/ranker_significance.json for the offline "
                       "comparison.")
        return out

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM requests")
            self._conn.commit()


def now_ms() -> float:
    return time.perf_counter() * 1000
