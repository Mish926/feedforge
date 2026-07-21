"""FeedForge serving API.

    uvicorn api.app:app --port 8000

Endpoints:
    GET  /                      the inspector UI
    GET  /api/users             sample of known user ids
    GET  /api/user/{id}         recent history for a user
    GET  /api/recommend/{id}    recommendations; arm auto-assigned or forced
    GET  /api/compare/{id}      both arms side by side (UI comparison view)
    GET  /api/metrics           traffic split + latency percentiles per arm
    GET  /healthz               liveness

The A/B arm is assigned deterministically from the user id unless the
caller forces one with ?arm=. Every /api/recommend call is logged for
the metrics endpoint; /api/compare deliberately is NOT logged, because
it runs both arms for display and would corrupt the traffic split.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from feedforge.experiment import (  # noqa: E402
    ExperimentLog,
    RequestRecord,
    assign_arm,
)
from feedforge.service import RecommenderService  # noqa: E402

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] loading artifacts and models...")
    t0 = time.time()
    state["service"] = RecommenderService(
        artifacts_path=os.environ.get("FF_ARTIFACTS", "artifacts/serving.pkl"),
        checkpoint_path=os.environ.get("FF_CHECKPOINT", "checkpoints/bert4rec_best.pt"),
        ranker_path=os.environ.get("FF_RANKER", "checkpoints/ranker.txt"),
        embeddings_path=os.environ.get("FF_EMBEDDINGS", "data/content_embeddings.npz"),
    )
    state["log"] = ExperimentLog(os.environ.get("FF_EXPLOG", "artifacts/experiment.db"))
    print(f"[startup] ready in {time.time() - t0:.1f}s")
    yield


app = FastAPI(title="FeedForge", version="1.0.0", lifespan=lifespan)

_posters = Path(os.environ.get("FF_POSTERS", "data/posters"))
if _posters.exists():
    app.mount("/posters", StaticFiles(directory=str(_posters)), name="posters")

_static = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    html = _static / "index.html"
    if not html.exists():
        raise HTTPException(status_code=404, detail="UI not built")
    return HTMLResponse(html.read_text())


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "users": len(state["service"].known_users())}


@app.get("/api/users")
async def users(limit: int = Query(60, ge=1, le=500)):
    svc = state["service"]
    ids = svc.known_users()
    step = max(1, len(ids) // limit)
    sample = ids[::step][:limit]
    return {"users": [{"user_id": u, "arm": assign_arm(u)} for u in sample],
            "total": len(ids)}


@app.get("/api/user/{user_id}")
async def user_detail(user_id: int, limit: int = Query(12, ge=1, le=50)):
    svc = state["service"]
    if user_id not in svc.user_row:
        raise HTTPException(status_code=404, detail=f"user {user_id} not found")
    return {"user_id": user_id, "arm": assign_arm(user_id),
            "history": svc.user_history(user_id, limit=limit)}


@app.get("/api/recommend/{user_id}")
async def recommend(user_id: int, n: int = Query(10, ge=1, le=50),
                    arm: str | None = None):
    svc, log = state["service"], state["log"]
    if user_id not in svc.user_row:
        raise HTTPException(status_code=404, detail=f"user {user_id} not found")
    chosen = arm or assign_arm(user_id)
    result = svc.recommend(user_id, arm=chosen, n=n)
    log.record(RequestRecord(
        user_id=user_id, arm=chosen, latency_ms=result["latency_ms"],
        n_results=len(result["items"]), cache_hit=result["cache_hit"], ts=time.time()))
    return result


@app.get("/api/compare/{user_id}")
async def compare(user_id: int, n: int = Query(10, ge=1, le=50)):
    """Both arms for one user. Not logged: this runs both pipelines for
    display, so counting it would distort the experiment's traffic
    split."""
    svc = state["service"]
    if user_id not in svc.user_row:
        raise HTTPException(status_code=404, detail=f"user {user_id} not found")
    baseline = svc.recommend(user_id, arm="baseline", n=n)
    reranked = svc.recommend(user_id, arm="reranked", n=n)
    base_ids = [i["item_id"] for i in baseline["items"]]
    rank_ids = [i["item_id"] for i in reranked["items"]]
    overlap = len(set(base_ids) & set(rank_ids))
    return {
        "user_id": user_id,
        "assigned_arm": assign_arm(user_id),
        "baseline": baseline,
        "reranked": reranked,
        "overlap_at_n": overlap,
        "identical_order": base_ids == rank_ids,
    }


@app.get("/api/metrics")
async def metrics():
    out = state["log"].metrics()
    out["cache"] = state["service"].cache.stats()
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 8000)), reload=False)
