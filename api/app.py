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
from pydantic import BaseModel  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from feedforge.experiment import (  # noqa: E402
    ExperimentLog,
    RequestRecord,
    assign_arm,
)
from feedforge.features import GENRES as GENRE_NAMES  # noqa: E402
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
                    arm: str | None = None,
                    diversity: float = Query(1.0, ge=0.1, le=1.0)):
    svc, log = state["service"], state["log"]
    if user_id not in svc.user_row:
        raise HTTPException(status_code=404, detail=f"user {user_id} not found")
    chosen = arm or assign_arm(user_id)
    result = svc.recommend(user_id, arm=chosen, n=n, diversity=diversity)
    log.record(RequestRecord(
        user_id=user_id, arm=chosen, latency_ms=result["latency_ms"],
        n_results=len(result["items"]), cache_hit=result["cache_hit"], ts=time.time()))
    return result


@app.get("/api/compare/{user_id}")
async def compare(user_id: int, n: int = Query(10, ge=1, le=50),
                  diversity: float = Query(1.0, ge=0.1, le=1.0)):
    """Both arms for one user. Not logged: this runs both pipelines for
    display, so counting it would distort the experiment's traffic
    split."""
    svc = state["service"]
    if user_id not in svc.user_row:
        raise HTTPException(status_code=404, detail=f"user {user_id} not found")
    baseline = svc.recommend(user_id, arm="baseline", n=n, diversity=diversity)
    reranked = svc.recommend(user_id, arm="reranked", n=n, diversity=diversity)
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


class ColdStartRequest(BaseModel):
    picks: list[int]
    n: int = 10
    diversity: float = 1.0


@app.get("/api/search")
async def search(q: str = Query(..., min_length=1), limit: int = Query(12, ge=1, le=30)):
    return {"results": state["service"].search_titles(q, limit)}


@app.get("/api/starters")
async def starters(n: int = Query(24, ge=1, le=60)):
    """Popular titles to seed the cold-start picker."""
    return {"items": state["service"].starter_items(n)}


@app.post("/api/coldstart")
async def coldstart(req: ColdStartRequest):
    """Recommendations for a visitor with no account, built from the
    titles they just picked."""
    if req.n < 1 or req.n > 50:
        raise HTTPException(status_code=400, detail="n must be 1..50")
    if not 0.1 <= req.diversity <= 1.0:
        raise HTTPException(status_code=400, detail="diversity must be 0.1..1.0")
    if len(req.picks) > 50:
        raise HTTPException(status_code=400, detail="too many picks")
    svc, log = state["service"], state["log"]
    result = svc.recommend_cold(req.picks, n=req.n, diversity=req.diversity)
    log.record(RequestRecord(
        user_id=-1, arm="coldstart", latency_ms=result["latency_ms"],
        n_results=len(result["items"]), cache_hit=False, ts=time.time()))
    return result


class ExplainRequest(BaseModel):
    item_id: int
    history: list[int]


@app.post("/api/explain")
async def explain(req: ExplainRequest):
    """Leave-one-out attribution: which history items drove this pick."""
    if len(req.history) > 50:
        raise HTTPException(status_code=400, detail="history too long")
    svc = state["service"]
    result = svc.explain(req.item_id, req.history)
    result["genres"] = [GENRE_NAMES[i] for i in result.pop("shared_genre_ids", [])
                        if i < len(GENRE_NAMES)]
    return result


@app.get("/api/explain/{user_id}/{item_id}")
async def explain_for_user(user_id: int, item_id: int):
    """Same attribution for a dataset viewer, using their real history."""
    svc = state["service"]
    if user_id not in svc.user_row:
        raise HTTPException(status_code=404, detail=f"user {user_id} not found")
    hist = svc.full_history[svc.user_row[user_id]]
    result = svc.explain(item_id, hist)
    result["genres"] = [GENRE_NAMES[i] for i in result.pop("shared_genre_ids", [])
                        if i < len(GENRE_NAMES)]
    return result


@app.get("/api/metrics")
async def metrics():
    out = state["log"].metrics()
    out["cache"] = state["service"].cache.stats()
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 8000)), reload=False)
