"""
FastAPI 服务入口
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from service.schemas import (
    RecommendRequest, RecommendResponse, RecommendItem,
    HealthResponse,
)
from service.cache import cache

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("=" * 50)
    print("启动中，加载 pipeline...")
    from service.pipeline import RecommendPipeline
    pipeline = RecommendPipeline()
    print("=" * 50)
    yield
    print("关闭服务")


app = FastAPI(
    title="小红书风格笔记推荐系统",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    if pipeline is None:
        raise HTTPException(503, "pipeline 未就绪")
    return HealthResponse(
        status="ok",
        n_notes=len(pipeline.notes),
        n_users=len(pipeline.user_history),
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if pipeline is None:
        raise HTTPException(503, "pipeline 未就绪")

    cache_key = f"rec:{req.user_id}:{req.k}"
    cached = cache.get(cache_key)
    if cached:
        return RecommendResponse(
            user_id=req.user_id,
            items=[RecommendItem(**x) for x in cached["items"]],
            latency_ms=0.5,
            from_cache=True,
        )

    items, latency = pipeline.recommend(req.user_id, k=req.k)

    output = []
    for r in items:
        info = pipeline.get_note_info(r["note_id"])
        output.append({
            "note_id": info["note_id"],
            "title": info["title"],
            "category": info["category"],
            "tags": info["tags"],
            "score": float(r["score"]),
            "source": r.get("source", "mmr"),
        })

    cache.set(cache_key, {"items": output})

    return RecommendResponse(
        user_id=req.user_id,
        items=[RecommendItem(**x) for x in output],
        latency_ms=latency,
        from_cache=False,
    )