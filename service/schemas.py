"""
Pydantic 请求/响应模型
"""
from pydantic import BaseModel
from typing import List, Optional


class RecommendRequest(BaseModel):
    user_id: int
    k: int = 10


class RecommendItem(BaseModel):
    note_id: int
    title: str
    category: str
    tags: str
    score: float
    source: str


class RecommendResponse(BaseModel):
    user_id: int
    items: List[RecommendItem]
    latency_ms: float
    from_cache: bool


class HealthResponse(BaseModel):
    status: str
    n_notes: int
    n_users: int