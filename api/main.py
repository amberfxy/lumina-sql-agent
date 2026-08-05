"""FastAPI gateway for LuminaSQL-Agent."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import DatabaseBackend, get_settings
from src.agent import LuminaSQLAgent
from src.db_executor import DatabaseExecutor

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Enterprise NL-to-SQL/NoSQL agent with self-correcting execution loop.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = LuminaSQLAgent(settings=settings)
db_executor = DatabaseExecutor(settings=settings)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language database question")
    backend: DatabaseBackend = DatabaseBackend.POSTGRES
    allow_mutations: bool = False
    stream: bool = True


class HealthResponse(BaseModel):
    status: str
    postgres: dict[str, Any]
    dynamodb: dict[str, Any]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    postgres = db_executor.health_check(DatabaseBackend.POSTGRES).to_dict()
    dynamodb = db_executor.health_check(DatabaseBackend.DYNAMODB).to_dict()
    overall = "ok" if postgres.get("success") or dynamodb.get("success") else "degraded"
    return HealthResponse(status=overall, postgres=postgres, dynamodb=dynamodb)


@app.post("/api/v1/query")
def run_query(request: QueryRequest) -> dict[str, Any]:
    """Run the agent synchronously and return the final structured result."""
    try:
        result = agent.run(
            user_query=request.query,
            backend=request.backend,
            allow_mutations=request.allow_mutations,
        )
        return result.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled query failure")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _event_stream(request: QueryRequest) -> AsyncIterator[str]:
    try:
        async for event in agent.astream_run(
            user_query=request.query,
            backend=request.backend,
            allow_mutations=request.allow_mutations,
        ):
            yield f"data: {json.dumps(event, default=str)}\n\n"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Streaming query failure")
        payload = {"type": "error", "content": {"message": str(exc)}}
        yield f"data: {json.dumps(payload)}\n\n"


@app.post("/api/v1/query/stream")
async def run_query_stream(request: QueryRequest) -> StreamingResponse:
    """Server-Sent Events stream for schema pruning, LLM tokens, and execution."""
    return StreamingResponse(
        _event_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.on_event("shutdown")
def shutdown() -> None:
    db_executor.close()
