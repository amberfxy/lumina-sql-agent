"""FastAPI gateway for LuminaSQL-Agent."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from config import DatabaseBackend, Settings, get_settings
from src.agent import LuminaSQLAgent
from src.cache import Cache, build_cache
from src.db_executor import DatabaseExecutor
from src.metrics import HTTP_LATENCY, HTTP_REQUESTS
from src.schema_manager import SchemaManager

logger = logging.getLogger(__name__)


class Runtime:
    """Process-wide dependencies. The agent is built lazily so the API can start
    (and pass liveness/readiness probes) before LLM credentials are configured."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache: Cache = build_cache(settings)
        self.db_executor = DatabaseExecutor(settings=settings)
        self.schema_manager = SchemaManager(settings=settings, db_executor=self.db_executor, cache=self.cache)
        self._agent: LuminaSQLAgent | None = None
        self._agent_lock = threading.Lock()

    def get_agent(self) -> LuminaSQLAgent:
        if self._agent is None:
            with self._agent_lock:
                if self._agent is None:
                    self._agent = LuminaSQLAgent(
                        settings=self.settings,
                        schema_manager=self.schema_manager,
                        db_executor=self.db_executor,
                        cache=self.cache,
                    )
        return self._agent

    def close(self) -> None:
        self.db_executor.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    app.state.runtime = Runtime(settings)
    yield
    app.state.runtime.close()


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.1.0",
    description="NL-to-SQL/PartiQL agent with a self-correcting execution loop.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def record_http_metrics(request: Request, call_next: Any) -> Response:
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        # Use the route template (not the raw path) to keep label cardinality bounded.
        route_label = getattr(route, "path", "unmatched")
        if route_label != "/metrics":
            HTTP_REQUESTS.labels(method=request.method, route=route_label, status=str(status)).inc()
            HTTP_LATENCY.labels(method=request.method, route=route_label).observe(time.perf_counter() - started)


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def get_agent(runtime: Runtime = Depends(get_runtime)) -> LuminaSQLAgent:
    try:
        return runtime.get_agent()
    except ValueError as exc:
        # Missing LLM credentials: the service is up but cannot answer queries.
        raise HTTPException(status_code=503, detail=str(exc)) from exc


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language database question")
    backend: DatabaseBackend = DatabaseBackend.POSTGRES
    allow_mutations: bool = False
    use_cache: bool = True


class HealthResponse(BaseModel):
    status: str
    postgres: dict[str, Any]
    dynamodb: dict[str, Any]
    redis: dict[str, Any]


@app.get("/livez")
def livez() -> dict[str, str]:
    """Liveness: the process is serving requests."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz(runtime: Runtime = Depends(get_runtime)) -> JSONResponse:
    """Readiness: at least one database backend is reachable."""
    postgres = runtime.db_executor.health_check(DatabaseBackend.POSTGRES)
    if postgres["success"]:
        return JSONResponse({"status": "ready", "postgres": postgres})
    dynamodb = runtime.db_executor.health_check(DatabaseBackend.DYNAMODB)
    if dynamodb["success"]:
        return JSONResponse({"status": "ready", "dynamodb": dynamodb})
    return JSONResponse({"status": "not_ready", "postgres": postgres, "dynamodb": dynamodb}, status_code=503)


@app.get("/health", response_model=HealthResponse)
def health(runtime: Runtime = Depends(get_runtime)) -> HealthResponse:
    """Detailed dependency status for humans and dashboards."""
    postgres = runtime.db_executor.health_check(DatabaseBackend.POSTGRES)
    dynamodb = runtime.db_executor.health_check(DatabaseBackend.DYNAMODB)
    redis_status = {"enabled": runtime.cache.enabled, "success": runtime.cache.ping()}
    overall = "ok" if postgres["success"] or dynamodb["success"] else "degraded"
    return HealthResponse(status=overall, postgres=postgres, dynamodb=dynamodb, redis=redis_status)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/query")
def run_query(request: QueryRequest, agent: LuminaSQLAgent = Depends(get_agent)) -> dict[str, Any]:
    """Run the agent synchronously and return the final structured result."""
    try:
        result = agent.run(
            user_query=request.query,
            backend=request.backend,
            allow_mutations=request.allow_mutations,
            use_cache=request.use_cache,
        )
        return result.to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled query failure")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _event_stream(agent: LuminaSQLAgent, request: QueryRequest) -> AsyncIterator[str]:
    try:
        async for event in agent.astream_run(
            user_query=request.query,
            backend=request.backend,
            allow_mutations=request.allow_mutations,
            use_cache=request.use_cache,
        ):
            yield f"data: {json.dumps(event, default=str)}\n\n"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Streaming query failure")
        payload = {"type": "error", "content": {"message": str(exc)}}
        yield f"data: {json.dumps(payload)}\n\n"


@app.post("/api/v1/query/stream")
async def run_query_stream(request: QueryRequest, agent: LuminaSQLAgent = Depends(get_agent)) -> StreamingResponse:
    """Server-Sent Events stream for schema pruning, LLM tokens, and execution."""
    return StreamingResponse(
        _event_stream(agent, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/cache/invalidate")
def invalidate_cache(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    """Drop cached schema catalogs and generated queries (e.g. after a migration)."""
    runtime.schema_manager.invalidate_cache()
    deleted_queries = runtime.cache.delete_prefix("query:")
    return {"status": "ok", "deleted_queries": deleted_queries}
