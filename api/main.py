"""FastAPI gateway for LuminaSQL-Agent."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from config import DatabaseBackend, Settings, get_settings
from src.admission import AdmissionController, AdmissionRejected, RateLimiter
from src.agent import AgentResult, LuminaSQLAgent
from src.cache import Cache, build_cache
from src.context import configure_logging, new_request_id, reset_request_id, set_request_id
from src.db_executor import DatabaseExecutor
from src.failures import Failure, FailureCategory, FailureSource
from src.metrics import ADMISSION_REJECTIONS, HTTP_LATENCY, HTTP_REQUESTS
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
        self.admission = AdmissionController(
            settings.max_concurrent_requests, settings.max_queued_requests, settings.queue_timeout_seconds
        )
        self.rate_limiter = RateLimiter(settings.rate_limit_per_minute)
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
    configure_logging(settings.log_level, settings.log_format)
    # Agent work runs in the loop's default executor; size it to the admission limit so the
    # admission controller, not an implicit thread cap, decides how much work runs at once.
    executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_requests + 4, thread_name_prefix="agent")
    asyncio.get_running_loop().set_default_executor(executor)
    app.state.runtime = Runtime(settings)
    yield
    app.state.runtime.close()
    executor.shutdown(wait=False, cancel_futures=True)


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description="Read-only NL-to-SQL/PartiQL agent with validated, policy-driven self-correction.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "Retry-After"],
)


_ADMISSION_ROUTES = {"/api/v1/query", "/api/v1/query/stream"}


@app.middleware("http")
async def request_context(request: Request, call_next: Any) -> Response:
    request_id = new_request_id(request.headers.get("x-request-id"))
    token = set_request_id(request_id)
    started = time.perf_counter()
    status = 500
    route_label: str | None = None
    try:
        runtime = getattr(request.app.state, "runtime", None)
        if (
            request.method == "POST"
            and request.url.path in _ADMISSION_ROUTES
            and runtime
            and runtime.admission.saturated
        ):
            # Shed before routing, body parsing, and dependency resolution: under overload the
            # rejection path must be nearly free or rejected traffic starves admitted requests.
            ADMISSION_REJECTIONS.labels(reason="queue_full").inc()
            route_label, status = request.url.path, 429
            return JSONResponse(
                {"detail": "Server busy; retry later."},
                status_code=429,
                headers={"Retry-After": "1", "X-Request-ID": request_id},
            )
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        route = request.scope.get("route")
        # Use the route template (not the raw path) to keep label cardinality bounded.
        route_label = route_label or getattr(route, "path", "unmatched")
        if route_label != "/metrics":
            HTTP_REQUESTS.labels(method=request.method, route=route_label, status=str(status)).inc()
            HTTP_LATENCY.labels(method=request.method, route=route_label).observe(time.perf_counter() - started)
        reset_request_id(token)


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def authenticate(request: Request, runtime: Runtime = Depends(get_runtime)) -> str:
    """Return the client identity used for rate limiting; enforce API keys when configured."""
    keys = runtime.settings.api_key_set
    supplied = request.headers.get("x-api-key") or ""
    authorization = request.headers.get("authorization", "")
    if not supplied and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if keys:
        if not supplied or not any(hmac.compare_digest(supplied, key) for key in keys):
            raise HTTPException(status_code=401, detail="Missing or invalid API key.")
        client = "key:" + supplied[-6:]
    else:
        client = "ip:" + (request.client.host if request.client else "unknown")
    if not runtime.rate_limiter.allow(client):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.", headers={"Retry-After": "60"})
    return client


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


def http_status_for(failure: Failure | None) -> int:
    """Request-level outcomes (answered, refused, unfixable query) are 200 with success=false;
    dependency failures are 5xx so load balancers, retries, and alerts can see them."""
    if failure is None:
        return 200
    if failure.source == FailureSource.LLM and failure.category != FailureCategory.UNANSWERABLE:
        return {FailureCategory.RATE_LIMIT: 503, FailureCategory.TIMEOUT: 504}.get(failure.category, 502)
    if failure.category in (FailureCategory.SCHEMA_UNAVAILABLE, FailureCategory.CONNECTION_ERROR):
        return 503
    if failure.source == FailureSource.AGENT and failure.category == FailureCategory.TIMEOUT:
        return 504
    if failure.category == FailureCategory.INTERNAL_ERROR:
        return 500
    return 200


def _check_mutations(request: QueryRequest, runtime: Runtime) -> None:
    if request.allow_mutations and not runtime.settings.mutations_enabled:
        raise HTTPException(status_code=403, detail="Mutations are disabled on this server.")


async def _admit(runtime: Runtime) -> None:
    try:
        await runtime.admission.acquire()
    except AdmissionRejected as exc:
        raise HTTPException(
            status_code=429,
            detail=f"Server busy ({exc.reason}); retry later.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc


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
async def run_query(
    request: QueryRequest,
    _client: str = Depends(authenticate),
    runtime: Runtime = Depends(get_runtime),
    agent: LuminaSQLAgent = Depends(get_agent),
) -> JSONResponse:
    """Run the agent and return the final structured result."""
    _check_mutations(request, runtime)
    await _admit(runtime)
    try:
        result: AgentResult = await asyncio.to_thread(
            agent.run,
            user_query=request.query,
            backend=request.backend,
            allow_mutations=request.allow_mutations,
            use_cache=request.use_cache,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled query failure")
        raise HTTPException(status_code=500, detail="Internal error.") from exc
    finally:
        runtime.admission.release()
    status = http_status_for(result.failure)
    headers = {"Retry-After": "5"} if status == 503 else None
    return JSONResponse(jsonable_encoder(result.to_dict()), status_code=status, headers=headers)


async def _event_stream(agent: LuminaSQLAgent, request: QueryRequest, runtime: Runtime) -> AsyncIterator[str]:
    # The slot is acquired inside the generator: a generator that never starts (client gone
    # before the first byte) never runs `finally`, so acquiring outside would leak the slot.
    try:
        await runtime.admission.acquire()
    except AdmissionRejected as exc:
        yield f"data: {json.dumps({'type': 'error', 'content': {'message': f'Server busy ({exc.reason})'}})}\n\n"
        return
    try:
        async for event in agent.astream_run(
            user_query=request.query,
            backend=request.backend,
            allow_mutations=request.allow_mutations,
            use_cache=request.use_cache,
        ):
            yield f"data: {json.dumps(event, default=str)}\n\n"
    except Exception:  # noqa: BLE001
        logger.exception("Streaming query failure")
        yield f"data: {json.dumps({'type': 'error', 'content': {'message': 'Internal error.'}})}\n\n"
    finally:
        runtime.admission.release()


@app.post("/api/v1/query/stream")
async def run_query_stream(
    request: QueryRequest,
    _client: str = Depends(authenticate),
    runtime: Runtime = Depends(get_runtime),
    agent: LuminaSQLAgent = Depends(get_agent),
) -> StreamingResponse:
    """Server-Sent Events stream for schema pruning, LLM tokens, and execution."""
    _check_mutations(request, runtime)
    if runtime.admission.saturated:
        raise HTTPException(status_code=429, detail="Server busy; retry later.", headers={"Retry-After": "1"})
    return StreamingResponse(
        _event_stream(agent, request, runtime),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/cache/invalidate")
def invalidate_cache(_client: str = Depends(authenticate), runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    """Drop cached schema catalogs and generated queries (e.g. after a migration)."""
    runtime.schema_manager.invalidate_cache()
    deleted_queries = runtime.cache.delete_prefix("query:")
    return {"status": "ok", "deleted_queries": deleted_queries}
