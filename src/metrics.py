"""Prometheus metrics shared by the API, agent, executor, and cache layers."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60)

HTTP_REQUESTS = Counter(
    "lumina_http_requests_total",
    "HTTP requests handled by the API.",
    ["method", "route", "status"],
)
HTTP_LATENCY = Histogram(
    "lumina_http_request_duration_seconds",
    "HTTP request latency (time to response headers for streaming routes).",
    ["method", "route"],
    buckets=_LATENCY_BUCKETS,
)

AGENT_RUNS = Counter(
    "lumina_agent_runs_total",
    "Agent runs by backend and outcome.",
    ["backend", "outcome"],
)
AGENT_LATENCY = Histogram(
    "lumina_agent_run_duration_seconds",
    "End-to-end agent run latency.",
    ["backend"],
    buckets=_LATENCY_BUCKETS,
)
AGENT_ATTEMPTS = Histogram(
    "lumina_agent_execution_attempts",
    "Database execution attempts per agent run (1 = no self-correction needed).",
    ["backend"],
    buckets=(1, 2, 3, 4, 5, 10),
)

LLM_CALLS = Counter(
    "lumina_llm_calls_total",
    "LLM calls by provider, phase, and status.",
    ["provider", "phase", "status"],
)
LLM_LATENCY = Histogram(
    "lumina_llm_call_duration_seconds",
    "LLM call latency.",
    ["provider", "phase"],
    buckets=_LATENCY_BUCKETS,
)

DB_QUERIES = Counter(
    "lumina_db_queries_total",
    "Database queries by backend and status (ok, error, blocked).",
    ["backend", "status"],
)
DB_LATENCY = Histogram(
    "lumina_db_query_duration_seconds",
    "Database query latency.",
    ["backend"],
    buckets=_LATENCY_BUCKETS,
)

CACHE_REQUESTS = Counter(
    "lumina_cache_requests_total",
    "Cache lookups by kind (schema, query) and result (hit, miss, error).",
    ["kind", "result"],
)
