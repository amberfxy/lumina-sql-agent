"""Prometheus metrics shared by the API, agent, executor, and cache layers."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

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
    "Agent runs by backend and outcome (success, self_corrected, cache_hit, rejected, failed).",
    ["backend", "outcome"],
)
AGENT_RUN_FAILURES = Counter(
    "lumina_agent_run_failures_total",
    "Agent runs that ended unsuccessfully, by terminal failure category.",
    ["backend", "category"],
)
AGENT_ATTEMPT_FAILURES = Counter(
    "lumina_agent_attempt_failures_total",
    "Failed generation attempts by failure category and retry policy.",
    ["backend", "category", "policy"],
)
SELF_CORRECTION = Counter(
    "lumina_agent_self_correction_total",
    "Runs that hit an LLM-correctable failure, by whether a retry recovered them.",
    ["backend", "result"],
)
TRANSIENT_RETRIES = Counter(
    "lumina_db_transient_retries_total",
    "Same-query database retries after transient failures (no LLM involved).",
    ["backend", "category"],
)
UNSAFE_REJECTIONS = Counter(
    "lumina_unsafe_query_rejections_total",
    "Queries refused for safety, by the layer that refused them (validator, database, model).",
    ["backend", "layer"],
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
LLM_TOKENS = Counter(
    "lumina_llm_tokens_total",
    "Tokens reported by the LLM provider, by kind (prompt, completion).",
    ["provider", "model", "kind"],
)
LLM_COST = Counter(
    "lumina_llm_estimated_cost_usd_total",
    "Estimated LLM spend in USD from reported tokens and configured prices.",
    ["provider", "model"],
)

IN_FLIGHT = Gauge("lumina_inflight_requests", "Agent requests currently executing.")
QUEUED = Gauge("lumina_queued_requests", "Agent requests waiting for an execution slot.")
ADMISSION_REJECTIONS = Counter(
    "lumina_admission_rejections_total",
    "Requests shed by admission control (queue_full, queue_timeout) or rate limiting (rate_limited).",
    ["reason"],
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
