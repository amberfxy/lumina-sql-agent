# Production AI Audit — LuminaSQL-Agent

Audited revision: `8961a64` (before the hardening work described in the other docs).
Method: read every module, then probe the running Docker Compose stack (PostgreSQL 16,
Redis 7, API) with hand-written queries. Every finding below references real files and,
where it matters, a reproduced behavior.

Ratings: **DONE** (production-adequate), **PARTIAL** (exists but has a gap that matters),
**MISSING** (not implemented), **N/A**.

## Executive summary

The core loop (schema pruning → generation → execution → error-driven regeneration) works
and the infra scaffolding (Compose, Kubernetes, CI, Prometheus) is solid. The serious
problems are in the **trust boundary between the LLM and the database**:

1. **Critical — read-only bypass to OS command execution.** The only statement guard is a
   keyword regex (`_MUTATING_PATTERN`, `src/db_executor.py:73`) plus
   `SET TRANSACTION READ ONLY` (`src/db_executor.py:217`). psycopg2 executes multi-statement
   strings, and the regex does not cover `SET`, `COPY`, or functions. Reproduced against the
   Compose database through `DatabaseExecutor.execute(..., allow_mutations=False)`:

   | Generated query | Result |
   | --- | --- |
   | `SELECT nextval('orders_order_id_seq')` | blocked (read-only txn) |
   | `SET TRANSACTION READ WRITE; SELECT nextval('orders_order_id_seq')` | **executed** (sequence advanced) |
   | `SET TRANSACTION READ WRITE; COPY (SELECT 1) TO PROGRAM 'true'` | **executed** (shell command ran on the DB host) |
   | `SELECT length(pg_read_file('/etc/hostname'))` | **executed** (server file read) |
   | `SELECT count(*) FROM pg_catalog.pg_authid` | **executed** (password hashes readable) |

   Since the SQL is LLM output and the LLM input is user text, any prompt injection that
   convinces the model to emit such a string becomes remote code execution on the database.
2. **High — the app connects as the PostgreSQL superuser** (`docker-compose.yml:6`,
   `deploy/k8s/config.yaml`), so the database provides no authorization boundary at all.
3. **High — `allow_mutations` is client-controlled** (`api/main.py:115`). Any caller can
   request write mode; there is no authentication.
4. **Medium — every execution error is sent back to the LLM**, including connection
   failures, timeouts, and the safety block itself (`src/agent.py:414-450`). This wastes
   tokens on unfixable errors and turns the model into an optimizer against the safety
   filter ("Mutation blocked by safety policy" is fed back as a hint).
5. **Medium — no measurements.** The 58-case gold set (`evaluation/dataset.jsonl`) has
   never been run against a model; there are no load, cache, or resilience measurements, and
   no token or cost accounting.

## Dimension ratings

| Dimension | Rating | Evidence |
| --- | --- | --- |
| LLM reliability | PARTIAL | SDK timeout and retries are configured (`src/agent.py:137-142`, `config.py:52-53`). Missing: provider errors are not classified (a 429 and a 401 look the same), malformed output is passed straight to the database (`_extract_query` falls back to the raw text, `src/agent.py:535`), and there is no per-request deadline. |
| Schema grounding | PARTIAL | Catalog extraction with PK/FK/comments and bag-of-words pruning (`src/schema_manager.py`). Missing: generated SQL is never checked against the catalog, so hallucinated tables or columns are only discovered by the database. Schema comments are injected into the prompt verbatim (an indirect-injection vector). |
| SQL correctness | PARTIAL | Gold set with an order- and column-tolerant comparator (`evaluation/compare.py`). Never executed against a model, so correctness is unknown. |
| Query safety | **MISSING (critical)** | See the bypass table above. No parser, no single-statement check, no statement allowlist, no function denylist. DynamoDB relies on the same regex only (`src/db_executor.py:261`). |
| Prompt injection | MISSING | User text and schema comments are interpolated into the prompt with no delimiting or instruction hierarchy (`src/agent.py:30-48`). Combined with the safety gap, injection is exploitable. |
| Authorization boundaries | MISSING | No API authentication; `allow_mutations` is a request field; DB user is `postgres` (superuser); no table/column allowlist; system catalogs are queryable. |
| Retry behavior | PARTIAL | Self-correction loop exists. Every `DatabaseExecutionError` is treated as LLM-correctable; transient failures are not retried without the LLM; no failure taxonomy. |
| Timeouts | PARTIAL | LLM 60 s, PG `statement_timeout` 15 s and `connect_timeout` 5 s, Redis 0.5 s, boto 5/15 s. Missing: end-to-end request deadline; SQLAlchemy `pool_timeout` is the 30 s default. |
| Concurrency | PARTIAL | Blocking work is offloaded to threads (`astream_run`, sync endpoints); engine and catalog init are lock-protected. Effective concurrency is set implicitly by AnyIO's 40-thread limiter and a 5+10 DB pool, never measured. |
| Backpressure | MISSING | No admission control. Excess requests queue invisibly in the thread pool or wait 30 s on the DB pool, then fail with 500. |
| Caching | PARTIAL | Redis schema and query cache with circuit breaker (`src/cache.py`). The key omits a prompt version, so prompt changes serve stale SQL. Value never benchmarked. |
| Rate limiting | MISSING | None; a single client can spend unbounded LLM budget. |
| Observability | PARTIAL | HTTP, agent, LLM, DB, and cache metrics plus 8 alerts and a dashboard (`src/metrics.py`, `observability/`). Missing: token/cost metrics, errors by category, unsafe-query rejections, in-flight and queue gauges, and correlation IDs (log lines from one request cannot be joined). |
| Evaluation quality | PARTIAL | 58 gold cases, concurrency-bounded harness. Missing: negative categories (unsafe, injection, unanswerable, ambiguous), per-category metrics, hallucination rate, latency/token/cost per request, CSV output. |
| Cost measurement | MISSING | Token usage from the provider response is discarded (`src/agent.py:154`). |
| Deployment | PARTIAL | Multi-stage non-root image, Compose, Kubernetes (probes, PDB, HPA, restricted securityContext), rolling restart verified on kind. Gaps: superuser DB credentials, no NetworkPolicy, Secret is a placeholder. |
| Testing | PARTIAL | 51 unit + 6 integration tests, 72% coverage, CI with 5 jobs. Missing: adversarial/security tests, failure-injection tests, load tests, deterministic end-to-end eval in CI. |

## Proposed improvements (minimal, high impact)

Ordered by risk reduction per line of code. No new frameworks; one new dependency
(`sqlglot`, a pure-Python SQL parser) is justified because regex-based SQL filtering is the
root cause of the critical finding.

1. **AST-based query validator** (`src/sql_guard.py`): parse with sqlglot; require exactly
   one statement whose root is `SELECT`/`UNION`/`WITH … SELECT`; reject any DML/DDL/`SET`/
   `COPY` node anywhere in the tree; deny dangerous functions (`pg_*`, `lo_*`, `dblink*`,
   `set_config`, `nextval`, …); restrict table references to the extracted catalog (which
   excludes `pg_catalog` and `information_schema`); optional table allowlist and column
   denylist. PartiQL: single `SELECT` statement on known tables.
2. **Least-privilege database role**: `lumina_reader` with `SELECT` on application tables
   only, `default_transaction_read_only=on`, and a role-level `statement_timeout`. The API
   connects as this role in Compose and Kubernetes. This makes the database itself the final
   authorization boundary, independent of application code.
3. **Server-side mutation gate**: `allow_mutations` honored only when `MUTATIONS_ENABLED`
   is set on the server; otherwise 403.
4. **Failure taxonomy and retry policy** (`src/failures.py`): classify by SQLSTATE /
   provider exception into categories with one of three policies: LLM-correctable,
   transient (retry the same query with backoff, no LLM), terminal. Unsafe queries are never
   sent back to the model.
5. **Prompt hardening**: delimit user text and schema as untrusted data; give the model an
   explicit refusal channel (`CANNOT_ANSWER:`) for unanswerable or write requests.
6. **Admission control**: bounded in-flight limit and bounded wait queue; shed with 429 and
   `Retry-After`; gauges for in-flight and queued requests.
7. **API keys and per-client rate limiting** (optional, enabled by config).
8. **Correlation IDs**: `X-Request-ID` propagated through a contextvar into every log line,
   the LLM call, and the executed SQL (as a SQL comment visible in `pg_stat_activity`).
9. **Token and cost accounting**: capture provider usage, expose metrics, and report
   tokens and estimated cost per request in the eval.
10. **Measurement**: categorized eval with negative cases, a deterministic scripted-LLM
    suite for CI, load test at 10/25/50/100 concurrency, cache hit/miss benchmark,
    failure-injection tests.

## Status after remediation

Re-rated after implementing the improvements above. Evidence lives in the referenced docs
and in `eval/results/`.

| Dimension | Before | After | Evidence |
| --- | --- | --- | --- |
| LLM reliability | PARTIAL | DONE | Provider errors classified (`RATE_LIMIT`/`TIMEOUT`/`MODEL_ERROR`), per-call timeout bounded by the run deadline, refusal channel; fault injection in `docs/failure-model.md` |
| Schema grounding | PARTIAL | DONE | Guard checks tables against the exposed catalog before execution; `ALLOWED_TABLES` / `DENIED_COLUMNS` |
| SQL correctness | PARTIAL | PARTIAL | Harness, 97-case categorized dataset and metrics are ready; real-model run pending an API key |
| Query safety | MISSING | DONE | AST guard (68/68 adversarial, 0/61 false positives) plus read-only role (5/5 audit payloads blocked with the guard disabled) |
| Prompt injection | MISSING | PARTIAL | Delimited untrusted input, refusal channel, guard on every output; model-level resistance unmeasured until the real-model eval runs |
| Authorization boundaries | MISSING | DONE | `lumina_reader` role, server-side mutation gate (403), optional API keys, NetworkPolicy (verified on kind) |
| Retry behavior | PARTIAL | DONE | Three-policy taxonomy (`src/failures.py`); 30/30 scripted scenarios |
| Timeouts | PARTIAL | DONE | Agent deadline, deadline-aware LLM timeout, pool timeout 5 s, statement timeout 15 s (role and session) |
| Concurrency | PARTIAL | DONE | Saturation measured: about 180-200 req/s per 1-vCPU replica (mock LLM), knee at 100-150 concurrent |
| Backpressure | MISSING | DONE | Bounded slots and queue, early 429 + `Retry-After`; trade-offs measured (`docs/evaluation.md`) |
| Caching | PARTIAL | DONE | Prompt version in key, re-validated hits; benchmark: hit 2.6 ms vs 408 ms, miss overhead within noise; kept |
| Rate limiting | MISSING | PARTIAL | Per-client token bucket per replica; global limits need the ingress or Redis |
| Observability | PARTIAL | DONE | Token/cost, failures by category, unsafe rejections by layer, in-flight/queue gauges, admission wait; 15 alert rules; request ID across HTTP, LLM, and SQL |
| Evaluation quality | PARTIAL | DONE | Four suites, negative categories, JSON/CSV reports, deterministic suites in CI |
| Cost measurement | MISSING | DONE | Provider-reported tokens, cost per request and per run, spend alert |
| Deployment | PARTIAL | DONE | Reader role in k8s, superuser secret isolated, NetworkPolicy, grace period covers the deadline; 0/1500 failed requests during a rolling restart |
| Testing | PARTIAL | DONE | 259 tests (86% coverage), security job, scripted eval, container e2e and kind jobs in CI |
