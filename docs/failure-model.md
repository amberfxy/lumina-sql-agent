# Failure model

Every unsuccessful attempt is classified into one category, and the category alone
decides what happens next. The goal is that retries are only spent where they can help:
the model is asked to fix *its own* mistakes, infrastructure blips are retried without
the model, and everything else stops immediately with a precise status.

Source: `src/failures.py` (taxonomy, policies, classifiers), `src/agent.py` (loop),
`api/main.py` (`http_status_for`).

## Taxonomy and policy

| Category | Typical source | Policy | HTTP | Rationale |
| --- | --- | --- | --- | --- |
| `SYNTAX_ERROR` | guard parse failure, SQLSTATE `42601` | LLM-correctable | 200* | The model wrote invalid SQL; the error message is actionable |
| `UNKNOWN_TABLE` | guard (table not in catalog), `42P01` | LLM-correctable | 200* | Hallucinated table; the guard's message lists the real tables |
| `UNKNOWN_COLUMN` | `42703` | LLM-correctable | 200* | Hallucinated column; PostgreSQL often suggests the right one |
| `SEMANTIC_ERROR` | other `42xxx`, `22xxx` (e.g. division by zero, grouping errors) | LLM-correctable | 200* | Query logic error the model can fix |
| `MALFORMED_OUTPUT` | no SQL in the response | LLM-correctable | 200* | Reformatting is cheap and usually works |
| `UNSAFE_QUERY` | guard, `25006` read-only violation | **Terminal** | 200* | Never let the model iterate toward a working exploit |
| `UNANSWERABLE` | model replied `CANNOT_ANSWER:` | Terminal | 200* | A correct refusal is a valid outcome |
| `PERMISSION_ERROR` | `42501` | Terminal | 200* | The role is intentionally limited; retrying cannot help |
| `TIMEOUT` | `57014` statement timeout | Terminal | 200* | Rewriting under a deadline rarely helps and doubles load |
| `TIMEOUT` | LLM SDK gave up | Terminal | 504 | The SDK has already retried |
| `TIMEOUT` | agent deadline | Terminal | 504 | Bounded end-to-end latency |
| `CONNECTION_ERROR` | DB `08xxx`/`53xxx`/`57Pxx`, pool timeout, `40001`/`40P01` | **Transient** | 503 | Same SQL retried with backoff (`DB_TRANSIENT_RETRIES`), no LLM call |
| `RATE_LIMIT` | DynamoDB throttling | Transient | 503 | Same as above |
| `RATE_LIMIT` | LLM `429` after SDK retries | Terminal | 503 | The SDK already backed off (honoring `Retry-After`) |
| `MODEL_ERROR` | LLM 5xx / unexpected SDK error | Terminal | 502 | Provider failure |
| `SCHEMA_UNAVAILABLE` | empty catalog (DB unreachable at schema load) | Terminal | 503 | Nothing to ground the model in |
| `INTERNAL_ERROR` | unclassified | Terminal | 500 | Bug; investigate by `request_id` |

\* Request-level outcomes return `200` with `success: false` and a `failure` object
(`category`, `source`, `policy`, `message`); they are answers about the question, not
service failures.

**Empty results are success.** An empty result set is a valid answer ("no customers in
Atlantis"). Treating it as an error would pressure the model into changing the question's
semantics until rows appear.

**Unsafe queries never reach the model again.** Feeding a guard rejection back would
turn the retry loop into an exploit-search loop.

## Retry budget

- LLM-correctable: up to `MAX_RETRY_ITERATIONS` (default 3) generations per run.
- Transient DB: up to `DB_TRANSIENT_RETRIES` (default 2) re-executions per attempt with
  `DB_RETRY_BACKOFF_SECONDS` exponential backoff.
- LLM transport: the provider SDK retries `429`/5xx/timeouts `LLM_MAX_RETRIES` times.
  Each try's timeout is `min(LLM_TIMEOUT_SECONDS, remaining_deadline / (LLM_MAX_RETRIES + 1))`,
  so one LLM call cannot outlive the run deadline.
- Whole run: `AGENT_DEADLINE_SECONDS` (default 90).

Before the deadline-aware timeout, the defaults (60s timeout, 3 SDK retries) allowed a
single LLM call to take up to 240s against a 90s deadline.

## Does self-correction help?

Measured so far (deterministic, `eval/datasets/scripted.jsonl`, simulated LLM):

| Scenario | Outcome |
| --- | --- |
| Unknown column, hallucinated table, syntax error, type error, division by zero, ambiguous column, grouping error, malformed output (`s02`-`s09`) | Recovered on the next attempt (exactly 1 corrective LLM call each) |
| Same mistake repeated 3 times (`s11`, `s30`) | Stops after 3 generations with the last category |
| Unsafe SQL, 11 variants (`s13`-`s23`) | Terminal on the first attempt, 0 database executions |
| Unsafe SQL after a correctable error (`s24`) | The correction is generated, the unsafe query is not executed or retried |
| Statement timeout (`s25`) | Terminal `TIMEOUT` after one execution |
| LLM 429 / timeout / 5xx (`s27`-`s29`) | Terminal, 0 database executions |
| Database connection blip (`tests/test_agent.py`) | Recovered by a same-SQL retry with no extra LLM call |

These prove the *mechanics*; they do not measure how often a real model makes
correctable mistakes or how often its corrections are right. That is what the real-model
ablation measures: `make eval` (3 attempts) versus `make eval-single-shot` (1 attempt) on
the same dataset; `self_correction_gain_pp` is the difference in execution accuracy and
`retry_recovery_rate` the share of correctable first-attempt failures that a retry fixed.
It requires an LLM API key and has **not** been run for this write-up.

## Observed behavior under injected faults

From `scripts/resilience.py` against the Compose `loadtest` stack (8 closed-loop clients,
1-CPU API, mock LLM 400 ms; `LLM_TIMEOUT_SECONDS=10`, `LLM_MAX_RETRIES=2`). Reports:
`eval/results/resilience-*.json`.

| Fault (duration) | During the fault | Recovery |
| --- | --- | --- |
| LLM returns 429 (20s) | `503 RATE_LIMIT` after SDK backoff, p95 2.8s | first success 0.4s after the fault cleared |
| LLM returns 500 (20s) | `502 MODEL_ERROR`, p95 2.7s | 0.5s |
| LLM hangs (20s) | requests ride it out via SDK retries; slowest 21.8s, all succeeded | immediate |
| LLM hangs (45s) | `504 TIMEOUT` at 31.5s (3 tries x 10s + backoff); slots stay bounded | 8.2s (in-flight hung calls drain) |
| LLM returns prose (20s) | `200` with `MALFORMED_OUTPUT` after 3 attempts, p95 1.4s | 0.5s |
| PostgreSQL stopped (20s) | `503 CONNECTION_ERROR`, fails fast (p95 1.1s, max 6.0s) | 0.6s after the container started |
| Redis stopped (20s) | 100% success; latency rises from cached (p50 13ms) to uncached (p50 383ms) | cache hits resume after the 30s cooldown |

Kubernetes disruptions (`scripts/k8s_disruption_test.sh`, kind, 2 replicas, 10 clients,
1,500 requests each; `eval/results/k8s-disruption-*.json`):

| Disruption | Failed requests |
| --- | --- |
| `kubectl rollout restart` (3 runs) | 0 / 4500 |
| Graceful pod delete | 0 / 1500 |
| Force delete (`--grace-period=0 --force`; kubelet still sends SIGTERM) | 0 / 1500 |
| SIGKILL of the API container (`crictl stop --timeout 0`) | 9 / 1500 (4 in-flight resets, 5 connects to the dead pod) |

Graceful termination drains instead of sleeping: the `preStop` hook creates `DRAIN_FILE`,
after which every response carries `Connection: close` and `/readyz` returns 503 for 10 s.
Keep-alive clients therefore reconnect through the Service while the old pod still
listens, rather than racing the connection close uvicorn performs on SIGTERM. This replaced
a plain 5 s sleep after one CI rollout run on a GitHub runner dropped requests (the drop did
not reproduce locally in 4 runs, 6,000 requests). The marker is removed at startup because a
liveness-triggered restart runs `preStop` but keeps the pod's `/tmp` volume.

Every query is read-only, so clients can safely retry transport errors; that is the
mitigation for hard crashes.
