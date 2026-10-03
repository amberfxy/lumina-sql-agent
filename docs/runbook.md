# Runbook

Operational guide for the LuminaSQL API. Every alert in
`observability/prometheus/alerts.yml` has a section here. Find any single request by its
`request_id` (response header `X-Request-ID`, JSON logs, `pg_stat_activity` query comment).

## Quick reference

| Symptom | First check | Likely cause |
| --- | --- | --- |
| Many `503` with `RATE_LIMIT` | `lumina_agent_run_failures_total{category="RATE_LIMIT"}` | LLM provider quota |
| Many `504` | LLM latency panel, `TIMEOUT` failures | Provider slow or hung |
| Many `502` | `MODEL_ERROR` failures | Provider 5xx |
| Many `503` with `CONNECTION_ERROR` / `SCHEMA_UNAVAILABLE` | `/readyz`, PostgreSQL | Database down or pool exhausted |
| Many `429` | "Shed requests / s" panel | Replica at its concurrency limit |
| Latency up, no errors | Admission wait p95, cache hit ratio | Saturation or Redis down |

## Alerts

### LuminaApiDown / LuminaHighHttpErrorRate
1. `kubectl -n lumina get pods`; check restarts and `kubectl -n lumina logs deploy/lumina-api`.
2. Break 5xx down by failure category on the dashboard: 502/503/504 are dependency
   failures (sections below); 500 is a bug, so find examples by `request_id` in the logs.

### LuminaLlmRateLimited
The provider returned 429 after the SDK's own backoff. Total LLM concurrency is
`replicas x MAX_CONCURRENT_REQUESTS`; lower it or raise the provider quota. The HPA can
make this worse by adding replicas, so cap `maxReplicas` to what the quota supports.

### LuminaLlmErrors / LuminaAgentLatencyP95High (provider slow or failing)
- Requests fail with `504` after at most `LLM_TIMEOUT_SECONDS x (LLM_MAX_RETRIES + 1)`
  (bounded by `AGENT_DEADLINE_SECONDS`). Slots are held during that time, so expect 429s.
- Mitigations: switch provider (`LLM_PROVIDER=anthropic`) or model; temporarily lower
  `LLM_MAX_RETRIES` to fail faster during a known outage.

### LuminaSchemaUnavailable (database unreachable)
1. `kubectl -n lumina get pods -l app.kubernetes.io/name=postgres`; check `/readyz` (503
   while PostgreSQL is unreachable, which takes the pod out of the Service).
2. Requests fail fast with `503 CONNECTION_ERROR` (measured p95 about 1 s); they recover
   by themselves once PostgreSQL is back (measured: first success 0.6 s after restart).
3. If PostgreSQL is up but errors persist, suspect pool exhaustion: raise
   `POSTGRES_POOL_SIZE` / `POSTGRES_MAX_OVERFLOW` or lower `MAX_CONCURRENT_REQUESTS`.

### LuminaDbLatencyP95High
Generated queries are scanning. Find slow statements in `pg_stat_activity` (each carries
`/* request_id=... */`). The role-level `statement_timeout` (15 s) caps the damage;
consider indexes for frequent patterns.

### LuminaValidatorBypass (critical)
The database refused a statement the SQL guard accepted (`layer="database"`). The read-only
role held, but the guard has a gap.
1. Find the request by `request_id` in logs and copy the SQL.
2. Add it to `eval/datasets/adversarial_sql.jsonl`, fix `src/sql_guard.py`, and confirm
   with `make eval-guard`.

### LuminaUnsafeQuerySpike
Someone is probing with injection attempts. The controls are working (these are
rejections). If it comes from one client, revoke its API key or rate-limit it.

### LuminaLoadShedding / LuminaAdmissionQueueWaitHigh
Replicas are at `MAX_CONCURRENT_REQUESTS`. Measured capacity of one 1-vCPU replica is
about 180-200 req/s of API work (mock LLM); with a real LLM the limit is concurrency x
(1 / LLM latency), e.g. 16 slots / 3 s ≈ 5 req/s per replica.
- Scale out (the HPA scales on CPU; for LLM-bound load, CPU stays low, so scale
  manually or add a custom-metrics HPA on `lumina_inflight_requests`).
- Do **not** raise `MAX_CONCURRENT_REQUESTS` past the provider quota.
- If a client ignores `Retry-After` (immediate-retry storms collapse goodput: measured
  60-73 req/s instead of about 140), rate-limit it at the ingress.

### LuminaAgentFailureRateHigh / LuminaSelfCorrectionSpike
A model regression or schema drift. Check "Failed attempts by category": a rise in
`UNKNOWN_COLUMN` / `UNKNOWN_TABLE` after a migration means stale schema. Run
`POST /api/v1/cache/invalidate` on **each replica** (the in-process schema cache is per
replica; otherwise it expires after `SCHEMA_CACHE_TTL_SECONDS`).

### LuminaCacheUnavailable
Redis is down. Requests keep succeeding without the cache (measured: 100% success,
latency moves to the uncached LLM path). The cache bypasses Redis for
`REDIS_FAILURE_COOLDOWN_SECONDS` after an error, so recovery lags Redis by up to 30 s.

### LuminaLlmSpendHigh
Check tokens/s and cost per run on the dashboard. A rise in attempts per run (self-correction)
multiplies cost; so does a drop in cache hit ratio.

## Procedures

### Deploy / rollback
`kubectl apply -k deploy/k8s` rolls out with `maxUnavailable: 0`, a 10 s `preStop` drain
(`Connection: close` on every response, readiness 503), and a 105 s grace period (longer than
the agent deadline). Measured: three rolling restarts under continuous traffic dropped 0 of
4,500 requests. A pod stuck reporting `{"status": "draining"}` on `/readyz` has a stale
`/tmp/lumina-draining`; restarting the container clears it. Roll back with
`kubectl -n lumina rollout undo deployment/lumina-api`.

### Rotate database credentials
`ALTER ROLE lumina_reader PASSWORD '...'` as the superuser, update the `lumina-secrets`
Secret, then `kubectl -n lumina rollout restart deployment/lumina-api`.

### After a schema migration
1. Grant `SELECT` on new tables to `lumina_reader` (`db/init/03_grants.sql`); new tables are
   invisible to the agent until granted and, if set, listed in `ALLOWED_TABLES`.
2. Invalidate caches on each replica (or wait for `SCHEMA_CACHE_TTL_SECONDS`). Cached queries
   are keyed on the pruned schema, so they change automatically once the catalog refreshes.
3. Run `make validate-gold` and `make eval-scripted`.

### Reproduce the measurements
```bash
make loadtest-up                        # mock LLM + 1-CPU API on :8002
make loadtest                           # concurrency sweep
make bench-cache                        # cache hit/miss
make resilience                         # LLM/DB/Redis fault injection
make k8s-up K8S_DIR=deploy/k8s-testing  # kind cluster with in-cluster mock LLM
make k8s-disruption                     # rollout/delete/kill/crash under traffic
```
