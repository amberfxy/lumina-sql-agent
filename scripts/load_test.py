"""Closed-loop concurrency sweep against /api/v1/query.

Each level runs `concurrency` workers that send requests back-to-back until the
request budget is spent. Questions cycle through the gold set with caching disabled,
so every request exercises schema pruning, one LLM call, validation, and PostgreSQL.
The LLM is whatever the target API is configured with (the mock in Compose's
`loadtest` profile); reports record the mock latency so results are not mistaken
for real-model latency.

Usage:
    python scripts/load_test.py --base-url http://api-loadtest:8000 --levels 10 25 50 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.harness import DATASETS, RESULTS, load_cases, percentile  # noqa: E402

_GAUGE = re.compile(r"^(lumina_inflight_requests|lumina_queued_requests) ([0-9.e+]+)$", re.MULTILINE)


async def poll_gauges(client: httpx.AsyncClient, stop: asyncio.Event, peaks: dict[str, float]) -> None:
    while not stop.is_set():
        try:
            body = (await client.get("/metrics", timeout=2)).text
            for name, value in _GAUGE.findall(body):
                peaks[name] = max(peaks.get(name, 0.0), float(value))
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.2)


async def run_level(
    base_url: str, questions: list[str], concurrency: int, total: int, honor_retry_after: bool = True
) -> dict:
    latencies_ok: list[float] = []
    statuses: Counter[str] = Counter()
    agent_success = 0
    next_index = 0

    async def worker() -> None:
        # One client and one connection per simulated user. A shared httpx pool becomes a hidden
        # client-side queue at high concurrency and inflates latency by seconds.
        nonlocal next_index, agent_success
        limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
        async with httpx.AsyncClient(base_url=base_url, timeout=120, limits=limits) as client:
            while next_index < total:
                index = next_index
                next_index += 1
                payload = {"query": questions[index % len(questions)], "use_cache": False}
                started = time.perf_counter()
                try:
                    response = await client.post("/api/v1/query", json=payload)
                except httpx.HTTPError as exc:
                    statuses[type(exc).__name__] += 1
                    continue
                elapsed = (time.perf_counter() - started) * 1000
                statuses[str(response.status_code)] += 1
                if response.status_code == 429 and honor_retry_after:
                    await asyncio.sleep(float(response.headers.get("Retry-After", "1")))
                if response.status_code == 200:
                    latencies_ok.append(elapsed)
                    agent_success += bool(response.json().get("success"))

    async with httpx.AsyncClient(base_url=base_url) as metrics_client:
        peaks: dict[str, float] = {}
        stop = asyncio.Event()
        poller = asyncio.create_task(poll_gauges(metrics_client, stop, peaks))
        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - started
        stop.set()
        await poller

    completed = sum(statuses.values())
    shed = statuses.get("429", 0)
    errors = completed - statuses.get("200", 0) - shed
    return {
        "concurrency": concurrency,
        "requests": completed,
        "wall_seconds": round(wall, 2),
        "throughput_ok_rps": round(len(latencies_ok) / wall, 2),
        "latency_ok_ms": {
            "p50": round(percentile(latencies_ok, 50), 1),
            "p95": round(percentile(latencies_ok, 95), 1),
            "p99": round(percentile(latencies_ok, 99), 1),
        },
        "shed_rate": round(100 * shed / completed, 1),
        "error_rate": round(100 * errors / completed, 1),
        "agent_success_rate": round(100 * agent_success / max(1, len(latencies_ok)), 1),
        "status_counts": dict(statuses),
        "peak_inflight": peaks.get("lumina_inflight_requests", 0.0),
        "peak_queued": peaks.get("lumina_queued_requests", 0.0),
    }


def _median_of(trials: list[dict]) -> dict:
    """Median of each headline metric across trials, plus the [min, max] range and raw trials."""

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    def pick(key: str, sub: str | None = None) -> list[float]:
        return [trial[key][sub] if sub else trial[key] for trial in trials]

    return {
        "concurrency": trials[0]["concurrency"],
        "trials": len(trials),
        "requests_per_trial": trials[0]["requests"],
        "throughput_ok_rps": median(pick("throughput_ok_rps")),
        "latency_ok_ms": {p: median(pick("latency_ok_ms", p)) for p in ("p50", "p95", "p99")},
        "shed_rate": median(pick("shed_rate")),
        "error_rate": median(pick("error_rate")),
        "agent_success_rate": median(pick("agent_success_rate")),
        "peak_inflight": max(pick("peak_inflight")),
        "peak_queued": max(pick("peak_queued")),
        "range": {
            "throughput_ok_rps": [min(pick("throughput_ok_rps")), max(pick("throughput_ok_rps"))],
            "p95": [min(pick("latency_ok_ms", "p95")), max(pick("latency_ok_ms", "p95"))],
        },
        "raw_trials": trials,
    }


async def main_async(args: argparse.Namespace) -> dict:
    questions = [case.question for case in load_cases(DATASETS / "nl2sql.jsonl") if case.expected == "match_gold"]
    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        for question in questions[:5]:  # warm the schema catalog and connection pool
            await client.post("/api/v1/query", json={"query": question, "use_cache": False})
        mock = {}
        if args.mock_url:
            try:
                mock = (await client.get(f"{args.mock_url}/admin/config")).json()
            except httpx.HTTPError:
                mock = {}

    levels = []
    for concurrency in args.levels:
        total = max(args.min_requests, concurrency * args.requests_per_worker)
        trials = []
        for _ in range(args.repeats):
            trials.append(await run_level(args.base_url, questions, concurrency, total, not args.ignore_retry_after))
            await asyncio.sleep(2)  # let in-flight work drain between trials
        result = _median_of(trials)
        levels.append(result)
        print(
            f"c={concurrency:>3}  ok_rps={result['throughput_ok_rps']:>6} {result['range']['throughput_ok_rps']}  "
            f"p50={result['latency_ok_ms']['p50']:>7}ms  p95={result['latency_ok_ms']['p95']:>7}ms "
            f"{result['range']['p95']}  shed={result['shed_rate']:>5}%  err={result['error_rate']:>5}%  "
            f"peak_inflight={result['peak_inflight']:.0f} peak_queued={result['peak_queued']:.0f}",
            flush=True,
        )
    return {
        "label": args.label,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": args.base_url,
        "clients_honor_retry_after": not args.ignore_retry_after,
        "mock_llm": {key: mock.get(key) for key in ("latency_ms", "jitter", "fault", "fault_rate")},
        "levels": levels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8002")
    parser.add_argument("--mock-url", default="http://mock-llm:9000")
    parser.add_argument("--levels", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--requests-per-worker", type=int, default=6)
    parser.add_argument("--min-requests", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3, help="trials per level; the median is reported")
    parser.add_argument(
        "--ignore-retry-after", action="store_true", help="retry 429s immediately (models misbehaving clients)"
    )
    parser.add_argument("--label", default="default")
    args = parser.parse_args()

    report = asyncio.run(main_async(args))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"load-{args.label}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
