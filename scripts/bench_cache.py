"""Measure whether the generated-query cache is worth keeping.

The cache stores generated SQL (never results), so a hit skips the LLM call but still
validates and executes the query. Phases, all sequential to keep queueing out of the
numbers:

1. no_cache  - every gold question with use_cache=false (baseline)
2. miss      - same questions after invalidation with use_cache=true (adds Redis GET+SET)
3. hit       - same questions again (served from cache)
4. replay    - a synthetic Zipf-distributed workload over the same questions, to show how
               hit rate drives mean latency. Real hit rates depend on how often production
               questions repeat verbatim (after case/whitespace normalization), which this
               repository has no data for.

    python scripts/bench_cache.py --base-url http://api-loadtest:8000
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.harness import DATASETS, RESULTS, load_cases, percentile  # noqa: E402


def _summary(samples: list[dict]) -> dict:
    latencies = [sample["ms"] for sample in samples]
    return {
        "requests": len(samples),
        "cache_hits": sum(sample["cache_hit"] for sample in samples),
        "success_rate": round(100 * sum(sample["success"] for sample in samples) / max(1, len(samples)), 1),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
        },
        "avg_llm_tokens": round(statistics.fmean(sample["tokens"] for sample in samples), 1) if samples else 0.0,
    }


def _ask(client: httpx.Client, question: str, use_cache: bool) -> dict:
    started = time.perf_counter()
    response = client.post("/api/v1/query", json={"query": question, "use_cache": use_cache})
    elapsed = (time.perf_counter() - started) * 1000
    body = response.json()
    usage = body.get("usage") or {}
    return {
        "ms": elapsed,
        "status": response.status_code,
        "success": bool(body.get("success")),
        "cache_hit": bool(body.get("cache_hit")),
        "tokens": (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0),
    }


def run(args: argparse.Namespace) -> dict:
    questions = [case.question for case in load_cases(DATASETS / "nl2sql.jsonl") if case.expected == "match_gold"]
    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        health = client.get("/health").json()
        if not health.get("redis", {}).get("success"):
            raise SystemExit("Redis is not reachable from the target API; the cache cannot be measured.")
        try:
            mock = client.get(f"{args.mock_url}/admin/config").json()
        except httpx.HTTPError:
            mock = {}

        for question in questions[:5]:
            _ask(client, question, use_cache=False)

        rounds: list[dict] = []
        for _ in range(args.rounds):
            client.post("/api/v1/cache/invalidate")
            no_cache = [_ask(client, q, use_cache=False) for q in questions]
            miss = [_ask(client, q, use_cache=True) for q in questions]
            hit = [_ask(client, q, use_cache=True) for q in questions]
            rounds.append({"no_cache": _summary(no_cache), "miss": _summary(miss), "hit": _summary(hit)})

        client.post("/api/v1/cache/invalidate")
        rng = random.Random(args.seed)
        weights = [1 / (rank**args.zipf_s) for rank in range(1, len(questions) + 1)]
        workload = rng.choices(questions, weights=weights, k=args.replay_requests)
        replay = [_ask(client, q, use_cache=True) for q in workload]
        replay_baseline = [_ask(client, q, use_cache=False) for q in workload]

    def median_of(phase: str, key: str) -> float:
        return statistics.median(r[phase]["latency_ms"][key] for r in rounds)

    phases = {
        phase: {
            "latency_ms": {key: median_of(phase, key) for key in ("mean", "p50", "p95")},
            "cache_hits": [r[phase]["cache_hits"] for r in rounds],
            "success_rate": [r[phase]["success_rate"] for r in rounds],
            "avg_llm_tokens": statistics.median(r[phase]["avg_llm_tokens"] for r in rounds),
        }
        for phase in ("no_cache", "miss", "hit")
    }
    replay_summary = _summary(replay)
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": args.base_url,
        "mock_llm": {key: mock.get(key) for key in ("latency_ms", "jitter", "fault")},
        "questions": len(questions),
        "rounds": args.rounds,
        "phases_median_of_rounds": phases,
        "miss_overhead_ms_p50": round(phases["miss"]["latency_ms"]["p50"] - phases["no_cache"]["latency_ms"]["p50"], 1),
        "hit_saving_ms_p50": round(phases["no_cache"]["latency_ms"]["p50"] - phases["hit"]["latency_ms"]["p50"], 1),
        "replay": {
            "distribution": f"zipf(s={args.zipf_s}) over {len(questions)} gold questions (synthetic)",
            "requests": args.replay_requests,
            "hit_rate": round(100 * replay_summary["cache_hits"] / max(1, replay_summary["requests"]), 1),
            "with_cache": replay_summary,
            "without_cache": _summary(replay_baseline),
        },
        "raw_rounds": rounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8002")
    parser.add_argument("--mock-url", default="http://mock-llm:9000")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replay-requests", type=int, default=300)
    parser.add_argument("--zipf-s", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label", default="default")
    args = parser.parse_args()

    report = run(args)
    phases = report["phases_median_of_rounds"]
    for phase in ("no_cache", "miss", "hit"):
        latency = phases[phase]["latency_ms"]
        print(
            f"{phase:>8}: p50={latency['p50']:>7}ms p95={latency['p95']:>7}ms "
            f"hits={phases[phase]['cache_hits']} tokens={phases[phase]['avg_llm_tokens']}"
        )
    replay = report["replay"]
    print(
        f"  replay: hit_rate={replay['hit_rate']}%  mean with cache={replay['with_cache']['latency_ms']['mean']}ms "
        f"vs without={replay['without_cache']['latency_ms']['mean']}ms"
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"cache-{args.label}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
