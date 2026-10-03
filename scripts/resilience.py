"""Fault-injection runs against the Compose `loadtest` stack under steady traffic.

Each scenario runs closed-loop traffic for three phases (baseline, fault, recovery) and
records every response's HTTP status, failure category, and latency. LLM faults are
injected through the mock LLM's /admin/config; dependency faults stop and start Compose
services. Run from the host (stdlib only, so it works with the system Python):

    make loadtest-up
    python3 scripts/resilience.py                      # all scenarios
    python3 scripts/resilience.py --only redis_down    # one scenario
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
# Must be gold-set questions verbatim: the mock LLM only answers those.
QUESTIONS = [
    "How many customers are there?",
    "How many products are currently active?",
    "How many orders were cancelled?",
    "What is the average list price of all products?",
]


def _post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except ValueError:
            return exc.code, {}


def mock_config(mock_url: str, **settings: object) -> None:
    _post_json(f"{mock_url}/admin/config", settings, timeout=5)


def compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], cwd=ROOT, check=True, capture_output=True)


SCENARIOS = {
    "llm_rate_limit": (
        "Provider returns 429 on every call",
        lambda a: mock_config(a.mock_url, fault="rate_limit", fault_rate=1.0),
        lambda a: mock_config(a.mock_url, fault="none", fault_rate=0.0),
    ),
    "llm_server_error": (
        "Provider returns 500 on every call",
        lambda a: mock_config(a.mock_url, fault="server_error", fault_rate=1.0),
        lambda a: mock_config(a.mock_url, fault="none", fault_rate=0.0),
    ),
    "llm_timeout": (
        "Provider accepts the request and never answers",
        lambda a: mock_config(a.mock_url, fault="timeout", fault_rate=1.0),
        lambda a: mock_config(a.mock_url, fault="none", fault_rate=0.0),
    ),
    "llm_malformed": (
        "Provider answers with prose instead of SQL",
        lambda a: mock_config(a.mock_url, fault="malformed", fault_rate=1.0),
        lambda a: mock_config(a.mock_url, fault="none", fault_rate=0.0),
    ),
    "postgres_down": (
        "PostgreSQL container stopped",
        lambda a: compose("stop", "postgres"),
        lambda a: compose("start", "postgres"),
    ),
    "redis_down": (
        "Redis container stopped (traffic uses the query cache)",
        lambda a: compose("stop", "redis"),
        lambda a: compose("start", "redis"),
    ),
}
CACHED_SCENARIOS = {"redis_down"}


class Traffic:
    def __init__(self, base_url: str, workers: int, timeout: float, use_cache: bool) -> None:
        self.base_url = base_url
        self.use_cache = use_cache
        self.workers = workers
        self.timeout = timeout
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def _worker(self, offset: int) -> None:
        index = offset
        while not self._stop.is_set():
            question = QUESTIONS[index % len(QUESTIONS)]
            index += 1
            started = time.monotonic()
            try:
                status, body = _post_json(
                    f"{self.base_url}/api/v1/query", {"query": question, "use_cache": self.use_cache}, self.timeout
                )
            except (urllib.error.URLError, OSError) as exc:
                status, body = 0, {"transport_error": type(exc).__name__}
            ended = time.monotonic()
            failure = body.get("failure") or {}
            sample = {
                "start": started,
                "end": ended,
                "status": status,
                "success": bool(body.get("success")),
                "category": failure.get("category") or body.get("transport_error") or "",
                "cache_hit": bool(body.get("cache_hit")),
            }
            with self._lock:
                self.samples.append(sample)
            if status in (0, 429):
                time.sleep(0.5)

    def start(self) -> None:
        for offset in range(self.workers):
            thread = threading.Thread(target=self._worker, args=(offset,), daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self, wait: float) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=wait)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))]


def _phase(samples: list[dict], start: float, end: float) -> dict:
    window = [s for s in samples if start <= s["end"] < end]
    latencies = [(s["end"] - s["start"]) * 1000 for s in window]
    return {
        "responses": len(window),
        "success_rate": round(100 * sum(s["success"] for s in window) / max(1, len(window)), 1),
        "status_counts": dict(Counter(str(s["status"]) for s in window)),
        "failure_categories": dict(Counter(s["category"] for s in window if s["category"])),
        "cache_hits": sum(s["cache_hit"] for s in window),
        "latency_ms": {"p50": round(_percentile(latencies, 50)), "p95": round(_percentile(latencies, 95))},
        "max_latency_ms": round(max(latencies)) if latencies else 0,
    }


def run_scenario(name: str, args: argparse.Namespace) -> dict:
    description, inject, clear = SCENARIOS[name]
    traffic = Traffic(args.base_url, args.workers, args.client_timeout, use_cache=name in CACHED_SCENARIOS)
    t0 = time.monotonic()
    traffic.start()
    time.sleep(args.baseline)
    t_fault = time.monotonic()
    inject(args)
    time.sleep(args.fault)
    t_clear = time.monotonic()
    clear(args)
    time.sleep(args.recovery)
    t_end = time.monotonic()
    traffic.stop(wait=args.client_timeout)
    samples = traffic.samples

    first_ok_after_clear = next(
        (s["end"] for s in sorted(samples, key=lambda s: s["end"]) if s["end"] >= t_clear and s["success"]), None
    )
    started_in_fault = [s for s in samples if t_fault <= s["start"] < t_clear]
    return {
        "scenario": name,
        "description": description,
        "phases": {
            "baseline": _phase(samples, t0, t_fault),
            "fault": _phase(samples, t_fault, t_clear),
            "recovery": _phase(samples, t_clear, t_end),
        },
        "requests_started_during_fault": {
            "count": len(started_in_fault),
            "status_counts": dict(Counter(str(s["status"]) for s in started_in_fault)),
            "max_latency_ms": round(max(((s["end"] - s["start"]) * 1000 for s in started_in_fault), default=0)),
        },
        "seconds_to_first_success_after_clear": (
            round(first_ok_after_clear - t_clear, 2) if first_ok_after_clear is not None else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8002")
    parser.add_argument("--mock-url", default="http://localhost:9000")
    parser.add_argument("--only", nargs="+", choices=sorted(SCENARIOS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--baseline", type=float, default=10)
    parser.add_argument("--fault", type=float, default=20)
    parser.add_argument("--recovery", type=float, default=30)
    parser.add_argument("--client-timeout", type=float, default=150)
    parser.add_argument("--label", default="default")
    args = parser.parse_args()

    mock_config(args.mock_url, fault="none", fault_rate=0.0)
    reports = []
    for name in args.only or list(SCENARIOS):
        report = run_scenario(name, args)
        reports.append(report)
        fault = report["phases"]["fault"]
        recovery = report["phases"]["recovery"]
        first_ok = report["seconds_to_first_success_after_clear"]
        print(
            f"{name:<17} fault: ok={fault['success_rate']:>5}% status={fault['status_counts']} "
            f"categories={fault['failure_categories']} p95={fault['latency_ms']['p95']}ms | "
            f"recovery: ok={recovery['success_rate']}% first_ok_after={first_ok}s",
            flush=True,
        )
        time.sleep(5)

    stamp = datetime.now(timezone.utc)  # noqa: UP017 - runs on the host's system Python
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"resilience-{args.label}-{stamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(
        json.dumps(
            {
                "timestamp": stamp.isoformat(timespec="seconds"),
                "target": args.base_url,
                "workers": args.workers,
                "phase_seconds": {"baseline": args.baseline, "fault": args.fault, "recovery": args.recovery},
                "scenarios": reports,
            },
            indent=2,
        )
    )
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
