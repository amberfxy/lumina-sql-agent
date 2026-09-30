"""Concurrent batch evaluation of LuminaSQL-Agent against the gold dataset.

Usage:
    python -m evaluation.run_eval --concurrency 8
    python -m evaluation.run_eval --validate-gold   # check the dataset only, no LLM calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import DatabaseBackend, Settings, get_settings
from evaluation.compare import results_match, to_matrix
from src.agent import LuminaSQLAgent
from src.db_executor import DatabaseExecutionError, DatabaseExecutor

logger = logging.getLogger("evaluation")

DEFAULT_DATASET = Path(__file__).parent / "dataset.jsonl"
DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


@dataclass(frozen=True)
class EvalCase:
    id: str
    difficulty: str
    question: str
    gold_sql: str
    ordered: bool


@dataclass
class CaseResult:
    id: str
    difficulty: str
    question: str
    gold_sql: str
    predicted_sql: str | None
    executed: bool
    correct: bool
    first_attempt_executed: bool
    first_attempt_correct: bool
    executions: int
    llm_calls: int
    latency_ms: float
    error: str | None = None


def load_cases(path: Path, limit: int | None = None) -> list[EvalCase]:
    cases = [EvalCase(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    return cases[:limit] if limit else cases


def compute_gold(executor: DatabaseExecutor, cases: list[EvalCase]) -> dict[str, list[tuple[Any, ...]]]:
    gold: dict[str, list[tuple[Any, ...]]] = {}
    for case in cases:
        result = executor.execute(case.gold_sql, backend=DatabaseBackend.POSTGRES)
        gold[case.id] = to_matrix(result.rows, result.columns)
    return gold


def evaluate_case(
    agent: LuminaSQLAgent,
    case: EvalCase,
    gold_rows: list[tuple[Any, ...]],
    max_attempts: int,
) -> CaseResult:
    started = time.perf_counter()
    try:
        result = agent.run(case.question, DatabaseBackend.POSTGRES, use_cache=False, max_attempts=max_attempts)
    except Exception as exc:  # noqa: BLE001 - LLM/API failures are recorded, not fatal
        return CaseResult(
            id=case.id,
            difficulty=case.difficulty,
            question=case.question,
            gold_sql=case.gold_sql,
            predicted_sql=None,
            executed=False,
            correct=False,
            first_attempt_executed=False,
            first_attempt_correct=False,
            executions=0,
            llm_calls=0,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{exc.__class__.__name__}: {exc}"[:500],
        )

    correct = False
    if result.success and result.execution is not None:
        predicted = to_matrix(result.execution.rows, result.execution.columns)
        correct = results_match(gold_rows, predicted, case.ordered)

    first_attempt_executed = result.success and not result.attempts
    return CaseResult(
        id=case.id,
        difficulty=case.difficulty,
        question=case.question,
        gold_sql=case.gold_sql,
        predicted_sql=result.final_query,
        executed=result.success,
        correct=correct,
        first_attempt_executed=first_attempt_executed,
        first_attempt_correct=correct and first_attempt_executed,
        executions=len(result.attempts) + (1 if result.success else 0),
        llm_calls=result.llm_calls,
        latency_ms=(time.perf_counter() - started) * 1000,
        error=None if result.success else result.message[:500],
    )


async def run_cases(
    agent: LuminaSQLAgent,
    cases: list[EvalCase],
    gold: dict[str, list[tuple[Any, ...]]],
    *,
    concurrency: int,
    max_attempts: int,
) -> list[CaseResult]:
    """Fan out cases over a bounded worker pool; the semaphore caps in-flight LLM/DB work."""
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="eval"))
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def run_one(case: EvalCase) -> CaseResult:
        nonlocal completed
        async with semaphore:
            outcome = await asyncio.to_thread(evaluate_case, agent, case, gold[case.id], max_attempts)
        completed += 1
        status = "PASS" if outcome.correct else ("EXEC" if outcome.executed else "FAIL")
        logger.info("[%d/%d] %s %s (%.0f ms)", completed, len(cases), status, case.id, outcome.latency_ms)
        return outcome

    return await asyncio.gather(*(run_one(case) for case in cases))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[rank]


def _ratio(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 1) if denominator else 0.0


def summarize(
    results: list[CaseResult],
    *,
    wall_seconds: float,
    concurrency: int,
    max_attempts: int,
    settings: Settings,
) -> dict[str, Any]:
    total = len(results)
    latencies = [result.latency_ms for result in results if result.llm_calls > 0]
    by_difficulty: dict[str, dict[str, Any]] = {}
    for difficulty in ("easy", "medium", "hard"):
        subset = [result for result in results if result.difficulty == difficulty]
        if subset:
            by_difficulty[difficulty] = {
                "cases": len(subset),
                "execution_accuracy": _ratio(sum(r.correct for r in subset), len(subset)),
                "first_attempt_accuracy": _ratio(sum(r.first_attempt_correct for r in subset), len(subset)),
            }

    correct = sum(result.correct for result in results)
    first_attempt_correct = sum(result.first_attempt_correct for result in results)
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": settings.llm_provider.value,
        "model": settings.active_llm_model,
        "cases": total,
        "concurrency": concurrency,
        "max_attempts": max_attempts,
        "execution_accuracy": _ratio(correct, total),
        "first_attempt_accuracy": _ratio(first_attempt_correct, total),
        "self_correction_gain_pp": round(_ratio(correct, total) - _ratio(first_attempt_correct, total), 1),
        "valid_sql_rate": _ratio(sum(result.executed for result in results), total),
        "first_attempt_valid_sql_rate": _ratio(sum(result.first_attempt_executed for result in results), total),
        "recovered_by_self_correction": sum(result.correct and not result.first_attempt_executed for result in results),
        "avg_llm_calls": round(sum(result.llm_calls for result in results) / total, 2) if total else 0.0,
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
        "wall_clock_seconds": round(wall_seconds, 2),
        "throughput_cases_per_second": round(total / wall_seconds, 3) if wall_seconds else 0.0,
        "infra_errors": sum(1 for result in results if result.llm_calls == 0 and result.error),
        "by_difficulty": by_difficulty,
        "incorrect_case_ids": [result.id for result in results if not result.correct],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"## Evaluation: {summary['model']} ({summary['cases']} cases, concurrency={summary['concurrency']})",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Execution accuracy | {summary['execution_accuracy']}% |",
        f"| First-attempt accuracy (no self-correction) | {summary['first_attempt_accuracy']}% |",
        f"| Self-correction gain | +{summary['self_correction_gain_pp']} pp |",
        f"| Valid SQL rate | {summary['valid_sql_rate']}% (first attempt {summary['first_attempt_valid_sql_rate']}%) |",
        f"| Cases recovered by self-correction | {summary['recovered_by_self_correction']} |",
        f"| Avg LLM calls per question | {summary['avg_llm_calls']} |",
        f"| Latency p50 / p95 | {summary['latency_ms']['p50']} ms / {summary['latency_ms']['p95']} ms |",
        f"| Wall clock | {summary['wall_clock_seconds']} s ({summary['throughput_cases_per_second']} cases/s) |",
        "",
        "| Difficulty | Cases | Accuracy | First-attempt |",
        "| --- | --- | --- | --- |",
    ]
    for difficulty, stats in summary["by_difficulty"].items():
        lines.append(
            f"| {difficulty} | {stats['cases']} | {stats['execution_accuracy']}% | {stats['first_attempt_accuracy']}% |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=None, help="defaults to MAX_RETRY_ITERATIONS")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases")
    parser.add_argument("--output", type=Path, default=None, help="results JSON path")
    parser.add_argument("--validate-gold", action="store_true", help="only check that gold SQL runs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("src", "httpx", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = get_settings().model_copy(
        update={"postgres_pool_size": max(5, args.concurrency), "redis_url": None, "log_level": "WARNING"}
    )
    cases = load_cases(args.dataset, args.limit)
    executor = DatabaseExecutor(settings=settings)

    try:
        gold = compute_gold(executor, cases)
    except DatabaseExecutionError as exc:
        logger.error("Gold SQL failed: %s", exc.raw_error)
        return 1

    if args.validate_gold:
        empty = [case_id for case_id, rows in gold.items() if not rows]
        logger.info("Validated %d gold queries (%d with empty results)", len(gold), len(empty))
        return 1 if empty else 0

    max_attempts = args.max_attempts or settings.max_retry_iterations
    agent = LuminaSQLAgent(settings=settings, db_executor=executor)

    started = time.perf_counter()
    results = asyncio.run(run_cases(agent, cases, gold, concurrency=args.concurrency, max_attempts=max_attempts))
    wall_seconds = time.perf_counter() - started

    summary = summarize(
        results,
        wall_seconds=wall_seconds,
        concurrency=args.concurrency,
        max_attempts=max_attempts,
        settings=settings,
    )
    output = args.output or DEFAULT_RESULTS_DIR / (
        f"{settings.active_llm_model.replace('/', '_')}-c{args.concurrency}-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2, default=str))

    print(render_markdown(summary))
    print(f"\nResults written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
