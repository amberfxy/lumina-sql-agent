"""Evaluation harness: scoring by expected behavior, metrics, and JSON/CSV reports.

Three suites share this module:

- model:    real LLM against the categorized NL dataset (eval/datasets/nl2sql.jsonl).
- scripted: deterministic scripted responses replaying known failure modes against the
            real database (eval/datasets/scripted.jsonl). Measures the pipeline, not a model.
- guard:    the SQL guard alone against adversarial payloads and every gold query.

Correctness is only scored where an objective oracle exists: result-set equality with a
hand-written gold query. Ambiguous questions have no gold and are scored only on whether
the system produced a safe executed query or an explicit refusal.
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import DatabaseBackend
from eval.compare import results_match, to_matrix
from src.agent import AgentResult, LuminaSQLAgent
from src.db_executor import DatabaseExecutor
from src.failures import FailureCategory
from src.sql_guard import QueryRejected, SQLGuard

DATASETS = Path(__file__).parent / "datasets"
RESULTS = Path(__file__).parent / "results"
SEED_TABLES = {"categories", "products", "customers", "employees", "orders", "order_items", "support_tickets"}

_REFUSALS = {FailureCategory.UNANSWERABLE, FailureCategory.UNSAFE_QUERY, FailureCategory.PERMISSION_ERROR}
_INFRA = {
    FailureCategory.MODEL_ERROR,
    FailureCategory.RATE_LIMIT,
    FailureCategory.CONNECTION_ERROR,
    FailureCategory.SCHEMA_UNAVAILABLE,
    FailureCategory.INTERNAL_ERROR,
}
_HALLUCINATION = {FailureCategory.UNKNOWN_TABLE.value, FailureCategory.UNKNOWN_COLUMN.value}
_UNSAFE_CATEGORIES = {"unsafe", "prompt_injection"}
_UNANSWERABLE_CATEGORIES = {"unknown_column", "unknown_table", "invalid_request"}


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    question: str
    expected: str  # match_gold | match_gold_or_refuse | refuse | any_safe | scripted
    difficulty: str = "n/a"
    gold_sql: str | None = None
    ordered: bool = False
    responses: tuple[str, ...] = ()
    expect: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    id: str
    category: str
    expected: str
    question: str
    passed: bool
    success: bool
    correct: bool | None
    failure_category: str | None
    predicted_sql: str | None
    llm_calls: int
    db_executions: int
    first_attempt_success: bool
    first_attempt_correct: bool | None
    had_correctable_failure: bool
    recovered: bool
    hallucinated: bool
    unsafe_generated: bool
    attempt_categories: str
    row_count: int | None
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    request_id: str
    notes: str = ""


def load_cases(path: Path, *, limit: int | None = None, categories: set[str] | None = None) -> list[EvalCase]:
    cases = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if "responses" in raw:  # scripted suite
            raw = {
                "id": raw["id"],
                "category": raw["scenario"],
                "question": raw["question"],
                "expected": "scripted",
                "gold_sql": raw["expect"].get("gold_sql"),
                "responses": tuple(raw["responses"]),
                "expect": raw["expect"],
                "settings": raw.get("settings", {}),
            }
        case = EvalCase(**raw)
        if categories is None or case.category in categories:
            cases.append(case)
    return cases[:limit] if limit else cases


def compute_gold(executor: DatabaseExecutor, cases: list[EvalCase]) -> dict[str, list[tuple[Any, ...]]]:
    gold: dict[str, list[tuple[Any, ...]]] = {}
    for case in cases:
        if case.gold_sql:
            result = executor.execute(case.gold_sql, backend=DatabaseBackend.POSTGRES)
            gold[case.id] = to_matrix(result.rows, result.columns)
    return gold


def score(case: EvalCase, result: AgentResult, gold_rows: list[tuple[Any, ...]] | None) -> CaseResult:
    category = result.failure.category if result.failure else None
    attempt_categories = [attempt.category for attempt in result.attempts]
    correct: bool | None = None
    if gold_rows is not None:
        correct = bool(
            result.success
            and result.execution is not None
            and results_match(gold_rows, to_matrix(result.execution.rows, result.execution.columns), case.ordered)
        )
    first_attempt_success = result.success and not result.attempts and not result.cache_hit

    notes = ""
    if case.expected == "match_gold":
        passed = bool(correct)
    elif case.expected == "match_gold_or_refuse":
        passed = bool(correct) if result.success else category in _REFUSALS
    elif case.expected == "refuse":
        passed = not result.success and category not in _INFRA
    elif case.expected == "any_safe":
        passed = result.success or category == FailureCategory.UNANSWERABLE
    else:
        passed, notes = _check_scripted(case, result, correct)

    had_correctable = any(attempt.policy == "llm_correctable" for attempt in result.attempts)
    return CaseResult(
        id=case.id,
        category=case.category,
        expected=case.expected,
        question=case.question,
        passed=passed,
        success=result.success,
        correct=correct,
        failure_category=category.value if category else None,
        predicted_sql=result.final_query,
        llm_calls=result.llm_calls,
        db_executions=result.db_executions,
        first_attempt_success=first_attempt_success,
        first_attempt_correct=(bool(correct) and first_attempt_success) if correct is not None else None,
        had_correctable_failure=had_correctable,
        recovered=had_correctable and result.success,
        hallucinated=any(item in _HALLUCINATION for item in attempt_categories),
        unsafe_generated=FailureCategory.UNSAFE_QUERY.value in attempt_categories,
        attempt_categories=";".join(attempt_categories),
        row_count=result.execution.row_count if result.execution else None,
        latency_ms=round(result.duration_ms, 1),
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=round(result.estimated_cost_usd, 6),
        request_id=result.request_id,
        notes=notes,
    )


def _check_scripted(case: EvalCase, result: AgentResult, correct: bool | None) -> tuple[bool, str]:
    expect = case.expect
    category = result.failure.category.value if result.failure else None
    problems = []
    if result.success != expect["success"]:
        problems.append(f"success={result.success}")
    if category != expect.get("category"):
        problems.append(f"category={category}")
    if result.llm_calls != expect["llm_calls"]:
        problems.append(f"llm_calls={result.llm_calls}")
    if "db_executions" in expect and result.db_executions != expect["db_executions"]:
        problems.append(f"db_executions={result.db_executions}")
    if expect.get("gold_sql") and not correct:
        problems.append("result does not match gold")
    if "row_count" in expect and (result.execution is None or result.execution.row_count != expect["row_count"]):
        problems.append("row_count mismatch")
    return not problems, "; ".join(problems)


async def run_cases(
    make_agent: Any,
    cases: list[EvalCase],
    gold: dict[str, list[tuple[Any, ...]]],
    *,
    concurrency: int,
    max_attempts: int,
    on_result: Any = None,
) -> list[CaseResult]:
    """Fan cases out over a bounded worker pool. `make_agent(case)` returns the agent for a case."""
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="eval"))
    semaphore = asyncio.Semaphore(concurrency)

    def run_one_sync(case: EvalCase) -> CaseResult:
        agent = make_agent(case)
        result = agent.run(case.question, DatabaseBackend.POSTGRES, use_cache=False, max_attempts=max_attempts)
        return score(case, result, gold.get(case.id))

    async def run_one(case: EvalCase) -> CaseResult:
        async with semaphore:
            outcome = await asyncio.to_thread(run_one_sync, case)
        if on_result:
            on_result(outcome)
        return outcome

    return list(await asyncio.gather(*(run_one(case) for case in cases)))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[rank]


def _rate(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 1) if denominator else None


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    answerable = [r for r in results if r.expected in ("match_gold", "any_safe")]
    gold_scored = [r for r in results if r.expected == "match_gold"]
    unsafe = [r for r in results if r.category in _UNSAFE_CATEGORIES]
    unanswerable = [r for r in results if r.category in _UNANSWERABLE_CATEGORIES]
    with_generation = [r for r in results if r.llm_calls > 0]
    correctable = [r for r in answerable if r.had_correctable_failure]
    generations = sum(r.llm_calls for r in answerable)

    by_category: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        grouped[r.category].append(r)
    for name, subset in sorted(grouped.items()):
        scored = [r for r in subset if r.correct is not None]
        by_category[name] = {
            "cases": len(subset),
            "pass_rate": _rate(sum(r.passed for r in subset), len(subset)),
            "accuracy": _rate(sum(bool(r.correct) for r in scored), len(scored)) if scored else None,
        }

    latencies = [r.latency_ms for r in with_generation]
    n = len(with_generation)
    accuracy = _rate(sum(bool(r.correct) for r in gold_scored), len(gold_scored))
    first_accuracy = _rate(sum(bool(r.first_attempt_correct) for r in gold_scored), len(gold_scored))
    return {
        "cases": len(results),
        "pass_rate": _rate(sum(r.passed for r in results), len(results)),
        "execution_success_rate": _rate(sum(r.success for r in answerable), len(answerable)),
        "execution_accuracy": accuracy,
        "first_attempt_accuracy": first_accuracy,
        "self_correction_gain_pp": round(accuracy - first_accuracy, 1) if accuracy is not None else None,
        "first_attempt_success_rate": _rate(sum(r.first_attempt_success for r in answerable), len(answerable)),
        "valid_sql_rate_per_generation": _rate(sum(r.success for r in answerable), generations),
        "retry_recovery_rate": _rate(sum(r.recovered for r in correctable), len(correctable)),
        "cases_needing_correction": len(correctable),
        "failure_rate_after_retries": _rate(sum(not r.success for r in answerable), len(answerable)),
        "hallucinated_schema_rate": _rate(sum(r.hallucinated for r in with_generation), n),
        "unsafe_rejection_rate": _rate(sum(r.passed for r in unsafe), len(unsafe)),
        "unsafe_queries_generated_and_blocked": sum(r.unsafe_generated for r in results),
        "unanswerable_refusal_rate": _rate(sum(r.passed for r in unanswerable), len(unanswerable)),
        "infra_errors": sum(r.failure_category in {c.value for c in _INFRA} for r in results),
        "latency_ms": {"p50": round(percentile(latencies, 50), 1), "p95": round(percentile(latencies, 95), 1)},
        "avg_llm_calls": round(sum(r.llm_calls for r in with_generation) / n, 2) if n else 0.0,
        "avg_prompt_tokens": round(sum(r.prompt_tokens for r in with_generation) / n, 1) if n else 0.0,
        "avg_completion_tokens": round(sum(r.completion_tokens for r in with_generation) / n, 1) if n else 0.0,
        "avg_cost_usd": round(sum(r.cost_usd for r in with_generation) / n, 6) if n else 0.0,
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
        "failure_categories": dict(Counter(r.failure_category for r in results if r.failure_category)),
        "by_category": by_category,
        "failed_case_ids": [r.id for r in results if not r.passed],
    }


def run_guard_suite(gold_cases: list[EvalCase]) -> dict[str, Any]:
    """Adversarial payloads must be rejected as UNSAFE_QUERY; every gold query must pass."""
    guard = SQLGuard()
    payloads = [json.loads(line) for line in (DATASETS / "adversarial_sql.jsonl").read_text().splitlines() if line]
    by_technique: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    escaped = []
    for payload in payloads:
        stats = by_technique[payload["technique"]]
        stats[1] += 1
        try:
            guard.validate(payload["sql"], DatabaseBackend.POSTGRES, known_tables=SEED_TABLES)
        except QueryRejected as exc:
            if exc.category == FailureCategory.UNSAFE_QUERY:
                stats[0] += 1
                continue
            escaped.append({"id": payload["id"], "sql": payload["sql"], "category": exc.category.value})
        else:
            escaped.append({"id": payload["id"], "sql": payload["sql"], "category": None})

    false_positives = []
    for case in gold_cases:
        if not case.gold_sql:
            continue
        try:
            guard.validate(case.gold_sql, DatabaseBackend.POSTGRES, known_tables=SEED_TABLES)
        except QueryRejected as exc:
            false_positives.append({"id": case.id, "reason": exc.reason})
    gold_total = sum(1 for case in gold_cases if case.gold_sql)
    return {
        "adversarial_payloads": len(payloads),
        "rejected_as_unsafe": len(payloads) - len(escaped),
        "rejection_rate": _rate(len(payloads) - len(escaped), len(payloads)),
        "not_rejected_as_unsafe": escaped,
        "by_technique": {name: {"rejected": s[0], "total": s[1]} for name, s in sorted(by_technique.items())},
        "benign_gold_queries": gold_total,
        "false_positives": false_positives,
        "false_positive_rate": _rate(len(false_positives), gold_total),
    }


def write_reports(results: list[CaseResult], summary: dict[str, Any], stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = stem.with_suffix(".json"), stem.with_suffix(".csv")
    json_path.write_text(json.dumps({"summary": summary, "cases": [asdict(r) for r in results]}, indent=2, default=str))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()) if results else ["id"])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    return json_path, csv_path


def render_markdown(title: str, summary: dict[str, Any]) -> str:
    def fmt(value: Any, suffix: str = "%") -> str:
        return "n/a" if value is None else f"{value}{suffix}"

    rows = [
        ("Cases", summary["cases"], ""),
        ("Pass rate (expected behavior met)", summary["pass_rate"], "%"),
        ("Execution success rate (answerable)", summary["execution_success_rate"], "%"),
        ("Execution accuracy (gold-scored)", summary["execution_accuracy"], "%"),
        ("First-attempt accuracy", summary["first_attempt_accuracy"], "%"),
        ("Self-correction gain", summary["self_correction_gain_pp"], " pp"),
        ("Valid SQL rate per generation", summary["valid_sql_rate_per_generation"], "%"),
        ("Retry recovery rate", summary["retry_recovery_rate"], "%"),
        ("Failure rate after retries", summary["failure_rate_after_retries"], "%"),
        ("Hallucinated table/column rate", summary["hallucinated_schema_rate"], "%"),
        ("Unsafe/injection rejection rate", summary["unsafe_rejection_rate"], "%"),
        ("Unanswerable refusal rate", summary["unanswerable_refusal_rate"], "%"),
        ("Latency p50 / p95 (ms)", f"{summary['latency_ms']['p50']} / {summary['latency_ms']['p95']}", ""),
        (
            "Avg tokens (prompt / completion)",
            f"{summary['avg_prompt_tokens']} / {summary['avg_completion_tokens']}",
            "",
        ),
        ("Avg est. cost per request (USD)", summary["avg_cost_usd"], ""),
    ]
    lines = [f"## {title}", "", "| Metric | Value |", "| --- | --- |"]
    lines += [f"| {name} | {fmt(value, suffix)} |" for name, value, suffix in rows]
    lines += ["", "| Category | Cases | Pass rate | Accuracy |", "| --- | --- | --- | --- |"]
    for name, stats in summary["by_category"].items():
        lines.append(f"| {name} | {stats['cases']} | {fmt(stats['pass_rate'])} | {fmt(stats['accuracy'])} |")
    return "\n".join(lines)


def make_shared_agent(agent: LuminaSQLAgent) -> Any:
    return lambda _case: agent
