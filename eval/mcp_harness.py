"""MCP evaluation: scripted tool-call sessions through a real FastMCP client.

Each case in eval/datasets/mcp.jsonl is a sequence of tool calls against the MCP server,
wired to the real database. ask_database runs the real agent with a ScriptedLLM, so this
suite measures the MCP interface and the pipeline behind it (validation, execution,
retry policy). It does not measure model quality or how well a model chooses tools:
the tool sequence is fixed by the dataset.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastmcp import Client

from config import DatabaseBackend, Settings
from eval.compare import results_match, to_matrix
from eval.scripted_llm import ScriptedLLM
from mcp_server.server import create_server
from mcp_server.tools import LuminaMCPService
from src.agent import LuminaSQLAgent
from src.cache import NullCache
from src.db_executor import DatabaseExecutor
from src.schema_manager import SchemaManager

_EXECUTED = {"executed", "success", "self_corrected", "cache_hit"}


@dataclass
class StepResult:
    case_id: str
    scenario: str
    index: int
    tool: str
    passed: bool
    is_error: bool
    status: str | None
    category: str | None
    reached_database: bool
    expected_success: bool
    expected_unsafe: bool
    expected_input_error: bool
    had_correctable_attempt: bool
    problems: str = ""


@dataclass
class McpCaseResult:
    id: str
    scenario: str
    passed: bool
    steps: list[StepResult] = field(default_factory=list)


class CountingExecutor:
    """Counts executions so the report can show unsafe calls never reached the database."""

    def __init__(self, inner: DatabaseExecutor) -> None:
        self.inner = inner
        self.calls = 0

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.inner.execute(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def load_mcp_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compute_step_gold(executor: DatabaseExecutor, cases: list[dict[str, Any]]) -> dict[tuple[str, int], list[Any]]:
    gold = {}
    for case in cases:
        for index, step in enumerate(case["steps"]):
            if sql := step["expect"].get("gold_sql"):
                result = executor.execute(sql, backend=DatabaseBackend.POSTGRES)
                gold[(case["id"], index)] = to_matrix(result.rows, result.columns)
    return gold


def _check(expect: dict[str, Any], result: Any, gold: list[Any] | None) -> list[str]:
    problems = []
    if result.is_error != expect.get("is_error", False):
        problems.append(f"is_error={result.is_error}")
    if result.is_error:
        return problems
    body = result.structured_content or {}
    failure = body.get("failure") or {}
    checks = {
        "valid": body.get("valid"),
        "status": body.get("status"),
        "category": failure.get("category"),
        "policy": failure.get("policy"),
        "llm_calls": body.get("llm_calls"),
        "row_count": body.get("row_count"),
    }
    for key, actual in checks.items():
        if key in expect and expect[key] != actual:
            problems.append(f"{key}={actual}")
    if "tables_include" in expect:
        names = {table["name"] for table in body.get("tables", [])}
        if missing := sorted(set(expect["tables_include"]) - names):
            problems.append(f"missing tables {missing}")
    if gold is not None:
        predicted = to_matrix(body.get("rows", []), body.get("columns", []))
        if not results_match(gold, predicted, expect.get("ordered", False)):
            problems.append("result does not match gold")
    return problems


async def run_mcp_cases(
    settings: Settings,
    executor: DatabaseExecutor,
    schema_manager: SchemaManager,
    cases: list[dict[str, Any]],
    gold: dict[tuple[str, int], list[Any]],
    *,
    on_result: Any = None,
) -> list[McpCaseResult]:
    results = []
    for case in cases:
        counting = CountingExecutor(executor)
        agent = LuminaSQLAgent(
            settings=settings,
            schema_manager=schema_manager,
            db_executor=counting,  # type: ignore[arg-type]
            llm_client=ScriptedLLM(list(case["responses"])),
            cache=NullCache(),
        )
        service = LuminaMCPService(
            settings,
            cache=NullCache(),
            db_executor=counting,  # type: ignore[arg-type]
            schema_manager=schema_manager,
            agent=agent,
        )
        outcome = McpCaseResult(id=case["id"], scenario=case["scenario"], passed=True)
        async with Client(create_server(service)) as client:
            for index, step in enumerate(case["steps"]):
                expect = step["expect"]
                before = counting.calls
                result = await client.call_tool(step["tool"], step["args"], raise_on_error=False)
                problems = _check(expect, result, gold.get((case["id"], index)))
                body = (result.structured_content or {}) if not result.is_error else {}
                outcome.steps.append(
                    StepResult(
                        case_id=case["id"],
                        scenario=case["scenario"],
                        index=index,
                        tool=step["tool"],
                        passed=not problems,
                        is_error=result.is_error,
                        status=body.get("status"),
                        category=(body.get("failure") or {}).get("category"),
                        reached_database=counting.calls > before,
                        expected_success=expect.get("status") in _EXECUTED,
                        expected_unsafe=expect.get("category") == "UNSAFE_QUERY",
                        expected_input_error=bool(expect.get("is_error")),
                        had_correctable_attempt=any(
                            attempt["policy"] == "llm_correctable" for attempt in body.get("attempts", [])
                        ),
                        problems="; ".join(problems),
                    )
                )
        outcome.passed = all(step.passed for step in outcome.steps)
        if on_result:
            on_result(outcome)
        results.append(outcome)
    return results


def _rate(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 1) if denominator else None


def summarize_mcp(results: list[McpCaseResult]) -> dict[str, Any]:
    steps = [step for result in results for step in result.steps]
    expected_success = [s for s in steps if s.expected_success]
    unsafe = [s for s in steps if s.expected_unsafe]
    malformed = [s for s in steps if s.expected_input_error]
    correctable = [s for s in steps if s.tool == "ask_database" and s.had_correctable_attempt]

    by_scenario: dict[str, list[McpCaseResult]] = defaultdict(list)
    for result in results:
        by_scenario[result.scenario].append(result)
    by_tool: dict[str, list[StepResult]] = defaultdict(list)
    for step in steps:
        by_tool[step.tool].append(step)

    return {
        "cases": len(results),
        "tool_calls": len(steps),
        "task_completion_rate": _rate(sum(r.passed for r in results), len(results)),
        "tool_call_success_rate": _rate(sum(s.passed for s in steps), len(steps)),
        "valid_query_execution_rate": _rate(sum(s.passed for s in expected_success), len(expected_success)),
        "valid_query_steps": len(expected_success),
        "unsafe_query_rejection_rate": _rate(
            sum(s.status == "rejected" and s.category == "UNSAFE_QUERY" and not s.reached_database for s in unsafe),
            len(unsafe),
        ),
        "unsafe_query_steps": len(unsafe),
        "unsafe_queries_reaching_database": sum(s.reached_database for s in unsafe),
        "malformed_input_rejection_rate": _rate(sum(s.is_error for s in malformed), len(malformed)),
        "self_correction_success_rate": _rate(sum(s.status in _EXECUTED for s in correctable), len(correctable)),
        "self_correction_steps": len(correctable),
        "by_scenario": {
            name: {"cases": len(subset), "pass_rate": _rate(sum(r.passed for r in subset), len(subset))}
            for name, subset in sorted(by_scenario.items())
        },
        "by_tool": {
            name: {"calls": len(subset), "pass_rate": _rate(sum(s.passed for s in subset), len(subset))}
            for name, subset in sorted(by_tool.items())
        },
        "failed_steps": [f"{s.case_id}#{s.index} {s.tool}: {s.problems}" for s in steps if not s.passed],
    }


def render_mcp_markdown(summary: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{value}%"

    rows = [
        ("Cases / tool calls", f"{summary['cases']} / {summary['tool_calls']}"),
        ("End-to-end task completion rate", fmt(summary["task_completion_rate"])),
        ("Tool-call success rate (expected outcome)", fmt(summary["tool_call_success_rate"])),
        (f"Valid-query execution rate (n={summary['valid_query_steps']})", fmt(summary["valid_query_execution_rate"])),
        (
            f"Unsafe-query rejection rate (n={summary['unsafe_query_steps']})",
            fmt(summary["unsafe_query_rejection_rate"]),
        ),
        ("Unsafe queries that reached the database", str(summary["unsafe_queries_reaching_database"])),
        ("Malformed-input rejection rate", fmt(summary["malformed_input_rejection_rate"])),
        (
            f"Self-correction success rate (n={summary['self_correction_steps']})",
            fmt(summary["self_correction_success_rate"]),
        ),
    ]
    lines = ["## MCP evaluation (scripted tool sessions, simulated LLM)", "", "| Metric | Value |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    lines += ["", "| Scenario | Cases | Pass rate |", "| --- | --- | --- |"]
    lines += [f"| {k} | {v['cases']} | {fmt(v['pass_rate'])} |" for k, v in summary["by_scenario"].items()]
    lines += ["", "| Tool | Calls | Pass rate |", "| --- | --- | --- |"]
    lines += [f"| {k} | {v['calls']} | {fmt(v['pass_rate'])} |" for k, v in summary["by_tool"].items()]
    return "\n".join(lines)


def run_mcp_suite(
    settings: Settings,
    executor: DatabaseExecutor,
    schema_manager: SchemaManager,
    cases: list[dict[str, Any]],
    **kw: Any,
) -> list[McpCaseResult]:
    gold = compute_step_gold(executor, cases)
    return asyncio.run(run_mcp_cases(settings, executor, schema_manager, cases, gold, **kw))
