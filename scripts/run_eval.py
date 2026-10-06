"""Run an evaluation suite and write JSON + CSV reports to eval/results/.

Usage:
    python scripts/run_eval.py guard                       # SQL guard vs adversarial corpus (no DB, no LLM)
    python scripts/run_eval.py validate-gold               # every gold query runs and returns rows (DB)
    python scripts/run_eval.py scripted                    # deterministic failure-mode suite (DB, no LLM)
    python scripts/run_eval.py mcp                         # MCP tool sessions through a FastMCP client (DB, no LLM)
    python scripts/run_eval.py model --concurrency 8       # real LLM on the categorized dataset (DB + key)
    python scripts/run_eval.py model --max-attempts 1      # same, with self-correction disabled
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from eval.harness import (  # noqa: E402
    DATASETS,
    RESULTS,
    CaseResult,
    EvalCase,
    compute_gold,
    load_cases,
    render_markdown,
    run_cases,
    run_guard_suite,
    summarize,
    write_reports,
)
from eval.mcp_harness import load_mcp_cases, render_mcp_markdown, run_mcp_suite, summarize_mcp  # noqa: E402
from eval.scripted_llm import ScriptedLLM  # noqa: E402
from src.agent import LuminaSQLAgent  # noqa: E402
from src.db_executor import DatabaseExecutionError, DatabaseExecutor  # noqa: E402
from src.schema_manager import SchemaManager  # noqa: E402

logger = logging.getLogger("eval")


def _progress(total: int):
    done = 0

    def report(result: CaseResult) -> None:
        nonlocal done
        done += 1
        status = "PASS" if result.passed else "FAIL"
        logger.info(
            "[%d/%d] %s %-5s %-22s %6.0f ms %s",
            done,
            total,
            status,
            result.id,
            result.category,
            result.latency_ms,
            result.failure_category or "",
        )

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite", choices=["guard", "validate-gold", "scripted", "mcp", "model"])
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=None, help="defaults to MAX_RETRY_ITERATIONS")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", action="append", default=None, help="restrict to a category (repeatable)")
    parser.add_argument("--output", type=Path, default=None, help="report path stem (no extension)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("src", "httpx", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    nl_cases = load_cases(DATASETS / "nl2sql.jsonl")
    if args.suite == "guard":
        report = run_guard_suite(nl_cases)
        stem = args.output or RESULTS / f"guard-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 0 if not report["not_rejected_as_unsafe"] and not report["false_positives"] else 1

    settings = get_settings().model_copy(
        update={"postgres_pool_size": max(10, args.concurrency), "redis_url": None, "log_level": "ERROR"}
    )
    executor = DatabaseExecutor(settings=settings)
    if args.suite == "validate-gold":
        try:
            gold = compute_gold(executor, nl_cases)
        except DatabaseExecutionError as exc:
            logger.error("Gold SQL failed: %s", exc.raw_error)
            return 1
        empty = [case_id for case_id, rows in gold.items() if not rows]
        logger.info("Validated %d gold queries (%d with empty results)", len(gold), len(empty))
        return 1 if empty else 0

    if args.suite == "mcp":
        return _run_mcp(args, settings, executor)

    dataset = args.dataset or DATASETS / ("scripted.jsonl" if args.suite == "scripted" else "nl2sql.jsonl")
    categories = set(args.category) if args.category else None
    cases = load_cases(dataset, limit=args.limit, categories=categories)
    gold = compute_gold(executor, cases)
    schema_manager = SchemaManager(settings=settings, db_executor=executor)
    max_attempts = args.max_attempts or settings.max_retry_iterations

    if args.suite == "scripted":
        executors: dict[str, DatabaseExecutor] = {}

        def make_agent(case: EvalCase) -> LuminaSQLAgent:
            case_executor = executor
            if case.settings:
                case_executor = DatabaseExecutor(settings=settings.model_copy(update=case.settings))
                executors[case.id] = case_executor
            return LuminaSQLAgent(
                settings=settings,
                schema_manager=schema_manager,
                db_executor=case_executor,
                llm_client=ScriptedLLM(list(case.responses)),
            )

        label = "scripted (deterministic, simulated LLM; token counts approximate)"
    else:
        agent = LuminaSQLAgent(settings=settings, schema_manager=schema_manager, db_executor=executor)

        def make_agent(case: EvalCase) -> LuminaSQLAgent:
            return agent

        label = f"model {settings.llm_provider.value}/{settings.active_llm_model}"

    started = time.perf_counter()
    results = asyncio.run(
        run_cases(
            make_agent,
            cases,
            gold,
            concurrency=args.concurrency,
            max_attempts=max_attempts,
            on_result=_progress(len(cases)),
        )
    )
    wall = time.perf_counter() - started
    order = {case.id: index for index, case in enumerate(cases)}
    results.sort(key=lambda result: order[result.id])

    summary = summarize(results)
    summary.update(
        {
            "suite": args.suite,
            "label": label,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset": str(dataset.relative_to(Path(__file__).resolve().parents[1])),
            "concurrency": args.concurrency,
            "max_attempts": max_attempts,
            "wall_clock_seconds": round(wall, 2),
        }
    )
    stem = args.output or RESULTS / f"{args.suite}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    json_path, csv_path = write_reports(results, summary, stem)
    print(render_markdown(f"Evaluation: {label}", summary))
    print(f"\nReports: {json_path} {csv_path}")
    return 0 if args.suite == "model" or all(result.passed for result in results) else 1


def _run_mcp(args: argparse.Namespace, settings, executor: DatabaseExecutor) -> int:
    dataset = args.dataset or DATASETS / "mcp.jsonl"
    cases = load_mcp_cases(dataset)[: args.limit] if args.limit else load_mcp_cases(dataset)
    schema_manager = SchemaManager(settings=settings, db_executor=executor)

    def report(result) -> None:
        logger.info("%s %-5s %s", "PASS" if result.passed else "FAIL", result.id, result.scenario)

    started = time.perf_counter()
    results = run_mcp_suite(settings, executor, schema_manager, cases, on_result=report)
    summary = summarize_mcp(results)
    summary.update(
        {
            "suite": "mcp",
            "label": "MCP tool sessions (FastMCP in-memory client, real PostgreSQL, scripted LLM)",
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset": str(dataset.relative_to(Path(__file__).resolve().parents[1])),
            "wall_clock_seconds": round(time.perf_counter() - started, 2),
        }
    )
    stem = args.output or RESULTS / f"mcp-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "cases": [asdict(result) for result in results]}
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str))
    print(render_mcp_markdown(summary))
    print(f"\nReport: {stem.with_suffix('.json')}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
