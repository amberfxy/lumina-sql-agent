"""Integration tests against the seeded PostgreSQL database and a real Redis.

The API connects as the least-privilege `lumina_reader` role (POSTGRES_USER in Compose/CI).
"""

from __future__ import annotations

import asyncio

import pytest

from config import DatabaseBackend, Settings
from eval.harness import DATASETS, compute_gold, load_cases, run_cases, summarize
from eval.scripted_llm import ScriptedLLM
from src.agent import LuminaSQLAgent
from src.cache import RedisCache
from src.db_executor import DatabaseExecutionError, DatabaseExecutor
from src.failures import FailureCategory
from src.schema_manager import SchemaManager

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pg_settings() -> Settings:
    return Settings(redis_url=None)


@pytest.fixture(scope="module")
def executor(pg_settings):
    executor = DatabaseExecutor(pg_settings)
    yield executor
    executor.close()


def test_api_connects_as_least_privilege_role(executor):
    result = executor.execute("SELECT current_user AS u", backend=DatabaseBackend.POSTGRES)
    assert result.rows == [{"u": "lumina_reader"}]


def test_schema_extraction_reads_keys_and_comments(pg_settings, executor):
    catalog = SchemaManager(settings=pg_settings, db_executor=executor)._get_catalog(DatabaseBackend.POSTGRES)
    tables = {table.name: table for table in catalog.tables}

    assert set(tables) >= {"customers", "orders", "order_items", "products", "categories", "employees"}
    order_customer = next(column for column in tables["orders"].columns if column.name == "customer_id")
    assert order_customer.foreign_key_target == "customers.customer_id"
    assert "Line revenue" in tables["order_items"].description


@pytest.mark.parametrize(
    "payload",
    [
        "SET TRANSACTION READ WRITE; SELECT nextval('orders_order_id_seq')",
        "SET TRANSACTION READ WRITE; COPY (SELECT 1) TO PROGRAM 'true'",
        "SELECT length(pg_read_file('/etc/hostname'))",
        "SELECT count(*) FROM pg_catalog.pg_authid",
        "SET TRANSACTION READ WRITE; UPDATE products SET stock_quantity = stock_quantity",
    ],
)
def test_database_role_blocks_audit_payloads_even_without_the_guard(executor, payload):
    """Layer 3 in isolation: call the driver path directly, skipping the guard and READ ONLY txn."""
    with pytest.raises(DatabaseExecutionError) as excinfo:
        executor._execute_postgres(payload, allow_mutations=True, parameters=None)
    assert excinfo.value.category in (FailureCategory.PERMISSION_ERROR, FailureCategory.UNSAFE_QUERY)


def test_read_only_transaction_blocks_writes_without_the_guard(executor):
    with pytest.raises(DatabaseExecutionError) as excinfo:
        executor._execute_postgres("SELECT nextval('orders_order_id_seq')", allow_mutations=False, parameters=None)
    assert excinfo.value.category in (FailureCategory.PERMISSION_ERROR, FailureCategory.UNSAFE_QUERY)


def test_statement_timeout_is_classified_as_terminal_timeout():
    executor = DatabaseExecutor(Settings(postgres_statement_timeout_ms=200, redis_url=None))
    try:
        with pytest.raises(DatabaseExecutionError) as excinfo:
            executor.execute(
                "SELECT COUNT(*) FROM order_items a, order_items b, order_items c", backend=DatabaseBackend.POSTGRES
            )
        assert excinfo.value.category == FailureCategory.TIMEOUT
    finally:
        executor.close()


def test_row_limit_reports_truncation(pg_settings):
    executor = DatabaseExecutor(pg_settings.model_copy(update={"max_result_rows": 10}))
    try:
        result = executor.execute("SELECT order_id FROM orders", backend=DatabaseBackend.POSTGRES)
        assert result.row_count == 10
        assert result.metadata == {"row_limit": 10, "truncated": True}
    finally:
        executor.close()


def test_every_gold_query_runs_and_returns_rows(executor):
    gold = compute_gold(executor, load_cases(DATASETS / "nl2sql.jsonl"))
    assert len(gold) >= 58
    assert all(rows for rows in gold.values())


def test_scripted_failure_mode_suite_passes(pg_settings, executor):
    cases = load_cases(DATASETS / "scripted.jsonl")
    gold = compute_gold(executor, cases)
    schema_manager = SchemaManager(settings=pg_settings, db_executor=executor)
    extra: list[DatabaseExecutor] = []

    def make_agent(case):
        case_executor = executor
        if case.settings:
            case_executor = DatabaseExecutor(pg_settings.model_copy(update=case.settings))
            extra.append(case_executor)
        return LuminaSQLAgent(
            settings=pg_settings,
            schema_manager=schema_manager,
            db_executor=case_executor,
            llm_client=ScriptedLLM(list(case.responses)),
        )

    results = asyncio.run(run_cases(make_agent, cases, gold, concurrency=4, max_attempts=3))
    for case_executor in extra:
        case_executor.close()
    failures = {result.id: result.notes for result in results if not result.passed}
    assert failures == {}
    summary = summarize(results)
    assert summary["unsafe_queries_generated_and_blocked"] >= 12


def test_health_check(executor):
    assert executor.health_check(DatabaseBackend.POSTGRES)["success"] is True


def test_real_redis_round_trip():
    cache = RedisCache.from_settings(Settings())
    assert cache.ping()
    cache.set("integration:k", "v", ttl_seconds=30)
    assert cache.get("integration:k", kind="query") == "v"
    cache.delete("integration:k")
