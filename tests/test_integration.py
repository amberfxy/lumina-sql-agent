"""Integration tests against the seeded PostgreSQL database and a real Redis."""

from __future__ import annotations

import pytest

from config import DatabaseBackend, Settings
from evaluation.run_eval import DEFAULT_DATASET, compute_gold, load_cases
from src.cache import RedisCache
from src.db_executor import DatabaseExecutionError, DatabaseExecutor
from src.schema_manager import SchemaManager

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pg_settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def executor(pg_settings):
    executor = DatabaseExecutor(pg_settings)
    yield executor
    executor.close()


def test_schema_extraction_reads_keys_and_comments(pg_settings, executor):
    catalog = SchemaManager(settings=pg_settings, db_executor=executor)._get_catalog(DatabaseBackend.POSTGRES)
    tables = {table.name: table for table in catalog.tables}

    assert set(tables) >= {"customers", "orders", "order_items", "products", "categories", "employees"}
    order_customer = next(column for column in tables["orders"].columns if column.name == "customer_id")
    assert order_customer.foreign_key_target == "customers.customer_id"
    assert "Line revenue" in tables["order_items"].description


def test_read_only_transaction_blocks_writes_the_keyword_filter_misses(executor):
    # nextval() mutates a sequence but contains no blocked keyword.
    with pytest.raises(DatabaseExecutionError) as excinfo:
        executor.execute("SELECT nextval('orders_order_id_seq')", backend=DatabaseBackend.POSTGRES)
    assert "read-only transaction" in excinfo.value.raw_error


def test_statement_timeout_is_enforced():
    executor = DatabaseExecutor(Settings(postgres_statement_timeout_ms=200))
    try:
        with pytest.raises(DatabaseExecutionError) as excinfo:
            executor.execute("SELECT pg_sleep(2)", backend=DatabaseBackend.POSTGRES)
        assert "statement timeout" in excinfo.value.raw_error
    finally:
        executor.close()


def test_every_gold_query_runs_and_returns_rows(executor):
    gold = compute_gold(executor, load_cases(DEFAULT_DATASET))
    assert len(gold) >= 50
    assert all(rows for rows in gold.values())


def test_health_check(executor):
    assert executor.health_check(DatabaseBackend.POSTGRES)["success"] is True


def test_real_redis_round_trip(pg_settings):
    cache = RedisCache.from_settings(pg_settings)
    assert cache.ping()
    cache.set("integration:k", "v", ttl_seconds=30)
    assert cache.get("integration:k", kind="query") == "v"
    cache.delete("integration:k")
