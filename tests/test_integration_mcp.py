"""MCP tools against the seeded PostgreSQL database, connected as `lumina_reader`.

These prove the database layers (READ ONLY transaction, least-privilege role, statement
timeout, row cap) still apply to MCP calls, including when both guards are disabled.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastmcp import Client

from config import Settings
from eval.harness import DATASETS
from eval.mcp_harness import load_mcp_cases, run_mcp_suite, summarize_mcp
from mcp_server.server import create_server
from mcp_server.tools import LuminaMCPService
from src.cache import NullCache
from src.db_executor import DatabaseExecutor
from src.schema_manager import SchemaManager
from src.sql_guard import SQLGuard

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pg_settings() -> Settings:
    return Settings(redis_url=None)


@pytest.fixture
def service(pg_settings):
    service = LuminaMCPService(pg_settings, cache=NullCache())
    yield service
    service.close()


def call(service: LuminaMCPService, tool: str, arguments: dict):
    async def run():
        async with Client(create_server(service)) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    return asyncio.run(run())


def test_get_schema_reads_the_seeded_catalog(service, pg_settings):
    body = call(service, "get_schema", {}).structured_content
    tables = {table["name"]: table for table in body["tables"]}
    assert set(tables) >= {"customers", "orders", "order_items", "products", "categories", "employees"}
    order_customer = next(column for column in tables["orders"]["columns"] if column["name"] == "customer_id")
    assert order_customer["foreign_key"] == "customers.customer_id"
    assert "Line revenue" in tables["order_items"]["description"]
    payload = json.dumps(body)
    assert pg_settings.postgres_password.get_secret_value() not in payload
    assert f"@{pg_settings.postgres_host}:" not in payload  # host is "postgres" in Compose, so check the URL form


def test_execute_runs_as_the_least_privilege_role(service):
    body = call(service, "execute_readonly_query", {"sql": "SELECT current_user AS u"}).structured_content
    assert body["status"] == "executed"
    assert body["rows"] == [{"u": "lumina_reader"}]


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE products SET stock_quantity = stock_quantity",
        "DELETE FROM order_items WHERE order_item_id = -1",
        "SELECT nextval('orders_order_id_seq')",
        "CREATE TABLE mcp_probe (id int)",
    ],
)
def test_database_blocks_writes_even_with_both_guards_disabled(service, monkeypatch, sql):
    """Layers 2 and 3 in isolation: no validator runs, so the READ ONLY transaction and the
    role's privileges are all that stand between the MCP call and the data."""
    monkeypatch.setattr(SQLGuard, "validate", lambda *args, **kwargs: None)
    body = call(service, "execute_readonly_query", {"sql": sql}).structured_content
    assert body["success"] is False
    assert body["status"] == "rejected"
    assert body["failure"]["category"] in ("UNSAFE_QUERY", "PERMISSION_ERROR")
    assert body["failure"]["source"] == "database"


def test_statement_timeout_applies_to_mcp_queries(pg_settings):
    service = LuminaMCPService(pg_settings.model_copy(update={"postgres_statement_timeout_ms": 200}), cache=NullCache())
    try:
        body = call(
            service,
            "execute_readonly_query",
            {"sql": "SELECT COUNT(*) FROM order_items a, order_items b, order_items c"},
        ).structured_content
    finally:
        service.close()
    assert body["status"] == "failed"
    assert body["failure"]["category"] == "TIMEOUT"
    assert body["failure"]["policy"] == "terminal"


def test_row_cap_applies_to_mcp_queries(pg_settings):
    service = LuminaMCPService(pg_settings.model_copy(update={"max_result_rows": 10}), cache=NullCache())
    try:
        body = call(service, "execute_readonly_query", {"sql": "SELECT order_id FROM orders"}).structured_content
    finally:
        service.close()
    assert body["row_count"] == 10
    assert body["truncated"] is True and body["row_limit"] == 10


def test_mcp_eval_suite_passes(pg_settings):
    executor = DatabaseExecutor(pg_settings)
    try:
        cases = load_mcp_cases(DATASETS / "mcp.jsonl")
        results = run_mcp_suite(pg_settings, executor, SchemaManager(settings=pg_settings, db_executor=executor), cases)
    finally:
        executor.close()
    summary = summarize_mcp(results)
    assert summary["failed_steps"] == []
    assert summary["unsafe_queries_reaching_database"] == 0
