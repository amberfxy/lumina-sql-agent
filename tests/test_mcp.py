"""MCP adapter tests: tools are called through a real FastMCP client (in-memory transport)
against the same agent, guard, and executor classes the REST API uses."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from decimal import Decimal
from typing import Any

import pytest
from fastmcp import Client
from pydantic import SecretStr

from config import DatabaseBackend
from eval.harness import DATASETS, SEED_TABLES
from eval.scripted_llm import ScriptedLLM
from mcp_server import server as server_module
from mcp_server.server import ApiKeyVerifier, create_server
from mcp_server.tools import LuminaMCPService
from src.agent import LuminaSQLAgent
from src.db_executor import DatabaseExecutor, ExecutionResult
from src.failures import FailureCategory
from src.schema_manager import ColumnMetadata, SchemaCatalog, SchemaManager, TableMetadata
from tests.conftest import FakeExecutor, FakeLLM, FakeSchemaManager, fenced

GOOD_SQL = "SELECT COUNT(*) AS n FROM customers"
BAD_SQL = "SELECT COUNT(nme) FROM customers"
ROWS = [{"n": 320}]

UNSAFE_SQL = [
    "DROP TABLE customers",
    "DELETE FROM orders",
    "UPDATE customers SET email = 'x'",
    "INSERT INTO customers (first_name) VALUES ('x')",
    "CREATE TABLE x AS SELECT * FROM customers",
    "ALTER TABLE customers DROP COLUMN email",
    "GRANT ALL ON customers TO PUBLIC",
    "TRUNCATE orders",
    "SELECT 1; DROP TABLE customers",
    "SELECT COUNT(*) FROM customers; DELETE FROM orders",
    "SELECT 1; /* harmless */ DROP TABLE orders --",
    "SELECT 1 -- comment\n; DELETE FROM orders",
    "WITH d AS (DELETE FROM orders RETURNING *) SELECT COUNT(*) FROM d",
    "SET TRANSACTION READ WRITE; SELECT 1",
    "SELECT * INTO backup FROM customers",
    "SELECT customer_id FROM customers FOR UPDATE",
    "SELECT pg_sleep(10)",
    "SELECT rolname, rolpassword FROM pg_catalog.pg_authid",
    "COPY (SELECT 1) TO PROGRAM 'id'",
]


class SpyExecutor:
    """Records every call; returns `rows` with `metadata` for any query."""

    def __init__(self, rows: list[dict[str, Any]] | None = None, metadata: dict[str, Any] | None = None) -> None:
        self.rows = rows if rows is not None else ROWS
        self.metadata = metadata or {"row_limit": 500, "truncated": False}
        self.calls: list[tuple[str, bool]] = []

    def execute(self, query: str, backend: DatabaseBackend, *, allow_mutations: bool = False, **_: Any):
        self.calls.append((query, allow_mutations))
        return ExecutionResult(
            success=True,
            backend=backend,
            query=query,
            rows=self.rows,
            row_count=len(self.rows),
            columns=list(self.rows[0]) if self.rows else [],
            metadata=self.metadata,
        )

    def close(self) -> None:
        pass


class ExplodingEngine:
    """Stands in for the SQLAlchemy engine; any database access fails the test."""

    def connect(self) -> Any:
        raise AssertionError("query reached the database")


def make_service(settings, *, executor=None, llm=None, tables=None, schema_manager=None) -> LuminaMCPService:
    executor = executor if executor is not None else FakeExecutor({GOOD_SQL: ROWS})
    schema_manager = schema_manager or FakeSchemaManager(tables=tables)
    agent = None
    if llm is not None:
        agent = LuminaSQLAgent(settings=settings, schema_manager=schema_manager, db_executor=executor, llm_client=llm)
    return LuminaMCPService(settings, db_executor=executor, schema_manager=schema_manager, agent=agent)


def call(service: LuminaMCPService, tool: str, arguments: dict[str, Any] | None = None):
    async def run():
        async with Client(create_server(service)) as client:
            return await client.call_tool(tool, arguments or {}, raise_on_error=False)

    return asyncio.run(run())


def error_text(result) -> str:
    return " ".join(getattr(block, "text", "") for block in result.content)


# --- discovery ---------------------------------------------------------------------------


def test_server_exposes_four_read_only_tools(settings):
    async def run():
        async with Client(create_server(make_service(settings))) as client:
            return await client.list_tools()

    tools = {tool.name: tool for tool in asyncio.run(run())}
    assert set(tools) == {"get_schema", "validate_sql", "execute_readonly_query", "ask_database"}
    for tool in tools.values():
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert "allow_mutations" not in json.dumps(tool.input_schema)
    assert tools["execute_readonly_query"].input_schema["required"] == ["sql"]
    assert tools["ask_database"].input_schema["required"] == ["question"]
    assert tools["validate_sql"].output_schema["properties"]["valid"]["type"] == "boolean"


# --- get_schema ----------------------------------------------------------------------------


def test_get_schema_returns_structured_tables(settings):
    result = call(make_service(settings), "get_schema")
    assert not result.is_error
    schema = result.structured_content
    assert schema["backend"] == "postgres"
    assert {table["name"] for table in schema["tables"]} == {"customers", "orders"}
    customers = next(table for table in schema["tables"] if table["name"] == "customers")
    assert customers["columns"][0] == {
        "name": "customer_id",
        "type": "INTEGER",
        "nullable": False,
        "primary_key": True,
        "foreign_key": None,
    }


def test_get_schema_applies_access_boundary_and_hides_connection_details(settings):
    secured = settings.model_copy(
        update={
            "allowed_tables": ["customers", "orders"],
            "denied_columns": ["customers.email"],
            "postgres_password": SecretStr("hunter2-secret"),
            "postgres_host": "db.internal.example",
        }
    )
    catalog = SchemaCatalog(
        backend=DatabaseBackend.POSTGRES,
        tables=[
            TableMetadata("customers", (ColumnMetadata("customer_id", "INTEGER"), ColumnMetadata("email", "TEXT"))),
            TableMetadata("orders", (ColumnMetadata("order_id", "INTEGER"),)),
            TableMetadata("payroll", (ColumnMetadata("salary", "NUMERIC"),)),
        ],
    )
    schema_manager = SchemaManager(settings=secured, db_executor=SpyExecutor())
    schema_manager._extract_postgres_schema = lambda: catalog  # type: ignore[method-assign]
    service = LuminaMCPService(secured, db_executor=SpyExecutor(), schema_manager=schema_manager)

    result = call(service, "get_schema")
    tables = {table["name"]: table for table in result.structured_content["tables"]}
    assert set(tables) == {"customers", "orders"}
    assert [column["name"] for column in tables["customers"]["columns"]] == ["customer_id"]
    payload = json.dumps(result.structured_content)
    assert "hunter2" not in payload and "db.internal" not in payload


def test_get_schema_reports_unavailable_catalog_as_tool_error(settings):
    result = call(make_service(settings, tables=set()), "get_schema")
    assert result.is_error
    assert "Schema catalog is unavailable" in error_text(result)


# --- validate_sql --------------------------------------------------------------------------


def test_validate_sql_accepts_select(settings):
    result = call(make_service(settings), "validate_sql", {"sql": "select count(*) as n from customers;"})
    assert result.structured_content == {
        "valid": True,
        "reason": None,
        "normalized_sql": "SELECT COUNT(*) AS n FROM customers",
        "failure": None,
    }


@pytest.mark.parametrize("sql", UNSAFE_SQL)
def test_validate_sql_rejects_unsafe_sql(settings, sql):
    result = call(make_service(settings, tables=SEED_TABLES), "validate_sql", {"sql": sql})
    body = result.structured_content
    assert body["valid"] is False
    assert body["failure"]["category"] == FailureCategory.UNSAFE_QUERY.value
    assert body["failure"]["policy"] == "terminal"
    assert body["reason"]


@pytest.mark.parametrize(
    ("sql", "category", "fragment"),
    [
        ("SELECT COUNT(*) FROM customers WHERE", "SYNTAX_ERROR", "parse"),
        ("SELECT COUNT(*) FROM clients", "UNKNOWN_TABLE", "Available tables: customers, orders"),
    ],
)
def test_validate_sql_explains_fixable_errors(settings, sql, category, fragment):
    body = call(make_service(settings), "validate_sql", {"sql": sql}).structured_content
    assert body["valid"] is False
    assert body["failure"]["category"] == category
    assert body["failure"]["policy"] == "llm_correctable"
    assert fragment in body["reason"]


# --- execute_readonly_query ----------------------------------------------------------------


def test_execute_runs_select_read_only(settings):
    executor = SpyExecutor()
    result = call(make_service(settings, executor=executor), "execute_readonly_query", {"sql": GOOD_SQL})
    body = result.structured_content
    assert body["success"] is True and body["status"] == "executed"
    assert body["rows"] == ROWS and body["columns"] == ["n"] and body["row_count"] == 1
    assert body["request_id"] not in ("", "-")
    assert executor.calls == [(GOOD_SQL, False)]


@pytest.mark.parametrize("sql", UNSAFE_SQL)
def test_execute_refuses_unsafe_sql_before_the_database(settings, sql):
    executor = SpyExecutor()
    body = call(
        make_service(settings, executor=executor, tables=SEED_TABLES), "execute_readonly_query", {"sql": sql}
    ).structured_content
    assert body["success"] is False and body["status"] == "rejected"
    assert body["failure"]["category"] == "UNSAFE_QUERY"
    assert executor.calls == []


def test_entire_adversarial_corpus_is_rejected_through_mcp(settings):
    payloads = [json.loads(line) for line in (DATASETS / "adversarial_sql.jsonl").read_text().splitlines() if line]
    executor = SpyExecutor()
    service = make_service(settings, executor=executor, tables=SEED_TABLES)

    async def run():
        async with Client(create_server(service)) as client:
            return [
                (await client.call_tool("execute_readonly_query", {"sql": p["sql"]}, raise_on_error=False))
                for p in payloads
            ]

    results = asyncio.run(run())
    escaped = [p["id"] for p, r in zip(payloads, results, strict=True) if r.structured_content["status"] != "rejected"]
    assert len(payloads) == 68
    assert escaped == []
    assert executor.calls == []


def test_executor_guard_still_blocks_when_the_mcp_layer_guard_is_bypassed(settings, monkeypatch):
    """Defense in depth: even if the adapter's validation were skipped, DatabaseExecutor
    re-validates before touching the database."""
    executor = DatabaseExecutor(settings)
    executor._postgres_engine = ExplodingEngine()  # type: ignore[assignment]
    service = make_service(settings, executor=executor, tables=SEED_TABLES)
    monkeypatch.setattr(service.guard, "validate", lambda *args, **kwargs: None)

    body = call(service, "execute_readonly_query", {"sql": "DELETE FROM orders"}).structured_content
    assert body["status"] == "rejected"
    assert body["failure"] == {
        "category": "UNSAFE_QUERY",
        "source": "database",
        "policy": "terminal",
        "message": "Statement type DELETE is not allowed; only SELECT is.",
    }


def test_execute_never_enables_mutations_even_when_the_server_allows_them(settings):
    executor = SpyExecutor()
    permissive = settings.model_copy(update={"mutations_enabled": True})
    service = make_service(permissive, executor=executor, tables=SEED_TABLES)
    assert call(service, "execute_readonly_query", {"sql": "DELETE FROM orders"}).structured_content["status"] == (
        "rejected"
    )
    call(service, "execute_readonly_query", {"sql": "SELECT 1"})
    assert executor.calls == [("SELECT 1", False)]


@pytest.mark.parametrize(
    ("category", "status", "policy"),
    [
        (FailureCategory.UNKNOWN_COLUMN, "failed", "llm_correctable"),
        (FailureCategory.TIMEOUT, "failed", "terminal"),
        (FailureCategory.CONNECTION_ERROR, "failed", "transient"),
        (FailureCategory.PERMISSION_ERROR, "rejected", "terminal"),
        (FailureCategory.UNSAFE_QUERY, "rejected", "terminal"),  # e.g. READ ONLY transaction (SQLSTATE 25006)
    ],
)
def test_execute_reports_classified_database_errors(settings, category, status, policy):
    executor = FakeExecutor({}, errors=[category])
    body = call(make_service(settings, executor=executor), "execute_readonly_query", {"sql": GOOD_SQL})
    body = body.structured_content
    assert body["success"] is False and body["status"] == status
    assert body["failure"]["category"] == category.value
    assert body["failure"]["source"] == "database"
    assert body["failure"]["policy"] == policy


def test_execute_encodes_rows_and_reports_truncation(settings):
    rows = [{"total": Decimal("12.50"), "day": dt.date(2026, 1, 2)}]
    executor = SpyExecutor(rows=rows, metadata={"row_limit": 1, "truncated": True})
    body = call(make_service(settings, executor=executor), "execute_readonly_query", {"sql": GOOD_SQL})
    body = body.structured_content
    assert body["rows"] == [{"total": 12.5, "day": "2026-01-02"}]
    assert body["truncated"] is True and body["row_limit"] == 1


def test_unexpected_errors_are_masked(settings):
    class BrokenExecutor(SpyExecutor):
        def execute(self, *args: Any, **kwargs: Any):
            raise RuntimeError("connection string postgres://admin:hunter2@db")

    result = call(make_service(settings, executor=BrokenExecutor()), "execute_readonly_query", {"sql": GOOD_SQL})
    assert result.is_error
    assert "hunter2" not in error_text(result)


# --- ask_database --------------------------------------------------------------------------


def test_ask_database_first_attempt(settings):
    service = make_service(settings, llm=FakeLLM([fenced(GOOD_SQL)]))
    body = call(service, "ask_database", {"question": "How many customers?"}).structured_content
    assert body["success"] is True and body["status"] == "success"
    assert body["generated_sql"] == body["final_sql"] == GOOD_SQL
    assert body["retry_count"] == 0 and body["llm_calls"] == 1 and body["attempts"] == []
    assert body["rows"] == ROWS
    assert body["prompt_tokens"] > 0
    assert "pruned_schema" not in body


def test_ask_database_self_corrects_through_the_agent_loop(settings):
    llm = FakeLLM([fenced(BAD_SQL), fenced(GOOD_SQL)])
    body = call(make_service(settings, llm=llm), "ask_database", {"question": "How many customers?"})
    body = body.structured_content
    assert body["success"] is True and body["status"] == "self_corrected"
    assert body["generated_sql"] == BAD_SQL and body["final_sql"] == GOOD_SQL
    assert body["retry_count"] == 1
    assert body["attempts"][0]["category"] == "UNKNOWN_COLUMN"
    assert body["attempts"][0]["policy"] == "llm_correctable"
    assert "does not exist" in llm.prompts[1]  # the database error went back to the model


def test_ask_database_recovers_from_malformed_output(settings):
    llm = FakeLLM(["Let me think about that.", fenced(GOOD_SQL)])
    body = call(make_service(settings, llm=llm), "ask_database", {"question": "How many customers?"})
    body = body.structured_content
    assert body["status"] == "self_corrected"
    assert body["attempts"][0]["category"] == "MALFORMED_OUTPUT"


@pytest.mark.parametrize(
    "generated",
    ["DROP TABLE customers", "SELECT COUNT(*) FROM customers; DELETE FROM orders", "DELETE FROM orders"],
)
def test_ask_database_rejects_unsafe_generated_sql_without_retry(settings, generated):
    executor = SpyExecutor()
    permissive = settings.model_copy(update={"mutations_enabled": True})
    llm = FakeLLM([fenced(generated), fenced(GOOD_SQL)])
    body = call(make_service(permissive, executor=executor, llm=llm), "ask_database", {"question": "q"})
    body = body.structured_content
    assert body["success"] is False and body["status"] == "rejected"
    assert body["failure"]["category"] == "UNSAFE_QUERY"
    assert body["llm_calls"] == 1  # unsafe output is terminal, never fed back to the model
    assert executor.calls == []


def test_ask_database_reports_refusal_and_exhausted_retries(settings):
    refusal = call(
        make_service(settings, llm=FakeLLM(["CANNOT_ANSWER: no phone column"])), "ask_database", {"question": "q"}
    ).structured_content
    assert refusal["status"] == "rejected" and refusal["failure"]["category"] == "UNANSWERABLE"

    exhausted = call(
        make_service(settings, llm=FakeLLM([fenced(BAD_SQL)] * 3)), "ask_database", {"question": "q"}
    ).structured_content
    assert exhausted["status"] == "failed" and exhausted["llm_calls"] == 3
    assert exhausted["failure"]["category"] == "UNKNOWN_COLUMN"
    assert [attempt["attempt_number"] for attempt in exhausted["attempts"]] == [1, 2, 3]


def test_ask_database_without_llm_credentials_is_a_tool_error(settings):
    unconfigured = settings.model_copy(update={"openai_api_key": None})
    service = LuminaMCPService(unconfigured, db_executor=SpyExecutor(), schema_manager=FakeSchemaManager())
    result = call(service, "ask_database", {"question": "q"})
    assert result.is_error
    assert "OPENAI_API_KEY is required" in error_text(result)
    # Schema, validation, and execution still work without an LLM.
    assert call(service, "validate_sql", {"sql": GOOD_SQL}).structured_content["valid"] is True


def test_ask_database_sheds_load_through_admission_control(settings):
    tight = settings.model_copy(update={"max_concurrent_requests": 1, "max_queued_requests": 0})
    llm = ScriptedLLM([fenced(GOOD_SQL)] * 2, latency_seconds=0.5)
    service = make_service(tight, llm=llm)

    async def run():
        async with Client(create_server(service)) as client:
            return await asyncio.gather(
                *(client.call_tool("ask_database", {"question": "q"}, raise_on_error=False) for _ in range(2))
            )

    results = asyncio.run(run())
    assert sorted(result.is_error for result in results) == [False, True]
    assert any("Server busy" in error_text(result) for result in results)


# --- malformed MCP inputs ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("execute_readonly_query", {}),
        ("execute_readonly_query", {"sql": ""}),
        ("execute_readonly_query", {"sql": "   \n "}),
        ("execute_readonly_query", {"sql": 42}),
        ("execute_readonly_query", {"sql": ["SELECT 1"]}),
        ("execute_readonly_query", {"sql": "SELECT 1", "allow_mutations": True}),
        ("execute_readonly_query", {"sql": "SELECT 1", "backend": "mysql"}),
        ("execute_readonly_query", {"sql": "SELECT 1" + " " * 10000 + "AS x"}),
        ("validate_sql", {}),
        ("validate_sql", {"sql": None}),
        ("ask_database", {}),
        ("ask_database", {"question": ""}),
        ("ask_database", {"question": {"text": "q"}}),
        ("ask_database", {"question": "x" * 2001}),
        ("get_schema", {"backend": "oracle"}),
    ],
)
def test_malformed_inputs_are_rejected_before_any_work(settings, tool, arguments):
    executor = SpyExecutor()
    llm = FakeLLM([fenced(GOOD_SQL)])
    result = call(make_service(settings, executor=executor, llm=llm), tool, arguments)
    assert result.is_error
    assert executor.calls == [] and llm.prompts == []


# --- HTTP transport auth -------------------------------------------------------------------


def test_api_key_verifier_matches_rest_api_keys():
    verifier = ApiKeyVerifier({"key-one-abcdef"})
    assert asyncio.run(verifier.verify_token("key-one-abcdef")).client_id == "key:abcdef"
    assert asyncio.run(verifier.verify_token("wrong")) is None


def test_http_transport_refuses_public_bind_without_api_keys(settings, monkeypatch):
    monkeypatch.setattr(server_module, "get_settings", lambda: settings)
    with pytest.raises(SystemExit, match="without API_KEYS"):
        server_module.main(["--transport", "http", "--host", "0.0.0.0"])
