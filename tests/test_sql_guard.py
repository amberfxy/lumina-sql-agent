from __future__ import annotations

import json

import pytest

from config import DatabaseBackend
from eval.harness import DATASETS, SEED_TABLES, load_cases, run_guard_suite
from src.failures import FailureCategory
from src.sql_guard import QueryRejected, SQLGuard

PG = DatabaseBackend.POSTGRES
DDB = DatabaseBackend.DYNAMODB
ADVERSARIAL = [json.loads(line) for line in (DATASETS / "adversarial_sql.jsonl").read_text().splitlines() if line]
GOLD = [case for case in load_cases(DATASETS / "nl2sql.jsonl") if case.gold_sql]


def rejection(sql: str, guard: SQLGuard | None = None, **kwargs) -> QueryRejected:
    with pytest.raises(QueryRejected) as excinfo:
        (guard or SQLGuard()).validate(sql, kwargs.pop("backend", PG), known_tables=SEED_TABLES, **kwargs)
    return excinfo.value


@pytest.mark.parametrize("payload", ADVERSARIAL, ids=[p["id"] + "-" + p["technique"] for p in ADVERSARIAL])
def test_adversarial_payload_is_rejected_as_unsafe(payload):
    assert rejection(payload["sql"]).category == FailureCategory.UNSAFE_QUERY


@pytest.mark.parametrize("case", GOLD, ids=[case.id for case in GOLD])
def test_every_gold_query_passes_the_guard(case):
    SQLGuard().validate(case.gold_sql, PG, known_tables=SEED_TABLES)


def test_guard_suite_report_has_no_escapes_or_false_positives():
    report = run_guard_suite(load_cases(DATASETS / "nl2sql.jsonl"))
    assert report["not_rejected_as_unsafe"] == []
    assert report["false_positives"] == []
    assert report["adversarial_payloads"] == len(ADVERSARIAL)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "SELECT COUNT(*) FROM customers;",
        "WITH recent AS (SELECT * FROM orders WHERE order_date > '2025-01-01') SELECT COUNT(*) FROM recent",
        "SELECT g FROM generate_series(1, 3) AS g",
        "SELECT created_at, last_update FROM support_tickets",  # keyword-like identifiers are fine
        "SELECT o.order_id FROM orders o JOIN customers c USING (customer_id) WHERE c.state = 'WA' LIMIT 10",
    ],
)
def test_benign_queries_pass(sql):
    SQLGuard().validate(sql, PG, known_tables=SEED_TABLES)


def test_unknown_table_is_llm_correctable_and_lists_alternatives():
    exc = rejection("SELECT * FROM clients")
    assert exc.category == FailureCategory.UNKNOWN_TABLE
    assert "customers" in exc.reason


def test_unparseable_sql_is_a_syntax_error():
    assert rejection("SELECT FROM WHERE (").category == FailureCategory.SYNTAX_ERROR


def test_empty_query_is_malformed_output():
    assert rejection("  ;  ").category == FailureCategory.MALFORMED_OUTPUT


def test_structural_checks_run_without_a_catalog():
    guard = SQLGuard()
    guard.validate("SELECT * FROM any_table", PG)
    with pytest.raises(QueryRejected):
        guard.validate("SELECT 1; DROP TABLE x", PG)


def test_denied_columns():
    guard = SQLGuard(denied_columns=["customers.email"])
    for sql in (
        "SELECT email FROM customers",
        "SELECT c.email FROM customers c",
        "SELECT * FROM customers",
        "SELECT c.* FROM customers c",
        "SELECT row_to_json(c) FROM customers c",
    ):
        assert rejection(sql, guard).category == FailureCategory.UNSAFE_QUERY, sql
    guard.validate("SELECT COUNT(*) FROM customers", PG, known_tables=SEED_TABLES)
    guard.validate("SELECT first_name, city FROM customers", PG, known_tables=SEED_TABLES)
    guard.validate("SELECT * FROM orders", PG, known_tables=SEED_TABLES)


def test_mutation_mode_allows_dml_but_never_ddl_or_multi_statement():
    guard = SQLGuard()
    guard.validate("UPDATE products SET stock_quantity = 0 WHERE product_id = 1", PG, allow_mutations=True)
    for sql in ("DROP TABLE products", "TRUNCATE orders", "UPDATE products SET stock_quantity = 0; DROP TABLE x"):
        with pytest.raises(QueryRejected):
            guard.validate(sql, PG, allow_mutations=True)


def test_partiql_rules():
    guard = SQLGuard()
    guard.validate('SELECT * FROM "orders" WHERE pk = 1', DDB, known_tables={"orders"})
    guard.validate('{"statement": "SELECT * FROM orders"}', DDB, known_tables={"orders"})
    assert rejection("DELETE FROM orders WHERE pk = 1", backend=DDB).category == FailureCategory.UNSAFE_QUERY
    assert rejection("SELECT * FROM orders; SELECT 1", backend=DDB).category == FailureCategory.UNSAFE_QUERY
    with pytest.raises(QueryRejected) as excinfo:
        guard.validate("SELECT * FROM missing", DDB, known_tables={"orders"})
    assert excinfo.value.category == FailureCategory.UNKNOWN_TABLE


def test_partiql_string_literals_do_not_trigger_structural_checks():
    guard = SQLGuard()
    guard.validate("SELECT pk FROM orders WHERE note = 'a; b'", DDB, known_tables={"orders"})
    guard.validate("SELECT pk FROM orders WHERE note = 'FROM missing'", DDB, known_tables={"orders"})
    guard.validate("SELECT pk FROM orders WHERE note = 'it''s; fine'", DDB, known_tables={"orders"})


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT pk FROM orders WHERE note = 'x'; DELETE FROM orders WHERE pk = 1",
        "SELECT pk FROM orders WHERE note = 'unterminated; DELETE FROM orders",
        "UPDATE orders SET status = 'x' WHERE pk = 1",
        "INSERT INTO orders VALUE {'pk': 1}",
        "EXISTS(SELECT * FROM orders)",
        '{"statement": "DELETE FROM orders WHERE pk = 1"}',
        "  delete from orders where pk = 1",
    ],
)
def test_partiql_writes_and_stacked_statements_are_unsafe(statement):
    assert rejection(statement, backend=DDB).category == FailureCategory.UNSAFE_QUERY


def test_partiql_write_targets_must_be_known_when_mutations_are_enabled():
    with pytest.raises(QueryRejected) as excinfo:
        SQLGuard().validate("INSERT INTO audit_log VALUE {'pk': 1}", DDB, known_tables={"orders"}, allow_mutations=True)
    assert excinfo.value.category == FailureCategory.UNKNOWN_TABLE


def test_partiql_denied_attributes():
    guard = SQLGuard(denied_columns=["customers.ssn"])
    guard.validate("SELECT pk, name FROM customers", DDB, known_tables={"customers", "orders"})
    guard.validate("SELECT * FROM orders", DDB, known_tables={"customers", "orders"})
    guard.validate("SELECT pk FROM customers WHERE name = 'ssn'", DDB, known_tables={"customers"})
    for statement in (
        "SELECT * FROM customers",
        "SELECT pk, ssn FROM customers",
        'SELECT pk, "SSN" FROM "customers"',
        "SELECT pk FROM customers WHERE ssn = '123'",
        "SELECT pk, profile.ssn FROM customers",
    ):
        with pytest.raises(QueryRejected) as excinfo:
            guard.validate(statement, DDB, known_tables={"customers"})
        assert excinfo.value.category == FailureCategory.UNSAFE_QUERY, statement
