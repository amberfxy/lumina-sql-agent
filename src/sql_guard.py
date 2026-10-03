"""Deterministic validation of model-generated SQL/PartiQL before execution.

The guard parses SQL into an AST (sqlglot) instead of pattern-matching text, so
multi-statement payloads, data-modifying CTEs, and side-effecting functions are
caught regardless of formatting. It is one of three independent layers: the
guard, a READ ONLY transaction, and a least-privilege database role.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from config import DatabaseBackend
from src.failures import FailureCategory

# Unsupported statements fall back to exp.Command, which is rejected; the parser's warning is noise.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

_SYSTEM_SCHEMAS = {"pg_catalog", "information_schema", "pg_toast"}

# Functions with side effects, filesystem/network access, or server introspection.
_DENIED_FUNCTIONS = {
    "nextval",
    "setval",
    "set_config",
    "current_setting",
    "query_to_xml",
    "query_to_xml_and_xmlschema",
    "table_to_xml",
    "cursor_to_xml",
    "txid_current",
    "version",
    "inet_server_addr",
    "inet_server_port",
}
_DENIED_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink", "file_")

_READ_ROOTS: tuple[type[exp.Expression], ...] = (exp.Select, exp.Union, exp.Intersect, exp.Except)
_WRITE_ROOTS: tuple[type[exp.Expression], ...] = (exp.Insert, exp.Update, exp.Delete)
_ALWAYS_FORBIDDEN = tuple(
    node
    for node in (
        getattr(exp, name, None)
        for name in (
            "Create",
            "Drop",
            "Alter",
            "TruncateTable",
            "Command",
            "Set",
            "Copy",
            "Grant",
            "Revoke",
            "Merge",
            "Transaction",
            "Commit",
            "Rollback",
            "Use",
            "LoadData",
            "Pragma",
        )
    )
    if node is not None
)

_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*")
_PARTIQL_TABLE = re.compile(r'\b(?:FROM|INTO|UPDATE)\s+"?([A-Za-z0-9_.\-]+)"?', re.IGNORECASE)
_PARTIQL_READ = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_PARTIQL_WRITE = re.compile(r"^\s*(INSERT|UPDATE|DELETE)\b", re.IGNORECASE)
_PARTIQL_STRING = re.compile(r"'(?:[^']|'')*'")
_PARTIQL_IDENTIFIER = re.compile(r'"([^"]+)"|\b([A-Za-z_][A-Za-z0-9_]*)\b')
_PARTIQL_STAR = re.compile(r"^\s*SELECT\s+\*", re.IGNORECASE)


class QueryRejected(Exception):
    """The query must not be executed. `category` decides whether the LLM may retry."""

    def __init__(self, category: FailureCategory, reason: str) -> None:
        super().__init__(reason)
        self.category = category
        self.reason = reason


class SQLGuard:
    def __init__(
        self,
        *,
        schema: str = "public",
        denied_columns: Iterable[str] = (),
    ) -> None:
        self.schema = schema.lower()
        self.denied_columns: dict[str, set[str]] = {}
        for entry in denied_columns:
            table, _, column = entry.lower().partition(".")
            if table and column:
                self.denied_columns.setdefault(table, set()).add(column)

    def validate(
        self,
        query: str,
        backend: DatabaseBackend,
        *,
        known_tables: Iterable[str] | None = None,
        allow_mutations: bool = False,
    ) -> None:
        """Raise QueryRejected if the query is unsafe or references unknown tables.

        `known_tables=None` skips the table-existence check (structural checks still run),
        which is how the executor applies the guard as defense in depth.
        """
        known = {table.lower() for table in known_tables} if known_tables is not None else None
        if backend == DatabaseBackend.DYNAMODB:
            self._validate_partiql(query, known, allow_mutations)
        else:
            self._validate_sql(query, known, allow_mutations)

    def _validate_sql(self, query: str, known: set[str] | None, allow_mutations: bool) -> None:
        text = query.strip().rstrip(";").strip()
        if not text:
            raise QueryRejected(FailureCategory.MALFORMED_OUTPUT, "Empty query.")
        try:
            statements = [statement for statement in sqlglot.parse(text, read="postgres") if statement is not None]
        except (ParseError, TokenError) as exc:
            raise QueryRejected(FailureCategory.SYNTAX_ERROR, f"Could not parse SQL: {exc}"[:500]) from exc

        if len(statements) != 1:
            raise QueryRejected(FailureCategory.UNSAFE_QUERY, "Exactly one SQL statement is allowed.")
        tree = statements[0]

        allowed_roots = _READ_ROOTS + (_WRITE_ROOTS if allow_mutations else ())
        if not isinstance(tree, allowed_roots):
            raise QueryRejected(
                FailureCategory.UNSAFE_QUERY, f"Statement type {tree.key.upper()} is not allowed; only SELECT is."
            )

        forbidden = _ALWAYS_FORBIDDEN + (() if allow_mutations else _WRITE_ROOTS)
        for node in tree.walk():
            if isinstance(node, forbidden):
                raise QueryRejected(
                    FailureCategory.UNSAFE_QUERY, f"{node.key.upper()} is not allowed inside a read-only query."
                )

        for select in tree.find_all(exp.Select):
            if select.args.get("into"):
                raise QueryRejected(FailureCategory.UNSAFE_QUERY, "SELECT ... INTO is not allowed.")
            if select.args.get("locks"):
                raise QueryRejected(
                    FailureCategory.UNSAFE_QUERY, "Row locking clauses (FOR UPDATE/SHARE) are not allowed."
                )

        for func in tree.find_all(exp.Func):
            # sqlglot normalizes known functions (version() -> CURRENT_VERSION), so check the
            # canonical name and the name as it renders back in the PostgreSQL dialect.
            names = {func.name.lower()} if isinstance(func, exp.Anonymous) else {func.sql_name().lower()}
            rendered = func.sql(dialect="postgres").split("(", 1)[0].strip().lower()
            if _IDENTIFIER.fullmatch(rendered):
                names.add(rendered)
            for name in names:
                if name in _DENIED_FUNCTIONS or name.startswith(_DENIED_FUNCTION_PREFIXES):
                    raise QueryRejected(FailureCategory.UNSAFE_QUERY, f"Function {name}() is not allowed.")

        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        referenced: dict[str, str] = {}  # alias -> table
        for table in tree.find_all(exp.Table):
            if isinstance(table.this, exp.Func):
                continue  # table-valued function such as generate_series(); checked above
            name = table.name.lower()
            schema = table.db.lower()
            if schema in _SYSTEM_SCHEMAS or name.startswith("pg_") or table.catalog:
                raise QueryRejected(
                    FailureCategory.UNSAFE_QUERY, f"Access to {table.sql()} is outside the schema boundary."
                )
            if schema and schema != self.schema:
                raise QueryRejected(FailureCategory.UNSAFE_QUERY, f"Schema {schema!r} is outside the allowed boundary.")
            if not schema and name in cte_names:
                continue
            if known is not None and name not in known:
                raise QueryRejected(
                    FailureCategory.UNKNOWN_TABLE,
                    f"Table {name!r} does not exist. Available tables: {', '.join(sorted(known))}.",
                )
            referenced[table.alias_or_name.lower()] = name

        self._check_denied_columns(tree, referenced)

    def _check_denied_columns(self, tree: exp.Expression, referenced: dict[str, str]) -> None:
        restricted = {alias: table for alias, table in referenced.items() if table in self.denied_columns}
        if not restricted:
            return
        denied_names = set().union(*(self.denied_columns[table] for table in restricted.values()))
        for select in tree.find_all(exp.Select):
            for projection in select.expressions:
                if isinstance(projection, exp.Star) or (
                    isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
                ):
                    raise QueryRejected(
                        FailureCategory.UNSAFE_QUERY, "SELECT * is not allowed on tables with restricted columns."
                    )
        for column in tree.find_all(exp.Column):
            name = column.name.lower()
            # A bare alias as a column (e.g. row_to_json(c)) exposes the whole row.
            if name in denied_names or (not column.table and name in restricted):
                raise QueryRejected(FailureCategory.UNSAFE_QUERY, f"Column {column.sql()} is restricted.")

    def _validate_partiql(self, query: str, known: set[str] | None, allow_mutations: bool) -> None:
        text = query.strip()
        if text.startswith("{"):
            try:
                text = str(json.loads(text).get("statement") or json.loads(text).get("Statement") or "")
            except (json.JSONDecodeError, AttributeError) as exc:
                raise QueryRejected(FailureCategory.MALFORMED_OUTPUT, "Invalid JSON PartiQL payload.") from exc
        text = text.strip().rstrip(";").strip()
        if not text:
            raise QueryRejected(FailureCategory.MALFORMED_OUTPUT, "Empty statement.")
        # String literals may legitimately contain ';' or keywords, so structural checks run on
        # the statement with literal contents removed. An unterminated literal is left intact
        # and therefore still trips the checks below.
        structure = _PARTIQL_STRING.sub("''", text)
        if ";" in structure:
            raise QueryRejected(FailureCategory.UNSAFE_QUERY, "Exactly one PartiQL statement is allowed.")
        if not _PARTIQL_READ.match(structure) and not (allow_mutations and _PARTIQL_WRITE.match(structure)):
            raise QueryRejected(FailureCategory.UNSAFE_QUERY, "Only PartiQL SELECT statements are allowed.")

        tables = [table.split(".")[0].lower() for table in _PARTIQL_TABLE.findall(structure)]
        if known is not None:
            for table in tables:
                if table not in known:
                    raise QueryRejected(
                        FailureCategory.UNKNOWN_TABLE,
                        f"Table {table!r} does not exist. Available tables: {', '.join(sorted(known))}.",
                    )

        restricted = set().union(*(self.denied_columns.get(table, set()) for table in tables))
        if restricted:
            if _PARTIQL_STAR.match(structure):
                raise QueryRejected(
                    FailureCategory.UNSAFE_QUERY, "SELECT * is not allowed on tables with restricted columns."
                )
            for quoted, bare in _PARTIQL_IDENTIFIER.findall(structure):
                name = (quoted or bare).lower()
                if name in restricted:
                    raise QueryRejected(FailureCategory.UNSAFE_QUERY, f"Attribute {name!r} is restricted.")
