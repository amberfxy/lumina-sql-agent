"""Transport-agnostic implementations of the MCP tools.

Every tool delegates to the services the FastAPI gateway uses:

- schema:     SchemaManager (same ALLOWED_TABLES / DENIED_COLUMNS boundary)
- validation: SQLGuard, with the same known-table check the agent applies
- execution:  DatabaseExecutor, which re-runs the guard, opens a READ ONLY transaction,
              and connects as the least-privilege role
- questions:  LuminaSQLAgent.run (generation, failure-taxonomy retries, cache, deadline)

Mutations are never enabled through MCP, regardless of MUTATIONS_ENABLED.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import sqlglot
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from sqlglot.errors import SqlglotError

from config import DatabaseBackend, Settings, get_settings
from src.agent import AgentResult, LuminaSQLAgent
from src.cache import Cache, build_cache
from src.context import get_request_id
from src.db_executor import DatabaseExecutionError, DatabaseExecutor, ExecutionResult
from src.failures import Failure, FailureCategory, FailureSource
from src.schema_manager import SchemaManager
from src.sql_guard import QueryRejected, SQLGuard

logger = logging.getLogger(__name__)

_MESSAGE_LIMIT = 500
_REJECTED = (FailureCategory.UNSAFE_QUERY, FailureCategory.UNANSWERABLE, FailureCategory.PERMISSION_ERROR)


class ServiceUnavailable(Exception):
    """A dependency the tool needs is missing (schema catalog, LLM credentials)."""


class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool
    primary_key: bool
    foreign_key: str | None = Field(None, description="Referenced `table.column`, if any.")


class TableInfo(BaseModel):
    name: str
    description: str
    columns: list[ColumnInfo]


class SchemaInfo(BaseModel):
    backend: str
    query_language: str
    tables: list[TableInfo]


class FailureInfo(BaseModel):
    category: str = Field(description="Failure taxonomy category, e.g. UNSAFE_QUERY, UNKNOWN_COLUMN, TIMEOUT.")
    source: str = Field(description="validator | database | llm | agent")
    policy: str = Field(description="llm_correctable | transient | terminal: whether retrying can help.")
    message: str


class ValidationResult(BaseModel):
    valid: bool
    reason: str | None = None
    normalized_sql: str | None = Field(
        None, description="Canonical rendering of the statement, for display. Execution runs the SQL as submitted."
    )
    failure: FailureInfo | None = None


class QueryResult(BaseModel):
    success: bool
    status: str = Field(description="executed | rejected | failed")
    sql: str
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    row_count: int = 0
    truncated: bool = False
    row_limit: int | None = None
    failure: FailureInfo | None = None
    request_id: str = "-"


class AttemptInfo(BaseModel):
    attempt_number: int
    query: str | None
    category: str
    policy: str
    source: str
    error: str


class AskResult(BaseModel):
    success: bool
    status: str = Field(description="success | self_corrected | cache_hit | rejected | failed")
    question: str
    generated_sql: str | None = Field(None, description="The first query the model produced.")
    final_sql: str | None = Field(None, description="The last query attempted (the executed one on success).")
    retry_count: int = Field(description="LLM correction calls after the first generation.")
    llm_calls: int
    db_executions: int
    transient_retries: int
    cache_hit: bool
    attempts: list[AttemptInfo] = []
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    row_count: int = 0
    truncated: bool = False
    failure: FailureInfo | None = None
    duration_ms: float
    prompt_tokens: int
    completion_tokens: int
    request_id: str


def _failure_info(failure: Failure) -> FailureInfo:
    return FailureInfo(
        category=failure.category.value,
        source=failure.source.value,
        policy=failure.policy.value,
        message=failure.message[:_MESSAGE_LIMIT],
    )


def _rows(execution: ExecutionResult | None) -> dict[str, Any]:
    if execution is None:
        return {}
    return {
        "columns": execution.columns,
        # Same encoding the REST API applies (Decimal -> float, dates -> ISO strings).
        "rows": jsonable_encoder(execution.rows),
        "row_count": execution.row_count,
        "truncated": bool(execution.metadata.get("truncated", False)),
    }


class LuminaMCPService:
    """Process-wide dependencies for the MCP server, wired like api.main.Runtime.

    The agent is built lazily so schema, validation, and execution tools work before
    LLM credentials are configured.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        cache: Cache | None = None,
        db_executor: DatabaseExecutor | None = None,
        schema_manager: SchemaManager | None = None,
        agent: LuminaSQLAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or build_cache(self.settings)
        self.db_executor = db_executor or DatabaseExecutor(settings=self.settings)
        self.schema_manager = schema_manager or SchemaManager(
            settings=self.settings, db_executor=self.db_executor, cache=self.cache
        )
        self.guard = SQLGuard(schema=self.settings.postgres_schema, denied_columns=self.settings.denied_columns)
        self._agent = agent
        self._agent_lock = threading.Lock()

    def get_agent(self) -> LuminaSQLAgent:
        if self._agent is None:
            with self._agent_lock:
                if self._agent is None:
                    try:
                        self._agent = LuminaSQLAgent(
                            settings=self.settings,
                            schema_manager=self.schema_manager,
                            db_executor=self.db_executor,
                            cache=self.cache,
                        )
                    except ValueError as exc:  # missing LLM credentials
                        raise ServiceUnavailable(str(exc)) from exc
        return self._agent

    def close(self) -> None:
        self.db_executor.close()

    def _known_tables(self, backend: DatabaseBackend) -> set[str]:
        tables = self.schema_manager.get_table_names(backend)
        if not tables:
            raise ServiceUnavailable("Schema catalog is unavailable; the database may be unreachable.")
        return tables

    def get_schema(self, backend: DatabaseBackend = DatabaseBackend.POSTGRES) -> SchemaInfo:
        catalog = self.schema_manager.get_catalog(backend)
        if not catalog.tables:
            raise ServiceUnavailable("Schema catalog is unavailable; the database may be unreachable.")
        return SchemaInfo(
            backend=backend.value,
            query_language="PartiQL" if backend == DatabaseBackend.DYNAMODB else "PostgreSQL SQL",
            tables=[
                TableInfo(
                    name=table.name,
                    description=table.description,
                    columns=[
                        ColumnInfo(
                            name=column.name,
                            type=column.data_type,
                            nullable=column.nullable,
                            primary_key=column.is_primary_key,
                            foreign_key=column.foreign_key_target,
                        )
                        for column in table.columns
                    ],
                )
                for table in catalog.tables
            ],
        )

    def validate_sql(self, sql: str, backend: DatabaseBackend = DatabaseBackend.POSTGRES) -> ValidationResult:
        known_tables = self._known_tables(backend)
        try:
            self.guard.validate(sql, backend, known_tables=known_tables, allow_mutations=False)
        except QueryRejected as exc:
            failure = Failure(exc.category, FailureSource.VALIDATOR, exc.reason)
            return ValidationResult(valid=False, reason=exc.reason[:_MESSAGE_LIMIT], failure=_failure_info(failure))
        return ValidationResult(valid=True, normalized_sql=self._normalize(sql, backend))

    def execute_readonly_query(self, sql: str, backend: DatabaseBackend = DatabaseBackend.POSTGRES) -> QueryResult:
        request_id = get_request_id()
        validation = self.validate_sql(sql, backend)
        if not validation.valid:
            return QueryResult(
                success=False, status="rejected", sql=sql, failure=validation.failure, request_id=request_id
            )
        try:
            execution = self.db_executor.execute(sql, backend=backend, allow_mutations=False)
        except DatabaseExecutionError as exc:
            failure = Failure(exc.category, FailureSource.DATABASE, exc.raw_error)
            status = "rejected" if exc.category in _REJECTED else "failed"
            return QueryResult(
                success=False, status=status, sql=sql, failure=_failure_info(failure), request_id=request_id
            )
        return QueryResult(
            success=True,
            status="executed",
            sql=execution.query,
            row_limit=execution.metadata.get("row_limit"),
            request_id=request_id,
            **_rows(execution),
        )

    def ask_database(self, question: str, backend: DatabaseBackend = DatabaseBackend.POSTGRES) -> AskResult:
        result = self.get_agent().run(question, backend, allow_mutations=False, use_cache=True)
        return self._ask_result(result)

    @staticmethod
    def _ask_result(result: AgentResult) -> AskResult:
        if result.failure is None:
            status = "cache_hit" if result.cache_hit else ("self_corrected" if result.attempts else "success")
        else:
            status = "rejected" if result.failure.category in _REJECTED else "failed"
        generated = result.attempts[0].query if result.attempts else result.final_query
        return AskResult(
            success=result.success,
            status=status,
            question=result.user_query,
            generated_sql=generated,
            final_sql=result.final_query,
            retry_count=max(0, result.llm_calls - 1),
            llm_calls=result.llm_calls,
            db_executions=result.db_executions,
            transient_retries=result.transient_retries,
            cache_hit=result.cache_hit,
            attempts=[
                AttemptInfo(
                    attempt_number=attempt.attempt_number,
                    query=attempt.query,
                    category=attempt.category,
                    policy=attempt.policy,
                    source=attempt.source,
                    error=attempt.error[:_MESSAGE_LIMIT],
                )
                for attempt in result.attempts
            ],
            failure=_failure_info(result.failure) if result.failure else None,
            duration_ms=round(result.duration_ms, 2),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            request_id=result.request_id,
            **_rows(result.execution),
        )

    @staticmethod
    def _normalize(sql: str, backend: DatabaseBackend) -> str:
        text = sql.strip().rstrip(";").strip()
        if backend == DatabaseBackend.DYNAMODB:
            return text
        try:
            return sqlglot.transpile(text, read="postgres", write="postgres")[0]
        except SqlglotError:
            return text
