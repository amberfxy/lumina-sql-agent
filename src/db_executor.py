"""Database execution layer for PostgreSQL and AWS DynamoDB."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import SQLAlchemyError

from config import DatabaseBackend, Settings, get_settings
from src.metrics import DB_LATENCY, DB_QUERIES

logger = logging.getLogger(__name__)

QueryLanguage = Literal["sql", "partiql"]


@dataclass(frozen=True)
class ExecutionResult:
    """Normalized execution payload returned to the agent and UI."""

    success: bool
    backend: DatabaseBackend
    query: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    raw_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "backend": self.backend.value,
            "query": self.query,
            "rows": self.rows,
            "row_count": self.row_count,
            "columns": self.columns,
            "error": self.error,
            "raw_error": self.raw_error,
            "metadata": self.metadata,
        }


class DatabaseExecutionError(Exception):
    """Raised when a query fails; carries the raw database error for the LLM loop."""

    def __init__(self, message: str, raw_error: str, backend: DatabaseBackend, query: str) -> None:
        super().__init__(message)
        self.raw_error = raw_error
        self.backend = backend
        self.query = query


class DatabaseExecutor:
    """Executes generated SQL/PartiQL against PostgreSQL or DynamoDB."""

    _READ_ONLY_PATTERN = re.compile(
        r"^\s*(with\b.*?select\b|select\b|explain\b|show\b|describe\b)",
        re.IGNORECASE | re.DOTALL,
    )
    _MUTATING_PATTERN = re.compile(
        r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke)\b",
        re.IGNORECASE,
    )
    _BLOCKED_ERROR = "Mutation blocked by safety policy"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._postgres_engine: Engine | None = None
        self._dynamodb_client: Any | None = None
        self._init_lock = threading.Lock()

    @property
    def postgres_engine(self) -> Engine:
        if self._postgres_engine is None:
            with self._init_lock:
                if self._postgres_engine is None:
                    self._postgres_engine = create_engine(
                        self.settings.postgres_url,
                        pool_pre_ping=True,
                        pool_size=self.settings.postgres_pool_size,
                        max_overflow=self.settings.postgres_max_overflow,
                        connect_args={
                            "connect_timeout": self.settings.postgres_connect_timeout_seconds,
                            "options": f"-c statement_timeout={self.settings.postgres_statement_timeout_ms}",
                        },
                    )
                    logger.info(
                        "Initialized PostgreSQL engine host=%s db=%s",
                        self.settings.postgres_host,
                        self.settings.postgres_db,
                    )
        return self._postgres_engine

    @property
    def dynamodb_client(self) -> Any:
        if self._dynamodb_client is None:
            with self._init_lock:
                if self._dynamodb_client is None:
                    self._dynamodb_client = self._create_dynamodb_client()
        return self._dynamodb_client

    def _create_dynamodb_client(self) -> Any:
        session_kwargs: dict[str, Any] = {"region_name": self.settings.aws_region}
        if self.settings.aws_access_key_id and self.settings.aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = self.settings.aws_access_key_id.get_secret_value()
            session_kwargs["aws_secret_access_key"] = self.settings.aws_secret_access_key.get_secret_value()

        session = boto3.session.Session(**session_kwargs)
        client_kwargs: dict[str, Any] = {
            "config": BotoConfig(
                retries={"max_attempts": 3, "mode": "standard"},
                connect_timeout=5,
                read_timeout=15,
            )
        }
        if self.settings.dynamodb_endpoint_url:
            client_kwargs["endpoint_url"] = self.settings.dynamodb_endpoint_url

        client = session.client("dynamodb", **client_kwargs)
        logger.info(
            "Initialized DynamoDB client region=%s endpoint=%s",
            self.settings.aws_region,
            self.settings.dynamodb_endpoint_url or "aws",
        )
        return client

    def execute(
        self,
        query: str,
        backend: DatabaseBackend,
        *,
        allow_mutations: bool = False,
        parameters: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Execute a query and return a normalized result or raise DatabaseExecutionError."""
        normalized_query = query.strip()
        if not normalized_query:
            raise DatabaseExecutionError(
                message="Query is empty.",
                raw_error="Empty query string",
                backend=backend,
                query=query,
            )

        logger.info("Executing query backend=%s query=%r", backend.value, normalized_query[:500])

        started = time.perf_counter()
        status = "ok"
        try:
            if backend == DatabaseBackend.POSTGRES:
                return self._execute_postgres(normalized_query, allow_mutations=allow_mutations, parameters=parameters)
            return self._execute_dynamodb(normalized_query, allow_mutations=allow_mutations, parameters=parameters)
        except DatabaseExecutionError as exc:
            status = "blocked" if exc.raw_error == self._BLOCKED_ERROR else "error"
            raise
        except Exception as exc:  # noqa: BLE001 - surface unknown failures uniformly
            status = "error"
            raw_error = repr(exc)
            logger.exception("Unexpected execution failure backend=%s", backend.value)
            raise DatabaseExecutionError(
                message=f"Unexpected database execution failure: {exc}",
                raw_error=raw_error,
                backend=backend,
                query=normalized_query,
            ) from exc
        finally:
            DB_QUERIES.labels(backend=backend.value, status=status).inc()
            DB_LATENCY.labels(backend=backend.value).observe(time.perf_counter() - started)

    def health_check(self, backend: DatabaseBackend) -> dict[str, Any]:
        """Connectivity probe used by health/readiness endpoints (not counted in query metrics)."""
        started = time.perf_counter()
        try:
            if backend == DatabaseBackend.POSTGRES:
                with self.postgres_engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
            else:
                self.dynamodb_client.list_tables(Limit=1)
        except Exception as exc:  # noqa: BLE001 - any failure means "not healthy"
            return {"success": False, "error": f"{exc.__class__.__name__}: {exc}"[:500]}
        return {"success": True, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}

    def _execute_postgres(
        self,
        query: str,
        *,
        allow_mutations: bool,
        parameters: dict[str, Any] | None,
    ) -> ExecutionResult:
        if not allow_mutations and self._MUTATING_PATTERN.search(query):
            raise DatabaseExecutionError(
                message="Mutating SQL statements are disabled by default.",
                raw_error=self._BLOCKED_ERROR,
                backend=DatabaseBackend.POSTGRES,
                query=query,
            )

        max_rows = self.settings.max_result_rows
        try:
            with self.postgres_engine.connect() as connection:
                if not allow_mutations:
                    # Database-enforced guard: catches writes the keyword filter misses
                    # (e.g. nextval(), side-effecting functions).
                    connection.execute(text("SET TRANSACTION READ ONLY"))
                result: Result[Any] = connection.execute(text(query), parameters or {})
                if result.returns_rows:
                    columns = self._dedupe_columns(list(result.keys()))
                    rows = [dict(zip(columns, tuple(row), strict=True)) for row in result.fetchmany(max_rows)]
                    connection.commit()
                    return ExecutionResult(
                        success=True,
                        backend=DatabaseBackend.POSTGRES,
                        query=query,
                        rows=rows,
                        row_count=len(rows),
                        columns=columns,
                        metadata={"truncated_to": max_rows},
                    )

                connection.commit()
                return ExecutionResult(
                    success=True,
                    backend=DatabaseBackend.POSTGRES,
                    query=query,
                    rows=[],
                    row_count=result.rowcount or 0,
                    columns=[],
                    metadata={"rowcount": result.rowcount},
                )
        except SQLAlchemyError as exc:
            raw_error = self._format_sqlalchemy_error(exc)
            logger.error("PostgreSQL execution failed raw_error=%s", raw_error)
            raise DatabaseExecutionError(
                message="PostgreSQL query execution failed.",
                raw_error=raw_error,
                backend=DatabaseBackend.POSTGRES,
                query=query,
            ) from exc

    def _execute_dynamodb(
        self,
        query: str,
        *,
        allow_mutations: bool,
        parameters: dict[str, Any] | None,
    ) -> ExecutionResult:
        statement = self._normalize_partiql(query)
        if not allow_mutations and self._MUTATING_PATTERN.search(statement):
            raise DatabaseExecutionError(
                message="Mutating PartiQL statements are disabled by default.",
                raw_error=self._BLOCKED_ERROR,
                backend=DatabaseBackend.DYNAMODB,
                query=query,
            )

        try:
            request: dict[str, Any] = {"Statement": statement}
            if parameters:
                request["Parameters"] = [self._to_dynamodb_parameter(key, value) for key, value in parameters.items()]

            response = self.dynamodb_client.execute_statement(**request)
            items = response.get("Items", [])
            rows = [self._deserialize_dynamodb_item(item) for item in items]
            columns = sorted({column for row in rows for column in row}) if rows else []

            return ExecutionResult(
                success=True,
                backend=DatabaseBackend.DYNAMODB,
                query=statement,
                rows=rows,
                row_count=len(rows),
                columns=columns,
                metadata={
                    "consumed_capacity": response.get("ConsumedCapacity"),
                    "next_token_present": bool(response.get("NextToken")),
                },
            )
        except (BotoCoreError, ClientError) as exc:
            raw_error = self._format_boto_error(exc)
            logger.error("DynamoDB execution failed raw_error=%s", raw_error)
            raise DatabaseExecutionError(
                message="DynamoDB PartiQL execution failed.",
                raw_error=raw_error,
                backend=DatabaseBackend.DYNAMODB,
                query=statement,
            ) from exc

    @staticmethod
    def _dedupe_columns(columns: list[str]) -> list[str]:
        """Make result column names unique (e.g. `c.name, p.name` -> `name, name_2`) so no values are dropped."""
        seen: dict[str, int] = {}
        unique: list[str] = []
        for column in columns:
            count = seen.get(column, 0) + 1
            seen[column] = count
            unique.append(column if count == 1 else f"{column}_{count}")
        return unique

    @staticmethod
    def _normalize_partiql(query: str) -> str:
        stripped = query.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatabaseExecutionError(
                    message="Invalid JSON PartiQL payload.",
                    raw_error=str(exc),
                    backend=DatabaseBackend.DYNAMODB,
                    query=query,
                ) from exc
            statement = payload.get("statement") or payload.get("Statement")
            if not statement:
                raise DatabaseExecutionError(
                    message="JSON payload must include a `statement` field.",
                    raw_error="Missing statement field",
                    backend=DatabaseBackend.DYNAMODB,
                    query=query,
                )
            return str(statement).strip()
        return stripped

    @staticmethod
    def _format_sqlalchemy_error(exc: SQLAlchemyError) -> str:
        original = getattr(exc, "orig", None)
        if original is not None:
            return f"{exc.__class__.__name__}: {original}"
        return f"{exc.__class__.__name__}: {exc}"

    @staticmethod
    def _format_boto_error(exc: Exception) -> str:
        if isinstance(exc, ClientError):
            error = exc.response.get("Error", {})
            return json.dumps(
                {
                    "code": error.get("Code"),
                    "message": error.get("Message"),
                    "response_metadata": exc.response.get("ResponseMetadata"),
                },
                default=str,
            )
        return repr(exc)

    @staticmethod
    def _to_dynamodb_parameter(name: str, value: Any) -> dict[str, Any]:
        return {"Name": name, "Value": DatabaseExecutor._serialize_dynamodb_value(value)}

    @staticmethod
    def _serialize_dynamodb_value(value: Any) -> dict[str, Any]:
        if value is None:
            return {"NULL": True}
        if isinstance(value, bool):
            return {"BOOL": value}
        if isinstance(value, int):
            return {"N": str(value)}
        if isinstance(value, float):
            return {"N": str(value)}
        if isinstance(value, (list, tuple)):
            return {"L": [DatabaseExecutor._serialize_dynamodb_value(item) for item in value]}
        if isinstance(value, dict):
            return {"M": {key: DatabaseExecutor._serialize_dynamodb_value(item) for key, item in value.items()}}
        return {"S": str(value)}

    @staticmethod
    def _deserialize_dynamodb_item(item: dict[str, Any]) -> dict[str, Any]:
        return {key: DatabaseExecutor._deserialize_dynamodb_value(value) for key, value in item.items()}

    @staticmethod
    def _deserialize_dynamodb_value(value: dict[str, Any]) -> Any:
        if "S" in value:
            return value["S"]
        if "N" in value:
            number = value["N"]
            return int(number) if number.isdigit() else float(number)
        if "BOOL" in value:
            return value["BOOL"]
        if "NULL" in value:
            return None
        if "L" in value:
            return [DatabaseExecutor._deserialize_dynamodb_value(item) for item in value["L"]]
        if "M" in value:
            return {key: DatabaseExecutor._deserialize_dynamodb_value(item) for key, item in value["M"].items()}
        return value

    def close(self) -> None:
        """Dispose underlying clients."""
        if self._postgres_engine is not None:
            self._postgres_engine.dispose()
            self._postgres_engine = None
        self._dynamodb_client = None
