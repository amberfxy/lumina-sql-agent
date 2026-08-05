"""Database execution layer for PostgreSQL and AWS DynamoDB."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import SQLAlchemyError

from config import DatabaseBackend, Settings, get_settings

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

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._postgres_engine: Engine | None = None
        self._dynamodb_client: Any | None = None

    @property
    def postgres_engine(self) -> Engine:
        if self._postgres_engine is None:
            self._postgres_engine = create_engine(
                self.settings.postgres_url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                future=True,
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
            session_kwargs: dict[str, Any] = {"region_name": self.settings.aws_region}
            if self.settings.aws_access_key_id and self.settings.aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = self.settings.aws_access_key_id.get_secret_value()
                session_kwargs["aws_secret_access_key"] = (
                    self.settings.aws_secret_access_key.get_secret_value()
                )

            session = boto3.session.Session(**session_kwargs)
            client_kwargs: dict[str, Any] = {
                "config": BotoConfig(retries={"max_attempts": 3, "mode": "standard"})
            }
            if self.settings.dynamodb_endpoint_url:
                client_kwargs["endpoint_url"] = self.settings.dynamodb_endpoint_url

            self._dynamodb_client = session.client("dynamodb", **client_kwargs)
            logger.info(
                "Initialized DynamoDB client region=%s endpoint=%s",
                self.settings.aws_region,
                self.settings.dynamodb_endpoint_url or "aws",
            )
        return self._dynamodb_client

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

        try:
            if backend == DatabaseBackend.POSTGRES:
                return self._execute_postgres(normalized_query, allow_mutations=allow_mutations, parameters=parameters)
            return self._execute_dynamodb(normalized_query, allow_mutations=allow_mutations, parameters=parameters)
        except DatabaseExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface unknown failures uniformly
            raw_error = repr(exc)
            logger.exception("Unexpected execution failure backend=%s", backend.value)
            raise DatabaseExecutionError(
                message=f"Unexpected database execution failure: {exc}",
                raw_error=raw_error,
                backend=backend,
                query=normalized_query,
            ) from exc

    def health_check(self, backend: DatabaseBackend) -> ExecutionResult:
        """Lightweight connectivity probe used by the API gateway."""
        if backend == DatabaseBackend.POSTGRES:
            query = "SELECT 1 AS ok"
        else:
            query = "SELECT 1"

        try:
            return self.execute(query, backend=backend, allow_mutations=False)
        except DatabaseExecutionError as exc:
            return ExecutionResult(
                success=False,
                backend=backend,
                query=query,
                error=str(exc),
                raw_error=exc.raw_error,
            )

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
                raw_error="Mutation blocked by safety policy",
                backend=DatabaseBackend.POSTGRES,
                query=query,
            )

        try:
            with self.postgres_engine.connect() as connection:
                result: Result[Any] = connection.execute(text(query), parameters or {})
                if result.returns_rows:
                    rows = [dict(row._mapping) for row in result.fetchmany(500)]
                    columns = list(result.keys())
                    connection.commit()
                    return ExecutionResult(
                        success=True,
                        backend=DatabaseBackend.POSTGRES,
                        query=query,
                        rows=rows,
                        row_count=len(rows),
                        columns=columns,
                        metadata={"truncated_to": 500},
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
                raw_error="Mutation blocked by safety policy",
                backend=DatabaseBackend.DYNAMODB,
                query=query,
            )

        try:
            request: dict[str, Any] = {"Statement": statement}
            if parameters:
                request["Parameters"] = [
                    self._to_dynamodb_parameter(key, value) for key, value in parameters.items()
                ]

            response = self.dynamodb_client.execute_statement(**request)
            items = response.get("Items", [])
            rows = [self._deserialize_dynamodb_item(item) for item in items]
            columns = sorted({column for row in rows for column in row.keys()}) if rows else []

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
            return {
                key: DatabaseExecutor._deserialize_dynamodb_value(item)
                for key, item in value["M"].items()
            }
        return value

    def close(self) -> None:
        """Dispose underlying clients."""
        if self._postgres_engine is not None:
            self._postgres_engine.dispose()
            self._postgres_engine = None
        self._dynamodb_client = None
