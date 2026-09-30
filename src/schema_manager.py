"""Database schema extraction and semantic pruning for token-efficient LLM context."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace

from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config import DatabaseBackend, Settings, get_settings
from src.cache import Cache, NullCache
from src.db_executor import DatabaseExecutor

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


@dataclass(frozen=True)
class ColumnMetadata:
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_target: str | None = None


@dataclass(frozen=True)
class TableMetadata:
    name: str
    columns: tuple[ColumnMetadata, ...]
    description: str = ""


@dataclass
class SchemaCatalog:
    backend: DatabaseBackend
    tables: list[TableMetadata] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"backend": self.backend.value, "tables": [asdict(table) for table in self.tables]})

    @classmethod
    def from_json(cls, payload: str) -> SchemaCatalog:
        data = json.loads(payload)
        tables = [
            TableMetadata(
                name=table["name"],
                description=table.get("description", ""),
                columns=tuple(ColumnMetadata(**column) for column in table["columns"]),
            )
            for table in data["tables"]
        ]
        return cls(backend=DatabaseBackend(data["backend"]), tables=tables)


class SchemaManager:
    """Extracts full schema metadata and prunes it to the tables relevant to a question.

    Catalogs are cached in-process and, when Redis is configured, shared across API
    replicas so each pod does not re-inspect the database on startup.
    """

    TOP_K_TABLES = 8
    FALLBACK_TABLES = 3

    def __init__(
        self,
        settings: Settings | None = None,
        db_executor: DatabaseExecutor | None = None,
        cache: Cache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db_executor = db_executor or DatabaseExecutor(self.settings)
        self.cache = cache or NullCache()
        self._catalog_cache: dict[DatabaseBackend, tuple[float, SchemaCatalog]] = {}
        self._lock = threading.Lock()

    def get_pruned_schema(self, user_query: str, backend: DatabaseBackend | None = None) -> str:
        """Return a markdown schema snippet relevant to the user query."""
        backend = backend or DatabaseBackend.POSTGRES
        catalog = self._get_catalog(backend)
        if not catalog.tables:
            return "_No schema metadata available._"

        ranked = self._rank_tables(user_query, catalog.tables)
        top_tables = [table for table, _score in ranked[: self.TOP_K_TABLES] if _score > 0.0]

        if not top_tables:
            top_tables = [table for table, _ in ranked[: self.FALLBACK_TABLES]]

        logger.info(
            "Pruned schema for backend=%s query=%r selected_tables=%s",
            backend.value,
            user_query[:120],
            [table.name for table in top_tables],
        )
        return self._format_schema_markdown(catalog.backend, top_tables)

    def _get_catalog(self, backend: DatabaseBackend) -> SchemaCatalog:
        ttl = self.settings.schema_cache_ttl_seconds
        cached = self._catalog_cache.get(backend)
        if cached and time.monotonic() - cached[0] < ttl:
            return cached[1]

        with self._lock:
            cached = self._catalog_cache.get(backend)
            if cached and time.monotonic() - cached[0] < ttl:
                return cached[1]

            catalog = self._load_shared_catalog(backend)
            if catalog is None:
                catalog = (
                    self._extract_postgres_schema()
                    if backend == DatabaseBackend.POSTGRES
                    else self._extract_dynamodb_schema()
                )
                if catalog.tables:
                    self.cache.set(self._cache_key(backend), catalog.to_json(), ttl_seconds=ttl)

            # Empty catalogs (e.g. database unreachable) are not cached so the next request retries.
            if catalog.tables:
                self._catalog_cache[backend] = (time.monotonic(), catalog)
            return catalog

    def _load_shared_catalog(self, backend: DatabaseBackend) -> SchemaCatalog | None:
        payload = self.cache.get(self._cache_key(backend), kind="schema")
        if payload is None:
            return None
        try:
            return SchemaCatalog.from_json(payload)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Discarding malformed cached schema for %s: %s", backend.value, exc)
            self.cache.delete(self._cache_key(backend))
            return None

    def _cache_key(self, backend: DatabaseBackend) -> str:
        if backend == DatabaseBackend.POSTGRES:
            source = f"{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"
            source += f"/{self.settings.postgres_schema}"
        else:
            source = f"{self.settings.aws_region}/{self.settings.dynamodb_endpoint_url or 'aws'}"
            source += f"/{self.settings.dynamodb_table_prefix}"
        return f"schema:{backend.value}:{source}"

    def invalidate_cache(self) -> None:
        """Clear cached schema snapshots (local and shared)."""
        with self._lock:
            self._catalog_cache.clear()
        self.cache.delete_prefix("schema:")

    def _extract_postgres_schema(self) -> SchemaCatalog:
        tables: list[TableMetadata] = []
        try:
            engine: Engine = self.db_executor.postgres_engine
            inspector = inspect(engine)
            schema = self.settings.postgres_schema

            for table_name in sorted(inspector.get_table_names(schema=schema)):
                pk_columns = set(inspector.get_pk_constraint(table_name, schema=schema).get("constrained_columns", []))
                fk_map: dict[str, str] = {}
                for fk in inspector.get_foreign_keys(table_name, schema=schema):
                    constrained = fk.get("constrained_columns") or []
                    referred_table = fk.get("referred_table")
                    referred_columns = fk.get("referred_columns") or []
                    if constrained and referred_table and referred_columns:
                        fk_map[constrained[0]] = f"{referred_table}.{referred_columns[0]}"

                columns: list[ColumnMetadata] = []
                for column in inspector.get_columns(table_name, schema=schema):
                    column_name = column["name"]
                    columns.append(
                        ColumnMetadata(
                            name=column_name,
                            data_type=str(column.get("type", "unknown")),
                            nullable=bool(column.get("nullable", True)),
                            is_primary_key=column_name in pk_columns,
                            is_foreign_key=column_name in fk_map,
                            foreign_key_target=fk_map.get(column_name),
                        )
                    )

                tables.append(TableMetadata(name=table_name, columns=tuple(columns)))

            with engine.connect() as connection:
                result = connection.execute(
                    text(
                        """
                        SELECT c.relname, obj_description(c.oid, 'pg_class')
                        FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE n.nspname = :schema AND c.relkind IN ('r', 'p', 'v', 'm')
                        """
                    ),
                    {"schema": schema},
                )
                descriptions = {name: description for name, description in result if description}
            tables = [
                replace(table, description=str(descriptions[table.name])) if table.name in descriptions else table
                for table in tables
            ]

        except SQLAlchemyError as exc:
            logger.exception("Failed to extract PostgreSQL schema: %s", exc)
            return SchemaCatalog(backend=DatabaseBackend.POSTGRES, tables=[])

        return SchemaCatalog(backend=DatabaseBackend.POSTGRES, tables=tables)

    def _extract_dynamodb_schema(self) -> SchemaCatalog:
        tables: list[TableMetadata] = []
        try:
            client = self.db_executor.dynamodb_client
            paginator = client.get_paginator("list_tables")
            prefix = self.settings.dynamodb_table_prefix

            table_names: list[str] = []
            for page in paginator.paginate():
                for name in page.get("TableNames", []):
                    if not prefix or name.startswith(prefix):
                        table_names.append(name)

            for table_name in sorted(table_names):
                response = client.describe_table(TableName=table_name)
                attribute_definitions = response["Table"].get("AttributeDefinitions", [])
                key_schema = response["Table"].get("KeySchema", [])
                key_names = {key["AttributeName"] for key in key_schema}

                columns = [
                    ColumnMetadata(
                        name=item["AttributeName"],
                        data_type=item.get("AttributeType", "S"),
                        nullable=False,
                        is_primary_key=item["AttributeName"] in key_names,
                    )
                    for item in attribute_definitions
                ]
                tables.append(
                    TableMetadata(
                        name=table_name,
                        columns=tuple(columns),
                        description="DynamoDB table (use PartiQL for queries)",
                    )
                )
        except (BotoCoreError, ClientError) as exc:
            logger.exception("Failed to extract DynamoDB schema: %s", exc)
            return SchemaCatalog(backend=DatabaseBackend.DYNAMODB, tables=[])

        return SchemaCatalog(backend=DatabaseBackend.DYNAMODB, tables=tables)

    def _rank_tables(self, user_query: str, tables: list[TableMetadata]) -> list[tuple[TableMetadata, float]]:
        query_tokens = self._tokenize(user_query)
        query_vector = Counter(query_tokens)

        scored: list[tuple[TableMetadata, float]] = []
        for table in tables:
            corpus = " ".join(
                [
                    table.name,
                    table.description,
                    " ".join(column.name for column in table.columns),
                    " ".join(column.data_type for column in table.columns),
                ]
            )
            doc_tokens = self._tokenize(corpus)
            doc_vector = Counter(doc_tokens)
            score = self._cosine_similarity(query_vector, doc_vector)
            scored.append((table, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    @staticmethod
    def _tokenize(text_value: str) -> list[str]:
        """Lowercase, split snake_case identifiers into parts, and fold simple plurals,
        so "customers", "customer_id", and "customer" share the token "customer"."""
        tokens: list[str] = []
        for raw in _TOKEN_PATTERN.findall(text_value):
            raw = raw.lower()
            parts = [raw, *raw.split("_")] if "_" in raw else [raw]
            tokens.extend(_stem(part) for part in parts if len(part) > 1)
        return tokens

    @staticmethod
    def _cosine_similarity(vec_a: Counter[str], vec_b: Counter[str]) -> float:
        if not vec_a or not vec_b:
            return 0.0

        intersection = set(vec_a) & set(vec_b)
        numerator = sum(vec_a[token] * vec_b[token] for token in intersection)
        magnitude_a = math.sqrt(sum(count * count for count in vec_a.values()))
        magnitude_b = math.sqrt(sum(count * count for count in vec_b.values()))
        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0
        return numerator / (magnitude_a * magnitude_b)

    @staticmethod
    def _format_schema_markdown(backend: DatabaseBackend, tables: list[TableMetadata]) -> str:
        lines = [f"## Pruned Schema ({backend.value})", ""]
        for table in tables:
            lines.append(f"### Table `{table.name}`")
            if table.description:
                lines.append(f"_Description_: {table.description}")
            lines.append("")
            lines.append("| Column | Type | PK | FK | Nullable |")
            lines.append("| --- | --- | --- | --- | --- |")
            for column in table.columns:
                fk_target = column.foreign_key_target or ""
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{column.name}`",
                            column.data_type,
                            "yes" if column.is_primary_key else "no",
                            fk_target or ("yes" if column.is_foreign_key else "no"),
                            "yes" if column.nullable else "no",
                        ]
                    )
                    + " |"
                )
            lines.append("")
        return "\n".join(lines).strip()
