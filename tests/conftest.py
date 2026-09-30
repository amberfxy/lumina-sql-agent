from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import fakeredis
import pytest

from config import DatabaseBackend, Settings
from src.cache import RedisCache
from src.db_executor import DatabaseExecutionError, ExecutionResult


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("LUMINA_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="set LUMINA_INTEGRATION=1 with Postgres/Redis running")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


class FakeLLM:
    """Returns scripted responses in order and records every prompt it receives."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        return self.responses.pop(0)

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        text = self.complete(system_prompt, user_prompt)
        midpoint = len(text) // 2
        yield text[:midpoint]
        yield text[midpoint:]


class FakeExecutor:
    """Maps exact SQL strings to rows; any other query raises the configured DB error."""

    def __init__(self, results: dict[str, list[dict[str, Any]]], error: str = 'column "nme" does not exist') -> None:
        self.results = results
        self.error = error
        self.executed: list[str] = []

    def execute(
        self,
        query: str,
        backend: DatabaseBackend,
        *,
        allow_mutations: bool = False,
        **_: Any,
    ) -> ExecutionResult:
        self.executed.append(query)
        if query not in self.results:
            raise DatabaseExecutionError("query failed", raw_error=self.error, backend=backend, query=query)
        rows = self.results[query]
        return ExecutionResult(
            success=True,
            backend=backend,
            query=query,
            rows=rows,
            row_count=len(rows),
            columns=list(rows[0].keys()) if rows else [],
        )


class FakeSchemaManager:
    def __init__(self, schema: str = "### Table `customers`\n| `customer_id` | integer |") -> None:
        self.schema = schema

    def get_pruned_schema(self, user_query: str, backend: DatabaseBackend | None = None) -> str:
        return self.schema


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, openai_api_key="test-key", max_retry_iterations=3, redis_url=None)


@pytest.fixture
def redis_cache() -> RedisCache:
    return RedisCache(fakeredis.FakeRedis(decode_responses=True), failure_cooldown_seconds=30)


def fenced(sql: str) -> str:
    return f"```sql\n{sql}\n```"
