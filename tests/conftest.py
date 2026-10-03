from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

import fakeredis
import pytest

from config import DatabaseBackend, Settings
from src.agent import LLMResponse
from src.cache import RedisCache
from src.db_executor import DatabaseExecutionError, ExecutionResult
from src.failures import FailureCategory


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("LUMINA_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="set LUMINA_INTEGRATION=1 with Postgres/Redis running")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


class FakeLLM:
    """Returns scripted responses in order and records every prompt it receives.
    A response may be an exception instance, which is raised instead."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.prompts.append(user_prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMResponse(response, prompt_tokens=100, completion_tokens=10)

    def stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, LLMResponse]:
        response = self.complete(system_prompt, user_prompt)
        midpoint = len(response.text) // 2
        yield response.text[:midpoint]
        yield response.text[midpoint:]
        return response


class FakeExecutor:
    """Maps exact SQL strings to rows; any other query raises the configured DB error.
    `errors` can script a sequence of categories to raise before results are served."""

    def __init__(
        self,
        results: dict[str, list[dict[str, Any]]],
        error: str = 'column "nme" does not exist',
        category: FailureCategory = FailureCategory.UNKNOWN_COLUMN,
        errors: list[FailureCategory] | None = None,
    ) -> None:
        self.results = results
        self.error = error
        self.category = category
        self.errors = list(errors or [])
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
        if self.errors:
            category = self.errors.pop(0)
            raise DatabaseExecutionError(
                "query failed", raw_error=category.value, backend=backend, query=query, category=category
            )
        if query not in self.results:
            raise DatabaseExecutionError(
                "query failed", raw_error=self.error, backend=backend, query=query, category=self.category
            )
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
    def __init__(
        self,
        schema: str = "### Table `customers`\n| `customer_id` | integer |",
        tables: set[str] | None = None,
    ) -> None:
        self.schema = schema
        self.tables = {"customers", "orders"} if tables is None else tables

    def get_table_names(self, backend: DatabaseBackend | None = None) -> set[str]:
        return self.tables

    def get_pruned_schema(self, user_query: str, backend: DatabaseBackend | None = None) -> str:
        return self.schema


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-key",
        max_retry_iterations=3,
        redis_url=None,
        db_retry_backoff_seconds=0,
    )


@pytest.fixture
def redis_cache() -> RedisCache:
    return RedisCache(fakeredis.FakeRedis(decode_responses=True), failure_cooldown_seconds=30)


def fenced(sql: str) -> str:
    return f"```sql\n{sql}\n```"
