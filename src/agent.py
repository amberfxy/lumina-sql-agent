"""Core LLM orchestration with self-correcting database execution loop."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from anthropic import Anthropic
from openai import OpenAI

from config import DatabaseBackend, LLMProvider, Settings, get_settings
from src.db_executor import DatabaseExecutionError, DatabaseExecutor, ExecutionResult
from src.schema_manager import SchemaManager

logger = logging.getLogger(__name__)

_CODE_FENCE_PATTERN = re.compile(r"```(?:sql|partiql|json)?\s*([\s\S]*?)```", re.IGNORECASE)


INITIAL_SQL_PROMPT = """You are LuminaSQL, an enterprise-grade database agent.

## Task
Convert the user's natural language request into a single executable {query_language} statement.

## Rules
- Use ONLY tables and columns present in the pruned schema below.
- Prefer explicit column lists over SELECT *.
- Add LIMIT 100 unless the user explicitly asks for all rows.
- Return ONLY the query inside a fenced code block.
- Do not include commentary outside the code block.
- For PostgreSQL, generate ANSI SQL compatible with PostgreSQL.
- For DynamoDB, generate PartiQL compatible with `execute_statement`.

## Pruned Schema
{pruned_schema}

## User Request
{user_query}
"""

DEBUG_SQL_PROMPT = """You are LuminaSQL Debugger. The previous query failed at execution time.

## Objective
Analyze the database error, identify the root cause, and return a corrected {query_language} query.

## Constraints
- Use ONLY objects from the pruned schema.
- Preserve the user's intent.
- Return ONLY the corrected query in a fenced code block.
- Do not repeat the same mistake.

## Pruned Schema
{pruned_schema}

## Original User Request
{user_query}

## Failed Query
{failed_query}

## Database Error Log
{error_log}

## Attempt
This is retry {attempt_number} of {max_attempts}.
"""


@dataclass
class DebugAttempt:
    attempt_number: int
    query: str
    error: str
    raw_error: str


@dataclass
class AgentResult:
    success: bool
    backend: DatabaseBackend
    user_query: str
    final_query: str | None
    execution: ExecutionResult | None
    pruned_schema: str
    attempts: list[DebugAttempt] = field(default_factory=list)
    llm_provider: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "backend": self.backend.value,
            "user_query": self.user_query,
            "final_query": self.final_query,
            "execution": self.execution.to_dict() if self.execution else None,
            "pruned_schema": self.pruned_schema,
            "attempts": [
                {
                    "attempt_number": attempt.attempt_number,
                    "query": attempt.query,
                    "error": attempt.error,
                    "raw_error": attempt.raw_error,
                }
                for attempt in self.attempts
            ],
            "llm_provider": self.llm_provider,
            "message": self.message,
        }


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...
    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]: ...


class OpenAILLMClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key.get_secret_value())

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class AnthropicLLMClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        self.settings = settings
        self.client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value())

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_blocks = [block.text for block in response.content if block.type == "text"]
        return "".join(text_blocks)

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.settings.anthropic_model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text


class LuminaSQLAgent:
    """Generates queries with an automatic self-debugging execution loop."""

    def __init__(
        self,
        settings: Settings | None = None,
        schema_manager: SchemaManager | None = None,
        db_executor: DatabaseExecutor | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.schema_manager = schema_manager or SchemaManager(settings=self.settings)
        self.db_executor = db_executor or DatabaseExecutor(settings=self.settings)
        self.llm_client = llm_client or self._build_llm_client()
        self.max_attempts = self.settings.max_retry_iterations

    def run(
        self,
        user_query: str,
        backend: DatabaseBackend = DatabaseBackend.POSTGRES,
        *,
        allow_mutations: bool = False,
    ) -> AgentResult:
        """Execute the full generate -> execute -> debug loop."""
        pruned_schema = self.schema_manager.get_pruned_schema(user_query, backend=backend)
        query_language = self._query_language(backend)
        system_prompt = self._system_prompt(backend)

        logger.info(
            "Agent run started backend=%s provider=%s max_attempts=%s",
            backend.value,
            self.settings.llm_provider.value,
            self.max_attempts,
        )

        initial_prompt = INITIAL_SQL_PROMPT.format(
            query_language=query_language,
            pruned_schema=pruned_schema,
            user_query=user_query,
        )
        current_query = self._extract_query(self.llm_client.complete(system_prompt, initial_prompt))
        attempts: list[DebugAttempt] = []

        for attempt_index in range(1, self.max_attempts + 1):
            logger.info("Execution attempt %s/%s query=%r", attempt_index, self.max_attempts, current_query[:300])
            try:
                execution = self.db_executor.execute(
                    current_query,
                    backend=backend,
                    allow_mutations=allow_mutations,
                )
                logger.info(
                    "Execution succeeded attempt=%s row_count=%s",
                    attempt_index,
                    execution.row_count,
                )
                return AgentResult(
                    success=True,
                    backend=backend,
                    user_query=user_query,
                    final_query=current_query,
                    execution=execution,
                    pruned_schema=pruned_schema,
                    attempts=attempts,
                    llm_provider=self.settings.llm_provider.value,
                    message="Query executed successfully.",
                )
            except DatabaseExecutionError as exc:
                debug_attempt = DebugAttempt(
                    attempt_number=attempt_index,
                    query=current_query,
                    error=str(exc),
                    raw_error=exc.raw_error,
                )
                attempts.append(debug_attempt)
                logger.warning(
                    "Execution failed attempt=%s raw_error=%s",
                    attempt_index,
                    exc.raw_error,
                )

                if attempt_index >= self.max_attempts:
                    break

                debug_prompt = DEBUG_SQL_PROMPT.format(
                    query_language=query_language,
                    pruned_schema=pruned_schema,
                    user_query=user_query,
                    failed_query=current_query,
                    error_log=exc.raw_error,
                    attempt_number=attempt_index + 1,
                    max_attempts=self.max_attempts,
                )
                current_query = self._extract_query(
                    self.llm_client.complete(system_prompt, debug_prompt)
                )

        return AgentResult(
            success=False,
            backend=backend,
            user_query=user_query,
            final_query=current_query,
            execution=None,
            pruned_schema=pruned_schema,
            attempts=attempts,
            llm_provider=self.settings.llm_provider.value,
            message=(
                f"Failed after {self.max_attempts} attempts. "
                f"Last error: {attempts[-1].raw_error if attempts else 'unknown'}"
            ),
        )

    def stream_run(
        self,
        user_query: str,
        backend: DatabaseBackend = DatabaseBackend.POSTGRES,
        *,
        allow_mutations: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield structured events for API/UI streaming consumers."""
        yield {"type": "status", "message": "Pruning schema metadata..."}
        pruned_schema = self.schema_manager.get_pruned_schema(user_query, backend=backend)
        yield {"type": "schema", "content": pruned_schema}

        query_language = self._query_language(backend)
        system_prompt = self._system_prompt(backend)
        initial_prompt = INITIAL_SQL_PROMPT.format(
            query_language=query_language,
            pruned_schema=pruned_schema,
            user_query=user_query,
        )

        yield {"type": "status", "message": "Generating initial query..."}
        generated_chunks: list[str] = []
        for chunk in self.llm_client.stream(system_prompt, initial_prompt):
            generated_chunks.append(chunk)
            yield {"type": "llm_token", "content": chunk}

        current_query = self._extract_query("".join(generated_chunks))
        yield {"type": "query", "content": current_query}

        attempts: list[DebugAttempt] = []
        for attempt_index in range(1, self.max_attempts + 1):
            yield {
                "type": "status",
                "message": f"Executing query (attempt {attempt_index}/{self.max_attempts})...",
            }
            try:
                execution = self.db_executor.execute(
                    current_query,
                    backend=backend,
                    allow_mutations=allow_mutations,
                )
                result = AgentResult(
                    success=True,
                    backend=backend,
                    user_query=user_query,
                    final_query=current_query,
                    execution=execution,
                    pruned_schema=pruned_schema,
                    attempts=attempts,
                    llm_provider=self.settings.llm_provider.value,
                    message="Query executed successfully.",
                )
                yield {"type": "result", "content": result.to_dict()}
                return
            except DatabaseExecutionError as exc:
                debug_attempt = DebugAttempt(
                    attempt_number=attempt_index,
                    query=current_query,
                    error=str(exc),
                    raw_error=exc.raw_error,
                )
                attempts.append(debug_attempt)
                yield {
                    "type": "error",
                    "content": {
                        "attempt": attempt_index,
                        "query": current_query,
                        "error": str(exc),
                        "raw_error": exc.raw_error,
                    },
                }

                if attempt_index >= self.max_attempts:
                    break

                debug_prompt = DEBUG_SQL_PROMPT.format(
                    query_language=query_language,
                    pruned_schema=pruned_schema,
                    user_query=user_query,
                    failed_query=current_query,
                    error_log=exc.raw_error,
                    attempt_number=attempt_index + 1,
                    max_attempts=self.max_attempts,
                )
                yield {
                    "type": "status",
                    "message": f"Self-debugging query (attempt {attempt_index + 1})...",
                }
                debug_chunks: list[str] = []
                for chunk in self.llm_client.stream(system_prompt, debug_prompt):
                    debug_chunks.append(chunk)
                    yield {"type": "llm_token", "content": chunk}
                current_query = self._extract_query("".join(debug_chunks))
                yield {"type": "query", "content": current_query}

        failed_result = AgentResult(
            success=False,
            backend=backend,
            user_query=user_query,
            final_query=current_query,
            execution=None,
            pruned_schema=pruned_schema,
            attempts=attempts,
            llm_provider=self.settings.llm_provider.value,
            message=(
                f"Failed after {self.max_attempts} attempts. "
                f"Last error: {attempts[-1].raw_error if attempts else 'unknown'}"
            ),
        )
        yield {"type": "result", "content": failed_result.to_dict()}

    async def astream_run(
        self,
        user_query: str,
        backend: DatabaseBackend = DatabaseBackend.POSTGRES,
        *,
        allow_mutations: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        for event in self.stream_run(
            user_query=user_query,
            backend=backend,
            allow_mutations=allow_mutations,
        ):
            yield event

    def _build_llm_client(self) -> LLMClient:
        if self.settings.llm_provider == LLMProvider.ANTHROPIC:
            return AnthropicLLMClient(self.settings)
        return OpenAILLMClient(self.settings)

    @staticmethod
    def _query_language(backend: DatabaseBackend) -> str:
        return "PartiQL" if backend == DatabaseBackend.DYNAMODB else "SQL"

    @staticmethod
    def _system_prompt(backend: DatabaseBackend) -> str:
        if backend == DatabaseBackend.DYNAMODB:
            return (
                "You generate safe, precise AWS DynamoDB PartiQL. "
                "Never invent tables or attributes."
            )
        return "You generate safe, precise PostgreSQL SQL. Never invent tables or columns."

    @staticmethod
    def _extract_query(llm_response: str) -> str:
        match = _CODE_FENCE_PATTERN.search(llm_response)
        if match:
            return match.group(1).strip()

        stripped = llm_response.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
                statement = payload.get("statement") or payload.get("Statement")
                if statement:
                    return str(statement).strip()
            except json.JSONDecodeError:
                pass
        return stripped
