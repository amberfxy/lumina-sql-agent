"""Core LLM orchestration with self-correcting database execution loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from anthropic import Anthropic
from openai import OpenAI

from config import DatabaseBackend, LLMProvider, Settings, get_settings
from src.cache import Cache, NullCache
from src.db_executor import DatabaseExecutionError, DatabaseExecutor, ExecutionResult
from src.metrics import AGENT_ATTEMPTS, AGENT_LATENCY, AGENT_RUNS, LLM_CALLS, LLM_LATENCY
from src.schema_manager import SchemaManager

logger = logging.getLogger(__name__)

_CODE_FENCE_PATTERN = re.compile(r"```(?:sql|partiql|json)?\s*([\s\S]*?)```", re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


INITIAL_SQL_PROMPT = """You are LuminaSQL, a database agent.

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
    cache_hit: bool = False
    llm_calls: int = 0
    duration_ms: float = 0.0

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
            "cache_hit": self.cache_hit,
            "llm_calls": self.llm_calls,
            "duration_ms": round(self.duration_ms, 2),
        }


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...
    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]: ...


class OpenAILLMClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

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
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class AnthropicLLMClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        self.settings = settings
        self.client = Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

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
            yield from stream.text_stream


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == LLMProvider.ANTHROPIC:
        return AnthropicLLMClient(settings)
    return OpenAILLMClient(settings)


class LuminaSQLAgent:
    """Generates queries with an automatic self-debugging execution loop.

    `run` and `stream_run` share one event generator so caching, retries, and
    metrics behave identically for synchronous and streaming callers.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        schema_manager: SchemaManager | None = None,
        db_executor: DatabaseExecutor | None = None,
        llm_client: LLMClient | None = None,
        cache: Cache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or NullCache()
        self.db_executor = db_executor or DatabaseExecutor(settings=self.settings)
        self.schema_manager = schema_manager or SchemaManager(
            settings=self.settings, db_executor=self.db_executor, cache=self.cache
        )
        self.llm_client = llm_client or build_llm_client(self.settings)
        self.max_attempts = self.settings.max_retry_iterations

    def run(
        self,
        user_query: str,
        backend: DatabaseBackend = DatabaseBackend.POSTGRES,
        *,
        allow_mutations: bool = False,
        use_cache: bool = True,
        max_attempts: int | None = None,
    ) -> AgentResult:
        """Execute the full generate -> execute -> debug loop and return the final result."""
        result: AgentResult | None = None
        for event in self._run_events(
            user_query,
            backend,
            allow_mutations=allow_mutations,
            use_cache=use_cache,
            max_attempts=max_attempts,
            stream_tokens=False,
        ):
            if event["type"] == "result":
                result = event["content"]
        assert result is not None
        return result

    async def arun(
        self,
        user_query: str,
        backend: DatabaseBackend = DatabaseBackend.POSTGRES,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the blocking agent loop in a worker thread so callers can fan out concurrently."""
        return await asyncio.to_thread(self.run, user_query, backend, **kwargs)

    def stream_run(
        self,
        user_query: str,
        backend: DatabaseBackend = DatabaseBackend.POSTGRES,
        *,
        allow_mutations: bool = False,
        use_cache: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Yield JSON-serializable events for API/UI streaming consumers."""
        for event in self._run_events(
            user_query,
            backend,
            allow_mutations=allow_mutations,
            use_cache=use_cache,
            max_attempts=None,
            stream_tokens=True,
        ):
            if event["type"] == "result":
                yield {"type": "result", "content": event["content"].to_dict()}
            else:
                yield event

    async def astream_run(
        self,
        user_query: str,
        backend: DatabaseBackend = DatabaseBackend.POSTGRES,
        *,
        allow_mutations: bool = False,
        use_cache: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async wrapper that advances the blocking generator in a worker thread.

        LLM and database calls are blocking; iterating them directly inside an async
        endpoint would stall the event loop and serialize every concurrent request.
        """
        iterator = self.stream_run(user_query, backend, allow_mutations=allow_mutations, use_cache=use_cache)
        sentinel = object()
        while True:
            event = await asyncio.to_thread(next, iterator, sentinel)
            if event is sentinel:
                return
            yield event

    def _run_events(
        self,
        user_query: str,
        backend: DatabaseBackend,
        *,
        allow_mutations: bool,
        use_cache: bool,
        max_attempts: int | None,
        stream_tokens: bool,
    ) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        max_attempts = max_attempts or self.max_attempts
        provider = self.settings.llm_provider.value
        attempts: list[DebugAttempt] = []
        llm_calls = 0
        executions = 0

        def finish(
            success: bool,
            final_query: str | None,
            execution: ExecutionResult | None,
            message: str,
            outcome: str,
            cache_hit: bool = False,
        ) -> dict[str, Any]:
            duration = time.perf_counter() - started
            AGENT_RUNS.labels(backend=backend.value, outcome=outcome).inc()
            AGENT_LATENCY.labels(backend=backend.value).observe(duration)
            if executions:
                AGENT_ATTEMPTS.labels(backend=backend.value).observe(executions)
            return {
                "type": "result",
                "content": AgentResult(
                    success=success,
                    backend=backend,
                    user_query=user_query,
                    final_query=final_query,
                    execution=execution,
                    pruned_schema=pruned_schema,
                    attempts=attempts,
                    llm_provider=provider,
                    message=message,
                    cache_hit=cache_hit,
                    llm_calls=llm_calls,
                    duration_ms=duration * 1000,
                ),
            }

        logger.info(
            "Agent run started backend=%s provider=%s max_attempts=%s",
            backend.value,
            provider,
            max_attempts,
        )

        yield {"type": "status", "message": "Pruning schema metadata..."}
        pruned_schema = self.schema_manager.get_pruned_schema(user_query, backend=backend)
        yield {"type": "schema", "content": pruned_schema}

        # Mutating runs are never served from or written to the cache.
        cacheable = use_cache and self.cache.enabled and not allow_mutations
        cache_key = self._query_cache_key(user_query, backend, pruned_schema) if cacheable else ""

        if cacheable:
            cached_query = self.cache.get(cache_key, kind="query")
            if cached_query:
                yield {"type": "status", "message": "Serving query from cache..."}
                yield {"type": "query", "content": cached_query}
                try:
                    executions += 1
                    execution = self.db_executor.execute(cached_query, backend=backend, allow_mutations=False)
                    yield finish(
                        True, cached_query, execution, "Query executed successfully (cached).", "cache_hit", True
                    )
                    return
                except DatabaseExecutionError as exc:
                    logger.warning("Cached query failed, regenerating: %s", exc.raw_error)
                    self.cache.delete(cache_key)
                    executions = 0

        query_language = self._query_language(backend)
        system_prompt = self._system_prompt(backend)

        yield {"type": "status", "message": "Generating initial query..."}
        initial_prompt = INITIAL_SQL_PROMPT.format(
            query_language=query_language,
            pruned_schema=pruned_schema,
            user_query=user_query,
        )
        llm_calls += 1
        response_text = yield from self._call_llm(system_prompt, initial_prompt, "generate", stream_tokens)
        current_query = self._extract_query(response_text)
        yield {"type": "query", "content": current_query}

        for attempt_index in range(1, max_attempts + 1):
            yield {"type": "status", "message": f"Executing query (attempt {attempt_index}/{max_attempts})..."}
            logger.info("Execution attempt %s/%s query=%r", attempt_index, max_attempts, current_query[:300])
            try:
                executions += 1
                execution = self.db_executor.execute(current_query, backend=backend, allow_mutations=allow_mutations)
            except DatabaseExecutionError as exc:
                attempts.append(
                    DebugAttempt(
                        attempt_number=attempt_index,
                        query=current_query,
                        error=str(exc),
                        raw_error=exc.raw_error,
                    )
                )
                logger.warning("Execution failed attempt=%s raw_error=%s", attempt_index, exc.raw_error)
                yield {
                    "type": "error",
                    "content": {
                        "attempt": attempt_index,
                        "query": current_query,
                        "error": str(exc),
                        "raw_error": exc.raw_error,
                    },
                }
                if attempt_index >= max_attempts:
                    break

                yield {"type": "status", "message": f"Self-debugging query (attempt {attempt_index + 1})..."}
                debug_prompt = DEBUG_SQL_PROMPT.format(
                    query_language=query_language,
                    pruned_schema=pruned_schema,
                    user_query=user_query,
                    failed_query=current_query,
                    error_log=exc.raw_error,
                    attempt_number=attempt_index + 1,
                    max_attempts=max_attempts,
                )
                llm_calls += 1
                response_text = yield from self._call_llm(system_prompt, debug_prompt, "debug", stream_tokens)
                current_query = self._extract_query(response_text)
                yield {"type": "query", "content": current_query}
                continue

            logger.info("Execution succeeded attempt=%s row_count=%s", attempt_index, execution.row_count)
            if cacheable:
                self.cache.set(cache_key, current_query, ttl_seconds=self.settings.query_cache_ttl_seconds)
            outcome = "success" if attempt_index == 1 else "self_corrected"
            yield finish(True, current_query, execution, "Query executed successfully.", outcome)
            return

        last_error = attempts[-1].raw_error if attempts else "unknown"
        yield finish(
            False,
            current_query,
            None,
            f"Failed after {max_attempts} attempts. Last error: {last_error}",
            "failed",
        )

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        phase: str,
        stream_tokens: bool,
    ) -> Iterator[dict[str, Any]]:
        """Call the LLM, optionally yielding token events; returns the full response text."""
        provider = self.settings.llm_provider.value
        started = time.perf_counter()
        status = "ok"
        try:
            if not stream_tokens:
                return self.llm_client.complete(system_prompt, user_prompt)
            chunks: list[str] = []
            for chunk in self.llm_client.stream(system_prompt, user_prompt):
                chunks.append(chunk)
                yield {"type": "llm_token", "content": chunk}
            return "".join(chunks)
        except Exception:
            status = "error"
            raise
        finally:
            LLM_CALLS.labels(provider=provider, phase=phase, status=status).inc()
            LLM_LATENCY.labels(provider=provider, phase=phase).observe(time.perf_counter() - started)

    def _query_cache_key(self, user_query: str, backend: DatabaseBackend, pruned_schema: str) -> str:
        # Keyed on the pruned schema so a schema change naturally invalidates stale queries.
        normalized_question = _WHITESPACE_PATTERN.sub(" ", user_query.strip().lower())
        digest = hashlib.sha256(
            "\x1f".join(
                [
                    backend.value,
                    self.settings.llm_provider.value,
                    self.settings.active_llm_model,
                    normalized_question,
                    pruned_schema,
                ]
            ).encode()
        ).hexdigest()
        return f"query:{digest}"

    @staticmethod
    def _query_language(backend: DatabaseBackend) -> str:
        return "PartiQL" if backend == DatabaseBackend.DYNAMODB else "SQL"

    @staticmethod
    def _system_prompt(backend: DatabaseBackend) -> str:
        if backend == DatabaseBackend.DYNAMODB:
            return "You generate safe, precise AWS DynamoDB PartiQL. Never invent tables or attributes."
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
