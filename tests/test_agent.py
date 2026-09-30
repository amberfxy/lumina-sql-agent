from __future__ import annotations

import asyncio

from config import DatabaseBackend
from src.agent import LuminaSQLAgent
from tests.conftest import FakeExecutor, FakeLLM, FakeSchemaManager, fenced

GOOD_SQL = "SELECT COUNT(*) AS n FROM customers"
BAD_SQL = "SELECT COUNT(nme) FROM customers"
ROWS = [{"n": 320}]


def make_agent(settings, llm, executor, cache=None) -> LuminaSQLAgent:
    return LuminaSQLAgent(
        settings=settings,
        schema_manager=FakeSchemaManager(),
        db_executor=executor,
        llm_client=llm,
        cache=cache,
    )


def test_first_attempt_success(settings):
    llm = FakeLLM([fenced(GOOD_SQL)])
    result = make_agent(settings, llm, FakeExecutor({GOOD_SQL: ROWS})).run("how many customers?")

    assert result.success
    assert result.final_query == GOOD_SQL
    assert result.attempts == []
    assert result.llm_calls == 1
    assert result.execution.rows == ROWS


def test_self_corrects_using_database_error(settings):
    llm = FakeLLM([fenced(BAD_SQL), fenced(GOOD_SQL)])
    executor = FakeExecutor({GOOD_SQL: ROWS})
    result = make_agent(settings, llm, executor).run("how many customers?")

    assert result.success
    assert result.llm_calls == 2
    assert [attempt.query for attempt in result.attempts] == [BAD_SQL]
    assert executor.executed == [BAD_SQL, GOOD_SQL]
    debug_prompt = llm.prompts[1]
    assert BAD_SQL in debug_prompt
    assert 'column "nme" does not exist' in debug_prompt


def test_gives_up_after_max_attempts(settings):
    llm = FakeLLM([fenced(BAD_SQL)] * 3)
    result = make_agent(settings, llm, FakeExecutor({})).run("how many customers?")

    assert not result.success
    assert len(result.attempts) == 3
    assert result.llm_calls == 3
    assert "Failed after 3 attempts" in result.message


def test_max_attempts_override_disables_retries(settings):
    llm = FakeLLM([fenced(BAD_SQL)])
    result = make_agent(settings, llm, FakeExecutor({GOOD_SQL: ROWS})).run("q", max_attempts=1)

    assert not result.success
    assert result.llm_calls == 1


def test_cache_hit_skips_llm(settings, redis_cache):
    executor = FakeExecutor({GOOD_SQL: ROWS})
    first = make_agent(settings, FakeLLM([fenced(GOOD_SQL)]), executor, redis_cache).run("How many customers?")
    assert first.success and not first.cache_hit

    no_llm = FakeLLM([])
    second = make_agent(settings, no_llm, executor, redis_cache).run("  how many   CUSTOMERS? ")
    assert second.success and second.cache_hit
    assert second.llm_calls == 0
    assert no_llm.prompts == []


def test_stale_cached_query_is_evicted_and_regenerated(settings, redis_cache):
    llm = FakeLLM([fenced(BAD_SQL), fenced(GOOD_SQL)])
    agent = make_agent(settings, llm, FakeExecutor({GOOD_SQL: ROWS}), redis_cache)
    key = agent._query_cache_key("q", DatabaseBackend.POSTGRES, FakeSchemaManager().schema)
    redis_cache.set(key, "SELECT removed_column FROM customers", ttl_seconds=60)

    result = agent.run("q")

    assert result.success and not result.cache_hit
    assert redis_cache.get(key, kind="query") == GOOD_SQL


def test_mutating_runs_bypass_cache(settings, redis_cache):
    executor = FakeExecutor({GOOD_SQL: ROWS})
    make_agent(settings, FakeLLM([fenced(GOOD_SQL)]), executor, redis_cache).run("q", allow_mutations=True)
    llm = FakeLLM([fenced(GOOD_SQL)])
    result = make_agent(settings, llm, executor, redis_cache).run("q")

    assert not result.cache_hit
    assert result.llm_calls == 1


def test_stream_run_emits_events_in_order(settings):
    llm = FakeLLM([fenced(BAD_SQL), fenced(GOOD_SQL)])
    events = list(make_agent(settings, llm, FakeExecutor({GOOD_SQL: ROWS})).stream_run("q"))
    types = [event["type"] for event in events]

    assert types[0] == "status"
    assert "schema" in types
    assert "llm_token" in types
    assert types.count("query") == 2
    assert types.count("error") == 1
    assert types[-1] == "result"
    assert events[-1]["content"]["success"] is True


def test_astream_run_matches_sync_stream(settings):
    llm = FakeLLM([fenced(GOOD_SQL)])
    agent = make_agent(settings, llm, FakeExecutor({GOOD_SQL: ROWS}))

    async def collect():
        return [event async for event in agent.astream_run("q")]

    events = asyncio.run(collect())
    assert events[-1]["type"] == "result"
    assert events[-1]["content"]["final_query"] == GOOD_SQL


def test_concurrent_runs_are_independent(settings):
    responses = [fenced(GOOD_SQL)] * 20

    class ThreadSafeLLM(FakeLLM):
        def complete(self, system_prompt, user_prompt):
            return responses[0]

    agent = make_agent(settings, ThreadSafeLLM([]), FakeExecutor({GOOD_SQL: ROWS}))

    async def fan_out():
        return await asyncio.gather(*(agent.arun(f"question {i}") for i in range(20)))

    results = asyncio.run(fan_out())
    assert all(result.success for result in results)
    assert len({result.user_query for result in results}) == 20


def test_extract_query_variants():
    assert LuminaSQLAgent._extract_query("```sql\nSELECT 1\n```") == "SELECT 1"
    assert LuminaSQLAgent._extract_query("Here:\n```\nSELECT 2\n```\nDone") == "SELECT 2"
    assert LuminaSQLAgent._extract_query('{"statement": "SELECT * FROM t"}') == "SELECT * FROM t"
    assert LuminaSQLAgent._extract_query("  SELECT 3  ") == "SELECT 3"
