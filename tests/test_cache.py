from __future__ import annotations

import redis

from src.cache import NullCache, RedisCache, build_cache


class BrokenRedis:
    def __init__(self) -> None:
        self.calls = 0

    def _fail(self, *args, **kwargs):
        self.calls += 1
        raise redis.ConnectionError("connection refused")

    get = set = delete = scan_iter = ping = _fail


def test_set_get_delete(redis_cache):
    redis_cache.set("k", "v", ttl_seconds=60)
    assert redis_cache.get("k", kind="query") == "v"
    redis_cache.delete("k")
    assert redis_cache.get("k", kind="query") is None


def test_ttl_is_applied(redis_cache):
    redis_cache.set("k", "v", ttl_seconds=60)
    ttl = redis_cache._client.ttl("lumina:k")
    assert 0 < ttl <= 60


def test_delete_prefix_only_removes_matching_keys(redis_cache):
    for index in range(3):
        redis_cache.set(f"query:{index}", "sql", ttl_seconds=60)
    redis_cache.set("schema:postgres", "{}", ttl_seconds=60)

    assert redis_cache.delete_prefix("query:") == 3
    assert redis_cache.get("schema:postgres", kind="schema") == "{}"


def test_redis_failure_degrades_gracefully_and_backs_off():
    client = BrokenRedis()
    cache = RedisCache(client, failure_cooldown_seconds=30)

    assert cache.get("k", kind="query") is None
    cache.set("k", "v", ttl_seconds=60)
    cache.delete("k")
    assert cache.get("k", kind="query") is None

    # Only the first call reached Redis; the rest were short-circuited by the cooldown.
    assert client.calls == 1


def test_build_cache_without_url_is_null(settings):
    assert isinstance(build_cache(settings), NullCache)
    assert not NullCache().enabled
