"""Optional Redis cache with graceful degradation.

Cache failures never fail a request: on a Redis error the cache logs, records a
metric, and bypasses Redis for a cooldown window so a dead Redis does not add a
socket timeout to every request.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

import redis

from config import Settings
from src.metrics import CACHE_REQUESTS

logger = logging.getLogger(__name__)


class Cache(Protocol):
    enabled: bool

    def get(self, key: str, *, kind: str) -> str | None: ...
    def set(self, key: str, value: str, *, ttl_seconds: int) -> None: ...
    def delete(self, key: str) -> None: ...
    def delete_prefix(self, prefix: str) -> int: ...
    def ping(self) -> bool: ...


class NullCache:
    """No-op cache used when Redis is not configured."""

    enabled = False

    def get(self, key: str, *, kind: str) -> str | None:
        return None

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        return None

    def delete(self, key: str) -> None:
        return None

    def delete_prefix(self, prefix: str) -> int:
        return 0

    def ping(self) -> bool:
        return False


class RedisCache:
    """Redis-backed cache; every operation is best-effort."""

    enabled = True

    def __init__(
        self,
        client: redis.Redis,
        *,
        namespace: str = "lumina",
        failure_cooldown_seconds: float = 30.0,
    ) -> None:
        self._client = client
        self._namespace = namespace
        self._failure_cooldown_seconds = failure_cooldown_seconds
        self._disabled_until = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> RedisCache:
        client = redis.Redis.from_url(
            settings.redis_url,
            socket_timeout=settings.redis_socket_timeout_seconds,
            socket_connect_timeout=settings.redis_socket_timeout_seconds,
            decode_responses=True,
        )
        return cls(client, failure_cooldown_seconds=settings.redis_failure_cooldown_seconds)

    def get(self, key: str, *, kind: str) -> str | None:
        if self._in_cooldown():
            CACHE_REQUESTS.labels(kind=kind, result="error").inc()
            return None
        try:
            value = self._client.get(self._key(key))
        except redis.RedisError as exc:
            self._trip(exc)
            CACHE_REQUESTS.labels(kind=kind, result="error").inc()
            return None
        CACHE_REQUESTS.labels(kind=kind, result="hit" if value is not None else "miss").inc()
        return value

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        if self._in_cooldown():
            return
        try:
            self._client.set(self._key(key), value, ex=ttl_seconds)
        except redis.RedisError as exc:
            self._trip(exc)

    def delete(self, key: str) -> None:
        if self._in_cooldown():
            return
        try:
            self._client.delete(self._key(key))
        except redis.RedisError as exc:
            self._trip(exc)

    def delete_prefix(self, prefix: str) -> int:
        if self._in_cooldown():
            return 0
        deleted = 0
        try:
            batch: list[str] = []
            for key in self._client.scan_iter(match=f"{self._key(prefix)}*", count=500):
                batch.append(key)
                if len(batch) >= 500:
                    deleted += self._client.delete(*batch)
                    batch.clear()
            if batch:
                deleted += self._client.delete(*batch)
        except redis.RedisError as exc:
            self._trip(exc)
        return deleted

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False

    def _key(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    def _in_cooldown(self) -> bool:
        with self._lock:
            return time.monotonic() < self._disabled_until

    def _trip(self, exc: Exception) -> None:
        with self._lock:
            self._disabled_until = time.monotonic() + self._failure_cooldown_seconds
        logger.warning(
            "Redis unavailable, bypassing cache for %.0fs: %s",
            self._failure_cooldown_seconds,
            exc,
        )


def build_cache(settings: Settings) -> Cache:
    if not settings.redis_url:
        logger.info("REDIS_URL not set; caching disabled")
        return NullCache()
    logger.info("Redis cache enabled")
    return RedisCache.from_settings(settings)
