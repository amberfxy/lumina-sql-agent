"""Admission control (bounded concurrency + bounded queue) and per-client rate limiting.

Each agent request holds a worker thread, a DB connection, and an in-flight LLM call
for seconds. Without a bound, overload shows up as unbounded latency and pool
timeouts; with one, excess load is shed early with 429 + Retry-After.
"""

from __future__ import annotations

import asyncio
import threading
import time

from src.metrics import ADMISSION_REJECTIONS, IN_FLIGHT, QUEUED


class AdmissionRejected(Exception):
    def __init__(self, reason: str, retry_after_seconds: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after_seconds = retry_after_seconds


class AdmissionController:
    def __init__(self, max_concurrent: int, max_queued: int, queue_timeout_seconds: float) -> None:
        self.max_concurrent = max_concurrent
        self.max_queued = max_queued
        self.queue_timeout_seconds = queue_timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._waiting = 0

    @property
    def saturated(self) -> bool:
        return self._semaphore.locked() and self._waiting >= self.max_queued

    async def acquire(self) -> None:
        if self.saturated:
            ADMISSION_REJECTIONS.labels(reason="queue_full").inc()
            raise AdmissionRejected("queue_full", retry_after_seconds=1)
        self._waiting += 1
        QUEUED.inc()
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.queue_timeout_seconds)
        except TimeoutError as exc:
            ADMISSION_REJECTIONS.labels(reason="queue_timeout").inc()
            raise AdmissionRejected("queue_timeout", retry_after_seconds=2) from exc
        finally:
            self._waiting -= 1
            QUEUED.dec()
        IN_FLIGHT.inc()

    def release(self) -> None:
        IN_FLIGHT.dec()
        self._semaphore.release()


class RateLimiter:
    """In-process token bucket per client. Limits are per replica (see docs/security.md)."""

    _MAX_CLIENTS = 10_000

    def __init__(self, per_minute: int) -> None:
        self.capacity = float(per_minute)
        self.refill_per_second = per_minute / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.capacity > 0

    def allow(self, client: str) -> bool:
        if not self.enabled:
            return True
        now = time.monotonic()
        with self._lock:
            tokens, updated = self._buckets.get(client, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - updated) * self.refill_per_second)
            allowed = tokens >= 1.0
            self._buckets[client] = (tokens - 1.0 if allowed else tokens, now)
            if len(self._buckets) > self._MAX_CLIENTS:
                oldest = min(self._buckets, key=lambda key: self._buckets[key][1])
                del self._buckets[oldest]
        if not allowed:
            ADMISSION_REJECTIONS.labels(reason="rate_limited").inc()
        return allowed
