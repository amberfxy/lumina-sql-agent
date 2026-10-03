from __future__ import annotations

import asyncio

import pytest

from src.admission import AdmissionController, AdmissionRejected, RateLimiter


def test_requests_beyond_concurrency_and_queue_are_shed():
    async def scenario():
        controller = AdmissionController(max_concurrent=2, max_queued=1, queue_timeout_seconds=1)
        await controller.acquire()
        await controller.acquire()
        queued = asyncio.create_task(controller.acquire())
        await asyncio.sleep(0)  # let the third request enter the queue
        with pytest.raises(AdmissionRejected) as excinfo:
            await controller.acquire()
        assert excinfo.value.reason == "queue_full"
        controller.release()
        await queued  # the queued request gets the freed slot
        controller.release()
        controller.release()

    asyncio.run(scenario())


def test_queued_requests_time_out():
    async def scenario():
        controller = AdmissionController(max_concurrent=1, max_queued=5, queue_timeout_seconds=0.05)
        await controller.acquire()
        with pytest.raises(AdmissionRejected) as excinfo:
            await controller.acquire()
        assert excinfo.value.reason == "queue_timeout"
        controller.release()
        await controller.acquire()  # the slot is usable again after the timeout
        controller.release()

    asyncio.run(scenario())


def test_rate_limiter_token_bucket():
    limiter = RateLimiter(per_minute=3)
    assert [limiter.allow("a") for _ in range(4)] == [True, True, True, False]
    assert limiter.allow("b")  # buckets are per client
    assert RateLimiter(per_minute=0).allow("a")  # disabled
