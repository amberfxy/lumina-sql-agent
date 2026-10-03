"""Deterministic LLM stand-ins for CI, fault injection, and load tests.

These measure the *system* (validation, retry policy, execution, concurrency), never
model quality. Token counts are approximations (characters / 4) and are labeled as such
in every report that uses them.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Generator

import httpx
import openai

from src.agent import LLMResponse

_FAULTS = {"!RATE_LIMIT", "!TIMEOUT", "!SERVER_ERROR"}
_USER_REQUEST = re.compile(r"<user_request>\n([\s\S]*?)\n</user_request>")
_REQUEST = httpx.Request("POST", "https://llm.invalid/v1/chat/completions")


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def raise_fault(marker: str) -> None:
    """Raise the same exception types the OpenAI SDK raises after its own retries give up."""
    if marker == "!RATE_LIMIT":
        raise openai.RateLimitError("Rate limit reached", response=httpx.Response(429, request=_REQUEST), body=None)
    if marker == "!TIMEOUT":
        raise openai.APITimeoutError(request=_REQUEST)
    if marker == "!SERVER_ERROR":
        raise openai.InternalServerError("Upstream error", response=httpx.Response(500, request=_REQUEST), body=None)


class ScriptedLLM:
    """Returns scripted responses in order; fault markers raise provider exceptions."""

    def __init__(self, responses: list[str], latency_seconds: float = 0.0) -> None:
        self.responses = list(responses)
        self.latency_seconds = latency_seconds
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def complete(self, system_prompt: str, user_prompt: str, timeout: float | None = None) -> LLMResponse:
        with self._lock:
            self.prompts.append(user_prompt)
            if not self.responses:
                raise AssertionError("ScriptedLLM ran out of responses")
            text = self.responses.pop(0)
        if self.latency_seconds:
            time.sleep(self.latency_seconds)
        if text in _FAULTS:
            raise_fault(text)
        return LLMResponse(text, approx_tokens(system_prompt + user_prompt), approx_tokens(text))

    def stream(
        self, system_prompt: str, user_prompt: str, timeout: float | None = None
    ) -> Generator[str, None, LLMResponse]:
        response = self.complete(system_prompt, user_prompt, timeout)
        yield response.text
        return response


def extract_user_request(prompt: str) -> str:
    # The rules also mention the tag names inline; only the data block puts a newline after the tag.
    matches = _USER_REQUEST.findall(prompt)
    return matches[-1].strip() if matches else ""
