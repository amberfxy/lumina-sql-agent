"""OpenAI-compatible mock LLM for load, cache, and resilience testing.

Answers questions from eval/datasets/nl2sql.jsonl with their gold SQL after a simulated
latency, so the API runs its real code path (OpenAI SDK, HTTP, retries, validation,
PostgreSQL) without a paid model. Results measure system behavior only.

Environment:
    MOCK_LLM_LATENCY_MS   mean simulated generation latency (default 400)
    MOCK_LLM_JITTER       +/- fraction of latency (default 0.2)
    MOCK_LLM_FAULT        none | rate_limit | server_error | timeout | malformed (default none)
    MOCK_LLM_FAULT_RATE   probability of the fault per call (default 0)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.scripted_llm import approx_tokens, extract_user_request  # noqa: E402

DATASET = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "nl2sql.jsonl"
ANSWERS = {
    case["question"]: case["gold_sql"]
    for case in map(json.loads, DATASET.read_text().splitlines())
    if case.get("gold_sql")
}

app = FastAPI(title="mock-llm")
state = {
    "latency_ms": float(os.environ.get("MOCK_LLM_LATENCY_MS", "400")),
    "jitter": float(os.environ.get("MOCK_LLM_JITTER", "0.2")),
    "fault": os.environ.get("MOCK_LLM_FAULT", "none"),
    "fault_rate": float(os.environ.get("MOCK_LLM_FAULT_RATE", "0")),
    "calls": 0,
}


@app.post("/admin/config")
async def configure(request: Request) -> dict:
    """Change latency/fault settings at runtime (used by resilience tests)."""
    state.update(await request.json())
    return state


@app.get("/admin/config")
async def get_config() -> dict:
    return state


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body = await request.json()
    state["calls"] += 1
    prompt = next((m["content"] for m in reversed(body["messages"]) if m["role"] == "user"), "")
    system = next((m["content"] for m in body["messages"] if m["role"] == "system"), "")

    latency = state["latency_ms"] * (1 + random.uniform(-state["jitter"], state["jitter"])) / 1000
    fault = state["fault"] if random.random() < state["fault_rate"] else "none"
    if fault == "timeout":
        await asyncio.sleep(3600)
    await asyncio.sleep(latency)
    if fault == "rate_limit":
        return JSONResponse({"error": {"message": "Rate limit reached", "type": "rate_limit"}}, status_code=429)
    if fault == "server_error":
        return JSONResponse({"error": {"message": "Upstream error", "type": "server_error"}}, status_code=500)

    question = extract_user_request(prompt)
    if fault == "malformed":
        text = "I think you should look at the orders table."
    elif question in ANSWERS:
        text = f"```sql\n{ANSWERS[question]}\n```"
    else:
        text = "CANNOT_ANSWER: question not in the mock answer set"

    prompt_tokens, completion_tokens = approx_tokens(system + prompt), approx_tokens(text)
    return JSONResponse(
        {
            "id": f"mock-{state['calls']}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9000")), log_level="warning")
