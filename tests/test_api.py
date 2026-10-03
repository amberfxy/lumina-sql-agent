from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from api.main import app, get_agent
from eval.scripted_llm import ScriptedLLM
from src.admission import AdmissionRejected, RateLimiter
from src.agent import LuminaSQLAgent
from tests.conftest import FakeExecutor, FakeSchemaManager, fenced

SQL = "SELECT 1 AS ok"


@pytest.fixture
def client(settings):
    responses: list[str] = []

    def fake_agent() -> LuminaSQLAgent:
        return LuminaSQLAgent(
            settings=settings,
            schema_manager=FakeSchemaManager(),
            db_executor=FakeExecutor({SQL: [{"ok": 1}]}),
            llm_client=ScriptedLLM(list(responses) or [fenced(SQL)]),
        )

    app.dependency_overrides[get_agent] = fake_agent
    with TestClient(app) as test_client:
        test_client.responses = responses  # tests may script the LLM for the next request
        yield test_client
    app.dependency_overrides.clear()


def test_livez(client):
    response = client.get("/livez")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_not_ready_without_database(client, monkeypatch):
    runtime = client.app.state.runtime
    monkeypatch.setattr(runtime.db_executor, "health_check", lambda backend: {"success": False, "error": "down"})
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_sync_query(client):
    response = client.post("/api/v1/query", json={"query": "is it ok?"})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["execution"]["rows"] == [{"ok": 1}]
    assert body["usage"]["prompt_tokens"] > 0


def test_stream_query_emits_sse_events(client):
    with client.stream("POST", "/api/v1/query/stream", json={"query": "is it ok?"}) as response:
        assert response.status_code == 200
        events = [json.loads(line[len("data: ") :]) for line in response.iter_lines() if line.startswith("data: ")]
    assert events[-1]["type"] == "result"
    assert events[-1]["content"]["success"] is True


def test_query_validation(client):
    assert client.post("/api/v1/query", json={"query": ""}).status_code == 422
    assert client.post("/api/v1/query", json={"query": "x" * 2001}).status_code == 422


def test_request_id_is_propagated_into_the_agent_thread(client):
    response = client.post("/api/v1/query", json={"query": "q"}, headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
    assert response.json()["request_id"] == "abc-123"


def test_unsafe_request_ids_are_replaced(client):
    response = client.post("/api/v1/query", json={"query": "q"}, headers={"X-Request-ID": "*/ DROP TABLE x; --"})
    request_id = response.headers["X-Request-ID"]
    assert request_id != "*/ DROP TABLE x; --"
    assert len(request_id) == 32 and request_id.isalnum()


def test_unsafe_generated_query_is_a_200_with_failure(client):
    client.responses[:] = [fenced("SELECT 1; DROP TABLE customers")]
    response = client.post("/api/v1/query", json={"query": "q"})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["failure"]["category"] == "UNSAFE_QUERY"
    assert body["failure"]["policy"] == "terminal"


@pytest.mark.parametrize(("marker", "status"), [("!RATE_LIMIT", 503), ("!TIMEOUT", 504), ("!SERVER_ERROR", 502)])
def test_llm_provider_failures_map_to_5xx(client, marker, status):
    client.responses[:] = [marker]
    response = client.post("/api/v1/query", json={"query": "q"})
    assert response.status_code == status
    assert response.json()["failure"]["source"] == "llm"


def test_mutations_require_server_side_enablement(client):
    response = client.post("/api/v1/query", json={"query": "q", "allow_mutations": True})
    assert response.status_code == 403


def test_api_keys_are_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(client.app.state.runtime.settings, "api_keys", SecretStr("k1,k2"))
    assert client.post("/api/v1/query", json={"query": "q"}).status_code == 401
    assert client.post("/api/v1/query", json={"query": "q"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/v1/query", json={"query": "q"}, headers={"X-API-Key": "k2"}).status_code == 200
    ok = client.post("/api/v1/query", json={"query": "q"}, headers={"Authorization": "Bearer k1"})
    assert ok.status_code == 200
    assert client.get("/livez").status_code == 200  # probes stay unauthenticated


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(client.app.state.runtime, "rate_limiter", RateLimiter(per_minute=2))
    statuses = [client.post("/api/v1/query", json={"query": "q"}).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]


def test_admission_rejection_returns_429_with_retry_after(client, monkeypatch):
    class Saturated:
        saturated = True

        async def acquire(self):
            raise AdmissionRejected("queue_full", retry_after_seconds=1)

        def release(self):
            raise AssertionError("release without acquire")

    monkeypatch.setattr(client.app.state.runtime, "admission", Saturated())
    response = client.post("/api/v1/query", json={"query": "q"})
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"
    assert client.post("/api/v1/query/stream", json={"query": "q"}).status_code == 429


def test_missing_llm_key_returns_503(monkeypatch):
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        runtime = test_client.app.state.runtime
        monkeypatch.setattr(runtime.settings, "openai_api_key", None)
        response = test_client.post("/api/v1/query", json={"query": "q"})
    assert response.status_code == 503


def test_metrics_exposes_request_and_failure_counters(client):
    client.get("/livez")
    client.post("/api/v1/query", json={"query": "is it ok?"})
    client.responses[:] = [fenced("DELETE FROM customers")]
    client.post("/api/v1/query", json={"query": "q"})
    body = client.get("/metrics").text
    assert 'lumina_http_requests_total{method="GET",route="/livez",status="200"}' in body
    assert "lumina_agent_runs_total" in body
    assert "lumina_llm_tokens_total" in body
    assert "lumina_inflight_requests" in body
    assert 'lumina_agent_run_failures_total{backend="postgres",category="UNSAFE_QUERY"}' in body
    assert 'lumina_unsafe_query_rejections_total{backend="postgres",layer="validator"}' in body
