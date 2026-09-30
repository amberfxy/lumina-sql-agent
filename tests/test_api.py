from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_agent
from src.agent import LuminaSQLAgent
from tests.conftest import FakeExecutor, FakeLLM, FakeSchemaManager, fenced

SQL = "SELECT 1 AS ok"


@pytest.fixture
def client(settings):
    def fake_agent() -> LuminaSQLAgent:
        return LuminaSQLAgent(
            settings=settings,
            schema_manager=FakeSchemaManager(),
            db_executor=FakeExecutor({SQL: [{"ok": 1}]}),
            llm_client=FakeLLM([fenced(SQL)]),
        )

    app.dependency_overrides[get_agent] = fake_agent
    with TestClient(app) as test_client:
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


def test_stream_query_emits_sse_events(client):
    with client.stream("POST", "/api/v1/query/stream", json={"query": "is it ok?"}) as response:
        assert response.status_code == 200
        events = [json.loads(line[len("data: ") :]) for line in response.iter_lines() if line.startswith("data: ")]
    assert events[-1]["type"] == "result"
    assert events[-1]["content"]["success"] is True


def test_query_validation(client):
    assert client.post("/api/v1/query", json={"query": ""}).status_code == 422


def test_missing_llm_key_returns_503(monkeypatch):
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        runtime = test_client.app.state.runtime
        monkeypatch.setattr(runtime.settings, "openai_api_key", None)
        response = test_client.post("/api/v1/query", json={"query": "q"})
    assert response.status_code == 503


def test_metrics_exposes_request_counters(client):
    client.get("/livez")
    client.post("/api/v1/query", json={"query": "is it ok?"})
    body = client.get("/metrics").text
    assert 'lumina_http_requests_total{method="GET",route="/livez",status="200"}' in body
    assert "lumina_agent_runs_total" in body
    assert "lumina_llm_calls_total" in body
