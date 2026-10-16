"""Tests for `app/api/auth.py`'s `require_api_token` dependency, applied
to `/api/projects/*` and `/api/agents/*` in `app/main.py`. Every other
test in `tests/api/`/`tests/integration/` exercises the normal,
authenticated path via the `client` fixture (which always sends the right
token); this file is the one place that fixture is deliberately not used,
so it can test what happens without it.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

TOKEN = "test-api-token-xyz789"  # matches conftest.py's _api_token fixture

anonymous_client = TestClient(app)
authenticated_client = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})

PROTECTED_ROUTES = [
    ("GET", "/api/projects"),
    ("POST", "/api/projects/register"),
    ("GET", "/api/agents"),
    ("POST", "/api/agents/register"),
]

UNPROTECTED_ROUTES = [
    ("GET", "/health"),
    ("GET", "/api/experiments"),
]


@pytest.fixture(autouse=True)
def _api_token(monkeypatch):
    monkeypatch.setenv("AI_ENV_API_TOKEN", TOKEN)
    monkeypatch.delenv("AI_ENV_API_ALLOW_NO_AUTH", raising=False)


class TestProtectedRoutes:
    @pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
    def test_missing_token_is_rejected(self, method, path):
        resp = anonymous_client.request(method, path, json={} if method == "POST" else None)
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"

    @pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
    def test_wrong_token_is_rejected(self, method, path):
        client = TestClient(app, headers={"Authorization": "Bearer not-the-token"})
        resp = client.request(method, path, json={} if method == "POST" else None)
        assert resp.status_code == 401

    def test_a_token_that_is_only_a_prefix_of_the_real_one_is_rejected(self):
        client = TestClient(app, headers={"Authorization": f"Bearer {TOKEN[:-1]}"})
        assert client.get("/api/projects").status_code == 401

    def test_non_bearer_scheme_is_rejected(self):
        client = TestClient(app, headers={"Authorization": f"Basic {TOKEN}"})
        assert client.get("/api/projects").status_code == 401

    def test_the_right_token_is_accepted(self):
        assert authenticated_client.get("/api/projects").status_code == 200
        assert authenticated_client.get("/api/agents").status_code == 200


class TestUnprotectedRoutes:
    @pytest.mark.parametrize("method,path", UNPROTECTED_ROUTES)
    def test_mode_a_and_health_routes_need_no_token(self, method, path):
        resp = anonymous_client.request(method, path)
        assert resp.status_code != 401


class TestFailClosed:
    def test_no_token_configured_returns_503_not_200(self, monkeypatch):
        monkeypatch.delenv("AI_ENV_API_TOKEN")
        resp = authenticated_client.get("/api/projects")
        assert resp.status_code == 503
        assert "AI_ENV_API_TOKEN" in resp.json()["detail"]

    def test_empty_token_counts_as_not_configured(self, monkeypatch):
        monkeypatch.setenv("AI_ENV_API_TOKEN", "")
        assert authenticated_client.get("/api/projects").status_code == 503

    def test_explicit_opt_out_allows_unauthenticated_access(self, monkeypatch):
        monkeypatch.delenv("AI_ENV_API_TOKEN")
        monkeypatch.setenv("AI_ENV_API_ALLOW_NO_AUTH", "1")
        assert anonymous_client.get("/api/projects").status_code == 200

    def test_opt_out_requires_exactly_1(self, monkeypatch):
        monkeypatch.delenv("AI_ENV_API_TOKEN")
        monkeypatch.setenv("AI_ENV_API_ALLOW_NO_AUTH", "true")
        assert anonymous_client.get("/api/projects").status_code == 503


class TestRejectionNeverReachesTheDatabase:
    def test_an_unauthenticated_register_creates_no_project(self):
        before = authenticated_client.get("/api/projects").json()
        resp = anonymous_client.post("/api/projects/register", json={"root_path": "/tmp/should-not-be-created"})
        assert resp.status_code == 401
        after = authenticated_client.get("/api/projects").json()
        assert after == before
