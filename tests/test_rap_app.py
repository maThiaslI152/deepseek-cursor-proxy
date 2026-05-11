"""Tests for the RAP FastAPI application (task 11.2)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from deepseek_cursor_proxy.rap.app import _load_rap_config, create_app
from deepseek_cursor_proxy.rap.config import RAPConfig


@pytest.fixture
def default_config() -> RAPConfig:
    """Create a default RAPConfig for testing."""
    return RAPConfig()


@pytest.fixture
def app(default_config: RAPConfig) -> TestClient:
    """Create a test client with default config."""
    fastapi_app = create_app(config=default_config)
    return TestClient(fastapi_app)


class TestHealthEndpoint:
    """Tests for GET /healthz endpoint."""

    def test_healthz_returns_200_when_healthy(self, app: TestClient) -> None:
        response = app.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline"] in ("healthy", "degraded")
        assert "phases" in data
        assert "config" in data

    def test_healthz_contains_phase_status(self, app: TestClient) -> None:
        response = app.get("/healthz")
        data = response.json()
        phases = data["phases"]
        assert "fidelity" in phases
        assert "security" in phases
        assert "toon" in phases
        assert "retrieval" in phases

    def test_healthz_contains_config_flags(self, app: TestClient) -> None:
        response = app.get("/healthz")
        data = response.json()
        config = data["config"]
        assert "phase_bridge" in config
        assert "phase_compression" in config
        assert "phase_retrieval" in config
        assert "phase_security" in config


class TestChatCompletionsEndpoint:
    """Tests for POST /v1/chat/completions endpoint."""

    def test_invalid_json_returns_400(self, app: TestClient) -> None:
        response = app.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "error" in response.json()

    def test_non_object_body_returns_400(self, app: TestClient) -> None:
        response = app.post(
            "/v1/chat/completions",
            content=json.dumps([1, 2, 3]).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "error" in response.json()

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    def test_non_streaming_success(
        self, mock_client_cls: AsyncMock, app: TestClient
    ) -> None:
        """Test non-streaming request flows through pipeline and returns response."""
        upstream_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Mock the httpx response (json() is synchronous in httpx)
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: upstream_response

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }

        response = app.post(
            "/v1/chat/completions",
            json=request_body,
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-123"
        assert data["choices"][0]["message"]["content"] == "Hello!"

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    def test_upstream_connection_error_returns_502(
        self, mock_client_cls: AsyncMock, app: TestClient
    ) -> None:
        """Test that upstream connection failures return 502."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }

        response = app.post(
            "/v1/chat/completions",
            json=request_body,
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 502
        assert "error" in response.json()

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    def test_upstream_timeout_returns_504(
        self, mock_client_cls: AsyncMock, app: TestClient
    ) -> None:
        """Test that upstream timeouts return 504."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ReadTimeout("Read timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }

        response = app.post(
            "/v1/chat/completions",
            json=request_body,
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 504
        assert "error" in response.json()

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    def test_upstream_error_passthrough(
        self, mock_client_cls: AsyncMock, app: TestClient
    ) -> None:
        """Test that upstream HTTP errors are passed through."""
        mock_response = AsyncMock()
        mock_response.status_code = 429
        mock_response.json = lambda: {"error": {"message": "Rate limit exceeded"}}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }

        response = app.post(
            "/v1/chat/completions",
            json=request_body,
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 429
        assert response.json()["error"]["message"] == "Rate limit exceeded"

    def test_authorization_header_forwarded(self, default_config: RAPConfig) -> None:
        """Test that the Authorization header is forwarded to upstream."""
        fastapi_app = create_app(config=default_config)

        # Verify app state is set correctly
        assert fastapi_app.state.config is default_config
        assert fastapi_app.state.pipeline is not None


class TestConfigLoading:
    """Tests for RAPConfig loading from config.yaml."""

    def test_load_rap_config_defaults(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test that loading from non-existent file returns defaults."""
        config = _load_rap_config(tmp_path / "nonexistent.yaml")  # type: ignore[arg-type]
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.upstream_base_url == "https://api.deepseek.com"

    def test_load_rap_config_from_yaml(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test that RAP fields are loaded from config.yaml."""
        config_file = tmp_path / "config.yaml"  # type: ignore[operator]
        config_file.write_text(
            "base_url: https://custom.api.com\n"
            "heartbeat_interval: 30\n"
            "phase_bridge: true\n"
            "phase_compression: true\n"
            "retrieval_top_k: 10\n"
        )
        config = _load_rap_config(config_file)
        assert config.upstream_base_url == "https://custom.api.com"
        assert config.heartbeat_interval == 30.0
        assert config.phase_bridge is True
        assert config.phase_compression is True
        assert config.retrieval_top_k == 10


class TestAppCreation:
    """Tests for app factory function."""

    def test_create_app_with_config(self, default_config: RAPConfig) -> None:
        """Test that create_app stores config and pipeline on app state."""
        app = create_app(config=default_config)
        assert app.state.config is default_config
        assert app.state.pipeline is not None

    def test_create_app_default_config(self) -> None:
        """Test that create_app works without explicit config."""
        app = create_app(config=RAPConfig())
        assert app.state.config is not None
        assert app.state.pipeline is not None
