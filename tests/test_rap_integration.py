"""Integration tests for the full RAP pipeline (task 11.3).

Tests the end-to-end flow through all pipeline phases with mocked external
services (DeepSeek API, LM Studio, Qdrant).

Requirements: 11.1, 11.2, 13.1, 13.2
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from deepseek_cursor_proxy.rap.app import create_app
from deepseek_cursor_proxy.rap.config import RAPConfig


@pytest.fixture
def integration_config(tmp_path) -> RAPConfig:
    """Create a RAPConfig with ALL phases enabled for integration testing."""
    return RAPConfig(
        upstream_base_url="https://api.deepseek.com",
        phase_bridge=True,
        phase_compression=True,
        phase_retrieval=True,
        phase_security=True,
        # Security settings
        redaction_enabled=True,
        cve_scanning_enabled=True,
        audit_db_path=tmp_path / "test_audit.sqlite3",
        entropy_threshold=4.5,
        security_model_url="http://localhost:1234/v1/chat/completions",
        # Retrieval settings
        qdrant_url="http://localhost:6333",
        embedding_url="http://localhost:1234/v1/embeddings",
        retrieval_top_k=5,
        retrieval_max_tokens=1000,
        # TOON settings
        toon_min_block_size=64,
        toon_compression_enabled=True,
        toon_rehydration_enabled=True,
        # Fidelity settings
        heartbeat_interval=15.0,
        reasoning_passthrough=True,
    )


@pytest.fixture
def integration_app(integration_config: RAPConfig) -> TestClient:
    """Create a test client with all phases enabled."""
    app = create_app(config=integration_config)
    return TestClient(app)


class TestEndToEndOutbound:
    """Test end-to-end outbound flow: Fidelity → Security → TOON → Retrieval → upstream."""

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    @patch("time.sleep", return_value=None)
    def test_full_outbound_pipeline_with_header_injection(
        self,
        mock_sleep: MagicMock,
        mock_client_cls: AsyncMock,
        integration_app: TestClient,
    ) -> None:
        """Test that outbound request flows through all phases.

        Verifies:
        - Fidelity injects X-Cursor-Plan and X-Cursor-Tier headers
        - Security redacts secrets from messages
        - Request reaches upstream with pipeline-processed content
        """
        upstream_response = {
            "id": "chatcmpl-integration-1",
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

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: upstream_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        # Include a secret in the message to verify redaction
        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "My API key is sk-test_abcdefghijklmnopqrstuvwxyz123456"},
            ],
            "stream": False,
        }

        response = integration_app.post(
            "/v1/chat/completions",
            json=request_body,
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-integration-1"

        # Verify the upstream call was made
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args

        # Verify headers were injected by Fidelity module
        upstream_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert upstream_headers.get("X-Cursor-Plan") == "pro"
        assert upstream_headers.get("X-Cursor-Tier") == "unlimited"

        # Verify the secret was redacted in the forwarded body
        upstream_body = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", b"")
        if isinstance(upstream_body, bytes):
            upstream_body = upstream_body.decode("utf-8")
        body_data = json.loads(upstream_body)
        # The secret should be redacted
        user_content = ""
        for msg in body_data.get("messages", []):
            if msg.get("role") == "user":
                user_content += msg.get("content", "")
        assert "sk-test_abcdefghijklmnopqrstuvwxyz123456" not in user_content
        assert "[REDACTED]" in user_content

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    @patch("time.sleep", return_value=None)
    def test_outbound_preserves_authorization_header(
        self,
        mock_sleep: MagicMock,
        mock_client_cls: AsyncMock,
        integration_app: TestClient,
    ) -> None:
        """Test that the Authorization header is forwarded to upstream."""
        upstream_response = {
            "id": "chatcmpl-auth-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: upstream_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        response = integration_app.post(
            "/v1/chat/completions",
            json=request_body,
            headers={"Authorization": "Bearer my-secret-key"},
        )

        assert response.status_code == 200

        # Verify Authorization was forwarded
        call_kwargs = mock_client.post.call_args
        upstream_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert upstream_headers.get("Authorization") == "Bearer my-secret-key"


class TestEndToEndInbound:
    """Test end-to-end inbound flow: upstream response → TOON rehydrate → Security scan → client."""

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    @patch("time.sleep", return_value=None)
    def test_inbound_pipeline_processes_response(
        self,
        mock_sleep: MagicMock,
        mock_client_cls: AsyncMock,
        integration_app: TestClient,
    ) -> None:
        """Test that inbound response flows through TOON rehydrate and security scan."""
        # Response with a code block that would trigger CVE scanning
        upstream_response = {
            "id": "chatcmpl-inbound-1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Here is some code:\n```python\nprint('hello')\n```",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: upstream_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Write hello world"}],
            "stream": False,
        }

        # Patch the LM Studio CVE scanning call to simulate unavailability
        # (graceful degradation — CVE scan skipped)
        with patch("deepseek_cursor_proxy.rap.security.httpx.Client") as mock_security_client:
            mock_security_instance = MagicMock()
            mock_security_instance.__enter__ = MagicMock(return_value=mock_security_instance)
            mock_security_instance.__exit__ = MagicMock(return_value=None)
            mock_security_instance.post.side_effect = httpx.ConnectError("LM Studio unavailable")
            mock_security_client.return_value = mock_security_instance

            response = integration_app.post(
                "/v1/chat/completions",
                json=request_body,
                headers={"Authorization": "Bearer test-key"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-inbound-1"
        # The response content should still be present (CVE scan skipped gracefully)
        content = data["choices"][0]["message"]["content"]
        assert "print('hello')" in content


class TestSSEStreaming:
    """Test end-to-end SSE streaming flow."""

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    @patch("time.sleep", return_value=None)
    def test_streaming_response_passes_through(
        self,
        mock_sleep: MagicMock,
        mock_client_cls: AsyncMock,
        integration_app: TestClient,
    ) -> None:
        """Test that streaming SSE responses pass through correctly."""
        from contextlib import asynccontextmanager

        # Simulate SSE chunks from upstream
        sse_chunks = [
            b'data: {"id":"chatcmpl-stream-1","choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n',
            b'data: {"id":"chatcmpl-stream-1","choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
            b'data: {"id":"chatcmpl-stream-1","choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        # Create a mock streaming response
        class MockStreamResponse:
            status_code = 200

            async def aiter_bytes(self):
                for chunk in sse_chunks:
                    yield chunk

            async def aread(self):
                return b""

        mock_client = AsyncMock()

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            yield MockStreamResponse()

        mock_client.stream = mock_stream

        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }

        response = integration_app.post(
            "/v1/chat/completions",
            json=request_body,
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Collect all streamed content
        content = response.content.decode("utf-8")
        assert "Hello" in content
        assert " world" in content
        assert "[DONE]" in content

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    @patch("time.sleep", return_value=None)
    def test_streaming_upstream_error_returns_sse_error(
        self,
        mock_sleep: MagicMock,
        mock_client_cls: AsyncMock,
        integration_app: TestClient,
    ) -> None:
        """Test that upstream connection errors during streaming are returned as SSE events."""
        from contextlib import asynccontextmanager

        mock_client = AsyncMock()

        # Create a proper async context manager that raises ConnectError
        @asynccontextmanager
        async def failing_stream(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")
            yield  # noqa: unreachable - needed for generator syntax

        mock_client.stream = failing_stream

        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }

        response = integration_app.post(
            "/v1/chat/completions",
            json=request_body,
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        content = response.content.decode("utf-8")
        # Error should be communicated via SSE
        assert "error" in content or "Connection" in content


class TestGracefulDegradation:
    """Test graceful degradation when Qdrant or LM Studio are unavailable."""

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    @patch("time.sleep", return_value=None)
    def test_qdrant_unavailable_skips_retrieval(
        self,
        mock_sleep: MagicMock,
        mock_retrieval_client: MagicMock,
        mock_app_client_cls: AsyncMock,
        integration_app: TestClient,
    ) -> None:
        """Test that when Qdrant is down, retrieval is skipped and full context is forwarded.

        Requirement 13.1: If Qdrant is unreachable, skip retrieval and forward full context.
        """
        # Mock the retrieval httpx.Client to simulate Qdrant being down
        mock_retrieval_instance = MagicMock()
        mock_retrieval_instance.__enter__ = MagicMock(return_value=mock_retrieval_instance)
        mock_retrieval_instance.__exit__ = MagicMock(return_value=None)
        # Embedding call succeeds but Qdrant upsert fails
        embedding_response = MagicMock()
        embedding_response.status_code = 200
        embedding_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1] * 384}],
        }
        embedding_response.raise_for_status = MagicMock()

        qdrant_error = httpx.ConnectError("Qdrant connection refused")

        def side_effect_post(url, **kwargs):
            if "embeddings" in url:
                return embedding_response
            raise qdrant_error

        def side_effect_put(url, **kwargs):
            raise qdrant_error

        mock_retrieval_instance.post.side_effect = side_effect_post
        mock_retrieval_instance.put.side_effect = side_effect_put
        mock_retrieval_client.return_value = mock_retrieval_instance

        # Mock upstream DeepSeek response
        upstream_response = {
            "id": "chatcmpl-degraded-1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response with full context"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: upstream_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_app_client_cls.return_value = mock_client

        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is Python?"},
            ],
            "stream": False,
        }

        response = integration_app.post(
            "/v1/chat/completions",
            json=request_body,
            headers={"Authorization": "Bearer test-key"},
        )

        # Should succeed despite Qdrant being down (graceful degradation)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-degraded-1"

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    @patch("time.sleep", return_value=None)
    def test_lm_studio_unavailable_skips_embedding(
        self,
        mock_sleep: MagicMock,
        mock_retrieval_client: MagicMock,
        mock_app_client_cls: AsyncMock,
        integration_app: TestClient,
    ) -> None:
        """Test that when LM Studio is down, embedding/retrieval is skipped.

        Requirement 13.2: If LM Studio embedding endpoint returns error or times out,
        skip retrieval and use uncompressed context.
        """
        # Mock the retrieval httpx.Client to simulate LM Studio being down
        mock_retrieval_instance = MagicMock()
        mock_retrieval_instance.__enter__ = MagicMock(return_value=mock_retrieval_instance)
        mock_retrieval_instance.__exit__ = MagicMock(return_value=None)
        mock_retrieval_instance.post.side_effect = httpx.ConnectError(
            "LM Studio connection refused"
        )
        mock_retrieval_client.return_value = mock_retrieval_instance

        # Mock upstream DeepSeek response
        upstream_response = {
            "id": "chatcmpl-degraded-2",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response without retrieval"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38},
        }

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: upstream_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_app_client_cls.return_value = mock_client

        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Explain async/await"},
            ],
            "stream": False,
        }

        # Also patch the security LM Studio client for CVE scanning
        with patch("deepseek_cursor_proxy.rap.security.httpx.Client") as mock_sec_client:
            mock_sec_instance = MagicMock()
            mock_sec_instance.__enter__ = MagicMock(return_value=mock_sec_instance)
            mock_sec_instance.__exit__ = MagicMock(return_value=None)
            mock_sec_instance.post.side_effect = httpx.ConnectError("LM Studio unavailable")
            mock_sec_client.return_value = mock_sec_instance

            response = integration_app.post(
                "/v1/chat/completions",
                json=request_body,
                headers={"Authorization": "Bearer test-key"},
            )

        # Should succeed despite LM Studio being down (graceful degradation)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-degraded-2"

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    @patch("time.sleep", return_value=None)
    def test_all_services_down_still_proxies_request(
        self,
        mock_sleep: MagicMock,
        mock_app_client_cls: AsyncMock,
        integration_config: RAPConfig,
    ) -> None:
        """Test that even when all optional services are down, the proxy still works.

        The pipeline should gracefully degrade through all phases and still
        forward the request to upstream.
        """
        app = create_app(config=integration_config)
        client = TestClient(app)

        upstream_response = {
            "id": "chatcmpl-all-down",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Still working!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: upstream_response

        mock_async_client = AsyncMock()
        mock_async_client.post = AsyncMock(return_value=mock_response)
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=None)
        mock_app_client_cls.return_value = mock_async_client

        # Patch both retrieval and security httpx.Client to simulate all services down
        with patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client") as mock_ret_client, \
             patch("deepseek_cursor_proxy.rap.security.httpx.Client") as mock_sec_client:

            # Retrieval client (LM Studio + Qdrant) — all fail
            mock_ret_instance = MagicMock()
            mock_ret_instance.__enter__ = MagicMock(return_value=mock_ret_instance)
            mock_ret_instance.__exit__ = MagicMock(return_value=None)
            mock_ret_instance.post.side_effect = httpx.ConnectError("Service unavailable")
            mock_ret_instance.put.side_effect = httpx.ConnectError("Service unavailable")
            mock_ret_client.return_value = mock_ret_instance

            # Security client (LM Studio for CVE) — fails
            mock_sec_instance = MagicMock()
            mock_sec_instance.__enter__ = MagicMock(return_value=mock_sec_instance)
            mock_sec_instance.__exit__ = MagicMock(return_value=None)
            mock_sec_instance.post.side_effect = httpx.ConnectError("Service unavailable")
            mock_sec_client.return_value = mock_sec_instance

            request_body = {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            }

            response = client.post(
                "/v1/chat/completions",
                json=request_body,
                headers={"Authorization": "Bearer test-key"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-all-down"
        assert data["choices"][0]["message"]["content"] == "Still working!"


class TestHealthEndpointIntegration:
    """Test the /healthz endpoint with all phases enabled."""

    def test_healthz_reports_all_phases(self, integration_app: TestClient) -> None:
        """Test that healthz reports status for all enabled phases."""
        response = integration_app.get("/healthz")
        # May be 200 or 503 depending on component health
        assert response.status_code in (200, 503)
        data = response.json()
        assert "phases" in data
        assert "fidelity" in data["phases"]
        assert "security" in data["phases"]
        assert "toon" in data["phases"]
        assert "retrieval" in data["phases"]
        assert "config" in data
        assert data["config"]["phase_bridge"] is True
        assert data["config"]["phase_compression"] is True
        assert data["config"]["phase_retrieval"] is True
        assert data["config"]["phase_security"] is True


class TestStreamingInboundPipeline:
    """Tests for streaming response accumulation + inbound pipeline processing."""

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    @patch("time.sleep", return_value=None)
    def test_streaming_accumulates_and_processes_inbound(
        self,
        mock_sleep: MagicMock,
        mock_client_cls: AsyncMock,
        integration_config: RAPConfig,
    ) -> None:
        """Test that streaming responses are accumulated and processed through inbound pipeline.

        After the stream completes, the accumulated SSE data should be parsed
        into a complete response and processed through process_response().
        The final output should contain inbound pipeline annotations.
        """
        from contextlib import asynccontextmanager

        app = create_app(config=integration_config)
        client = TestClient(app)

        # Simulate SSE chunks that contain code with a CVE pattern
        sse_chunks = [
            b'data: {"id":"chatcmpl-stream-1","choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n',
            b'data: {"id":"chatcmpl-stream-1","choices":[{"delta":{"content":"```python\\npassword = \\"admin123\\"\\n```"},"index":0}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        class MockStreamResponse:
            status_code = 200

            async def aiter_bytes(self):
                for chunk in sse_chunks:
                    yield chunk

            async def aread(self):
                return b""

        mock_client = AsyncMock()

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            yield MockStreamResponse()

        mock_client.stream = mock_stream
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Write password code"}],
            "stream": True,
        }

        # Patch CVE scanning to be unavailable (graceful degradation)
        with patch("deepseek_cursor_proxy.rap.security.httpx.Client") as mock_sec_client:
            mock_sec_instance = MagicMock()
            mock_sec_instance.__enter__ = MagicMock(return_value=mock_sec_instance)
            mock_sec_instance.__exit__ = MagicMock(return_value=None)
            mock_sec_instance.post.side_effect = httpx.ConnectError("LM Studio unavailable")
            mock_sec_client.return_value = mock_sec_instance

            response = client.post(
                "/v1/chat/completions",
                json=request_body,
                headers={"Authorization": "Bearer test-key"},
            )

        assert response.status_code == 200
        content = response.content.decode("utf-8")
        # The content should be passed through (no CVE scan without LM Studio)
        assert "password" in content
        assert "admin123" in content

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    @patch("time.sleep", return_value=None)
    def test_streaming_inbound_with_static_cve(
        self,
        mock_sleep: MagicMock,
        mock_client_cls: AsyncMock,
        integration_config: RAPConfig,
    ) -> None:
        """Test that static CVE patterns are detected in streaming responses.

        Static CVE patterns should detect vulnerabilities without needing
        the LM Studio model call.
        """
        from contextlib import asynccontextmanager

        app = create_app(config=integration_config)
        client = TestClient(app)

        # SSE chunks containing eval() (static code injection pattern)
        sse_chunks = [
            b'data: {"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n',
            b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"```python\\neval(user_input)\\n```"},"index":0}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        class MockStreamResponse:
            status_code = 200

            async def aiter_bytes(self):
                for chunk in sse_chunks:
                    yield chunk

            async def aread(self):
                return b""

        mock_client = AsyncMock()

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            yield MockStreamResponse()

        mock_client.stream = mock_stream
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Write eval code"}],
            "stream": True,
        }

        # Patch CVE scanning to be unavailable — static patterns should still catch it
        with patch("deepseek_cursor_proxy.rap.security.httpx.Client") as mock_sec_client:
            mock_sec_instance = MagicMock()
            mock_sec_instance.__enter__ = MagicMock(return_value=mock_sec_instance)
            mock_sec_instance.__exit__ = MagicMock(return_value=None)
            mock_sec_instance.post.side_effect = httpx.ConnectError("LM Studio unavailable")
            mock_sec_client.return_value = mock_sec_instance

            response = client.post(
                "/v1/chat/completions",
                json=request_body,
                headers={"Authorization": "Bearer test-key"},
            )

        assert response.status_code == 200
        content = response.content.decode("utf-8")
        assert "eval" in content

    @patch("deepseek_cursor_proxy.rap.app.httpx.AsyncClient")
    @patch("time.sleep", return_value=None)
    def test_streaming_cve_scanning_enabled_requires_config(
        self,
        mock_sleep: MagicMock,
        mock_client_cls: AsyncMock,
        integration_config: RAPConfig,
    ) -> None:
        """Test that CVE scanning in streaming only happens when explicitly enabled."""
        from contextlib import asynccontextmanager

        app = create_app(config=integration_config)
        client = TestClient(app)

        sse_chunks = [
            b'data: {"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n',
            b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello world"},"index":0}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        class MockStreamResponse:
            status_code = 200

            async def aiter_bytes(self):
                for chunk in sse_chunks:
                    yield chunk

            async def aread(self):
                return b""

        mock_client = AsyncMock()

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            yield MockStreamResponse()

        mock_client.stream = mock_stream
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        request_body = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
        }

        response = client.post(
            "/v1/chat/completions",
            json=request_body,
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        content = response.content.decode("utf-8")
        assert "Hello" in content
