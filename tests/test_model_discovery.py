"""Unit tests for LM Studio model discovery module.

Tests model discovery API calls, model classification (embedding vs chat),
and model selection helper functions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import pytest

from deepseek_cursor_proxy.rap.model_discovery import (
    LMStudioModels,
    _classify_model,
    discover_lmstudio_models,
    get_chat_model,
    get_embedding_model,
)


class TestModelClassification:
    """Tests for _classify_model() heuristic."""

    def test_classifies_embedding_model_by_keyword(self) -> None:
        """Models with 'embedding' in the key are classified as embedding."""
        assert _classify_model("text-embedding-nomic-embed-text-v1.5") == "embedding"

    def test_classifies_nomic_embed(self) -> None:
        """Nomic embed models are classified as embedding."""
        assert _classify_model("nomic-embed-text-v1") == "embedding"

    def test_classifies_bge_model(self) -> None:
        """BGE models are classified as embedding."""
        assert _classify_model("BAAI/bge-small-en-v1.5") == "embedding"

    def test_classifies_e5_model(self) -> None:
        """E5 models are classified as embedding."""
        assert _classify_model("intfloat/e5-large-v2") == "embedding"

    def test_classifies_gte_model(self) -> None:
        """GTE models are classified as embedding."""
        assert _classify_model("thenlper/gte-small") == "embedding"

    def test_classifies_instructor_model(self) -> None:
        """Instructor models are classified as embedding."""
        assert _classify_model("hkunlp/instructor-xl") == "embedding"

    def test_classifies_chat_model_by_keyword(self) -> None:
        """Models with chat keywords are classified as chat."""
        assert _classify_model("gpt-4") == "chat"
        assert _classify_model("llama-3-8b") == "chat"
        assert _classify_model("qwen2.5-coder-7b") == "chat"
        assert _classify_model("deepseek-v4-flash") == "chat"

    def test_defaults_to_chat_for_unknown(self) -> None:
        """Unknown model keys default to chat classification."""
        assert _classify_model("some-random-model") == "chat"


class TestDiscoverModels:
    """Tests for discover_lmstudio_models()."""

    @pytest.mark.asyncio
    @patch("deepseek_cursor_proxy.rap.model_discovery.httpx.AsyncClient")
    async def test_discover_success_with_both_types(
        self, mock_client_cls: AsyncMock
    ) -> None:
        """Successfully discovers both embedding and chat models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "key": "text-embedding-nomic-embed-text-v1.5",
                    "name": "Nomic Embed Text v1.5",
                    "loaded_instances": [{"id": "instance-1"}],
                },
                {
                    "key": "qwen2.5-coder-7b",
                    "name": "Qwen 2.5 Coder 7B",
                    "loaded_instances": [{"id": "instance-1"}, {"id": "instance-2"}],
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await discover_lmstudio_models("http://localhost:1234")

        assert result.lm_studio_available is True
        assert len(result.all_models) == 2
        assert len(result.embedding_models) == 1
        assert len(result.chat_models) == 1
        assert result.embedding_models[0].key == "text-embedding-nomic-embed-text-v1.5"
        assert result.chat_models[0].key == "qwen2.5-coder-7b"
        assert result.embedding_models[0].loaded_instances == 1
        assert result.chat_models[0].loaded_instances == 2

    @pytest.mark.asyncio
    @patch("deepseek_cursor_proxy.rap.model_discovery.httpx.AsyncClient")
    async def test_discover_skips_unloaded_models(
        self, mock_client_cls: AsyncMock
    ) -> None:
        """Models with no loaded instances are skipped."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "key": "model-with-loaded-instances",
                    "name": "Loaded Model",
                    "loaded_instances": [{"id": "instance-1"}],
                },
                {
                    "key": "model-without-loaded-instances",
                    "name": "Empty Model",
                    "loaded_instances": [],
                },
                {
                    "key": "model-with-non-list-instances",
                    "name": "Bad Model",
                    "loaded_instances": None,
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await discover_lmstudio_models("http://localhost:1234")

        assert result.lm_studio_available is True
        assert len(result.all_models) == 1
        assert result.all_models[0].key == "model-with-loaded-instances"

    @pytest.mark.asyncio
    @patch("deepseek_cursor_proxy.rap.model_discovery.httpx.AsyncClient")
    async def test_discover_connection_error(self, mock_client_cls: AsyncMock) -> None:
        """Connection error returns available=False."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await discover_lmstudio_models("http://localhost:1234")

        assert result.lm_studio_available is False
        assert len(result.all_models) == 0
        assert len(result.embedding_models) == 0
        assert len(result.chat_models) == 0

    @pytest.mark.asyncio
    @patch("deepseek_cursor_proxy.rap.model_discovery.httpx.AsyncClient")
    async def test_discover_timeout(self, mock_client_cls: AsyncMock) -> None:
        """Timeout returns available=False."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await discover_lmstudio_models("http://localhost:1234")

        assert result.lm_studio_available is False

    @pytest.mark.asyncio
    @patch("deepseek_cursor_proxy.rap.model_discovery.httpx.AsyncClient")
    async def test_discover_invalid_json(self, mock_client_cls: AsyncMock) -> None:
        """Invalid JSON response returns available=True but no models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await discover_lmstudio_models("http://localhost:1234")

        assert result.lm_studio_available is True
        assert len(result.all_models) == 0

    @pytest.mark.asyncio
    @patch("deepseek_cursor_proxy.rap.model_discovery.httpx.AsyncClient")
    async def test_discover_empty_models_list(self, mock_client_cls: AsyncMock) -> None:
        """Empty models list returns no models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await discover_lmstudio_models("http://localhost:1234")

        assert result.lm_studio_available is True
        assert len(result.all_models) == 0


class TestModelSelection:
    """Tests for get_embedding_model() and get_chat_model()."""

    def test_get_embedding_model_returns_first_embedding(self) -> None:
        """Returns the first embedding model."""
        models = LMStudioModels(
            lm_studio_available=True,
            embedding_models=[
                MockDiscoveredModel("emb-model-1", "embedding"),
                MockDiscoveredModel("emb-model-2", "embedding"),
            ],
            chat_models=[
                MockDiscoveredModel("chat-model", "chat"),
            ],
        )
        assert get_embedding_model(models) == "emb-model-1"

    def test_get_embedding_model_falls_back_to_any_model(self) -> None:
        """Falls back to first loaded model if no embedding models found."""
        models = LMStudioModels(
            lm_studio_available=True,
            all_models=[MockDiscoveredModel("some-model", "chat")],
            chat_models=[MockDiscoveredModel("some-model", "chat")],
        )
        assert get_embedding_model(models) == "some-model"

    def test_get_embedding_model_returns_none_if_empty(self) -> None:
        """Returns None if no models available."""
        models = LMStudioModels(lm_studio_available=True)
        assert get_embedding_model(models) is None

    def test_get_chat_model_returns_first_chat(self) -> None:
        """Returns the first chat model."""
        models = LMStudioModels(
            lm_studio_available=True,
            chat_models=[
                MockDiscoveredModel("chat-1", "chat"),
                MockDiscoveredModel("chat-2", "chat"),
            ],
        )
        assert get_chat_model(models) == "chat-1"

    def test_get_chat_model_returns_none_if_empty(self) -> None:
        """Returns None if no chat models available."""
        models = LMStudioModels(lm_studio_available=True)
        assert get_chat_model(models) is None


class MockDiscoveredModel:
    """Minimal mock for DiscoveredModel used in selection tests."""

    def __init__(self, key: str, model_type: str) -> None:
        self.key = key
        self.name = key
        self.loaded_instances = 1
        self.model_type = model_type
