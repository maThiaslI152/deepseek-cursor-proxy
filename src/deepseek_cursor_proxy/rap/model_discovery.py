"""LM Studio model discovery — query /api/v1/models and classify models.

This module provides the ability to query LM Studio's model management
endpoint to discover which models are loaded and available, and classify
them by type (embedding vs chat/completion).

Used by PipelineOrchestrator to auto-select models when configured
model names are empty or unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredModel:
    """A single model discovered from LM Studio.

    Attributes:
        key: The model key/identifier (e.g. 'text-embedding-nomic-embed-text-v1.5').
        name: Human-readable model name.
        loaded_instances: Number of active loaded instances.
        model_type: Classification — 'embedding' or 'chat'.
    """

    key: str
    name: str
    loaded_instances: int
    model_type: str  # "embedding" | "chat"


@dataclass
class LMStudioModels:
    """Result of a model discovery request.

    Attributes:
        all_models: All discovered models with loaded instances.
        embedding_models: Filtered to embedding-type models only.
        chat_models: Filtered to chat/completion models only.
        lm_studio_available: Whether LM Studio responded successfully.
    """

    all_models: list[DiscoveredModel] = field(default_factory=list)
    embedding_models: list[DiscoveredModel] = field(default_factory=list)
    chat_models: list[DiscoveredModel] = field(default_factory=list)
    lm_studio_available: bool = False


KNOWN_EMBEDDING_KEYWORDS = [
    "embedding",
    "nomic-embed",
    "text-embedding",
    "bge-",
    "e5-",
    "gte-",
    "instructor",
    "sentence-transformers",
]

KNOWN_CHAT_KEYWORDS = [
    "grok",
    "gpt",
    "llama",
    "qwen",
    "deepseek",
    "mistral",
    "gemma",
    "phi",
    "codestral",
    "starcoder",
    "codegemma",
    "codeqwen",
    "codellama",
    "dbrx",
    "command-r",
    "nemotron",
    "yi-",
    "minicpm",
    "bloom",
    "falcon",
    "olmo",
    "solar",
    "stablelm",
    "aya",
    "granite",
    "merlinite",
    "zephyr",
]


def _classify_model(key: str) -> str:
    """Classify a model as 'embedding' or 'chat' based on its key.

    Uses heuristic keyword matching. Prefers embedding detection since
    embedding keywords are more distinctive.
    """
    key_lower = key.lower()
    for kw in KNOWN_EMBEDDING_KEYWORDS:
        if kw in key_lower:
            return "embedding"
    for kw in KNOWN_CHAT_KEYWORDS:
        if kw in key_lower:
            return "chat"
    # Default to chat if unknown
    return "chat"


def _test_embedding_endpoint(base_url: str, timeout: float = 5.0) -> bool:
    """Test if the LM Studio embedding endpoint is functional.

    Sends a minimal embedding request to verify the endpoint works.
    """
    embedding_url = f"{base_url.rstrip('/')}/v1/embeddings"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                embedding_url,
                json={"model": "", "input": ["test"]},
            )
            # If the model is wrong we still get a valid response shape
            return resp.status_code in (200, 422)
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


async def discover_lmstudio_models(base_url: str) -> LMStudioModels:
    """Query LM Studio's /api/v1/models to discover available models.

    Args:
        base_url: The base URL of the LM Studio server (e.g. 'http://localhost:1234').

    Returns:
        An LMStudioModels dataclass with discovered models classified by type.
    """
    api_url = f"{base_url.rstrip('/')}/api/v1/models"
    result = LMStudioModels()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(api_url)
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        logger.warning("LM Studio model discovery failed: %s", exc)
        result.lm_studio_available = False
        return result

    result.lm_studio_available = True

    try:
        data = response.json()
        models_raw = data.get("models", [])
    except Exception as exc:
        logger.warning("Failed to parse LM Studio model response: %s", exc)
        return result

    if not models_raw:
        logger.info("LM Studio returned no models.")
        return result

    for model_raw in models_raw:
        key = model_raw.get("key", "")
        if not key:
            continue
        loaded_instances = model_raw.get("loaded_instances", [])
        if not isinstance(loaded_instances, list) or len(loaded_instances) == 0:
            continue

        name = model_raw.get("name", key)
        model_type = _classify_model(key)
        discovered = DiscoveredModel(
            key=key,
            name=name,
            loaded_instances=len(loaded_instances),
            model_type=model_type,
        )

        result.all_models.append(discovered)
        if model_type == "embedding":
            result.embedding_models.append(discovered)
        else:
            result.chat_models.append(discovered)

    # If no embedding models found by keyword, try a test call
    if not result.embedding_models:
        logger.info("No embedding models found by keyword; trying test embedding call...")
        if _test_embedding_endpoint(base_url):
            logger.info(
                "Embedding endpoint is functional but no model was classified; "
                "defaulting to first loaded model for embeddings."
            )

    logger.info(
        "Discovered %d LM Studio models (%d embedding, %d chat): %s",
        len(result.all_models),
        len(result.embedding_models),
        len(result.chat_models),
        [m.key for m in result.all_models],
    )

    return result


def get_embedding_model(models: LMStudioModels) -> str | None:
    """Select the best embedding model from discovered models.

    Returns:
        The first discovered embedding model's key, or None.
    """
    if models.embedding_models:
        return models.embedding_models[0].key
    if models.all_models:
        # Fall back to first loaded model
        return models.all_models[0].key
    return None


def get_chat_model(models: LMStudioModels) -> str | None:
    """Select the best chat model from discovered models.

    Returns:
        The first discovered chat model's key, or None.
    """
    if models.chat_models:
        return models.chat_models[0].key
    return None
