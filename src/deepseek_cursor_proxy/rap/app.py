"""RAP FastAPI application integrating the pipeline orchestrator.

Provides:
- POST /v1/chat/completions — routes requests through the RAP pipeline
- GET /healthz — exposes pipeline component health status

Requirements: 11.1, 11.2, 13.5
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from deepseek_cursor_proxy.config import load_config_file, resolve_config_path
from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.pipeline import PipelineOrchestrator

logger = logging.getLogger(__name__)


def _load_rap_config(config_path: str | Path | None = None) -> RAPConfig:
    """Load RAPConfig from config.yaml, falling back to defaults.

    Reads the existing config.yaml (if present) and extracts RAP-relevant
    fields. Unknown fields are ignored; missing fields use RAPConfig defaults.
    """
    resolved = resolve_config_path(config_path)
    settings = load_config_file(resolved)

    # Map config.yaml keys to RAPConfig field names
    field_mapping: dict[str, str] = {
        "host": "host",
        "port": "port",
        "base_url": "upstream_base_url",
        "upstream_base_url": "upstream_base_url",
        "model": "upstream_model",
        "upstream_model": "upstream_model",
        "heartbeat_interval": "heartbeat_interval",
        "toon_compression_enabled": "toon_compression_enabled",
        "toon_rehydration_enabled": "toon_rehydration_enabled",
        "toon_min_block_size": "toon_min_block_size",
        "qdrant_url": "qdrant_url",
        "qdrant_collection": "qdrant_collection",
        "embedding_url": "embedding_url",
        "embedding_model": "embedding_model",
        "retrieval_top_k": "retrieval_top_k",
        "retrieval_max_tokens": "retrieval_max_tokens",
        "use_msgpack": "use_msgpack",
        "redaction_enabled": "redaction_enabled",
        "cve_scanning_enabled": "cve_scanning_enabled",
        "audit_db_path": "audit_db_path",
        "entropy_threshold": "entropy_threshold",
        "security_model_url": "security_model_url",
        "phase_bridge": "phase_bridge",
        "phase_compression": "phase_compression",
        "phase_retrieval": "phase_retrieval",
        "phase_security": "phase_security",
        "spoof_pro_headers": "spoof_pro_headers",
        "reasoning_passthrough": "reasoning_passthrough",
    }

    kwargs: dict[str, Any] = {}
    for yaml_key, config_field in field_mapping.items():
        if yaml_key in settings:
            value = settings[yaml_key]
            # Handle Path conversion for audit_db_path
            if config_field == "audit_db_path" and not isinstance(value, Path):
                value = Path(str(value))
            kwargs[config_field] = value

    # Strip trailing slash from upstream_base_url if present
    if "upstream_base_url" in kwargs and isinstance(kwargs["upstream_base_url"], str):
        kwargs["upstream_base_url"] = kwargs["upstream_base_url"].rstrip("/")

    return RAPConfig(**kwargs)


def create_app(config: RAPConfig | None = None) -> FastAPI:
    """Create and configure the RAP FastAPI application.

    Args:
        config: Optional pre-built RAPConfig. If None, loads from config.yaml.

    Returns:
        Configured FastAPI application with pipeline endpoints.
    """
    if config is None:
        config = _load_rap_config()

    pipeline = PipelineOrchestrator(config)

    app = FastAPI(
        title="DeepSeek RAP Proxy",
        description="Smart Retrieval-Augmented Proxy for DeepSeek",
        version="0.1.0",
    )

    # Store references on app state for testability
    app.state.config = config
    app.state.pipeline = pipeline

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        """Return pipeline component health status (Requirement 13.5)."""
        health = pipeline.health_check()
        status_code = 200 if health.get("pipeline") == "healthy" else 503
        return JSONResponse(content=health, status_code=status_code)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
        """Route chat completion requests through the RAP pipeline.

        1. Parse the incoming OpenAI-format request
        2. Run process_request() through the pipeline (outbound)
        3. Forward to upstream DeepSeek API
        4. Run process_response() on the result (inbound) — non-streaming only
        5. Return the response to the client

        Supports both streaming (SSE) and non-streaming responses.
        """
        # Extract authorization from incoming request
        auth_header = request.headers.get("authorization", "")

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Invalid JSON request body"}},
            )

        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Request body must be a JSON object"}},
            )

        # Determine if streaming is requested
        is_streaming = body.get("stream", False)

        # Run outbound pipeline
        processed_request = pipeline.process_request(body)

        # Extract pipeline-injected headers (from fidelity module)
        pipeline_headers = processed_request.pop("_headers", {})

        # Build upstream headers
        upstream_headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if is_streaming else "application/json",
            "User-Agent": "DeepSeekRAPProxy/0.1",
        }
        if auth_header:
            upstream_headers["Authorization"] = auth_header

        # Merge pipeline-injected headers (e.g., spoofed Pro headers)
        if isinstance(pipeline_headers, dict):
            upstream_headers.update(pipeline_headers)

        # Build upstream URL
        upstream_url = f"{config.upstream_base_url}/chat/completions"

        # Serialize request body
        upstream_body = json.dumps(
            processed_request, ensure_ascii=False, separators=(",", ":")
        )

        if is_streaming:
            return await _handle_streaming(
                upstream_url, upstream_headers, upstream_body, config
            )
        else:
            return await _handle_non_streaming(
                upstream_url, upstream_headers, upstream_body, pipeline, config
            )

    return app


async def _handle_streaming(
    upstream_url: str,
    headers: dict[str, str],
    body: str,
    config: RAPConfig,
) -> StreamingResponse | JSONResponse:
    """Handle streaming (SSE) responses by passing through from upstream."""

    async def stream_generator() -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            try:
                async with client.stream(
                    "POST",
                    upstream_url,
                    headers=headers,
                    content=body.encode("utf-8"),
                ) as response:
                    if response.status_code != 200:
                        # Yield error as SSE event
                        error_body = await response.aread()
                        yield b"data: " + error_body + b"\n\n"
                        yield b"data: [DONE]\n\n"
                        return

                    async for chunk in response.aiter_bytes():
                        yield chunk
            except httpx.ConnectError as exc:
                error_msg = json.dumps(
                    {"error": {"message": f"Upstream connection failed: {exc}"}}
                )
                yield b"data: " + error_msg.encode("utf-8") + b"\n\n"
                yield b"data: [DONE]\n\n"
            except httpx.ReadTimeout as exc:
                error_msg = json.dumps(
                    {"error": {"message": f"Upstream read timeout: {exc}"}}
                )
                yield b"data: " + error_msg.encode("utf-8") + b"\n\n"
                yield b"data: [DONE]\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _handle_non_streaming(
    upstream_url: str,
    headers: dict[str, str],
    body: str,
    pipeline: PipelineOrchestrator,
    config: RAPConfig,
) -> JSONResponse:
    """Handle non-streaming responses with full pipeline processing."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        try:
            response = await client.post(
                upstream_url,
                headers=headers,
                content=body.encode("utf-8"),
            )
        except httpx.ConnectError as exc:
            logger.warning("Upstream connection failed: %s", exc)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {"message": f"Upstream connection failed: {exc}"}
                },
            )
        except httpx.ReadTimeout as exc:
            logger.warning("Upstream read timeout: %s", exc)
            return JSONResponse(
                status_code=504,
                content={"error": {"message": f"Upstream read timeout: {exc}"}},
            )

    if response.status_code != 200:
        # Pass through upstream error
        try:
            error_content = response.json()
        except Exception:
            error_content = {"error": {"message": response.text}}
        return JSONResponse(
            status_code=response.status_code,
            content=error_content,
        )

    # Parse upstream response
    try:
        upstream_response = response.json()
    except Exception:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "Invalid JSON from upstream"}},
        )

    # Run inbound pipeline (response processing)
    processed_response = pipeline.process_response(upstream_response)

    return JSONResponse(content=processed_response)
