# Smart Retrieval-Augmented Proxy (RAP) — Developer Guide

## Overview

The Smart RAP extends the DeepSeek Cursor Proxy with five middleware modules that optimize DeepSeek V4 integration within Cursor while maintaining local security and performance.

**Architecture:**
```
Cursor → [Fidelity] → [Security] → [TOON] → [Retrieval] → DeepSeek API
                                                              ↓
Cursor ← [Fidelity] ← [Security] ← [TOON] ← [Stream Health] ← DeepSeek API
```

**Modules:**
| Module | Purpose | Phase Config |
|--------|---------|--------------|
| Fidelity | Header spoofing, reasoning pass-through, heartbeat keep-alive | `phase_bridge` |
| Security Gateway | Secret redaction (outbound), CVE scanning (inbound), audit logging | `phase_security` |
| TOON Engine | Structured data compression/re-hydration | `phase_compression` |
| Retrieval Layer | Context chunking, embedding, vector search via Qdrant | `phase_retrieval` |
| Pipeline Orchestrator | Coordinates all modules, graceful degradation | Always active |

## Quick Start

### Prerequisites

- Python 3.11+
- Podman (for Qdrant)
- LM Studio (for embeddings and CVE scanning)
- The embedding model `text-embedding-nomic-embed-text-v1.5-embedding` downloaded in LM Studio

### Start Everything

```bash
./scripts/start-rap.sh
```

This starts Qdrant (Podman), LM Studio server + embedding model, and the RAP proxy.

### Stop Everything

```bash
./scripts/start-rap.sh --stop
```

### Manual Start

```bash
# 1. Start Qdrant
podman run -d --name rap-qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v rap_qdrant_storage:/qdrant/storage:z \
  docker.io/qdrant/qdrant:latest

# 2. Start LM Studio server and load embedding model
lms server start --port 1234
lms load text-embedding-nomic-embed-text-v1.5-embedding --gpu max

# 3. Start the RAP proxy
python -m uvicorn deepseek_cursor_proxy.rap.app:create_app \
  --factory --host 127.0.0.1 --port 9000
```

## API Endpoints

### POST /v1/chat/completions

Routes OpenAI-format chat completion requests through the RAP pipeline, then forwards to DeepSeek.

```bash
curl -X POST http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_DEEPSEEK_KEY" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

Supports both streaming (SSE) and non-streaming responses.

### GET /healthz

Returns pipeline component health status.

```bash
curl http://127.0.0.1:9000/healthz
```

Response:
```json
{
  "pipeline": "healthy",
  "phases": {
    "fidelity": "healthy",
    "security": "disabled",
    "toon": "disabled",
    "retrieval": "healthy"
  },
  "config": {
    "phase_bridge": true,
    "phase_compression": false,
    "phase_retrieval": true,
    "phase_security": false
  }
}
```

Retrieval health states:
- `healthy` — LM Studio reachable, embedding model loaded, Qdrant reachable
- `model_not_loaded` — LM Studio reachable but embedding model not loaded
- `degraded` — one service reachable, one not
- `unhealthy` — both services unreachable

## Configuration

Configuration is loaded from `config.yaml` (if present) or uses defaults. All RAP settings are in `RAPConfig`:

```yaml
# config.yaml
base_url: https://api.deepseek.com
heartbeat_interval: 15

# Phase toggles (enable/disable modules)
phase_bridge: true
phase_compression: false
phase_retrieval: false
phase_security: false

# Retrieval settings
qdrant_url: http://localhost:6333
qdrant_collection: rap_context
embedding_url: http://localhost:1234/v1/embeddings
embedding_model: text-embedding-nomic-embed-text-v1.5-embedding
retrieval_top_k: 5
retrieval_max_tokens: 1000

# Security settings
redaction_enabled: true
cve_scanning_enabled: true
entropy_threshold: 4.5
security_model_url: http://localhost:1234/v1/chat/completions

# TOON settings
toon_min_block_size: 64
toon_compression_enabled: true
toon_rehydration_enabled: true
```

### Validation Rules

| Field | Constraint |
|-------|-----------|
| `heartbeat_interval` | (0, 60] seconds |
| `retrieval_top_k` | [1, 50] |
| `retrieval_max_tokens` | [100, 10000] |
| `entropy_threshold` | [3.0, 8.0] |
| `toon_min_block_size` | >= 64 bytes |
| `qdrant_url` | Valid HTTP URL |
| `embedding_url` | Valid HTTP URL |

## Project Structure

```
src/deepseek_cursor_proxy/rap/
├── __init__.py
├── app.py              # FastAPI application (endpoints)
├── config.py           # RAPConfig dataclass with validation
├── fidelity.py         # Header spoofing, reasoning pass-through, heartbeat
├── pipeline.py         # Pipeline Orchestrator (wires all modules)
├── retrieval.py        # Context chunking, embedding, Qdrant vector search
├── security.py         # Secret redaction, CVE scanning, audit logging
└── toon.py             # TOON compression/re-hydration engine

scripts/
└── start-rap.sh        # Start/stop all infrastructure

tests/
├── test_rap_app.py                              # FastAPI endpoint tests
├── test_rap_config.py                           # Config validation tests
├── test_rap_config_properties.py                # Config property tests
├── test_rap_integration.py                      # End-to-end integration tests
├── test_rap_pipeline.py                         # Pipeline orchestration tests
├── test_rap_pipeline_properties.py              # Pipeline property tests
├── test_fidelity.py                             # Fidelity module tests
├── test_fidelity_properties.py                  # Fidelity property tests
├── test_fidelity_stream.py                      # Stream health tests
├── test_fidelity_stream_properties.py           # Stream property tests
├── test_retrieval.py                            # Retrieval chunking tests
├── test_retrieval_properties.py                 # Chunking property tests
├── test_retrieval_embed.py                      # Embedding tests
├── test_retrieval_embed_properties.py           # Embedding property tests
├── test_retrieval_qdrant.py                     # Qdrant storage tests
├── test_retrieval_qdrant_properties.py          # Qdrant property tests
├── test_retrieval_build_context.py              # build_reduced_context tests
├── test_retrieval_message_preservation_properties.py  # Message preservation PBT
├── test_security.py                             # Redaction tests
├── test_security_properties.py                  # Redaction property tests
├── test_security_cve.py                         # CVE scanning tests
├── test_security_cve_properties.py              # CVE property tests
├── test_security_audit.py                       # Audit logging tests
├── test_security_audit_properties.py            # Audit property tests
├── test_toon.py                                 # TOON compression tests
└── test_toon_properties.py                      # TOON property tests
```

## Module Details

### Fidelity Module (`fidelity.py`)

**Outbound:**
- Injects `X-Cursor-Plan: pro` and `X-Cursor-Tier: unlimited` headers
- Idempotent (applying twice = applying once)
- Preserves all original headers

**Inbound:**
- Extracts `reasoning_content` from SSE chunks as a distinct stream
- Injects `: heartbeat\n\n` SSE comments during long reasoning cycles
- Closes stream after 60s of no data

### Security Gateway (`security.py`)

**Outbound (scan_outbound):**
- Regex patterns: API keys, AWS keys, SSH keys, env vars, JWT tokens, GitHub tokens
- Shannon entropy detection (>= threshold, 16+ chars)
- Replaces matches with `[REDACTED]` in a copy (never mutates original)

**Inbound (scan_inbound):**
- Extracts code blocks from responses
- Calls local LM Studio for vulnerability analysis
- Produces `CVEFinding` with type, severity, code_snippet, line_range, recommendation

**Audit Logging:**
- SQLite database at `~/.deepseek-cursor-proxy/audit.sqlite3`
- File permissions: 0o600 (owner-only)
- Stores only metadata (no secrets): timestamp, direction, request_hash, counts, status
- Graceful degradation on DB corruption/full

### TOON Engine (`toon.py`)

**Compression:**
- Detects structured blocks: file trees, symbol maps, multi-file diffs
- Converts to pipe-delimited TOON format (30%+ compression)
- Only processes blocks >= `toon_min_block_size` bytes
- Preserves message count and roles

**Re-hydration:**
- Converts TOON format back to original JSON structure
- Graceful failure: skips block, forwards original content

### Retrieval Layer (`retrieval.py`)

**Pipeline:** chunk → embed → upsert → query → assemble

- Chunks context into 512-token windows with 64-token overlap (tiktoken)
- Generates embeddings via LM Studio `/v1/embeddings`
- Stores vectors in Qdrant using MessagePack serialization
- Retrieves top-k chunks with score > 0.5
- Assembles reduced context within `retrieval_max_tokens` budget
- Preserves system messages and latest user message unchanged
- Exponential backoff retry (max 5 minutes)
- Graceful degradation when services unavailable

**Health Check:**
- Queries LM Studio `/api/v1/models` to verify embedding model is loaded
- Checks Qdrant `/collections` endpoint reachability

## Graceful Degradation

The proxy is designed to work even when optional services are down:

| Service Down | Behavior |
|-------------|----------|
| LM Studio | Retrieval skipped, full context forwarded; CVE scanning skipped |
| Qdrant | Retrieval skipped, full context forwarded |
| Both | Proxy still works (just header injection + pass-through) |
| Audit DB corrupt | Logging skipped, warning emitted, processing continues |

Each pipeline phase is wrapped in try/except. A failing phase is logged and skipped — subsequent phases still execute.

## Testing

```bash
# Run all tests
python -m pytest tests/ -q

# Run only RAP tests (fast, ~10s)
python -m pytest tests/test_rap_*.py tests/test_fidelity*.py tests/test_retrieval*.py tests/test_security*.py tests/test_toon*.py -q

# Run property-based tests only
python -m pytest tests/ -k "properties" -q
```

**Test stats:** 520 tests, ~96 seconds (property tests use reduced examples for speed).

All external services are mocked in tests — no real network calls to LM Studio or Qdrant.

## Infrastructure

### Qdrant

- Runs in Podman container `rap-qdrant`
- Persistent volume: `rap_qdrant_storage`
- REST API: `http://localhost:6333`
- Dashboard: `http://localhost:6333/dashboard`
- Communication uses MessagePack serialization (smaller than JSON)
- Localhost-only (127.0.0.1) — no external network

### LM Studio

- Server on `http://localhost:1234`
- Embedding model: `text-embedding-nomic-embed-text-v1.5-embedding`
- Status API: `GET /api/v1/models` (shows loaded_instances)
- CLI: `lms server start`, `lms load <model>`, `lms ps`
- Models stored on external M.2 — loading takes time

## Troubleshooting

**Proxy shows `model_not_loaded`:**
```bash
lms load text-embedding-nomic-embed-text-v1.5-embedding --gpu max
```

**Qdrant not starting:**
```bash
podman logs rap-qdrant
```

**Memory issues with retry loop:**
The retry logic uses exponential backoff (1s, 2s, 4s... capped at 60s per delay, 5 min total). If you see rapid retries, check that the fix in `_retry_with_backoff()` is applied.

**Test failures in `test_transform.py`:**
This is a pre-existing flaky test in the original proxy (test ordering issue). Not related to RAP.
