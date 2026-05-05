# Project Index

## Summary

**deepseek-cursor-proxy** — A local HTTP proxy that sits between Cursor IDE and the DeepSeek API, fixing the `reasoning_content` missing-field bug in tool-call requests when thinking mode is enabled.

## Quick Links

| Document | Description |
|----------|-------------|
| [README.md](README.md) | User-facing setup, usage, and debugging guide |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and design decisions |
| [pyproject.toml](pyproject.toml) | Package metadata and dependencies |
| [LICENSE](LICENSE) | MIT License |

## Package Structure

```
deepseek-cursor-proxy/
├── src/deepseek_cursor_proxy/
│   ├── __init__.py          # Version: 0.1.0
│   ├── __main__.py          # python -m entry point (→ server.main)
│   ├── server.py            # HTTP server, request handler, CLI, main()
│   ├── config.py            # ProxyConfig dataclass, YAML load/save
│   ├── transform.py         # Request/response transformation pipeline
│   ├── streaming.py         # StreamAccumulator, CursorReasoningDisplayAdapter
│   ├── reasoning_store.py   # SQLite-backed reasoning_content cache
│   ├── tunnel.py            # ngrok subprocess management
│   └── trace.py             # Structured request tracing (JSON dumps)
├── tests/
│   ├── test_config.py               # Config loading & defaults
│   ├── test_server.py               # CLI parsing, gzip, client disconnect
│   ├── test_streaming.py            # StreamAccumulator & display adapter
│   ├── test_transform.py            # Request normalization & cache patching
│   ├── test_tunnel.py               # ngrok URL parsing
│   ├── test_trace.py                # Trace file format & auth redaction
│   ├── test_reasoning_store.py      # SQLite CRUD & pruning
│   ├── test_live_deepseek_cursor_proxy.py  # Live integration test (opt-in)
│   └── test_proxy_end_to_end.py     # Full end-to-end tests with fake upstream
├── assets/
│   ├── logo.png / logo.svg          # Project logo
│   ├── cursor_chat.png              # Chat in Cursor screenshot
│   ├── cursor_config.png            # Cursor config UI screenshot
│   └── error_400.png                # The 400 error this proxy fixes
├── .github/workflows/ci.yml        # GitHub Actions CI (lint + matrix test)
├── .pre-commit-config.yaml         # Pre-commit hooks (black, ruff)
├── pyproject.toml                  # Build config & runtime deps
├── uv.lock                         # uv lockfile
├── README.md                       # User documentation
├── ARCHITECTURE.md                 # Architecture documentation
└── Index.md                        # This file
```

## Runtime Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | >= 3.10 | Runtime |
| PyYAML | >= 6.0 | Config file parsing |
| ngrok | external | Public HTTPS tunnel (optional) |

**Dev dependencies:** black, ruff, pre-commit

## Module Responsibilities

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `server.py` | ~1280 | HTTP server, request lifecycle, CLI, logging |
| `transform.py` | ~500 | Request normalization, reasoning patching, recovery |
| `reasoning_store.py` | ~350 | SQLite cache for reasoning_content |
| `streaming.py` | ~300 | SSE chunk accumulation, think-block wrapping |
| `config.py` | ~270 | Configuration loading, defaults, type coercion |
| `trace.py` | ~120 | Structured request/response debug dumps |
| `tunnel.py` | ~100 | ngrok subprocess lifecycle |

## Key Concepts

### Reasoning Content Cache

The proxy stores `reasoning_content` from DeepSeek responses in a local SQLite database (`~/.deepseek-cursor-proxy/reasoning_content.sqlite3`). Each entry is stored under three key types for priority-ordered lookup:

| Priority | Key Pattern | Survives Context Reset? | Scope |
|----------|-------------|-------------------------|-------|
| 1 | `scope:{prefix_hash}:{type}:{id}` | Same conversation only | Exact conversation |
| 2 | `namespace:{ns}:turn:{turn_sig}:{type}:{id}` | Same message tail | Same turn context |
| 3 | `namespace:{ns}:{type}:{id}` | **Yes — any context reset** | Any conversation, same API key |

Key types for each level:
- **signature** — SHA-256 of the assistant message content
- **tool_call:{id}** — The `id` field from each tool call
- **tool_call_signature:{sig}** — SHA-256 of the function name + arguments

### Request Transformation Pipeline

Each incoming request passes through these stages in `prepare_upstream_request()`:

1. **Model resolution** — Use Cursor's requested model or config fallback
2. **Thinking mode** — Set/override the thinking mode field
3. **Reasoning effort** — Normalize alias values
4. **Tool normalization** — Convert `functions`/`function_call` → `tools`/`tool_choice`
5. **Message normalization** — Strip think blocks, extract content, patch reasoning from cache
6. **Recovery** — If required reasoning is missing, drop history and restart from latest user message

### Response Rewriting

Each DeepSeek response is rewritten before being sent to Cursor:
1. **Model name restored** — Replace upstream model with the original requested model
2. **Reasoning recorded** — Store `reasoning_content` in the SQLite cache
3. **Think blocks** — Optionally mirror reasoning into `<think>` tags for Cursor UI
4. **Recovery notice** — Prepend a notice if recovery mode was triggered

### Streaming vs Non-Streaming

| Aspect | Non-Streaming | Streaming |
|--------|---------------|-----------|
| Processing | Single pass | Incremental SSE chunks |
| Cache storage | After full response | Incremental (early on tool call detection) |
| Think blocks | Applied to final body | Applied per-chunk via adapter |
| Recovery notice | Prepended to content | Injected into first content delta |

## CLI Reference

```bash
deepseek-cursor-proxy [options]

Options:
  --config PATH                     YAML config file
  --host HOST                       Bind host
  --port PORT                       Bind port (default: 9000)
  --model MODEL                     Fallback model name
  --base-url URL                    DeepSeek API base URL
  --thinking {enabled,disabled,pass-through}
  --reasoning-effort {low,medium,high,max,xhigh}
  --reasoning-content-path PATH     SQLite cache path
  --ngrok / --no-ngrok             Enable/disable ngrok tunnel
  --verbose / --no-verbose         Detailed logging
  --display-reasoning / --no-display-reasoning  Show think blocks in Cursor
  --cors / --no-cors               CORS headers
  --request-timeout SECONDS        Upstream timeout (default: 900)
  --max-request-body-bytes BYTES   Max request size (default: 50MB)
  --missing-reasoning-strategy {recover,reject}
  --clear-reasoning-cache          Clear SQLite cache and exit
  --trace-dir PATH                 Write structured traces to directory
```

## Test Suites

| Target | File | Coverage |
|--------|------|----------|
| Config | `test_config.py` | File creation, loading, defaults, errors, path resolution |
| Server | `test_server.py` | CLI parsing, response decompression, client disconnect |
| Transform | `test_transform.py` | Request normalization, cache patching, recovery, scope isolation |
| Streaming | `test_streaming.py` | SSE accumulation, think-block wrapping, tool-call merging |
| Reasoning Store | `test_reasoning_store.py` | SQLite CRUD, pruning, permissions |
| Tunnel | `test_tunnel.py` | ngrok URL parsing (both API versions) |
| Trace | `test_trace.py` | File creation, sequence numbering, auth redaction |
| End-to-End | `test_proxy_end_to_end.py` | Full lifecycle with fake upstream |
| Live | `test_live_deepseek_cursor_proxy.py` | Real DeepSeek API (opt-in) |

Run with: `uv run python -m unittest discover -s tests`
