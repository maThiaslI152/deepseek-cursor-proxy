# Architecture

## Overview

**deepseek-cursor-proxy** is a local HTTP proxy that sits between Cursor IDE and the DeepSeek API. Its primary purpose is to fix a compatibility issue: when DeepSeek's thinking mode is enabled, Cursor omits the `reasoning_content` field from tool-call messages in subsequent requests, causing DeepSeek to return a 400 error.

The proxy caches `reasoning_content` from DeepSeek responses in a local SQLite database and injects it back into outgoing requests before forwarding them.

## Data Flow

```
┌─────────┐   HTTPS    ┌──────────┐   HTTP    ┌────────────┐   HTTPS    ┌──────────────┐
│  Cursor  │ ────────→ │  ngrok   │ ────────→ │  Proxy     │ ────────→ │  DeepSeek    │
│   IDE    │ ←──────── │  Tunnel  │ ←──────── │  Server    │ ←──────── │  API         │
└─────────┘            └──────────┘            └─────┬──────┘            └──────────────┘
                                                      │
                                                      │  ┌────────────────────┐
                                                      ├─→ │  Reasoning Store   │
                                                      │   │  (SQLite)          │
                                                      │   └────────────────────┘
                                                      │
                                                      └─→ ┌────────────────────┐
                                                          │  Trace Writer      │
                                                          │  (JSON files)      │
                                                          └────────────────────┘
```

1. **Cursor** sends a chat completion request to the proxy's ngrok URL
2. **ngrok** forwards the request to the local proxy server
3. **Proxy Server** transforms the request (injects cached reasoning, normalizes fields) and forwards to the real DeepSeek API
4. **DeepSeek API** responds with a chat completion (streaming or regular)
5. **Proxy Server** records reasoning content from the response into the SQLite cache, rewrites the response (restores original model name, injects think blocks), and sends it back to Cursor via ngrok

## Module Breakdown

### `server.py` — HTTP Server & Request Handler (~1290 lines)

The entry point and main orchestrator.

- **`DeepSeekProxyServer`** — Subclass of `ThreadingHTTPServer`. Holds shared state: `config`, `reasoning_store`, `trace_writer`, and `RequestMetrics` (tracks cache hit/miss, latency, success/failure counts).
- **`DeepSeekProxyHandler`** — Subclass of `BaseHTTPRequestHandler`. Handles all HTTP verbs:
  - `do_GET()` — Health check (`/healthz`) and model listing (`/models`). `/healthz` exposes metrics snapshot.
  - `do_POST()` — Main pipeline for `/chat/completions`:
    1. Parse JSON body and validate auth
    2. Call `prepare_upstream_request()` to transform the request
    3. Forward to DeepSeek API via `urllib.request.urlopen()`
    4. Proxy response back via `_proxy_streaming_response()` or `_proxy_regular_response()`
    5. `RequestMetrics.record()` updates hit/miss counters for each completed request
  - `do_OPTIONS()` — CORS preflight handling
- **`main()`** — Entry point: loads config, applies CLI overrides, initializes store/traces, optionally starts ngrok, starts the server.
- **Logging helpers** — Unicode box-drawing format for structured terminal output:
  ```
  ┌ cursor  model=deepseek-v4-flash messages=42 tools=8
  ├ context status=ok reasoning_context=5
  ├ send    user_msgs=3 messages=42 tools=8 reasoning_content=5
  └ stats   prompt=12,345 output=5,678 reasoning=1,234 api_cache=72.3% proxy_cache=85%
  ```
  The `proxy_cache` field shows the proxy's own reasoning_content cache hit rate (across all requests since startup), distinct from DeepSeek's API-side prompt cache hit rate (`api_cache`).

### `config.py` — Configuration Management (~270 lines)

- **`ProxyConfig`** — Frozen dataclass with all settings: host, port, upstream URL, model, thinking mode, reasoning effort, ngrok, CORS, tracing, cache limits, etc.
- **`settings_from_config()`** — Loads YAML from `~/.deepseek-cursor-proxy/config.yaml`, auto-populates defaults on first run.
- **Type coercion helpers** — `as_str()`, `as_bool()`, `as_int()`, `as_float()`, `as_path()` with safe fallbacks for missing/invalid values.
- **Path helpers** — `default_app_dir()`, `default_config_path()`, `default_reasoning_content_path()`.

### `transform.py` — Request/Response Transformation (~500 lines)

The core compatibility layer.

- **`prepare_upstream_request()`** — Main transformation pipeline:
  1. Set model (respects Cursor's requested model, falls back to config default)
  2. Configure thinking mode (enabled/disabled/pass-through)
  3. Normalize `reasoning_effort` (accepts aliases: high/xhigh/max → same value)
  4. Convert legacy `functions`/`function_call` → `tools`/`tool_choice`
  5. Normalize each message: strip mirrored `<think>` blocks, extract text content, patch `reasoning_content` from cache
  6. Recover from missing reasoning (drop unreachable history if cache is cold)
  7. Return `PreparedRequest` dataclass with metadata about what was transformed
- **`normalize_message()`** — Per-message normalization and cache lookup
- **`recover_messages_from_missing_reasoning()`** — Recovery strategies when the cache doesn't have required reasoning
- **`rewrite_response_body()`** / **`record_response_reasoning()`** — Post-process DeepSeek responses: record reasoning to cache, restore original model name
- **`reasoning_cache_namespace()`** — Creates isolated cache namespaces per API key + config combination

### `streaming.py` — Streaming Response Handling (~300 lines)

- **`StreamAccumulator`** — Accumulates SSE chunks into full assistant messages. Supports multi-choice indexing. Methods:
  - `ingest_chunk()` — Process each incoming SSE data chunk
  - `store_reasoning()` — Store all accumulated reasoning to the SQLite cache
  - `store_ready_reasoning()` — Store early when tool calls are detected before `finish_reason`
  - `store_finished_reasoning()` — Final storage on stream completion
  - Stage-based deduplication prevents re-storing already-cached reasoning
- **`CursorReasoningDisplayAdapter`** — Mirrors `reasoning_content` into `content` wrapped in `<think>...</think>` tags so Cursor displays the thinking tokens in its UI. Tracks open/close state per choice index.

### `reasoning_store.py` — SQLite Cache (~430 lines)

Thread-safe SQLite-backed cache for reasoning content, using schema v2 for deduplication and WAL mode for concurrent performance.

- **Schema v2** — Two normalized tables:
  - `reasoning_texts(hash, reasoning, created_at)` — Stores reasoning content once, keyed by SHA-256 hash
  - `reasoning_cache(key, reasoning_hash, created_at)` — Maps cache keys to reasoning text via hash reference
  - This eliminates the per-key duplication of reasoning text (previously N rows × full text per assistant message)
- **WAL mode + pragmas** — `PRAGMA journal_mode=WAL`, `synchronous=NORMAL`, `mmap_size=256MB`, `cache_size=64MB`. WAL enables concurrent readers without blocking on writes, critical for the threaded proxy.
- **`ReasoningStore`** — Core store with methods:
  - `put()` / `get()` — Basic CRUD (v2-aware with hash dedup)
  - `batch_lookup(keys)` — **Single SQL query** replacing up to ~15 individual SELECTs:
    - Uses `WHERE key IN (...)` with `ORDER BY CASE` to maintain caller priority ordering
    - Eliminates lock acquisition overhead of N separate round-trips
  - `store_assistant_message()` — Batch `executemany` INSERT instead of N individual `put()` calls
  - `warm_cache()` — Pre-writes scope + portable keys when a namespace key hits (context reset detected), so the next turn hits at Priority 1
  - `lookup_for_message()` — Priority-ordered lookup, now delegates to `batch_lookup()`
  - `clear()` — Clears both tables
  - `prune()` — Age/row-limit cleanup with orphaned reasoning_texts removal
- **Key types (priority order on lookup):**
  - `scope:{scope}:{type}:{id}` — Exact conversation scope (priority 1)
  - `namespace:{ns}:turn:{turn_sig}:{type}:{id}` — Portable turn-context keys (priority 2)
  - `namespace:{ns}:{type}:{id}` — Broad namespace keys, survives Cursor context resets (priority 3)
- **`conversation_scope()`** — SHA-256 of canonical conversation prefix (roles + content + tool calls, excluding reasoning_content)
- **`namespace_reasoning_keys()`** — Broad keys keyed only by API config + message content, allowing recall across arbitrary conversations
- **`message_signature()`**, **`tool_call_signature()`**, **`turn_context_signature()`** — Hashing utilities

### `tunnel.py` — ngrok Tunnel Management (~100 lines)

- **`NgrokTunnel`** — Spawns an ngrok subprocess, polls the ngrok local API for the public URL, provides graceful shutdown.
  - `parse_ngrok_public_url()` — Parses both current (`/api/endpoints`) and legacy (`/api/tunnels`) ngrok API response formats
  - `local_tunnel_target()` — Formats the local URL for ngrok, handles IPv6 and wildcard bind addresses
- Supports `SIGTERM` for clean subprocess teardown.

### `trace.py` — Structured Request Tracing (~120 lines)

- **`TraceWriter`** — Creates a timestamp+pid session directory, writes one JSON file per proxied request.
- **`TraceRequest`** — Accumulates full request/response lifecycle data including cursor body, transform result, upstream request/response, stream chunks, usage, completion status.
- Auth headers are SHA-256 hashed (never stored in plaintext).
- Files created with `0o600` permissions.

## Key Design Decisions

1. **ThreadingHTTPServer over async** — Simple synchronous model is sufficient for a local proxy. Each request gets its own thread. The bottleneck is the upstream API call, not concurrency.

2. **SQLite over in-memory cache** — Reasoning content must survive proxy restarts. SQLite provides persistence with zero infrastructure. The cache uses multiple key types (message signature, tool call ID, tool call signature, portable turn keys) to maximize cache hit rate across different conversation contexts.

3. **Scope isolation via SHA-256** — Each conversation gets a unique scope derived from a canonical representation of its message history. This prevents tool call ID collisions across concurrent conversations while allowing byte-identical clones to share cache.

4. **Priority-ordered cache key lookup** — The cache stores each reasoning entry under three key types, and looks them up in priority order:
   - **Priority 1 (scope keys):** Exact conversation match — `scope:{full_prefix_hash}:{type}:{id}`. Guarantees the same conversation gets the exact same reasoning.
   - **Priority 2 (portable turn keys):** Same message tail, different prefix — `namespace:{ns}:turn:{turn_sig}:{type}:{id}`. Handles mode switches (agent ↔ plan) where the message suffix is identical.
   - **Priority 3 (broad namespace keys):** Any conversation with the same API config — `namespace:{ns}:{type}:{id}`. Only depends on API config + message content, so the same tool call in a completely new Cursor session finds cached reasoning, surviving arbitrary context resets.
   
   This eliminates the need for a separate `backfill_portable_aliases()` step — all key types are written at store time, and priority-order lookup ensures correctness by trying more specific matches first.

5. **Missing reasoning recovery** — When the cache is cold (proxy restart, model switch), the proxy can either:
   - **Recover** (default): Drop unreachable history, continue from the latest user message, prefix the next response with a notice
   - **Reject** (strict mode): Return HTTP 409, useful for debugging

6. **No synthetic thread IDs** — The proxy preserves DeepSeek's context caching by never injecting synthetic identifiers, timestamps, or cache-control messages. Reasoning content is restored as the exact original string.

7. **Dual response handling** — Streaming and non-streaming responses use different code paths:
   - Streaming: SSE chunks are accumulated and stored incrementally (tool-call detection triggers early storage)
   - Non-streaming: The full response is decoded, processed, and stored in one pass

8. **Single-query batch lookup** — When patching reasoning into an upstream request, the proxy previously iterated through ~15 cache keys per assistant message, each requiring an individual SQLite SELECT query. For a 10-message conversation, that was ~150 round-trips. The `batch_lookup()` method collapses this into a single `SELECT ... WHERE key IN (...) ORDER BY CASE ... LIMIT 1`, improving TTFT after context resets by ~10-50ms.

9. **Cache warming on context reset** — When Cursor truncates conversation history (around 200K tokens), the proxy's scope keys (Priority 1) and portable keys (Priority 2) all miss because the conversation prefix changed. The namespace keys (Priority 3) still hit because they depend only on message content + API config. After a namespace hit, `warm_cache()` pre-writes scope + portable keys for the current conversation, so the *next* turn hits at Priority 1 — zero wasted lookups.

10. **Schema v2 deduplication** — Each assistant message's `reasoning_content` is stored under ~11-15 different cache keys (scope + portable + namespace × signature + tool_call_id + tool_call_signature + tool_name). V1 stored the full reasoning text in every row. V2 normalizes reasoning text into a separate `reasoning_texts` table keyed by SHA-256 hash, with `reasoning_cache` referencing the hash. This eliminates redundant storage and keeps the cache table lean for faster scans.

11. **WAL mode for concurrent access** — The `ThreadingHTTPServer` dispatches requests across threads. WAL mode allows readers to proceed without blocking writers, preventing lock contention when one thread is recording a response while another is looking up cached reasoning for the next request.

## Configuration

All settings can be configured via:
1. **YAML config file** — `~/.deepseek-cursor-proxy/config.yaml` (auto-created on first run)
2. **CLI flags** — Override any config value at runtime (e.g., `--port 9000`, `--no-ngrok`)
3. **CLI flags override config file values** — When both are provided

Key config options:

| Setting | Default | Description |
|---------|---------|-------------|
| `base_url` | `https://api.deepseek.com` | Upstream DeepSeek API base URL |
| `model` | `deepseek-v4-flash` | Fallback model when request has none |
| `thinking` | `enabled` | Thinking mode: enabled/disabled/pass-through |
| `reasoning_effort` | `high` | Reasoning effort: low/medium/high/max/xhigh |
| `display_reasoning` | `true` | Mirror reasoning into Cursor-visible think blocks |
| `ngrok` | `true` | Start ngrok tunnel for public HTTPS access |
| `missing_reasoning_strategy` | `recover` | recover/reject when cache is cold |

## Error Handling

- **Upstream HTTP errors** — Forwarded to Cursor as-is with appropriate status codes
- **Upstream connection failures** → HTTP 502
- **Missing/empty auth** → HTTP 401
- **Oversized requests** → HTTP 413
- **Missing reasoning in strict mode** → HTTP 409
- **Invalid JSON bodies** → HTTP 400
- **Client disconnects** — Gracefully handled at header send and body write points (both streaming and non-streaming)
- **Unsupported paths** → HTTP 404

## Security

- Auth tokens are forwarded as-is to the upstream API (proxy never stores them)
- Trace files SHA-256 hash auth headers instead of storing them in plaintext
- Config file and app directory are created with `0o600`/`0o700` permissions
- Trace files written with `0o600` permissions
- Insecure HTTP upstream to non-localhost hosts triggers a warning
