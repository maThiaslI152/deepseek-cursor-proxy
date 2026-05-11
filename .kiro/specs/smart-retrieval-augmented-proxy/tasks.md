# Implementation Plan: Smart Retrieval-Augmented Proxy (RAP)

## Overview

This plan implements the Smart RAP in four phases: Bridge & Fidelity, Compression (TOON), Retrieval (Qdrant), and Security. Each phase builds incrementally on the previous, with the Pipeline Orchestrator wiring everything together. The existing proxy is migrated from stdlib `http.server` to FastAPI/uvicorn as the foundation for async middleware.

## Tasks

- [x] 1. Project foundation and FastAPI migration
  - [x] 1.1 Add RAP dependencies to pyproject.toml
    - Add `fastapi>=0.100`, `uvicorn>=0.23`, `msgpack>=1.0`, `qdrant-client>=1.7`, `httpx>=0.25`, `tiktoken>=0.5` to project dependencies
    - Add `hypothesis>=6.0`, `testcontainers>=3.7`, `pytest-asyncio>=0.23` to dev dependencies
    - _Requirements: 14.1, 14.2_

  - [x] 1.2 Create RAP configuration model with validation
    - Create `src/deepseek_cursor_proxy/rap/config.py` with `RAPConfig` dataclass
    - Implement validation for `heartbeat_interval` (0, 60], `retrieval_top_k` [1, 50], `retrieval_max_tokens` [100, 10000], `entropy_threshold` [3.0, 8.0], `toon_min_block_size` >= 64, URL validation for `qdrant_url` and `embedding_url`
    - Raise specific `ValidationError` with field name and reason on failure
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_

  - [x] 1.3 Write property test for configuration validation
    - **Property 22: Configuration Validation Correctness**
    - **Validates: Requirements 12.1, 12.3, 12.4, 12.5, 12.6**

  - [x] 1.4 Create Pipeline Orchestrator skeleton
    - Create `src/deepseek_cursor_proxy/rap/pipeline.py` with `PipelineOrchestrator` class
    - Implement phase-enabled/disabled logic and graceful degradation (try/except per phase)
    - Implement `process_request()` and `process_response()` methods with phase ordering
    - Implement `health_check()` endpoint returning component status
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 13.5_

  - [x] 1.5 Write property tests for pipeline orchestration
    - **Property 19: Pipeline Phase Skip on Disable**
    - **Property 20: Pipeline Graceful Degradation on Phase Failure**
    - **Property 21: Message Format Preservation Through Pipeline**
    - **Validates: Requirements 11.3, 11.4, 11.5**

- [x] 2. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Phase 1 — Bridge & Fidelity Module
  - [x] 3.1 Implement header spoofing and endpoint routing
    - Create `src/deepseek_cursor_proxy/rap/fidelity.py` with `FidelityModule` class
    - Implement `inject_spoof_headers()` that adds `X-Cursor-Plan: pro` and `X-Cursor-Tier: unlimited`
    - Ensure idempotency (applying twice produces same result as once)
    - Preserve all original headers in output
    - Route to configured BYOK endpoint
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 3.2 Write property tests for header injection
    - **Property 1: Header Injection Completeness**
    - **Property 2: Header Injection Idempotency**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4**

  - [x] 3.3 Implement reasoning token pass-through
    - Add `extract_reasoning_stream()` to `FidelityModule`
    - Extract `reasoning_content` field from SSE chunks and emit as distinct stream
    - Forward all reasoning tokens without modification when enabled
    - Handle missing `reasoning_content` gracefully (return None, no error)
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.4 Write property test for reasoning extraction
    - **Property 3: Reasoning Token Extraction Integrity**
    - **Validates: Requirements 2.1, 2.2, 2.3**

  - [x] 3.5 Implement stream health monitoring with heartbeat injection
    - Add `heartbeat_wrapper()` async generator to `FidelityModule`
    - Inject `: heartbeat\n\n` SSE comment when no data received within configured interval
    - Preserve all original upstream chunks in order without modification
    - Close stream gracefully after 60s of no data despite heartbeats
    - Ensure time since last emitted byte never exceeds 2× heartbeat interval
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 3.6 Write property test for stream integrity
    - **Property 4: Stream Integrity Under Heartbeat Injection**
    - **Validates: Requirement 3.2**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Phase 2 — TOON Compression Engine
  - [x] 5.1 Implement structured block detection
    - Create `src/deepseek_cursor_proxy/rap/toon.py` with `TOONEngine` class
    - Implement `detect_structured_blocks()` to find file trees, symbol maps, and multi-file diffs
    - Return non-overlapping `StructuredBlock` instances sorted by offset
    - Only detect blocks >= `toon_min_block_size` bytes
    - _Requirements: 4.1, 4.2, 4.6_

  - [x] 5.2 Write property test for non-overlapping detection
    - **Property 9: Non-Overlapping Structured Block Detection**
    - **Validates: Requirement 4.6**

  - [x] 5.3 Implement TOON compression
    - Implement `to_toon()` converting structured blocks to pipe-delimited TOON format
    - Implement `compress()` method that processes message lists
    - Ensure output is at most 70% the size of original structured block
    - Preserve message count and role assignments
    - Leave messages shorter than `toon_min_block_size` unchanged
    - _Requirements: 4.1, 4.3, 4.4, 4.5_

  - [x] 5.4 Write property tests for TOON compression
    - **Property 6: TOON Compression Ratio Bound**
    - **Property 7: Message Count and Role Preservation Under Compression**
    - **Property 8: Short Message Identity Under Compression**
    - **Validates: Requirements 4.3, 4.4, 4.5**

  - [x] 5.5 Implement TOON re-hydration
    - Implement `rehydrate()` method converting TOON format back to original JSON structure
    - Handle re-hydration failure gracefully (skip block, forward original content)
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 5.6 Write property test for TOON round-trip
    - **Property 5: TOON Compression Round-Trip**
    - **Validates: Requirements 5.1, 5.2**

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Phase 3 — Retrieval Layer (Qdrant + LM Studio)
  - [x] 7.1 Implement context chunking
    - Create `src/deepseek_cursor_proxy/rap/retrieval.py` with `RetrievalLayer` class
    - Implement `chunk_context()` splitting messages into chunks of `chunk_size_tokens` with `chunk_overlap_tokens` overlap
    - Use tiktoken for accurate token counting
    - Preserve system messages and latest user message unchanged
    - _Requirements: 6.1, 6.4_

  - [x] 7.2 Write property test for chunking
    - **Property 10: Chunking Produces Correct Size and Overlap**
    - **Validates: Requirement 6.1**

  - [x] 7.3 Implement embedding generation via LM Studio
    - Implement `embed()` method calling LM Studio `/v1/embeddings` endpoint via httpx
    - Validate embedding dimensionality consistency across batch
    - Reject embeddings containing NaN or Inf values
    - Handle LM Studio unavailability gracefully (skip retrieval)
    - _Requirements: 6.2, 6.3, 6.5, 13.2_

  - [x] 7.4 Write property test for embedding consistency
    - **Property 11: Embedding Dimensionality Consistency**
    - **Validates: Requirements 6.3, 6.5**

  - [x] 7.5 Implement Qdrant vector storage with MessagePack transport
    - Implement `upsert_chunks()` using MessagePack serialization (`use_bin_type=True`)
    - Implement `retrieve()` querying top-k chunks with score > 0.5 threshold
    - Ensure all communication over localhost only (127.0.0.1)
    - Handle Qdrant unavailability gracefully (skip retrieval, forward full context)
    - _Requirements: 7.1, 7.2, 7.4, 7.5, 13.1, 14.1, 14.3_

  - [x] 7.6 Write property tests for retrieval
    - **Property 13: Retrieval Token Budget Compliance**
    - **Property 14: Retrieval Score Threshold**
    - **Property 23: MessagePack Size Efficiency**
    - **Validates: Requirements 7.3, 7.4, 14.2**

  - [x] 7.7 Implement `build_reduced_context()` assembly
    - Implement full context reduction pipeline: chunk → embed → upsert → query → assemble
    - Ensure total retrieved token count does not exceed `retrieval_max_tokens`
    - Preserve system messages and latest user message in output
    - Implement exponential backoff retry for service connections (max 5 min)
    - _Requirements: 7.3, 6.4, 13.4_

  - [x] 7.8 Write property test for message preservation
    - **Property 12: System and User Message Preservation**
    - **Validates: Requirement 6.4**

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Phase 4 — Security Gateway
  - [x] 9.1 Implement outbound secret redaction
    - Create `src/deepseek_cursor_proxy/rap/security.py` with `SecurityGateway` class
    - Implement `scan_outbound()` with regex patterns for API keys, AWS keys, SSH keys, env vars, JWT tokens, GitHub tokens
    - Implement Shannon entropy detection for high-entropy substrings (>= threshold, 16+ chars)
    - Replace matches with `[REDACTED]` — return a copy, never mutate original
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 9.2 Write property tests for redaction
    - **Property 15: Redaction Completeness**
    - **Property 16: Redaction Immutability**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4**

  - [x] 9.3 Implement inbound CVE scanning
    - Implement `scan_inbound()` extracting code blocks from responses
    - Call local LM Studio model for vulnerability analysis (no external network)
    - Produce `CVEFinding` with type, severity, code_snippet, line_range, recommendation
    - Annotate response with findings for developer visibility
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 9.4 Write property test for CVE finding structure
    - **Property 17: CVE Finding Structural Completeness**
    - **Validates: Requirement 9.3**

  - [x] 9.5 Implement audit logging with SQLite
    - Implement `log_transaction()` writing `AuditEntry` to SQLite database
    - Create audit schema with proper indexes (timestamp, direction, status)
    - Set database file permissions to 0o600 (owner-only)
    - Ensure no secret content stored (only metadata and counts)
    - Handle database corruption/full gracefully (continue without logging, emit warning)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 9.6 Write property test for audit logging
    - **Property 18: Audit Entry Completeness and No Secret Leakage**
    - **Validates: Requirements 10.1, 10.2, 10.4**

- [x] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Integration and wiring
  - [x] 11.1 Wire all modules into Pipeline Orchestrator
    - Connect FidelityModule, SecurityGateway, TOONEngine, RetrievalLayer into `PipelineOrchestrator`
    - Implement outbound order: Fidelity → Security (outbound) → TOON (compress) → Retrieval
    - Implement inbound order: Stream Health → TOON (re-hydrate) → Security (inbound scan)
    - Ensure graceful degradation when any phase fails
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [x] 11.2 Integrate RAP pipeline into FastAPI application
    - Create `src/deepseek_cursor_proxy/rap/app.py` with FastAPI app mounting the pipeline
    - Add `/v1/chat/completions` endpoint routing through the pipeline
    - Add `/healthz` endpoint exposing component health status
    - Wire RAPConfig loading from existing config.yaml with new RAP fields
    - _Requirements: 11.1, 11.2, 13.5_

  - [x] 11.3 Write integration tests for full pipeline
    - Test end-to-end outbound flow with mock DeepSeek API
    - Test end-to-end inbound flow with SSE streaming
    - Test graceful degradation when Qdrant/LM Studio unavailable
    - _Requirements: 11.1, 11.2, 13.1, 13.2_

- [x] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation between phases
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The existing proxy functionality is preserved; RAP modules are additive middleware
- Qdrant requires Podman to be installed and running locally
- LM Studio must be running with an embedding model loaded for retrieval/CVE features

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3", "1.4"] },
    { "id": 3, "tasks": ["1.5", "3.1"] },
    { "id": 4, "tasks": ["3.2", "3.3", "3.5"] },
    { "id": 5, "tasks": ["3.4", "3.6"] },
    { "id": 6, "tasks": ["5.1"] },
    { "id": 7, "tasks": ["5.2", "5.3"] },
    { "id": 8, "tasks": ["5.4", "5.5"] },
    { "id": 9, "tasks": ["5.6"] },
    { "id": 10, "tasks": ["7.1", "9.1"] },
    { "id": 11, "tasks": ["7.2", "7.3", "9.2"] },
    { "id": 12, "tasks": ["7.4", "7.5", "9.3"] },
    { "id": 13, "tasks": ["7.6", "7.7", "9.4", "9.5"] },
    { "id": 14, "tasks": ["7.8", "9.6"] },
    { "id": 15, "tasks": ["11.1"] },
    { "id": 16, "tasks": ["11.2"] },
    { "id": 17, "tasks": ["11.3"] }
  ]
}
```
