# Requirements Document

## Introduction

The Smart Retrieval-Augmented Proxy (RAP) extends the existing DeepSeek Cursor Proxy with five integrated modules that optimize DeepSeek V4 integration within Cursor while maintaining local security and performance. The system adds orchestration fidelity for reasoning token pass-through, context optimization via selective compression, local hybrid intelligence using vector search, cybersecurity protections for data integrity, and a pipeline orchestrator to coordinate all middleware stages.

## Glossary

- **Proxy**: The DeepSeek Cursor Proxy application that intercepts requests from Cursor IDE and forwards them to the DeepSeek API
- **Fidelity_Module**: The component responsible for header spoofing, reasoning token pass-through, and stream health monitoring
- **TOON_Engine**: Token-Oriented Object Notation engine that compresses structured data blocks and re-hydrates responses
- **Retrieval_Layer**: The component that chunks context, generates embeddings, and retrieves relevant snippets via Qdrant vector search
- **Security_Gateway**: The component that redacts secrets from outbound requests and scans inbound code for CVEs
- **Pipeline_Orchestrator**: The component that coordinates the execution order of all middleware modules
- **Structured_Block**: A detected region of content containing file trees, symbol maps, or multi-file diffs
- **TOON_Format**: A compact pipe-delimited notation that eliminates redundant JSON keys and brackets
- **Qdrant**: A local vector database running in a Podman container for similarity search
- **LM_Studio**: A local LLM inference application used for embedding generation and CVE scanning
- **MessagePack**: A binary serialization format used for efficient proxy-to-Qdrant communication
- **SSE**: Server-Sent Events streaming protocol used for DeepSeek API responses
- **Heartbeat**: An SSE comment line (`: heartbeat\n\n`) injected to keep connections alive during long reasoning cycles
- **Shannon_Entropy**: A measure of randomness in a string, used to detect potential secrets
- **Redaction**: The replacement of sensitive content with `[REDACTED]` placeholder text
- **CVE_Finding**: A detected vulnerability in AI-generated code with type, severity, and recommendation
- **Audit_Entry**: A record in the local SQLite database logging transaction metadata

## Requirements

### Requirement 1: Header Spoofing and Endpoint Routing

**User Story:** As a developer using Cursor, I want the proxy to inject Pro/Unlimited headers and route to the BYOK endpoint, so that I can access full DeepSeek V4 capabilities without plan restrictions.

#### Acceptance Criteria

1. WHEN an outbound request passes through the Fidelity_Module, THE Fidelity_Module SHALL inject an `X-Cursor-Plan: pro` header into the request
2. WHEN an outbound request passes through the Fidelity_Module, THE Fidelity_Module SHALL inject an `X-Cursor-Tier: unlimited` header into the request
3. WHEN the Fidelity_Module injects spoofed headers, THE Fidelity_Module SHALL preserve all original headers from the incoming request
4. WHEN `inject_spoof_headers()` is applied multiple times to the same request, THE Fidelity_Module SHALL produce the same result as applying it once (idempotency)
5. THE Fidelity_Module SHALL route inference requests to the configured BYOK endpoint

### Requirement 2: Reasoning Token Pass-Through

**User Story:** As a developer, I want to see the model's reasoning process in a separate stream, so that I can understand how DeepSeek arrived at its answer.

#### Acceptance Criteria

1. WHEN an SSE chunk contains a `reasoning_content` field, THE Fidelity_Module SHALL extract it and emit it as a distinct stream to the client
2. WHILE `reasoning_passthrough` is enabled in configuration, THE Fidelity_Module SHALL forward all reasoning tokens without modification
3. IF the `reasoning_content` field is absent from an SSE chunk, THEN THE Fidelity_Module SHALL continue processing without error

### Requirement 3: Stream Health Monitoring

**User Story:** As a developer, I want the proxy to keep my connection alive during long reasoning cycles, so that my IDE does not time out waiting for a response.

#### Acceptance Criteria

1. WHILE no data has been received from the upstream for the configured heartbeat interval, THE Fidelity_Module SHALL inject an SSE comment heartbeat (`: heartbeat\n\n`) into the client stream
2. WHEN heartbeat comments are injected, THE Fidelity_Module SHALL preserve all original upstream data chunks in their original order without modification
3. IF no data is received from DeepSeek for more than 60 seconds despite heartbeats, THEN THE Fidelity_Module SHALL close the stream gracefully and return a partial response
4. THE Fidelity_Module SHALL ensure the time since the last emitted byte to the client does not exceed twice the configured heartbeat interval

### Requirement 4: TOON Compression of Structured Data

**User Story:** As a developer, I want the proxy to compress structured data in my context (file trees, symbol maps, diffs), so that I use fewer tokens and reduce API costs.

#### Acceptance Criteria

1. WHEN a message content contains a structured block of at least `toon_min_block_size` bytes, THE TOON_Engine SHALL detect and convert it to TOON_Format
2. WHEN detecting structured blocks, THE TOON_Engine SHALL identify file trees, symbol maps, and multi-file diffs
3. WHEN converting to TOON_Format, THE TOON_Engine SHALL produce output that is at most 70% the size of the original structured block
4. WHEN compressing messages, THE TOON_Engine SHALL preserve the message count and role assignments unchanged
5. WHEN a message content is shorter than `toon_min_block_size` bytes, THE TOON_Engine SHALL leave it unchanged
6. THE TOON_Engine SHALL ensure detected structured blocks do not overlap in character range

### Requirement 5: TOON Re-Hydration

**User Story:** As a developer, I want the proxy to convert simplified model responses back to the JSON format Cursor expects, so that my IDE renders responses correctly.

#### Acceptance Criteria

1. WHEN a response contains TOON_Format content, THE TOON_Engine SHALL re-hydrate it back to the original JSON structure
2. FOR ALL valid structured blocks, compressing to TOON_Format then re-hydrating SHALL produce content equivalent to the original (round-trip property)
3. IF re-hydration of a TOON block fails, THEN THE TOON_Engine SHALL skip that block and forward the original content unchanged

### Requirement 6: Context Chunking and Embedding

**User Story:** As a developer, I want the proxy to split my context into chunks and generate embeddings, so that only the most relevant parts are sent to the model.

#### Acceptance Criteria

1. WHEN processing context messages, THE Retrieval_Layer SHALL split them into chunks of the configured `chunk_size_tokens` with `chunk_overlap_tokens` overlap
2. WHEN chunks are created, THE Retrieval_Layer SHALL generate embeddings via the configured LM_Studio endpoint
3. WHEN embeddings are generated, THE Retrieval_Layer SHALL produce vectors with consistent dimensionality matching the Qdrant collection configuration
4. THE Retrieval_Layer SHALL preserve system messages and the latest user message unchanged during context reduction
5. WHEN embedding generation returns results, THE Retrieval_Layer SHALL ensure no NaN or Inf values exist in the embedding vectors

### Requirement 7: Vector Storage and Retrieval

**User Story:** As a developer, I want the proxy to store and retrieve context vectors locally via Qdrant, so that relevant code snippets are found quickly without external network calls.

#### Acceptance Criteria

1. WHEN chunks are embedded, THE Retrieval_Layer SHALL upsert them to Qdrant using MessagePack serialization
2. WHEN a user query is received, THE Retrieval_Layer SHALL retrieve the top-k most semantically relevant chunks from Qdrant
3. WHEN assembling reduced context, THE Retrieval_Layer SHALL ensure the total token count of retrieved content does not exceed `retrieval_max_tokens`
4. WHEN retrieving chunks, THE Retrieval_Layer SHALL only include chunks with a relevance score above 0.5
5. THE Retrieval_Layer SHALL communicate with Qdrant exclusively over the localhost interface (127.0.0.1)

### Requirement 8: Outbound Secret Redaction

**User Story:** As a developer, I want the proxy to detect and redact secrets from my code before it reaches external servers, so that my API keys and credentials are never leaked.

#### Acceptance Criteria

1. WHEN outbound messages pass through the Security_Gateway with redaction enabled, THE Security_Gateway SHALL scan for patterns matching API keys, AWS keys, SSH keys, environment variables, JWT tokens, and GitHub tokens
2. WHEN a secret pattern is matched, THE Security_Gateway SHALL replace it with `[REDACTED]`
3. WHEN scanning for secrets, THE Security_Gateway SHALL detect high-entropy substrings (Shannon entropy >= configured threshold) of 16 or more characters
4. WHEN redacting content, THE Security_Gateway SHALL not mutate the original payload (returns a copy)
5. THE Security_Gateway SHALL perform all redaction before any data is transmitted to the DeepSeek API
6. WHEN redaction is performed, THE Security_Gateway SHALL record the count and type of redactions in the audit log

### Requirement 9: Inbound CVE Scanning

**User Story:** As a developer, I want the proxy to scan AI-generated code for common vulnerabilities, so that I am warned about security issues before incorporating suggested code.

#### Acceptance Criteria

1. WHEN a response contains code blocks and CVE scanning is enabled, THE Security_Gateway SHALL extract and scan each code block for vulnerabilities
2. WHEN scanning code, THE Security_Gateway SHALL use the local LM_Studio model without making external network calls
3. WHEN a vulnerability is detected, THE Security_Gateway SHALL produce a CVE_Finding with a valid type, severity level, code snippet, line range, and recommendation
4. WHEN vulnerabilities are found, THE Security_Gateway SHALL annotate the response with findings for the developer

### Requirement 10: Audit Logging

**User Story:** As a developer, I want all proxy transactions logged locally, so that I can review what data was processed and what security actions were taken.

#### Acceptance Criteria

1. WHILE `audit_logging_enabled` is True, THE Security_Gateway SHALL write exactly one Audit_Entry to the SQLite database for each processed request
2. WHEN writing an Audit_Entry, THE Security_Gateway SHALL include a valid timestamp, direction (outbound/inbound), request hash, model used, token counts, redaction count, CVE finding count, and status
3. THE Security_Gateway SHALL store the audit database with owner-only read/write permissions (0o600)
4. THE Security_Gateway SHALL not store any secret content in the audit log (only metadata and counts)
5. IF the audit database becomes corrupted or full, THEN THE Security_Gateway SHALL continue processing requests without audit logging and emit a warning

### Requirement 11: Pipeline Orchestration

**User Story:** As a developer, I want the proxy modules to execute in a defined order with proper error propagation, so that the system behaves predictably and degrades gracefully.

#### Acceptance Criteria

1. WHEN processing an outbound request, THE Pipeline_Orchestrator SHALL execute modules in order: Fidelity_Module, Security_Gateway (outbound), TOON_Engine (compress), Retrieval_Layer
2. WHEN processing an inbound response, THE Pipeline_Orchestrator SHALL execute modules in order: Stream Health Monitor, TOON_Engine (re-hydrate), Security_Gateway (inbound scan)
3. WHEN a pipeline phase is disabled in configuration, THE Pipeline_Orchestrator SHALL skip that phase and pass data to the next enabled phase
4. IF a pipeline phase fails, THEN THE Pipeline_Orchestrator SHALL skip the failing phase, log the error, and continue with remaining phases (graceful degradation)
5. WHEN processing messages through any pipeline stage, THE Pipeline_Orchestrator SHALL maintain valid OpenAI chat completion format (messages have `role` and either `content` or `tool_calls`)

### Requirement 12: Configuration Validation

**User Story:** As a developer, I want the proxy to validate my configuration at startup, so that I am informed of invalid settings before the system begins processing requests.

#### Acceptance Criteria

1. WHEN loading configuration, THE Proxy SHALL validate that `heartbeat_interval` is greater than 0 and at most 60 seconds
2. WHEN loading configuration, THE Proxy SHALL validate that `qdrant_url` and `embedding_url` are valid HTTP URLs
3. WHEN loading configuration, THE Proxy SHALL validate that `retrieval_top_k` is between 1 and 50 inclusive
4. WHEN loading configuration, THE Proxy SHALL validate that `retrieval_max_tokens` is between 100 and 10000 inclusive
5. WHEN loading configuration, THE Proxy SHALL validate that `entropy_threshold` is between 3.0 and 8.0 inclusive
6. WHEN loading configuration, THE Proxy SHALL validate that `toon_min_block_size` is at least 64 bytes
7. IF any configuration value fails validation, THEN THE Proxy SHALL report the specific validation error and refuse to start

### Requirement 13: Graceful Degradation

**User Story:** As a developer, I want the proxy to continue functioning when optional services (Qdrant, LM Studio) are unavailable, so that my workflow is not blocked by infrastructure issues.

#### Acceptance Criteria

1. IF Qdrant is unreachable, THEN THE Retrieval_Layer SHALL skip the retrieval phase and forward the full context to DeepSeek
2. IF LM_Studio embedding endpoint returns an error or times out, THEN THE Retrieval_Layer SHALL skip retrieval and use uncompressed context
3. IF TOON compression produces invalid output (fails re-hydration test), THEN THE TOON_Engine SHALL skip compression for that block and forward original content
4. WHEN a service becomes unavailable, THE Pipeline_Orchestrator SHALL log a warning and retry connection with exponential backoff up to a maximum of 5 minutes
5. THE Pipeline_Orchestrator SHALL expose a health check endpoint reporting the status of each component (Qdrant, LM_Studio, audit database)

### Requirement 14: MessagePack Transport Efficiency

**User Story:** As a developer, I want the proxy to use binary-efficient serialization for local service communication, so that overhead on the loopback interface is minimized.

#### Acceptance Criteria

1. WHEN communicating with Qdrant, THE Retrieval_Layer SHALL serialize payloads using MessagePack with `use_bin_type=True`
2. WHEN using MessagePack serialization, THE Retrieval_Layer SHALL produce payloads that are smaller than equivalent JSON representations
3. THE Retrieval_Layer SHALL only transmit MessagePack payloads over the localhost interface (no external network)
