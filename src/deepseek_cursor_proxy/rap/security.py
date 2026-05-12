"""Security Gateway for the RAP middleware stack.

Protects sensitive data from leaving the local machine by scanning outbound
requests for secrets and redacting them before transmission. Also scans
inbound AI-generated code for common vulnerabilities via local LM Studio.
Provides audit logging of all transactions to a local SQLite database.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3, 10.4, 10.5
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# Precompiled patterns for common secret formats
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("api_key", re.compile(r"(?:sk|pk|api[_\-]?key)[_\-][\w]{20,}", re.IGNORECASE)),
    ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("ssh_key", re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----")),
    ("env_var", re.compile(r"(?:export\s+)?[A-Z_]{2,}=['\"]?[\w/+=]{8,}['\"]?")),
    ("jwt_token", re.compile(
        r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+"
    )),
    ("github_token", re.compile(r"gh[ps]_[A-Za-z0-9_]{36,}")),
]


@dataclass(frozen=True)
class SecurityConfig:
    """Configuration for the Security Gateway.

    Attributes:
        redaction_enabled: Whether outbound secret redaction is active.
        cve_scanning_enabled: Whether inbound CVE scanning is active.
        audit_logging_enabled: Whether audit logging is active.
        audit_db_path: Path to the SQLite audit database.
        local_security_model_url: URL for the local LM Studio security model.
        entropy_threshold: Shannon entropy threshold for high-entropy detection.
        redaction_patterns: Additional custom redaction patterns.
    """

    redaction_enabled: bool = True
    cve_scanning_enabled: bool = False
    audit_logging_enabled: bool = True
    audit_db_path: str = "~/.deepseek-cursor-proxy/audit.sqlite3"
    local_security_model_url: str = "http://localhost:1234/v1/chat/completions"
    security_model_name: str = ""
    entropy_threshold: float = 4.5
    redaction_patterns: list[str] = field(default_factory=list)


@dataclass
class Redaction:
    """Record of a single redaction performed on outbound content.

    Attributes:
        pattern_name: Name of the pattern that matched (e.g. 'api_key').
        original_length: Length of the original matched text.
        position: Start and end offsets of the match in the content.
        replacement: The replacement text used.
    """

    pattern_name: str
    original_length: int
    position: tuple[int, int]
    replacement: str = "[REDACTED]"


@dataclass
class CVEFinding:
    """A detected vulnerability in AI-generated code.

    Attributes:
        cve_type: Type of vulnerability (e.g. 'sql_injection', 'buffer_overflow').
        severity: Severity level ('low', 'medium', 'high', 'critical').
        code_snippet: The code fragment containing the vulnerability.
        line_range: Start and end line numbers of the vulnerable code.
        recommendation: Suggested fix or mitigation.
    """

    cve_type: str
    severity: str
    code_snippet: str
    line_range: tuple[int, int]
    recommendation: str


@dataclass
class AuditEntry:
    """A record of a proxy transaction for audit logging.

    Contains only metadata and counts — no secret content is stored.

    Attributes:
        timestamp: Unix timestamp of the transaction.
        direction: Either 'outbound' or 'inbound'.
        request_hash: Hash identifying the request (no secret content).
        redactions_count: Number of redactions performed.
        cve_findings_count: Number of CVE findings detected.
        model_used: The model name used for the request.
        token_count: Token count for the transaction.
        status: Transaction status ('success', 'error', 'redacted').
        metadata: Optional JSON-serializable dict for additional stats
                  (e.g. compression before/after token counts).
    """

    timestamp: float
    direction: str  # "outbound" | "inbound"
    request_hash: str
    redactions_count: int
    cve_findings_count: int
    model_used: str
    token_count: int
    status: str = "success"  # "success" | "error" | "redacted"
    metadata: str | None = None


# SQL schema for the audit log database
AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('outbound', 'inbound')),
    request_hash TEXT NOT NULL,
    model_used TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    redactions_count INTEGER DEFAULT 0,
    cve_findings_count INTEGER DEFAULT 0,
    status TEXT NOT NULL CHECK(status IN ('success', 'error', 'redacted')),
    metadata TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_direction ON audit_log(direction);
CREATE INDEX IF NOT EXISTS idx_audit_status ON audit_log(status);
"""


# Static CVE patterns — fast-path detection without LLM call
STATIC_CVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("code_injection", re.compile(r"\b(exec|eval|compile)\s*\(")),
    ("sql_injection", re.compile(
        r"cursor\.execute\(f['\"]|\.execute\(['\"].*\+|\.raw\(|\.query\(.*\+.*user"
    )),
    ("command_injection", re.compile(
        r"os\.system\(|subprocess\.(call|Popen|run)\(.*shell=True"
    )),
    ("hardcoded_credential", re.compile(
        r"password\s*=\s*['\"][^'\"]+['\"]"
        r"|secret\s*=\s*['\"][^'\"]+['\"]"
        r"|api_key\s*=\s*['\"][^'\"]+['\"]"
    )),
    ("pickle_deserialization", re.compile(r"pickle\.loads\(")),
    ("path_traversal", re.compile(
        r"os\.path\.join\(.*request|open\(.*request|Path\(.*request"
    )),
    ("insecure_hash", re.compile(r"\b(md5|sha1)\s*\(")),
]


def _check_static_cve_patterns(code: str) -> list[CVEFinding]:
    """Check a code block against static CVE patterns.

    Runs regex patterns on the code. For each match, creates a CVEFinding
    with the matching line range. This fast-path runs before any LLM call.
    """
    findings: list[CVEFinding] = []
    code_lines = code.split("\n")
    total_lines = len(code_lines)

    for pattern_name, pattern in STATIC_CVE_PATTERNS:
        for match in pattern.finditer(code):
            # Compute line range for this match
            match_start = match.start()
            line_start = code[:match_start].count("\n") + 1
            line_end = line_start
            # Extend to include any multi-line match
            match_end = match.end()
            line_end = code[:match_end].count("\n") + 1

            line_start = max(1, min(total_lines, line_start))
            line_end = max(line_start, min(total_lines, line_end))

            # Severity based on pattern type
            severity_map = {
                "code_injection": "critical",
                "sql_injection": "critical",
                "command_injection": "critical",
                "hardcoded_credential": "high",
                "pickle_deserialization": "high",
                "path_traversal": "high",
                "insecure_hash": "medium",
            }
            recommendation_map = {
                "code_injection": "Avoid using exec/eval/compile with untrusted input. Use safe alternatives like ast.literal_eval().",
                "sql_injection": "Use parameterized queries or an ORM. Never interpolate user input directly into SQL.",
                "command_injection": "Avoid shell=True in subprocess calls. Use subprocess.run with a list of arguments.",
                "hardcoded_credential": "Use environment variables or a secret manager for credentials.",
                "pickle_deserialization": "Avoid pickle.loads() on untrusted data. Use a safe serialization format like JSON.",
                "path_traversal": "Validate and sanitize file paths. Use os.path.realpath() and check for path traversal.",
                "insecure_hash": "Use SHA-256 or stronger hashing. MD5 and SHA-1 are cryptographically broken.",
            }

            snippet_lines = code_lines[line_start - 1 : line_end]
            code_snippet = "\n".join(snippet_lines)

            findings.append(CVEFinding(
                cve_type=pattern_name,
                severity=severity_map.get(pattern_name, "medium"),
                code_snippet=code_snippet,
                line_range=(line_start, line_end),
                recommendation=recommendation_map.get(
                    pattern_name, "Review this code for security issues."
                ),
            ))

    return findings


class SecurityGateway:
    """Scans and redacts secrets from outbound requests.

    The gateway uses regex-based pattern matching and Shannon entropy
    analysis to detect potential secrets in message content. All matches
    are replaced with [REDACTED] in a deep copy of the payload — the
    original is never mutated.

    Also provides audit logging of all transactions to a local SQLite
    database with owner-only permissions.
    """

    def __init__(self, config: SecurityConfig) -> None:
        self._config = config
        self._db_initialized = False
        self._db_conn: sqlite3.Connection | None = None

    @property
    def config(self) -> SecurityConfig:
        """Return the current security configuration."""
        return self._config

    def log_transaction(self, transaction: AuditEntry) -> None:
        """Write an audit entry to the local SQLite database.

        Only stores metadata and counts — no secret content is logged.
        If the database is corrupted or full, continues processing without
        logging and emits a warning.

        Requirements:
            10.1 — Writes exactly one AuditEntry per processed request
            10.2 — Includes timestamp, direction, request_hash, model_used,
                    token_count, redactions_count, cve_findings_count, status
            10.3 — Database file has owner-only permissions (0o600)
            10.4 — No secret content stored (only metadata and counts)
            10.5 — Handles corruption/full gracefully (continue without logging)

        Args:
            transaction: The AuditEntry to write to the database.
        """
        if not self._config.audit_logging_enabled:
            return

        try:
            conn = self._get_db_connection()
            conn.execute(
                """INSERT INTO audit_log
                   (timestamp, direction, request_hash, model_used,
                    token_count, redactions_count, cve_findings_count,
                    status, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    transaction.timestamp,
                    transaction.direction,
                    transaction.request_hash,
                    transaction.model_used,
                    transaction.token_count,
                    transaction.redactions_count,
                    transaction.cve_findings_count,
                    transaction.status,
                    transaction.metadata,
                ),
            )
            conn.commit()
        except (sqlite3.DatabaseError, OSError) as exc:
            # Requirement 10.5: continue without logging, emit warning
            logger.warning(
                "Audit logging failed (continuing without logging): %s", exc
            )
            # Reset connection so next attempt tries fresh
            self._close_db_connection()

    def _get_db_connection(self) -> sqlite3.Connection:
        """Get or create the SQLite database connection.

        Creates the database file and schema on first use.
        Sets file permissions to 0o600 (owner-only read/write).

        Returns:
            An open sqlite3.Connection.

        Raises:
            sqlite3.DatabaseError: If the database is corrupted.
            OSError: If file operations fail.
        """
        if self._db_conn is not None:
            return self._db_conn

        db_path = Path(self._config.audit_db_path).expanduser()

        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create/open the database
        # check_same_thread=False is needed because the proxy uses ThreadingHTTPServer
        # and requests come from different threads
        self._db_conn = sqlite3.connect(str(db_path), check_same_thread=False)

        # Create schema
        self._db_conn.executescript(AUDIT_SCHEMA)

        # Set file permissions to 0o600 (owner-only read/write)
        # Requirement 10.3
        os.chmod(db_path, 0o600)

        self._db_initialized = True
        return self._db_conn

    def _close_db_connection(self) -> None:
        """Close the database connection and reset state."""
        if self._db_conn is not None:
            try:
                self._db_conn.close()
            except Exception:
                pass
            self._db_conn = None
            self._db_initialized = False

    def scan_outbound(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], list[Redaction]]:
        """Scan and redact secrets from an outbound request payload.

        This method performs a deep copy of the payload, then scans all
        message content strings for secret patterns and high-entropy
        substrings. Matches are replaced with [REDACTED].

        Requirements:
            8.1 — Scans for API keys, AWS keys, SSH keys, env vars,
                   JWT tokens, GitHub tokens
            8.2 — Replaces matches with [REDACTED]
            8.3 — Detects high-entropy substrings (Shannon >= threshold, 16+ chars)
            8.4 — Does not mutate the original payload (returns a copy)
            8.5 — Performs redaction before data is transmitted

        Args:
            payload: The outbound request dict containing a 'messages' list.

        Returns:
            A tuple of (redacted_payload_copy, list_of_redactions).
        """
        if not self._config.redaction_enabled:
            return copy.deepcopy(payload), []

        redactions: list[Redaction] = []
        result = copy.deepcopy(payload)

        for i, message in enumerate(result.get("messages", [])):
            content = message.get("content", "")
            if not isinstance(content, str):
                continue

            content, msg_redactions = self._redact_content(content)
            redactions.extend(msg_redactions)
            result["messages"][i]["content"] = content

        return result, redactions

    def scan_inbound(
        self, response: dict[str, Any]
    ) -> tuple[dict[str, Any], list[CVEFinding]]:
        """Scan AI-generated code blocks in a response for vulnerabilities.

        Extracts code blocks from response choices[].message.content using
        regex, then calls the local LM Studio model for vulnerability analysis.
        Annotates the response with findings for developer visibility.

        Requirements:
            9.1 — Extracts and scans code blocks when CVE scanning is enabled
            9.2 — Uses local LM Studio model (no external network calls)
            9.3 — Produces CVEFinding with type, severity, snippet, line_range,
                   recommendation
            9.4 — Annotates response with findings for developer visibility

        Args:
            response: The inbound response dict (OpenAI chat completion format).

        Returns:
            A tuple of (annotated_response_copy, list_of_cve_findings).
        """
        if not self._config.cve_scanning_enabled:
            return copy.deepcopy(response), []

        result = copy.deepcopy(response)
        all_findings: list[CVEFinding] = []

        choices = result.get("choices", [])
        for i, choice in enumerate(choices):
            message = choice.get("message", {})
            content = message.get("content", "")
            if not isinstance(content, str):
                continue

            code_blocks = self._extract_code_blocks(content)
            if not code_blocks:
                continue

            for code_block in code_blocks:
                findings = self._analyze_code_block(code_block)
                all_findings.extend(findings)

            # Annotate the response with findings
            if all_findings:
                annotation = self._format_findings_annotation(all_findings)
                result["choices"][i]["message"]["content"] = (
                    content + "\n\n" + annotation
                )

        return result, all_findings

    def _extract_code_blocks(self, content: str) -> list[str]:
        """Extract fenced code blocks from markdown content.

        Matches ```...``` patterns (with optional language identifier).

        Args:
            content: The markdown content to extract code blocks from.

        Returns:
            List of code block contents (without the fence markers).
        """
        pattern = re.compile(r"```(?:\w*)\n?(.*?)```", re.DOTALL)
        return [match.group(1).strip() for match in pattern.finditer(content)]

    def _analyze_code_block(self, code: str) -> list[CVEFinding]:
        """Analyze a code block for vulnerabilities.

        Two-stage approach:
        1. Fast-path: Check static CVE patterns (regex-based, no model required)
        2. Fallback: If no static patterns matched AND a model is configured,
           call the local LM Studio model for deeper analysis.

        Args:
            code: The code snippet to analyze.

        Returns:
            List of CVEFinding instances for detected vulnerabilities.
        """
        # Stage 1: Static CVE pattern analysis (fast-path, no LLM needed)
        static_findings = _check_static_cve_patterns(code)
        if static_findings:
            logger.debug(
                "Static CVE patterns matched %d vulnerabilities (skipping LLM)",
                len(static_findings),
            )
            return static_findings

        # Stage 2: LLM-based analysis (only if model is configured)
        if not self._config.security_model_name:
            logger.debug(
                "No security model configured; skipping LLM-based CVE analysis"
            )
            return []

        prompt = (
            "Analyze the following code for security vulnerabilities. "
            "For each vulnerability found, respond with a JSON array of objects "
            "with these fields: cve_type (string like 'sql_injection', "
            "'buffer_overflow', 'hardcoded_credential', 'xss', 'path_traversal'), "
            "severity ('low', 'medium', 'high', 'critical'), "
            "line_start (int), line_end (int), recommendation (string). "
            "If no vulnerabilities are found, respond with an empty array [].\n\n"
            f"```\n{code}\n```"
        )

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    self._config.local_security_model_url,
                    json={
                        "model": self._config.security_model_name,
                        "messages": [
                            {"role": "system", "content": "You are a security code reviewer. Respond only with valid JSON."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.0,
                    },
                )
                resp.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException, OSError) as exc:
            logger.warning("LM Studio unavailable for CVE scanning: %s", exc)
            return []

        return self._parse_cve_response(resp.json(), code)

    def _parse_cve_response(
        self, response_data: dict[str, Any], code: str
    ) -> list[CVEFinding]:
        """Parse the LM Studio response into CVEFinding instances.

        Args:
            response_data: The JSON response from LM Studio.
            code: The original code snippet for context.

        Returns:
            List of CVEFinding instances.
        """
        findings: list[CVEFinding] = []

        try:
            content = (
                response_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            # Try to extract JSON from the response content
            vulnerabilities = json.loads(content)
            if not isinstance(vulnerabilities, list):
                return []
        except (json.JSONDecodeError, IndexError, KeyError):
            return []

        code_lines = code.split("\n")
        total_lines = len(code_lines)

        for vuln in vulnerabilities:
            if not isinstance(vuln, dict):
                continue

            cve_type = vuln.get("cve_type", "unknown")
            severity = vuln.get("severity", "medium")
            line_start = max(1, min(total_lines, int(vuln.get("line_start", 1))))
            line_end = max(line_start, min(total_lines, int(vuln.get("line_end", line_start))))
            recommendation = vuln.get("recommendation", "Review this code for security issues.")

            # Validate severity
            if severity not in ("low", "medium", "high", "critical"):
                severity = "medium"

            # Extract the relevant code snippet
            snippet_lines = code_lines[line_start - 1 : line_end]
            code_snippet = "\n".join(snippet_lines)

            findings.append(CVEFinding(
                cve_type=cve_type,
                severity=severity,
                code_snippet=code_snippet,
                line_range=(line_start, line_end),
                recommendation=recommendation,
            ))

        return findings

    def _format_findings_annotation(self, findings: list[CVEFinding]) -> str:
        """Format CVE findings as a developer-visible annotation.

        Args:
            findings: List of CVEFinding instances to format.

        Returns:
            A formatted string annotation to append to the response.
        """
        lines = ["⚠️ **Security Scan Results:**"]
        for i, finding in enumerate(findings, 1):
            lines.append(
                f"\n{i}. **[{finding.severity.upper()}] {finding.cve_type}** "
                f"(lines {finding.line_range[0]}-{finding.line_range[1]})\n"
                f"   Recommendation: {finding.recommendation}"
            )
        return "\n".join(lines)

    def _redact_content(self, content: str) -> tuple[str, list[Redaction]]:
        """Redact secrets from a single content string.

        Applies pattern-based redaction first, then entropy-based detection
        on the remaining content. Uses reverse-order replacement to preserve
        position accuracy.

        Args:
            content: The text content to scan.

        Returns:
            A tuple of (redacted_content, list_of_redactions).
        """
        redactions: list[Redaction] = []

        # Collect all pattern matches
        pattern_matches: list[tuple[str, int, int]] = []
        for pattern_name, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(content):
                pattern_matches.append((pattern_name, match.start(), match.end()))

        # Collect entropy matches (only on regions not already matched by patterns)
        entropy_matches = self._find_entropy_matches(content, pattern_matches)

        # Combine all matches and sort by start position descending
        # so we can replace from end to start without offset issues
        all_matches: list[tuple[str, int, int]] = pattern_matches + entropy_matches
        all_matches.sort(key=lambda m: m[1], reverse=True)

        # Remove overlapping matches (keep the one that starts first)
        # Since sorted descending, we process from end to start
        filtered = self._remove_overlaps(all_matches)

        # Apply replacements from end to start
        for pattern_name, start, end in filtered:
            redactions.append(Redaction(
                pattern_name=pattern_name,
                original_length=end - start,
                position=(start, end),
            ))
            content = content[:start] + "[REDACTED]" + content[end:]

        return content, redactions

    def _find_entropy_matches(
        self,
        content: str,
        pattern_matches: list[tuple[str, int, int]],
    ) -> list[tuple[str, int, int]]:
        """Find high-entropy substrings not already covered by pattern matches.

        Uses a sliding window approach to detect substrings of 16+ characters
        with Shannon entropy >= the configured threshold.

        Args:
            content: The text to scan.
            pattern_matches: Already-detected pattern matches to exclude.

        Returns:
            List of (pattern_name, start, end) tuples for entropy matches.
        """
        entropy_matches: list[tuple[str, int, int]] = []
        min_window = 16
        threshold = self._config.entropy_threshold

        if len(content) < min_window:
            return entropy_matches

        # Build a set of covered positions from pattern matches
        covered = set()
        for _, start, end in pattern_matches:
            covered.update(range(start, end))

        # Scan for high-entropy windows using word-boundary-aware approach
        # We look for contiguous non-whitespace tokens that are high entropy
        i = 0
        while i <= len(content) - min_window:
            # Skip positions already covered by pattern matches
            if i in covered:
                i += 1
                continue

            # Find the end of the current non-whitespace run
            run_end = i
            while run_end < len(content) and not content[run_end].isspace():
                run_end += 1

            run_length = run_end - i
            if run_length >= min_window:
                # Check entropy of this run
                substring = content[i:run_end]
                entropy = shannon_entropy(substring)
                if entropy >= threshold:
                    # Check it's not overlapping with pattern matches
                    if not any(pos in covered for pos in range(i, run_end)):
                        entropy_matches.append(("high_entropy", i, run_end))
                        # Skip past this match
                        i = run_end
                        continue

            # Move to next position
            if run_length >= min_window:
                i = run_end
            else:
                i += 1

        return entropy_matches

    def _remove_overlaps(
        self, matches: list[tuple[str, int, int]]
    ) -> list[tuple[str, int, int]]:
        """Remove overlapping matches, keeping earlier-starting ones.

        Args:
            matches: List of (pattern_name, start, end) sorted descending by start.

        Returns:
            Filtered list with no overlapping ranges, sorted descending by start.
        """
        if not matches:
            return []

        # Sort by start ascending for overlap detection
        sorted_asc = sorted(matches, key=lambda m: m[1])
        filtered: list[tuple[str, int, int]] = []

        for match in sorted_asc:
            _, start, end = match
            # Check if this overlaps with any already-accepted match
            overlaps = False
            for _, accepted_start, accepted_end in filtered:
                if start < accepted_end and end > accepted_start:
                    overlaps = True
                    break
            if not overlaps:
                filtered.append(match)

        # Return sorted descending by start for end-to-start replacement
        filtered.sort(key=lambda m: m[1], reverse=True)
        return filtered

    def detect_high_entropy(self, text: str) -> list[tuple[int, int, float]]:
        """Find high-entropy substrings in text.

        Scans the text for contiguous non-whitespace runs of 16+ characters
        with Shannon entropy >= the configured threshold.

        Args:
            text: The text to scan.

        Returns:
            List of (start, end, entropy) tuples for detected high-entropy spans.
        """
        results: list[tuple[int, int, float]] = []
        min_window = 16
        threshold = self._config.entropy_threshold

        if len(text) < min_window:
            return results

        i = 0
        while i <= len(text) - min_window:
            # Find the end of the current non-whitespace run
            run_end = i
            while run_end < len(text) and not text[run_end].isspace():
                run_end += 1

            run_length = run_end - i
            if run_length >= min_window:
                substring = text[i:run_end]
                entropy = shannon_entropy(substring)
                if entropy >= threshold:
                    results.append((i, run_end, entropy))
                    i = run_end
                    continue

            if run_length >= min_window:
                i = run_end
            else:
                i += 1

        return results


def shannon_entropy(text: str) -> float:
    """Calculate the Shannon entropy of a string.

    Measures the information density / randomness of the text.
    Higher values indicate more randomness (likely secrets).

    Preconditions:
        - text is a string (may be empty)

    Postconditions:
        - Returns float in range [0, log2(alphabet_size)]
        - Returns 0.0 for empty strings

    Args:
        text: The string to calculate entropy for.

    Returns:
        The Shannon entropy value as a float.
    """
    if not text:
        return 0.0

    freq: dict[str, int] = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )
