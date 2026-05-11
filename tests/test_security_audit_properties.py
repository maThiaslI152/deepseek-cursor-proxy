"""Property-based tests for the Security Gateway audit logging.

**Validates: Requirements 10.1, 10.2, 10.4**

Property 18: Audit Entry Completeness and No Secret Leakage
For any request processed with `audit_logging_enabled=True`, exactly one
Audit_Entry SHALL be written with all required fields (timestamp, direction,
request_hash, model_used, token counts, redaction count, CVE finding count,
status), and no field in the entry SHALL contain content matching secret
patterns.
"""

from __future__ import annotations

import sqlite3
import string
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from deepseek_cursor_proxy.rap.security import (
    SECRET_PATTERNS,
    AuditEntry,
    SecurityConfig,
    SecurityGateway,
)


# --- Strategies ---

# Valid directions for audit entries
valid_directions = st.sampled_from(["outbound", "inbound"])

# Valid statuses for audit entries
valid_statuses = st.sampled_from(["success", "error", "redacted"])

# Generate valid timestamps (positive floats)
valid_timestamps = st.floats(
    min_value=0.001,
    max_value=2_000_000_000.0,
    allow_nan=False,
    allow_infinity=False,
)

# Generate non-empty request hashes (alphanumeric, no secrets)
valid_request_hashes = st.text(
    alphabet=string.ascii_lowercase + string.digits,
    min_size=1,
    max_size=64,
)

# Generate model names (non-empty, no secrets)
valid_model_names = st.text(
    alphabet=string.ascii_lowercase + string.digits + "-_.",
    min_size=1,
    max_size=50,
).filter(lambda s: len(s.strip()) > 0)

# Generate non-negative token counts
valid_token_counts = st.integers(min_value=0, max_value=1_000_000)

# Generate non-negative redaction counts
valid_redaction_counts = st.integers(min_value=0, max_value=1000)

# Generate non-negative CVE finding counts
valid_cve_counts = st.integers(min_value=0, max_value=1000)

# Strategy for generating a valid AuditEntry
audit_entry_strategy = st.builds(
    AuditEntry,
    timestamp=valid_timestamps,
    direction=valid_directions,
    request_hash=valid_request_hashes,
    redactions_count=valid_redaction_counts,
    cve_findings_count=valid_cve_counts,
    model_used=valid_model_names,
    token_count=valid_token_counts,
    status=valid_statuses,
)

# Strategy for generating secret-like content that should NOT appear in audit DB
secret_content_strategy = st.one_of(
    # API key patterns
    st.builds(
        lambda suffix: f"sk_{suffix}",
        suffix=st.text(
            alphabet=string.ascii_letters + string.digits,
            min_size=20,
            max_size=40,
        ),
    ),
    # AWS key patterns
    st.builds(
        lambda suffix: f"AKIA{suffix}",
        suffix=st.text(
            alphabet=string.ascii_uppercase + string.digits,
            min_size=16,
            max_size=16,
        ),
    ),
    # GitHub token patterns
    st.builds(
        lambda suffix: f"ghp_{suffix}",
        suffix=st.text(
            alphabet=string.ascii_letters + string.digits,
            min_size=36,
            max_size=50,
        ),
    ),
)


# --- Property 18: Audit Entry Completeness and No Secret Leakage ---


class TestProperty18AuditEntryCompletenessAndNoSecretLeakage:
    """For any request processed with audit_logging_enabled=True, exactly one
    Audit_Entry SHALL be written with all required fields, and no field in the
    entry SHALL contain content matching secret patterns.

    **Validates: Requirements 10.1, 10.2, 10.4**
    """

    @given(entry=audit_entry_strategy)
    @settings(max_examples=30)
    def test_exactly_one_row_written_per_log_transaction(
        self, entry: AuditEntry
    ) -> None:
        """Each log_transaction() call writes exactly one row to the database.

        **Validates: Requirements 10.1**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "audit.sqlite3")
            config = SecurityConfig(
                audit_logging_enabled=True,
                audit_db_path=db_path,
            )
            gateway = SecurityGateway(config)

            gateway.log_transaction(entry)

            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM audit_log")
            count = cursor.fetchone()[0]
            conn.close()

            assert count == 1, (
                f"Expected exactly 1 row, got {count} after log_transaction()"
            )

    @given(entry=audit_entry_strategy)
    @settings(max_examples=30)
    def test_all_required_fields_present_and_valid(
        self, entry: AuditEntry
    ) -> None:
        """Every audit entry has valid timestamp (> 0), valid direction,
        non-empty request_hash, non-empty model_used, non-negative token_count,
        non-negative redactions_count, non-negative cve_findings_count, valid status.

        **Validates: Requirements 10.2**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "audit.sqlite3")
            config = SecurityConfig(
                audit_logging_enabled=True,
                audit_db_path=db_path,
            )
            gateway = SecurityGateway(config)

            gateway.log_transaction(entry)

            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT timestamp, direction, request_hash, model_used, "
                "token_count, redactions_count, cve_findings_count, status "
                "FROM audit_log"
            )
            row = cursor.fetchone()
            conn.close()

            assert row is not None, "No row found in audit_log"

            timestamp, direction, request_hash, model_used, token_count, \
                redactions_count, cve_findings_count, status = row

            # Validate timestamp > 0
            assert timestamp > 0, f"Timestamp must be > 0, got {timestamp}"

            # Validate direction is valid
            assert direction in ("outbound", "inbound"), (
                f"Invalid direction: {direction!r}"
            )

            # Validate request_hash is non-empty
            assert request_hash and len(request_hash) > 0, (
                f"request_hash must be non-empty, got {request_hash!r}"
            )

            # Validate model_used is non-empty
            assert model_used and len(model_used) > 0, (
                f"model_used must be non-empty, got {model_used!r}"
            )

            # Validate token_count is non-negative
            assert token_count >= 0, (
                f"token_count must be >= 0, got {token_count}"
            )

            # Validate redactions_count is non-negative
            assert redactions_count >= 0, (
                f"redactions_count must be >= 0, got {redactions_count}"
            )

            # Validate cve_findings_count is non-negative
            assert cve_findings_count >= 0, (
                f"cve_findings_count must be >= 0, got {cve_findings_count}"
            )

            # Validate status is valid
            assert status in ("success", "error", "redacted"), (
                f"Invalid status: {status!r}"
            )

    @given(entry=audit_entry_strategy)
    @settings(max_examples=30)
    def test_no_secret_patterns_in_stored_data(
        self, entry: AuditEntry
    ) -> None:
        """No field in the audit entry contains content matching secret patterns.

        **Validates: Requirements 10.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "audit.sqlite3")
            config = SecurityConfig(
                audit_logging_enabled=True,
                audit_db_path=db_path,
            )
            gateway = SecurityGateway(config)

            gateway.log_transaction(entry)

            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT * FROM audit_log")
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            conn.close()

            assert row is not None, "No row found in audit_log"

            # Check every field value for secret patterns
            for col_name, value in zip(columns, row):
                if value is None:
                    continue
                value_str = str(value)
                for pattern_name, pattern in SECRET_PATTERNS:
                    assert not pattern.search(value_str), (
                        f"Secret pattern '{pattern_name}' found in column "
                        f"'{col_name}': {value_str!r}"
                    )

    @given(
        entry=audit_entry_strategy,
        secret=secret_content_strategy,
    )
    @settings(max_examples=30)
    def test_no_secret_columns_in_schema(
        self, entry: AuditEntry, secret: str
    ) -> None:
        """The audit database schema has no columns that could store raw
        message content, API keys, passwords, or other secret material.
        The AuditEntry dataclass only contains metadata fields.

        **Validates: Requirements 10.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "audit.sqlite3")
            config = SecurityConfig(
                audit_logging_enabled=True,
                audit_db_path=db_path,
            )
            gateway = SecurityGateway(config)

            gateway.log_transaction(entry)

            # Verify the database schema has no content/message/payload columns
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT * FROM audit_log")
            columns = {desc[0] for desc in cursor.description}
            conn.close()

            # These columns should never exist in the audit schema
            forbidden_columns = {
                "content", "message", "payload", "body",
                "secret", "key", "password", "token_value",
            }
            leaked_columns = forbidden_columns.intersection(columns)
            assert not leaked_columns, (
                f"Audit schema contains forbidden columns that could leak "
                f"secrets: {leaked_columns}"
            )
