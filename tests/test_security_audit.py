"""Unit tests for the Security Gateway — audit logging with SQLite.

Tests audit entry writing, schema creation, file permissions,
no-secret-content guarantee, and graceful handling of database errors.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import os
import sqlite3
import stat
import time
from pathlib import Path

import pytest

from deepseek_cursor_proxy.rap.security import (
    AUDIT_SCHEMA,
    AuditEntry,
    SecurityConfig,
    SecurityGateway,
)


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    """Return a temporary database path for testing."""
    return str(tmp_path / "test_audit.sqlite3")


@pytest.fixture
def gateway(tmp_db_path: str) -> SecurityGateway:
    """Create a SecurityGateway with audit logging enabled."""
    config = SecurityConfig(
        audit_logging_enabled=True,
        audit_db_path=tmp_db_path,
    )
    return SecurityGateway(config)


@pytest.fixture
def sample_entry() -> AuditEntry:
    """Create a sample AuditEntry for testing."""
    return AuditEntry(
        timestamp=time.time(),
        direction="outbound",
        request_hash="abc123def456",
        redactions_count=2,
        cve_findings_count=0,
        model_used="deepseek-v4-flash",
        token_count=1500,
        status="success",
    )


class TestAuditEntry:
    """Tests for the AuditEntry dataclass."""

    def test_default_status_is_success(self) -> None:
        """AuditEntry defaults to 'success' status."""
        entry = AuditEntry(
            timestamp=1000.0,
            direction="outbound",
            request_hash="hash123",
            redactions_count=0,
            cve_findings_count=0,
            model_used="test-model",
            token_count=100,
        )
        assert entry.status == "success"

    def test_all_fields_stored(self) -> None:
        """All fields are accessible on the dataclass."""
        entry = AuditEntry(
            timestamp=1234.5,
            direction="inbound",
            request_hash="xyz789",
            redactions_count=3,
            cve_findings_count=1,
            model_used="deepseek-v4",
            token_count=2000,
            status="redacted",
        )
        assert entry.timestamp == 1234.5
        assert entry.direction == "inbound"
        assert entry.request_hash == "xyz789"
        assert entry.redactions_count == 3
        assert entry.cve_findings_count == 1
        assert entry.model_used == "deepseek-v4"
        assert entry.token_count == 2000
        assert entry.status == "redacted"


class TestLogTransaction:
    """Tests for SecurityGateway.log_transaction()."""

    def test_writes_entry_to_database(
        self, gateway: SecurityGateway, sample_entry: AuditEntry, tmp_db_path: str
    ) -> None:
        """log_transaction writes exactly one row to the audit_log table."""
        gateway.log_transaction(sample_entry)

        conn = sqlite3.connect(tmp_db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM audit_log")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_stores_correct_fields(
        self, gateway: SecurityGateway, sample_entry: AuditEntry, tmp_db_path: str
    ) -> None:
        """All AuditEntry fields are stored correctly in the database."""
        gateway.log_transaction(sample_entry)

        conn = sqlite3.connect(tmp_db_path)
        cursor = conn.execute(
            "SELECT timestamp, direction, request_hash, model_used, "
            "token_count, redactions_count, cve_findings_count, status "
            "FROM audit_log"
        )
        row = cursor.fetchone()
        conn.close()

        assert row[0] == sample_entry.timestamp
        assert row[1] == sample_entry.direction
        assert row[2] == sample_entry.request_hash
        assert row[3] == sample_entry.model_used
        assert row[4] == sample_entry.token_count
        assert row[5] == sample_entry.redactions_count
        assert row[6] == sample_entry.cve_findings_count
        assert row[7] == sample_entry.status

    def test_multiple_entries(
        self, gateway: SecurityGateway, tmp_db_path: str
    ) -> None:
        """Multiple log_transaction calls write multiple rows."""
        for i in range(5):
            entry = AuditEntry(
                timestamp=1000.0 + i,
                direction="outbound" if i % 2 == 0 else "inbound",
                request_hash=f"hash_{i}",
                redactions_count=i,
                cve_findings_count=0,
                model_used="test-model",
                token_count=100 * i,
                status="success",
            )
            gateway.log_transaction(entry)

        conn = sqlite3.connect(tmp_db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM audit_log")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 5

    def test_disabled_logging_does_not_write(self, tmp_path: Path) -> None:
        """When audit_logging_enabled is False, nothing is written."""
        db_path = str(tmp_path / "disabled.sqlite3")
        config = SecurityConfig(
            audit_logging_enabled=False,
            audit_db_path=db_path,
        )
        gw = SecurityGateway(config)

        entry = AuditEntry(
            timestamp=time.time(),
            direction="outbound",
            request_hash="test",
            redactions_count=0,
            cve_findings_count=0,
            model_used="model",
            token_count=0,
        )
        gw.log_transaction(entry)

        # Database file should not even be created
        assert not Path(db_path).exists()

    def test_inbound_direction(
        self, gateway: SecurityGateway, tmp_db_path: str
    ) -> None:
        """Inbound direction is stored correctly."""
        entry = AuditEntry(
            timestamp=time.time(),
            direction="inbound",
            request_hash="inbound_hash",
            redactions_count=0,
            cve_findings_count=3,
            model_used="deepseek-v4",
            token_count=500,
            status="success",
        )
        gateway.log_transaction(entry)

        conn = sqlite3.connect(tmp_db_path)
        cursor = conn.execute("SELECT direction FROM audit_log")
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "inbound"


class TestDatabaseSchema:
    """Tests for the audit database schema and indexes."""

    def test_schema_creates_table(self, tmp_db_path: str) -> None:
        """AUDIT_SCHEMA creates the audit_log table."""
        conn = sqlite3.connect(tmp_db_path)
        conn.executescript(AUDIT_SCHEMA)

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_schema_creates_indexes(self, tmp_db_path: str) -> None:
        """AUDIT_SCHEMA creates the expected indexes."""
        conn = sqlite3.connect(tmp_db_path)
        conn.executescript(AUDIT_SCHEMA)

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "idx_audit_timestamp" in indexes
        assert "idx_audit_direction" in indexes
        assert "idx_audit_status" in indexes

    def test_schema_enforces_direction_check(self, tmp_db_path: str) -> None:
        """Schema CHECK constraint rejects invalid direction values."""
        conn = sqlite3.connect(tmp_db_path)
        conn.executescript(AUDIT_SCHEMA)

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO audit_log (timestamp, direction, request_hash, "
                "model_used, status) VALUES (1.0, 'invalid', 'hash', 'model', 'success')"
            )
        conn.close()

    def test_schema_enforces_status_check(self, tmp_db_path: str) -> None:
        """Schema CHECK constraint rejects invalid status values."""
        conn = sqlite3.connect(tmp_db_path)
        conn.executescript(AUDIT_SCHEMA)

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO audit_log (timestamp, direction, request_hash, "
                "model_used, status) VALUES (1.0, 'outbound', 'hash', 'model', 'bad')"
            )
        conn.close()

    def test_schema_is_idempotent(self, tmp_db_path: str) -> None:
        """Running AUDIT_SCHEMA multiple times does not error."""
        conn = sqlite3.connect(tmp_db_path)
        conn.executescript(AUDIT_SCHEMA)
        conn.executescript(AUDIT_SCHEMA)  # Should not raise
        conn.close()


class TestFilePermissions:
    """Tests for database file permissions (Requirement 10.3)."""

    def test_db_file_has_owner_only_permissions(
        self, gateway: SecurityGateway, sample_entry: AuditEntry, tmp_db_path: str
    ) -> None:
        """Database file is created with 0o600 permissions."""
        gateway.log_transaction(sample_entry)

        file_stat = os.stat(tmp_db_path)
        # Mask out file type bits, check only permission bits
        permissions = stat.S_IMODE(file_stat.st_mode)
        assert permissions == 0o600

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Parent directory is created if it does not exist."""
        db_path = str(tmp_path / "nested" / "dir" / "audit.sqlite3")
        config = SecurityConfig(
            audit_logging_enabled=True,
            audit_db_path=db_path,
        )
        gw = SecurityGateway(config)

        entry = AuditEntry(
            timestamp=time.time(),
            direction="outbound",
            request_hash="test",
            redactions_count=0,
            cve_findings_count=0,
            model_used="model",
            token_count=0,
        )
        gw.log_transaction(entry)

        assert Path(db_path).exists()


class TestNoSecretContent:
    """Tests ensuring no secret content is stored (Requirement 10.4)."""

    def test_no_message_content_in_db(
        self, gateway: SecurityGateway, tmp_db_path: str
    ) -> None:
        """The audit log does not store any message content or secrets."""
        # The AuditEntry only has metadata fields — no content field exists
        entry = AuditEntry(
            timestamp=time.time(),
            direction="outbound",
            request_hash="sha256_hash_of_request",
            redactions_count=5,
            cve_findings_count=0,
            model_used="deepseek-v4",
            token_count=3000,
            status="redacted",
        )
        gateway.log_transaction(entry)

        # Read all columns from the database
        conn = sqlite3.connect(tmp_db_path)
        cursor = conn.execute("SELECT * FROM audit_log")
        columns = [desc[0] for desc in cursor.description]
        conn.close()

        # Verify no column stores actual content/secrets
        secret_columns = {"content", "message", "payload", "body", "secret", "key"}
        assert not secret_columns.intersection(set(columns))


class TestGracefulDegradation:
    """Tests for graceful handling of database errors (Requirement 10.5)."""

    def test_corrupted_db_continues_without_logging(
        self, tmp_path: Path
    ) -> None:
        """If the database is corrupted, processing continues with a warning."""
        db_path = str(tmp_path / "corrupted.sqlite3")

        # Create a corrupted database file
        Path(db_path).write_bytes(b"this is not a valid sqlite database")

        config = SecurityConfig(
            audit_logging_enabled=True,
            audit_db_path=db_path,
        )
        gw = SecurityGateway(config)

        entry = AuditEntry(
            timestamp=time.time(),
            direction="outbound",
            request_hash="test",
            redactions_count=0,
            cve_findings_count=0,
            model_used="model",
            token_count=0,
        )

        # Should not raise — continues without logging
        gw.log_transaction(entry)

    def test_readonly_db_continues_without_logging(
        self, tmp_path: Path
    ) -> None:
        """If the database file is read-only, processing continues with a warning."""
        db_path = str(tmp_path / "readonly.sqlite3")

        # Create a valid database then make it read-only
        conn = sqlite3.connect(db_path)
        conn.executescript(AUDIT_SCHEMA)
        conn.close()
        os.chmod(db_path, 0o400)

        config = SecurityConfig(
            audit_logging_enabled=True,
            audit_db_path=db_path,
        )
        gw = SecurityGateway(config)

        entry = AuditEntry(
            timestamp=time.time(),
            direction="outbound",
            request_hash="test",
            redactions_count=0,
            cve_findings_count=0,
            model_used="model",
            token_count=0,
        )

        # Should not raise — continues without logging
        gw.log_transaction(entry)

        # Restore permissions for cleanup
        os.chmod(db_path, 0o600)

    def test_full_db_continues_without_logging(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the database is full, processing continues with a warning."""
        db_path = str(tmp_path / "full.sqlite3")
        config = SecurityConfig(
            audit_logging_enabled=True,
            audit_db_path=db_path,
        )
        gw = SecurityGateway(config)

        # First call to initialize the database
        entry = AuditEntry(
            timestamp=time.time(),
            direction="outbound",
            request_hash="init",
            redactions_count=0,
            cve_findings_count=0,
            model_used="model",
            token_count=0,
        )
        gw.log_transaction(entry)

        # Simulate a "database full" error by replacing _get_db_connection
        # with one that returns a connection whose execute raises
        class FakeConn:
            def execute(self, *args, **kwargs):
                raise sqlite3.OperationalError("database or disk is full")

            def commit(self, *args, **kwargs):
                pass

        monkeypatch.setattr(gw, "_get_db_connection", lambda: FakeConn())

        # Should not raise
        entry2 = AuditEntry(
            timestamp=time.time(),
            direction="inbound",
            request_hash="full_test",
            redactions_count=0,
            cve_findings_count=0,
            model_used="model",
            token_count=0,
        )
        gw.log_transaction(entry2)
