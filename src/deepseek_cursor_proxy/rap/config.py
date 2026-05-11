"""RAP configuration model with validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


class ValidationError(Exception):
    """Raised when a configuration value fails validation.

    Attributes:
        field: The name of the configuration field that failed validation.
        reason: A human-readable explanation of why validation failed.
    """

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"Invalid configuration for '{field}': {reason}")


def _validate_url(value: str, field_name: str) -> None:
    """Validate that a string is a valid HTTP or HTTPS URL."""
    try:
        parsed = urlparse(value)
    except Exception:
        raise ValidationError(field_name, "not a valid URL")
    if parsed.scheme not in ("http", "https"):
        raise ValidationError(
            field_name, f"URL scheme must be 'http' or 'https', got '{parsed.scheme}'"
        )
    if not parsed.hostname:
        raise ValidationError(field_name, "URL must have a hostname")


@dataclass(frozen=True)
class RAPConfig:
    """Extended proxy configuration for RAP features.

    All fields are validated on construction via ``__post_init__``.
    A ``ValidationError`` is raised with the offending field name and
    reason if any value is outside its allowed range.
    """

    # Existing proxy config (inherited)
    host: str = "127.0.0.1"
    port: int = 9000
    upstream_base_url: str = "https://api.deepseek.com"
    upstream_model: str = "deepseek-v4-flash"

    # Fidelity settings
    spoof_pro_headers: bool = True
    heartbeat_interval: float = 15.0
    reasoning_passthrough: bool = True

    # TOON settings
    toon_compression_enabled: bool = True
    toon_rehydration_enabled: bool = True
    toon_min_block_size: int = 256

    # Retrieval settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rap_context"
    embedding_url: str = "http://localhost:1234/v1/embeddings"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5-embedding"
    retrieval_top_k: int = 5
    retrieval_max_tokens: int = 1000
    use_msgpack: bool = True

    # Security settings
    redaction_enabled: bool = True
    cve_scanning_enabled: bool = True
    audit_db_path: Path = field(
        default_factory=lambda: Path("~/.deepseek-cursor-proxy/audit.sqlite3")
    )
    entropy_threshold: float = 4.5
    security_model_url: str = "http://localhost:1234/v1/chat/completions"
    security_model_name: str = "ibm-grok4-ultrafast-coder-1b"

    # Phase control
    phase_bridge: bool = True
    phase_compression: bool = True
    phase_retrieval: bool = True
    phase_security: bool = True

    def __post_init__(self) -> None:
        """Validate all configuration fields after construction."""
        # heartbeat_interval: (0, 60]
        if self.heartbeat_interval <= 0 or self.heartbeat_interval > 60:
            raise ValidationError(
                "heartbeat_interval",
                "must be greater than 0 and at most 60 seconds",
            )

        # retrieval_top_k: [1, 50]
        if self.retrieval_top_k < 1 or self.retrieval_top_k > 50:
            raise ValidationError(
                "retrieval_top_k",
                "must be between 1 and 50 inclusive",
            )

        # retrieval_max_tokens: [100, 10000]
        if self.retrieval_max_tokens < 100 or self.retrieval_max_tokens > 10000:
            raise ValidationError(
                "retrieval_max_tokens",
                "must be between 100 and 10000 inclusive",
            )

        # entropy_threshold: [3.0, 8.0]
        if self.entropy_threshold < 3.0 or self.entropy_threshold > 8.0:
            raise ValidationError(
                "entropy_threshold",
                "must be between 3.0 and 8.0 inclusive",
            )

        # toon_min_block_size: >= 64
        if self.toon_min_block_size < 64:
            raise ValidationError(
                "toon_min_block_size",
                "must be at least 64 bytes",
            )

        # URL validations
        _validate_url(self.qdrant_url, "qdrant_url")
        _validate_url(self.embedding_url, "embedding_url")
