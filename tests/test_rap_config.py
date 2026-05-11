"""Unit tests for RAP configuration model and validation."""

from __future__ import annotations

import unittest

from deepseek_cursor_proxy.rap.config import RAPConfig, ValidationError


class TestRAPConfigDefaults(unittest.TestCase):
    """Test that RAPConfig can be constructed with all defaults."""

    def test_default_construction_succeeds(self) -> None:
        config = RAPConfig()
        self.assertEqual(config.heartbeat_interval, 15.0)
        self.assertEqual(config.retrieval_top_k, 5)
        self.assertEqual(config.retrieval_max_tokens, 1000)
        self.assertEqual(config.entropy_threshold, 4.5)
        self.assertEqual(config.toon_min_block_size, 256)
        self.assertEqual(config.qdrant_url, "http://localhost:6333")
        self.assertEqual(config.embedding_url, "http://localhost:1234/v1/embeddings")

    def test_config_is_frozen(self) -> None:
        config = RAPConfig()
        with self.assertRaises(AttributeError):
            config.heartbeat_interval = 10.0  # type: ignore[misc]


class TestHeartbeatIntervalValidation(unittest.TestCase):
    """Requirement 12.1: heartbeat_interval must be > 0 and <= 60."""

    def test_zero_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(heartbeat_interval=0.0)
        self.assertEqual(ctx.exception.field, "heartbeat_interval")

    def test_negative_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(heartbeat_interval=-1.0)
        self.assertEqual(ctx.exception.field, "heartbeat_interval")

    def test_above_60_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(heartbeat_interval=60.1)
        self.assertEqual(ctx.exception.field, "heartbeat_interval")

    def test_exactly_60_accepted(self) -> None:
        config = RAPConfig(heartbeat_interval=60.0)
        self.assertEqual(config.heartbeat_interval, 60.0)

    def test_small_positive_accepted(self) -> None:
        config = RAPConfig(heartbeat_interval=0.1)
        self.assertEqual(config.heartbeat_interval, 0.1)


class TestRetrievalTopKValidation(unittest.TestCase):
    """Requirement 12.3: retrieval_top_k must be between 1 and 50 inclusive."""

    def test_zero_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(retrieval_top_k=0)
        self.assertEqual(ctx.exception.field, "retrieval_top_k")

    def test_negative_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(retrieval_top_k=-5)
        self.assertEqual(ctx.exception.field, "retrieval_top_k")

    def test_above_50_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(retrieval_top_k=51)
        self.assertEqual(ctx.exception.field, "retrieval_top_k")

    def test_exactly_1_accepted(self) -> None:
        config = RAPConfig(retrieval_top_k=1)
        self.assertEqual(config.retrieval_top_k, 1)

    def test_exactly_50_accepted(self) -> None:
        config = RAPConfig(retrieval_top_k=50)
        self.assertEqual(config.retrieval_top_k, 50)


class TestRetrievalMaxTokensValidation(unittest.TestCase):
    """Requirement 12.4: retrieval_max_tokens must be between 100 and 10000."""

    def test_below_100_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(retrieval_max_tokens=99)
        self.assertEqual(ctx.exception.field, "retrieval_max_tokens")

    def test_above_10000_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(retrieval_max_tokens=10001)
        self.assertEqual(ctx.exception.field, "retrieval_max_tokens")

    def test_exactly_100_accepted(self) -> None:
        config = RAPConfig(retrieval_max_tokens=100)
        self.assertEqual(config.retrieval_max_tokens, 100)

    def test_exactly_10000_accepted(self) -> None:
        config = RAPConfig(retrieval_max_tokens=10000)
        self.assertEqual(config.retrieval_max_tokens, 10000)


class TestEntropyThresholdValidation(unittest.TestCase):
    """Requirement 12.5: entropy_threshold must be between 3.0 and 8.0."""

    def test_below_3_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(entropy_threshold=2.9)
        self.assertEqual(ctx.exception.field, "entropy_threshold")

    def test_above_8_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(entropy_threshold=8.1)
        self.assertEqual(ctx.exception.field, "entropy_threshold")

    def test_exactly_3_accepted(self) -> None:
        config = RAPConfig(entropy_threshold=3.0)
        self.assertEqual(config.entropy_threshold, 3.0)

    def test_exactly_8_accepted(self) -> None:
        config = RAPConfig(entropy_threshold=8.0)
        self.assertEqual(config.entropy_threshold, 8.0)


class TestToonMinBlockSizeValidation(unittest.TestCase):
    """Requirement 12.6: toon_min_block_size must be >= 64."""

    def test_below_64_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(toon_min_block_size=63)
        self.assertEqual(ctx.exception.field, "toon_min_block_size")

    def test_zero_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(toon_min_block_size=0)
        self.assertEqual(ctx.exception.field, "toon_min_block_size")

    def test_exactly_64_accepted(self) -> None:
        config = RAPConfig(toon_min_block_size=64)
        self.assertEqual(config.toon_min_block_size, 64)

    def test_large_value_accepted(self) -> None:
        config = RAPConfig(toon_min_block_size=1024)
        self.assertEqual(config.toon_min_block_size, 1024)


class TestURLValidation(unittest.TestCase):
    """Requirement 12.2: qdrant_url and embedding_url must be valid HTTP URLs."""

    def test_invalid_qdrant_url_no_scheme(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(qdrant_url="not-a-url")
        self.assertEqual(ctx.exception.field, "qdrant_url")

    def test_invalid_qdrant_url_ftp_scheme(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(qdrant_url="ftp://localhost:6333")
        self.assertEqual(ctx.exception.field, "qdrant_url")

    def test_invalid_embedding_url_no_scheme(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(embedding_url="localhost:1234")
        self.assertEqual(ctx.exception.field, "embedding_url")

    def test_invalid_embedding_url_empty(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RAPConfig(embedding_url="")
        self.assertEqual(ctx.exception.field, "embedding_url")

    def test_valid_http_url_accepted(self) -> None:
        config = RAPConfig(qdrant_url="http://192.168.1.100:6333")
        self.assertEqual(config.qdrant_url, "http://192.168.1.100:6333")

    def test_valid_https_url_accepted(self) -> None:
        config = RAPConfig(embedding_url="https://localhost:1234/v1/embeddings")
        self.assertEqual(config.embedding_url, "https://localhost:1234/v1/embeddings")


class TestValidationErrorAttributes(unittest.TestCase):
    """Requirement 12.7: ValidationError includes field name and reason."""

    def test_error_has_field_and_reason(self) -> None:
        try:
            RAPConfig(heartbeat_interval=0.0)
            self.fail("Expected ValidationError")
        except ValidationError as e:
            self.assertEqual(e.field, "heartbeat_interval")
            self.assertIn("greater than 0", e.reason)
            # The string representation includes both field and reason
            self.assertIn("heartbeat_interval", str(e))

    def test_error_is_exception(self) -> None:
        with self.assertRaises(Exception):
            RAPConfig(retrieval_top_k=0)


if __name__ == "__main__":
    unittest.main()
