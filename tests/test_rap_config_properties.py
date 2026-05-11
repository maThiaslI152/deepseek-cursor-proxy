"""Property-based tests for RAP configuration validation.

**Validates: Requirements 12.1, 12.3, 12.4, 12.5, 12.6**

Property 22: Configuration Validation Correctness
For any numeric configuration value, the validator SHALL accept values within
the specified valid range and reject values outside it.
"""

from __future__ import annotations

import pytest
from hypothesis import given, assume
from hypothesis import strategies as st

from deepseek_cursor_proxy.rap.config import RAPConfig, ValidationError


# --- Strategies for valid values ---

valid_heartbeat_interval = st.floats(
    min_value=0.0, max_value=60.0, exclude_min=True
).filter(lambda x: x == x)  # exclude NaN

valid_retrieval_top_k = st.integers(min_value=1, max_value=50)

valid_retrieval_max_tokens = st.integers(min_value=100, max_value=10000)

valid_entropy_threshold = st.floats(
    min_value=3.0, max_value=8.0
).filter(lambda x: x == x)  # exclude NaN

valid_toon_min_block_size = st.integers(min_value=64, max_value=100000)


# --- Strategies for invalid values ---

invalid_heartbeat_interval = st.one_of(
    st.floats(max_value=0.0),  # <= 0
    st.floats(min_value=60.0, exclude_min=True),  # > 60
).filter(lambda x: x == x and not (x == float("inf") or x == float("-inf")))

invalid_retrieval_top_k = st.one_of(
    st.integers(max_value=0),  # < 1
    st.integers(min_value=51),  # > 50
)

invalid_retrieval_max_tokens = st.one_of(
    st.integers(max_value=99),  # < 100
    st.integers(min_value=10001),  # > 10000
)

invalid_entropy_threshold = st.one_of(
    st.floats(max_value=3.0, exclude_max=True),  # < 3.0
    st.floats(min_value=8.0, exclude_min=True),  # > 8.0
).filter(lambda x: x == x and not (x == float("inf") or x == float("-inf")))

invalid_toon_min_block_size = st.integers(max_value=63)  # < 64


class TestProperty22ValidValuesAccepted:
    """Valid values within ranges always construct successfully.

    **Validates: Requirements 12.1, 12.3, 12.4, 12.5, 12.6**
    """

    @given(value=valid_heartbeat_interval)
    def test_valid_heartbeat_interval_constructs(self, value: float) -> None:
        """Any heartbeat_interval in (0, 60] constructs successfully."""
        config = RAPConfig(heartbeat_interval=value)
        assert config.heartbeat_interval == value

    @given(value=valid_retrieval_top_k)
    def test_valid_retrieval_top_k_constructs(self, value: int) -> None:
        """Any retrieval_top_k in [1, 50] constructs successfully."""
        config = RAPConfig(retrieval_top_k=value)
        assert config.retrieval_top_k == value

    @given(value=valid_retrieval_max_tokens)
    def test_valid_retrieval_max_tokens_constructs(self, value: int) -> None:
        """Any retrieval_max_tokens in [100, 10000] constructs successfully."""
        config = RAPConfig(retrieval_max_tokens=value)
        assert config.retrieval_max_tokens == value

    @given(value=valid_entropy_threshold)
    def test_valid_entropy_threshold_constructs(self, value: float) -> None:
        """Any entropy_threshold in [3.0, 8.0] constructs successfully."""
        config = RAPConfig(entropy_threshold=value)
        assert config.entropy_threshold == value

    @given(value=valid_toon_min_block_size)
    def test_valid_toon_min_block_size_constructs(self, value: int) -> None:
        """Any toon_min_block_size >= 64 constructs successfully."""
        config = RAPConfig(toon_min_block_size=value)
        assert config.toon_min_block_size == value

    @given(
        heartbeat=valid_heartbeat_interval,
        top_k=valid_retrieval_top_k,
        max_tokens=valid_retrieval_max_tokens,
        entropy=valid_entropy_threshold,
        block_size=valid_toon_min_block_size,
    )
    def test_all_valid_values_together_construct(
        self,
        heartbeat: float,
        top_k: int,
        max_tokens: int,
        entropy: float,
        block_size: int,
    ) -> None:
        """Any combination of valid values constructs successfully."""
        config = RAPConfig(
            heartbeat_interval=heartbeat,
            retrieval_top_k=top_k,
            retrieval_max_tokens=max_tokens,
            entropy_threshold=entropy,
            toon_min_block_size=block_size,
        )
        assert config.heartbeat_interval == heartbeat
        assert config.retrieval_top_k == top_k
        assert config.retrieval_max_tokens == max_tokens
        assert config.entropy_threshold == entropy
        assert config.toon_min_block_size == block_size


class TestProperty22InvalidValuesRejected:
    """Invalid values outside ranges always raise ValidationError with correct field name.

    **Validates: Requirements 12.1, 12.3, 12.4, 12.5, 12.6**
    """

    @given(value=invalid_heartbeat_interval)
    def test_invalid_heartbeat_interval_raises(self, value: float) -> None:
        """Any heartbeat_interval outside (0, 60] raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RAPConfig(heartbeat_interval=value)
        assert exc_info.value.field == "heartbeat_interval"

    @given(value=invalid_retrieval_top_k)
    def test_invalid_retrieval_top_k_raises(self, value: int) -> None:
        """Any retrieval_top_k outside [1, 50] raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RAPConfig(retrieval_top_k=value)
        assert exc_info.value.field == "retrieval_top_k"

    @given(value=invalid_retrieval_max_tokens)
    def test_invalid_retrieval_max_tokens_raises(self, value: int) -> None:
        """Any retrieval_max_tokens outside [100, 10000] raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RAPConfig(retrieval_max_tokens=value)
        assert exc_info.value.field == "retrieval_max_tokens"

    @given(value=invalid_entropy_threshold)
    def test_invalid_entropy_threshold_raises(self, value: float) -> None:
        """Any entropy_threshold outside [3.0, 8.0] raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RAPConfig(entropy_threshold=value)
        assert exc_info.value.field == "entropy_threshold"

    @given(value=invalid_toon_min_block_size)
    def test_invalid_toon_min_block_size_raises(self, value: int) -> None:
        """Any toon_min_block_size < 64 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RAPConfig(toon_min_block_size=value)
        assert exc_info.value.field == "toon_min_block_size"


class TestProperty22BoundaryValues:
    """Boundary values are handled correctly at the edges of valid ranges.

    **Validates: Requirements 12.1, 12.3, 12.4, 12.5, 12.6**
    """

    @given(epsilon=st.floats(min_value=1e-10, max_value=1e-3))
    def test_heartbeat_just_above_zero_accepted(self, epsilon: float) -> None:
        """Values just above 0 are accepted for heartbeat_interval."""
        config = RAPConfig(heartbeat_interval=epsilon)
        assert config.heartbeat_interval == epsilon

    def test_heartbeat_exactly_60_accepted(self) -> None:
        """Boundary value 60.0 is accepted (inclusive upper bound)."""
        config = RAPConfig(heartbeat_interval=60.0)
        assert config.heartbeat_interval == 60.0

    @given(epsilon=st.floats(min_value=1e-10, max_value=1e-3))
    def test_heartbeat_just_above_60_rejected(self, epsilon: float) -> None:
        """Values just above 60 are rejected for heartbeat_interval."""
        with pytest.raises(ValidationError) as exc_info:
            RAPConfig(heartbeat_interval=60.0 + epsilon)
        assert exc_info.value.field == "heartbeat_interval"

    def test_retrieval_top_k_boundary_1_accepted(self) -> None:
        """Lower boundary 1 is accepted for retrieval_top_k."""
        config = RAPConfig(retrieval_top_k=1)
        assert config.retrieval_top_k == 1

    def test_retrieval_top_k_boundary_50_accepted(self) -> None:
        """Upper boundary 50 is accepted for retrieval_top_k."""
        config = RAPConfig(retrieval_top_k=50)
        assert config.retrieval_top_k == 50

    def test_retrieval_top_k_boundary_0_rejected(self) -> None:
        """Value just below lower boundary is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RAPConfig(retrieval_top_k=0)
        assert exc_info.value.field == "retrieval_top_k"

    def test_retrieval_top_k_boundary_51_rejected(self) -> None:
        """Value just above upper boundary is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RAPConfig(retrieval_top_k=51)
        assert exc_info.value.field == "retrieval_top_k"

    def test_retrieval_max_tokens_boundary_100_accepted(self) -> None:
        """Lower boundary 100 is accepted for retrieval_max_tokens."""
        config = RAPConfig(retrieval_max_tokens=100)
        assert config.retrieval_max_tokens == 100

    def test_retrieval_max_tokens_boundary_10000_accepted(self) -> None:
        """Upper boundary 10000 is accepted for retrieval_max_tokens."""
        config = RAPConfig(retrieval_max_tokens=10000)
        assert config.retrieval_max_tokens == 10000

    def test_retrieval_max_tokens_boundary_99_rejected(self) -> None:
        """Value just below lower boundary is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RAPConfig(retrieval_max_tokens=99)
        assert exc_info.value.field == "retrieval_max_tokens"

    def test_retrieval_max_tokens_boundary_10001_rejected(self) -> None:
        """Value just above upper boundary is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RAPConfig(retrieval_max_tokens=10001)
        assert exc_info.value.field == "retrieval_max_tokens"

    def test_entropy_threshold_boundary_3_accepted(self) -> None:
        """Lower boundary 3.0 is accepted for entropy_threshold."""
        config = RAPConfig(entropy_threshold=3.0)
        assert config.entropy_threshold == 3.0

    def test_entropy_threshold_boundary_8_accepted(self) -> None:
        """Upper boundary 8.0 is accepted for entropy_threshold."""
        config = RAPConfig(entropy_threshold=8.0)
        assert config.entropy_threshold == 8.0

    @given(epsilon=st.floats(min_value=1e-10, max_value=0.5))
    def test_entropy_threshold_just_below_3_rejected(self, epsilon: float) -> None:
        """Values just below 3.0 are rejected for entropy_threshold."""
        with pytest.raises(ValidationError) as exc_info:
            RAPConfig(entropy_threshold=3.0 - epsilon)
        assert exc_info.value.field == "entropy_threshold"

    @given(epsilon=st.floats(min_value=1e-10, max_value=0.5))
    def test_entropy_threshold_just_above_8_rejected(self, epsilon: float) -> None:
        """Values just above 8.0 are rejected for entropy_threshold."""
        with pytest.raises(ValidationError) as exc_info:
            RAPConfig(entropy_threshold=8.0 + epsilon)
        assert exc_info.value.field == "entropy_threshold"

    def test_toon_min_block_size_boundary_64_accepted(self) -> None:
        """Lower boundary 64 is accepted for toon_min_block_size."""
        config = RAPConfig(toon_min_block_size=64)
        assert config.toon_min_block_size == 64

    def test_toon_min_block_size_boundary_63_rejected(self) -> None:
        """Value just below lower boundary is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RAPConfig(toon_min_block_size=63)
        assert exc_info.value.field == "toon_min_block_size"
