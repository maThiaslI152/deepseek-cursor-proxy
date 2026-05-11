"""Property-based tests for RetrievalLayer.embed() — embedding consistency.

**Validates: Requirements 6.3, 6.5**

Property 11: Embedding Dimensionality Consistency
For any batch of embeddings returned by embed():
- All vectors have the same dimensionality
- No value in any embedding vector is NaN or Inf
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.retrieval import RetrievalLayer


# --- Strategies ---


@st.composite
def _consistent_embeddings(draw: st.DrawFn) -> list[list[float]]:
    """Generate a batch of embedding vectors with consistent dimensionality.

    All vectors have the same dimension and contain only finite float values.
    """
    num_vectors = draw(st.integers(min_value=1, max_value=10))
    dimension = draw(st.integers(min_value=1, max_value=256))
    vectors = draw(
        st.lists(
            st.lists(
                st.floats(
                    min_value=-10.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=dimension,
                max_size=dimension,
            ),
            min_size=num_vectors,
            max_size=num_vectors,
        )
    )
    return vectors


@st.composite
def _inconsistent_dimension_embeddings(draw: st.DrawFn) -> list[list[float]]:
    """Generate a batch of embeddings where at least one has a different dimension.

    The first vector sets the expected dimension, and at least one subsequent
    vector has a different dimension.
    """
    num_vectors = draw(st.integers(min_value=2, max_value=8))
    base_dimension = draw(st.integers(min_value=2, max_value=128))

    # Generate the first vector with base_dimension
    first_vector = draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=base_dimension,
            max_size=base_dimension,
        )
    )

    # Pick a different dimension for at least one vector
    alt_dimension = draw(
        st.integers(min_value=1, max_value=128).filter(lambda d: d != base_dimension)
    )

    # Pick which index will have the wrong dimension
    bad_index = draw(st.integers(min_value=1, max_value=num_vectors - 1))

    vectors = [first_vector]
    for i in range(1, num_vectors):
        dim = alt_dimension if i == bad_index else base_dimension
        vec = draw(
            st.lists(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                min_size=dim,
                max_size=dim,
            )
        )
        vectors.append(vec)

    return vectors


@st.composite
def _embeddings_with_nan_or_inf(draw: st.DrawFn) -> list[list[float]]:
    """Generate a batch of embeddings where at least one value is NaN or Inf.

    All vectors have consistent dimensionality, but at least one contains
    a NaN or Inf value.
    """
    num_vectors = draw(st.integers(min_value=1, max_value=8))
    dimension = draw(st.integers(min_value=2, max_value=64))

    # Generate valid vectors first
    vectors: list[list[float]] = []
    for _ in range(num_vectors):
        vec = draw(
            st.lists(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                min_size=dimension,
                max_size=dimension,
            )
        )
        vectors.append(vec)

    # Inject a NaN or Inf into a random position
    bad_vector_idx = draw(st.integers(min_value=0, max_value=num_vectors - 1))
    bad_position = draw(st.integers(min_value=0, max_value=dimension - 1))
    bad_value = draw(st.sampled_from([float("nan"), float("inf"), float("-inf")]))
    vectors[bad_vector_idx][bad_position] = bad_value

    return vectors


def _mock_embedding_response(embeddings: list[list[float]]) -> dict:
    """Build a mock LM Studio API response from embedding vectors."""
    return {
        "data": [
            {"index": i, "embedding": emb}
            for i, emb in enumerate(embeddings)
        ],
        "model": "nomic-embed-text",
    }


def _setup_mock_client(mock_client_cls: MagicMock, embeddings: list[list[float]]) -> None:
    """Configure the mock httpx.Client to return the given embeddings."""
    mock_response = MagicMock()
    mock_response.json.return_value = _mock_embedding_response(embeddings)
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client_cls.return_value = mock_client


# --- Property Tests ---


class TestEmbeddingDimensionalityConsistency:
    """Property 11: Embedding Dimensionality Consistency."""

    @given(embeddings=_consistent_embeddings())
    @settings(max_examples=20, deadline=5000)
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_consistent_embeddings_all_same_dimension(
        self,
        mock_client_cls: MagicMock,
        embeddings: list[list[float]],
    ) -> None:
        """For any batch of valid embeddings, all returned vectors have the same dimensionality.

        **Validates: Requirements 6.3**
        """
        _setup_mock_client(mock_client_cls, embeddings)

        config = RAPConfig()
        layer = RetrievalLayer(config)
        texts = [f"text_{i}" for i in range(len(embeddings))]

        result = layer.embed(texts)

        # All vectors must have the same dimension
        assert len(result) == len(embeddings)
        if result:
            expected_dim = len(result[0])
            for i, vec in enumerate(result):
                assert len(vec) == expected_dim, (
                    f"Vector at index {i} has dimension {len(vec)}, "
                    f"expected {expected_dim}"
                )

    @given(embeddings=_consistent_embeddings())
    @settings(max_examples=20, deadline=5000)
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_consistent_embeddings_no_nan_or_inf(
        self,
        mock_client_cls: MagicMock,
        embeddings: list[list[float]],
    ) -> None:
        """For any valid embedding vector, no value is NaN or Inf.

        **Validates: Requirements 6.5**
        """
        _setup_mock_client(mock_client_cls, embeddings)

        config = RAPConfig()
        layer = RetrievalLayer(config)
        texts = [f"text_{i}" for i in range(len(embeddings))]

        result = layer.embed(texts)

        # No value should be NaN or Inf
        for i, vec in enumerate(result):
            for j, val in enumerate(vec):
                assert not math.isnan(val), (
                    f"NaN found at vector {i}, position {j}"
                )
                assert not math.isinf(val), (
                    f"Inf found at vector {i}, position {j}"
                )

    @given(embeddings=_inconsistent_dimension_embeddings())
    @settings(max_examples=20, deadline=5000)
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_inconsistent_dimensions_raises_value_error(
        self,
        mock_client_cls: MagicMock,
        embeddings: list[list[float]],
    ) -> None:
        """When embeddings have inconsistent dimensionality, embed() raises ValueError.

        **Validates: Requirements 6.3**
        """
        _setup_mock_client(mock_client_cls, embeddings)

        config = RAPConfig()
        layer = RetrievalLayer(config)
        texts = [f"text_{i}" for i in range(len(embeddings))]

        import pytest

        with pytest.raises(ValueError, match="dimensionality mismatch"):
            layer.embed(texts)

    @given(embeddings=_embeddings_with_nan_or_inf())
    @settings(max_examples=20, deadline=5000)
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_nan_or_inf_values_raises_value_error(
        self,
        mock_client_cls: MagicMock,
        embeddings: list[list[float]],
    ) -> None:
        """When any embedding contains NaN or Inf, embed() raises ValueError.

        **Validates: Requirements 6.5**
        """
        _setup_mock_client(mock_client_cls, embeddings)

        config = RAPConfig()
        layer = RetrievalLayer(config)
        texts = [f"text_{i}" for i in range(len(embeddings))]

        import pytest

        with pytest.raises(ValueError, match="invalid value"):
            layer.embed(texts)
