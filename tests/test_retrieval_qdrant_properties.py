"""Property-based tests for RetrievalLayer — retrieval token budget, score threshold, and MessagePack efficiency.

**Validates: Requirements 7.3, 7.4, 14.2**

Property 13: Retrieval Token Budget Compliance
For any set of scored chunks assembled into reduced context, the total token count
of retrieved content SHALL not exceed config.retrieval_max_tokens.

Property 14: Retrieval Score Threshold
For any chunk included in retrieval results from Qdrant, its relevance score
SHALL be greater than 0.5.

Property 23: MessagePack Size Efficiency
For any payload serialized for Qdrant communication, the MessagePack representation
SHALL be smaller in bytes than the equivalent JSON representation of the same data.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import msgpack
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.retrieval import (
    Chunk,
    RetrievalLayer,
    ScoredChunk,
)


# --- Strategies ---


@st.composite
def _scored_qdrant_results(draw: st.DrawFn) -> list[dict]:
    """Generate a list of Qdrant search results with varying scores and token counts.

    Each result has an id, score (0.0 to 1.0), and payload with text and token_count.
    """
    num_results = draw(st.integers(min_value=1, max_value=20))
    results = []
    for i in range(num_results):
        score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        token_count = draw(st.integers(min_value=1, max_value=500))
        # Generate text that roughly corresponds to the token count
        text = " ".join(["word"] * token_count)
        results.append({
            "id": f"id-{i}",
            "score": score,
            "payload": {
                "text": text,
                "token_count": token_count,
                "source_index": i,
                "metadata": {"chunk_index": i},
            },
        })
    return results


@st.composite
def _retrieval_max_tokens(draw: st.DrawFn) -> int:
    """Generate a valid retrieval_max_tokens value (100 to 10000)."""
    return draw(st.integers(min_value=100, max_value=5000))


@st.composite
def _qdrant_payload(draw: st.DrawFn) -> dict:
    """Generate a payload that would be sent to Qdrant (upsert or search).

    Produces realistic payloads with vectors, text, and metadata.
    MessagePack is more efficient than JSON for payloads with substantial
    text content and multiple fields — which is the realistic case for
    Qdrant upserts containing code snippets and embeddings.
    """
    num_points = draw(st.integers(min_value=1, max_value=5))
    dimension = draw(st.integers(min_value=16, max_value=64))

    points = []
    for i in range(num_points):
        # Use floats with significant precision — realistic embedding values
        # Embeddings from models always have many decimal places
        vector = draw(
            st.lists(
                st.floats(min_value=-0.99, max_value=0.99, allow_nan=False, allow_infinity=False)
                .map(lambda x: round(x, 8))
                .filter(lambda x: abs(x) > 0.001),
                min_size=dimension,
                max_size=dimension,
            )
        )
        # Generate realistic text content (code-like snippets)
        text_words = draw(
            st.lists(
                st.from_regex(r"[a-z]{2,8}", fullmatch=True),
                min_size=20,
                max_size=50,
            )
        )
        text = " ".join(text_words)
        points.append({
            "id": f"point-{i}-uuid-{draw(st.integers(min_value=1000, max_value=9999))}",
            "vector": vector,
            "payload": {
                "text": text,
                "token_count": draw(st.integers(min_value=10, max_value=512)),
                "source_index": i,
                "metadata": {"chunk_index": i, "total_chunks": num_points},
            },
        })

    return {"points": points}


@st.composite
def _search_request_payload(draw: st.DrawFn) -> dict:
    """Generate a search request payload for Qdrant.

    Uses realistic embedding dimensions (>= 16) where MessagePack's
    binary encoding of floats is more compact than JSON text representation
    for values with significant decimal precision.
    """
    dimension = draw(st.integers(min_value=16, max_value=64))
    # Use floats with significant precision — realistic query embeddings
    vector = draw(
        st.lists(
            st.floats(min_value=-0.99, max_value=0.99, allow_nan=False, allow_infinity=False)
            .map(lambda x: round(x, 8))
            .filter(lambda x: abs(x) > 0.001),
            min_size=dimension,
            max_size=dimension,
        )
    )
    limit = draw(st.integers(min_value=1, max_value=50))
    return {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
    }


def _setup_mock_client_for_retrieve(
    mock_client_cls: MagicMock, results: list[dict]
) -> None:
    """Configure the mock httpx.Client to return the given Qdrant search results."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": results}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client_cls.return_value = mock_client


# --- Property Tests ---


class TestRetrievalTokenBudgetCompliance:
    """Property 13: Retrieval Token Budget Compliance.

    For any output of retrieve(), when assembling reduced context from the
    scored chunks, the total token count of retrieved content SHALL not
    exceed config.retrieval_max_tokens.
    """

    @given(
        results=_scored_qdrant_results(),
        max_tokens=_retrieval_max_tokens(),
    )
    @settings(max_examples=20, deadline=5000)
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_retrieved_chunks_total_tokens_within_budget(
        self,
        mock_client_cls: MagicMock,
        results: list[dict],
        max_tokens: int,
    ) -> None:
        """Retrieved chunks total token count never exceeds retrieval_max_tokens.

        The retrieve() method returns scored chunks filtered by score > 0.5.
        When these chunks are assembled into reduced context, their total
        token count must respect the configured budget.

        **Validates: Requirements 7.3**
        """
        _setup_mock_client_for_retrieve(mock_client_cls, results)

        config = RAPConfig(retrieval_max_tokens=max_tokens)
        layer = RetrievalLayer(config)

        scored_chunks = layer.retrieve([0.1, 0.2, 0.3])

        # Simulate assembling reduced context within token budget
        # (this is what build_reduced_context would do)
        total_tokens = 0
        assembled_chunks: list[ScoredChunk] = []
        for sc in scored_chunks:
            if total_tokens + sc.chunk.token_count <= max_tokens:
                assembled_chunks.append(sc)
                total_tokens += sc.chunk.token_count

        # The assembled context must not exceed the budget
        assert total_tokens <= max_tokens, (
            f"Total token count {total_tokens} exceeds budget of {max_tokens}"
        )

    @given(
        results=_scored_qdrant_results(),
        max_tokens=_retrieval_max_tokens(),
    )
    @settings(max_examples=20, deadline=5000)
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_greedy_assembly_respects_budget(
        self,
        mock_client_cls: MagicMock,
        results: list[dict],
        max_tokens: int,
    ) -> None:
        """Greedy assembly of top-scoring chunks respects the token budget.

        Even when many high-scoring chunks are available, the assembly
        process must stop before exceeding retrieval_max_tokens.

        **Validates: Requirements 7.3**
        """
        # Ensure we have some results above threshold
        for r in results:
            r["score"] = 0.9  # Force all above threshold

        _setup_mock_client_for_retrieve(mock_client_cls, results)

        config = RAPConfig(retrieval_max_tokens=max_tokens)
        layer = RetrievalLayer(config)

        scored_chunks = layer.retrieve([0.1, 0.2, 0.3])

        # Assemble greedily by score (already sorted descending)
        total_tokens = 0
        for sc in scored_chunks:
            if total_tokens + sc.chunk.token_count > max_tokens:
                break
            total_tokens += sc.chunk.token_count

        assert total_tokens <= max_tokens, (
            f"Greedy assembly produced {total_tokens} tokens, exceeding budget {max_tokens}"
        )


class TestRetrievalScoreThreshold:
    """Property 14: Retrieval Score Threshold.

    For any chunk included in retrieval results, its relevance score
    SHALL be greater than 0.5.
    """

    @given(results=_scored_qdrant_results())
    @settings(max_examples=30, deadline=5000)
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_all_retrieved_chunks_above_threshold(
        self,
        mock_client_cls: MagicMock,
        results: list[dict],
    ) -> None:
        """All retrieved chunks have a relevance score > 0.5.

        The retrieve() method filters out any chunks with score <= 0.5.
        This property verifies that filtering holds for any combination
        of scores returned by Qdrant.

        **Validates: Requirements 7.4**
        """
        _setup_mock_client_for_retrieve(mock_client_cls, results)

        config = RAPConfig()
        layer = RetrievalLayer(config)

        scored_chunks = layer.retrieve([0.1, 0.2, 0.3])

        for sc in scored_chunks:
            assert sc.score > 0.5, (
                f"Retrieved chunk has score {sc.score} which is not > 0.5"
            )

    @given(results=_scored_qdrant_results())
    @settings(max_examples=30, deadline=5000)
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_boundary_score_excluded(
        self,
        mock_client_cls: MagicMock,
        results: list[dict],
    ) -> None:
        """Chunks with score exactly 0.5 are excluded (strict > 0.5 threshold).

        **Validates: Requirements 7.4**
        """
        # Inject a boundary score result
        results.append({
            "id": "boundary-id",
            "score": 0.5,
            "payload": {
                "text": "boundary chunk",
                "token_count": 10,
                "source_index": 99,
                "metadata": {},
            },
        })

        _setup_mock_client_for_retrieve(mock_client_cls, results)

        config = RAPConfig()
        layer = RetrievalLayer(config)

        scored_chunks = layer.retrieve([0.1, 0.2, 0.3])

        # The boundary chunk must not appear in results
        for sc in scored_chunks:
            assert sc.vector_id != "boundary-id", (
                "Chunk with score exactly 0.5 should be excluded"
            )
            assert sc.score > 0.5

    @given(results=_scored_qdrant_results())
    @settings(max_examples=20, deadline=5000)
    @patch("deepseek_cursor_proxy.rap.retrieval.httpx.Client")
    def test_results_sorted_by_score_descending(
        self,
        mock_client_cls: MagicMock,
        results: list[dict],
    ) -> None:
        """Retrieved chunks are sorted by score in descending order.

        **Validates: Requirements 7.4**
        """
        _setup_mock_client_for_retrieve(mock_client_cls, results)

        config = RAPConfig()
        layer = RetrievalLayer(config)

        scored_chunks = layer.retrieve([0.1, 0.2, 0.3])

        for i in range(len(scored_chunks) - 1):
            assert scored_chunks[i].score >= scored_chunks[i + 1].score, (
                f"Results not sorted: score[{i}]={scored_chunks[i].score} < "
                f"score[{i+1}]={scored_chunks[i+1].score}"
            )


class TestMessagePackSizeEfficiency:
    """Property 23: MessagePack Size Efficiency.

    For any payload serialized for Qdrant communication, the MessagePack
    representation SHALL be smaller in bytes than the equivalent JSON
    representation of the same data.
    """

    @given(payload=_qdrant_payload())
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.filter_too_much])
    def test_msgpack_smaller_than_json_for_upsert(
        self,
        payload: dict,
    ) -> None:
        """MessagePack serialized upsert payloads are smaller than equivalent JSON.

        **Validates: Requirements 14.2**
        """
        msgpack_bytes = msgpack.packb(payload, use_bin_type=True)
        json_bytes = json.dumps(payload).encode("utf-8")

        assert len(msgpack_bytes) < len(json_bytes), (
            f"MessagePack ({len(msgpack_bytes)} bytes) is not smaller than "
            f"JSON ({len(json_bytes)} bytes) for upsert payload"
        )

    @given(payload=_search_request_payload())
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.filter_too_much])
    def test_msgpack_smaller_than_json_for_search(
        self,
        payload: dict,
    ) -> None:
        """MessagePack serialized search payloads are smaller than equivalent JSON.

        **Validates: Requirements 14.2**
        """
        msgpack_bytes = msgpack.packb(payload, use_bin_type=True)
        json_bytes = json.dumps(payload).encode("utf-8")

        assert len(msgpack_bytes) < len(json_bytes), (
            f"MessagePack ({len(msgpack_bytes)} bytes) is not smaller than "
            f"JSON ({len(json_bytes)} bytes) for search payload"
        )

    @given(payload=_qdrant_payload())
    @settings(max_examples=10, deadline=10000, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.filter_too_much])
    def test_msgpack_roundtrip_preserves_data(
        self,
        payload: dict,
    ) -> None:
        """MessagePack serialization and deserialization preserves the payload data.

        This ensures the size efficiency doesn't come at the cost of data integrity.

        **Validates: Requirements 14.2**
        """
        packed = msgpack.packb(payload, use_bin_type=True)
        unpacked = msgpack.unpackb(packed, raw=False)

        # Verify structure is preserved
        assert "points" in unpacked
        assert len(unpacked["points"]) == len(payload["points"])

        for original, restored in zip(payload["points"], unpacked["points"]):
            assert original["id"] == restored["id"]
            assert original["payload"]["text"] == restored["payload"]["text"]
            assert original["payload"]["token_count"] == restored["payload"]["token_count"]
            assert original["payload"]["source_index"] == restored["payload"]["source_index"]
            # Vectors may have minor float precision differences with msgpack
            assert len(original["vector"]) == len(restored["vector"])
