"""Property-based tests for the TOON Engine — structured block detection.

**Validates: Requirement 4.6**

Property 9: Non-Overlapping Structured Block Detection
For any content string that contains structured blocks, the detected blocks:
- Never overlap in character range
- Are always sorted by start_offset
- Each block's start_offset < end_offset
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.toon import TOONEngine


# --- Strategies ---


def _file_tree_entry() -> st.SearchStrategy[dict[str, object]]:
    """Generate a single file tree entry with path, type, and size."""
    return st.fixed_dictionaries({
        "path": st.from_regex(r"[a-z][a-z0-9_/]{5,30}\.(py|ts|js|rs|go)", fullmatch=True),
        "type": st.sampled_from(["file", "directory"]),
        "size": st.integers(min_value=0, max_value=100000),
    })


def _symbol_map_entry() -> st.SearchStrategy[dict[str, object]]:
    """Generate a single symbol map entry with name, kind, and location."""
    return st.fixed_dictionaries({
        "name": st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{2,20}", fullmatch=True),
        "kind": st.sampled_from(["class", "function", "constant", "method", "variable"]),
        "location": st.builds(
            lambda path, line: f"{path}:{line}",
            path=st.from_regex(r"[a-z][a-z0-9_/]{3,20}\.(py|ts|js)", fullmatch=True),
            line=st.integers(min_value=1, max_value=500),
        ),
    })


def _file_tree_json() -> st.SearchStrategy[str]:
    """Generate a JSON file tree array string (at least 3 entries to meet min size)."""
    return st.lists(
        _file_tree_entry(),
        min_size=3,
        max_size=10,
    ).map(json.dumps)


def _symbol_map_json() -> st.SearchStrategy[str]:
    """Generate a JSON symbol map array string (at least 3 entries to meet min size)."""
    return st.lists(
        _symbol_map_entry(),
        min_size=3,
        max_size=10,
    ).map(json.dumps)


def _diff_block() -> st.SearchStrategy[str]:
    """Generate a multi-file diff block."""
    return st.builds(
        _build_diff,
        filename=st.from_regex(r"[a-z][a-z0-9_]{2,15}\.(py|ts|js)", fullmatch=True),
        added_lines=st.lists(
            st.from_regex(r"[a-z_ ]{5,40}", fullmatch=True),
            min_size=2,
            max_size=5,
        ),
    )


def _build_diff(filename: str, added_lines: list[str]) -> str:
    """Build a realistic diff block from components."""
    lines = [
        f"diff --git a/src/{filename} b/src/{filename}",
        "index abc1234..def5678 100644",
        f"--- a/src/{filename}",
        f"+++ b/src/{filename}",
        "@@ -1,5 +1,8 @@",
        " import os",
        " import sys",
    ]
    for added in added_lines:
        lines.append(f"+{added}")
    lines.append(" ")
    lines.append(" def main():")
    lines.append("     pass")
    return "\n".join(lines)


def _separator() -> st.SearchStrategy[str]:
    """Generate separator text between structured blocks."""
    return st.from_regex(r"\n\n[A-Za-z ]{5,30}:\n\n", fullmatch=True)


def _content_with_blocks() -> st.SearchStrategy[str]:
    """Generate content containing multiple structured blocks separated by text."""
    block = st.one_of(_file_tree_json(), _symbol_map_json(), _diff_block())
    return st.builds(
        _assemble_content,
        prefix=st.from_regex(r"[A-Za-z ]{10,50}\n\n", fullmatch=True),
        blocks=st.lists(block, min_size=1, max_size=4),
        separators=st.lists(_separator(), min_size=3, max_size=6),
        suffix=st.from_regex(r"\n\n[A-Za-z .]{10,40}", fullmatch=True),
    )


def _assemble_content(
    prefix: str,
    blocks: list[str],
    separators: list[str],
    suffix: str,
) -> str:
    """Assemble content from prefix, blocks with separators, and suffix."""
    parts = [prefix]
    for i, block in enumerate(blocks):
        parts.append(block)
        if i < len(blocks) - 1 and i < len(separators):
            parts.append(separators[i])
    parts.append(suffix)
    return "".join(parts)


# Also test with arbitrary text that may or may not contain blocks
_arbitrary_content = st.one_of(
    _content_with_blocks(),
    st.text(min_size=0, max_size=500),
    # Content with just a file tree
    _file_tree_json(),
    # Content with just a symbol map
    _symbol_map_json(),
    # Content with just a diff
    _diff_block(),
)


# --- Fixtures ---


def _make_engine() -> TOONEngine:
    """Create a TOONEngine with min_block_size=64 for testing."""
    return TOONEngine(RAPConfig(toon_min_block_size=64))


# --- Property 9: Non-Overlapping Structured Block Detection ---


class TestProperty9NonOverlappingDetection:
    """For any content string that contains structured blocks, the detected
    blocks never overlap in character range, are sorted by start_offset,
    and each block's start_offset < end_offset.

    **Validates: Requirement 4.6**
    """

    @given(content=_content_with_blocks())
    @settings(max_examples=30)
    def test_blocks_never_overlap(self, content: str) -> None:
        """Detected blocks never overlap in character range.

        **Validates: Requirement 4.6**
        """
        engine = _make_engine()
        blocks = engine.detect_structured_blocks(content)

        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                assert blocks[i].end_offset <= blocks[j].start_offset or \
                    blocks[j].end_offset <= blocks[i].start_offset, (
                    f"Blocks overlap: [{blocks[i].start_offset}, {blocks[i].end_offset}) "
                    f"and [{blocks[j].start_offset}, {blocks[j].end_offset})"
                )

    @given(content=_content_with_blocks())
    @settings(max_examples=30)
    def test_blocks_sorted_by_start_offset(self, content: str) -> None:
        """Detected blocks are always sorted by start_offset.

        **Validates: Requirement 4.6**
        """
        engine = _make_engine()
        blocks = engine.detect_structured_blocks(content)

        for i in range(len(blocks) - 1):
            assert blocks[i].start_offset < blocks[i + 1].start_offset, (
                f"Blocks not sorted: block {i} starts at {blocks[i].start_offset} "
                f"but block {i+1} starts at {blocks[i+1].start_offset}"
            )

    @given(content=_content_with_blocks())
    @settings(max_examples=30)
    def test_each_block_has_valid_offsets(self, content: str) -> None:
        """Each block's start_offset < end_offset.

        **Validates: Requirement 4.6**
        """
        engine = _make_engine()
        blocks = engine.detect_structured_blocks(content)

        for block in blocks:
            assert block.start_offset < block.end_offset, (
                f"Invalid block offsets: start={block.start_offset}, "
                f"end={block.end_offset}, type={block.block_type}"
            )

    @given(content=_arbitrary_content)
    @settings(max_examples=30)
    def test_non_overlap_on_arbitrary_content(self, content: str) -> None:
        """Non-overlapping property holds for arbitrary content strings.

        **Validates: Requirement 4.6**
        """
        engine = _make_engine()
        blocks = engine.detect_structured_blocks(content)

        # All three properties must hold regardless of input
        for i in range(len(blocks)):
            # Valid offsets
            assert blocks[i].start_offset < blocks[i].end_offset

            # Sorted
            if i > 0:
                assert blocks[i - 1].start_offset < blocks[i].start_offset

            # Non-overlapping with all subsequent blocks
            for j in range(i + 1, len(blocks)):
                assert blocks[i].end_offset <= blocks[j].start_offset

    @given(content=_content_with_blocks())
    @settings(max_examples=30)
    def test_block_offsets_within_content_bounds(self, content: str) -> None:
        """All block offsets are within the bounds of the content string.

        **Validates: Requirement 4.6**
        """
        engine = _make_engine()
        blocks = engine.detect_structured_blocks(content)

        for block in blocks:
            assert block.start_offset >= 0, (
                f"start_offset {block.start_offset} is negative"
            )
            assert block.end_offset <= len(content), (
                f"end_offset {block.end_offset} exceeds content length {len(content)}"
            )


# --- Property 6: TOON Compression Ratio Bound ---


def _file_tree_json_large() -> st.SearchStrategy[str]:
    """Generate a JSON file tree array large enough to exercise compression.

    Produces arrays with enough entries to exceed min_block_size=64 bytes
    and demonstrate meaningful compression.
    """
    return st.lists(
        _file_tree_entry(),
        min_size=5,
        max_size=30,
    ).map(lambda entries: json.dumps(entries, indent=2))


def _symbol_map_json_large() -> st.SearchStrategy[str]:
    """Generate a JSON symbol map array large enough to exercise compression.

    Produces arrays with enough entries to exceed min_block_size=64 bytes
    and demonstrate meaningful compression.
    """
    return st.lists(
        _symbol_map_entry(),
        min_size=5,
        max_size=30,
    ).map(lambda entries: json.dumps(entries, indent=2))


class TestProperty6CompressionRatioBound:
    """For any structured block that is detected and compressed, the TOON
    output is at most 70% the size of the original.

    **Validates: Requirement 4.3**
    """

    @given(tree_json=_file_tree_json_large())
    @settings(max_examples=30)
    def test_file_tree_compression_ratio(self, tree_json: str) -> None:
        """TOON output for file trees is at most 70% the size of the original.

        **Validates: Requirement 4.3**
        """
        engine = _make_engine()
        blocks = engine.detect_structured_blocks(tree_json)

        for block in blocks:
            assert block.block_type == "file_tree"
            toon_output = engine.to_toon(block)
            ratio = engine.compression_ratio(block.raw_content, toon_output)
            assert ratio <= 0.70, (
                f"Compression ratio {ratio:.3f} exceeds 0.70 for file_tree block "
                f"(original={len(block.raw_content)} bytes, "
                f"compressed={len(toon_output.encode('utf-8'))} bytes)"
            )

    @given(sym_json=_symbol_map_json_large())
    @settings(max_examples=30)
    def test_symbol_map_compression_ratio(self, sym_json: str) -> None:
        """TOON output for symbol maps is at most 70% the size of the original.

        **Validates: Requirement 4.3**
        """
        engine = _make_engine()
        blocks = engine.detect_structured_blocks(sym_json)

        for block in blocks:
            assert block.block_type == "symbol_map"
            toon_output = engine.to_toon(block)
            ratio = engine.compression_ratio(block.raw_content, toon_output)
            assert ratio <= 0.70, (
                f"Compression ratio {ratio:.3f} exceeds 0.70 for symbol_map block "
                f"(original={len(block.raw_content)} bytes, "
                f"compressed={len(toon_output.encode('utf-8'))} bytes)"
            )


# --- Property 7: Message Count and Role Preservation Under Compression ---


_ROLES = st.sampled_from(["system", "user", "assistant", "tool"])


def _message_with_structured_content() -> st.SearchStrategy[dict[str, object]]:
    """Generate a message whose content contains a structured block."""
    content = st.one_of(
        _file_tree_json_large(),
        _symbol_map_json_large(),
    )
    return st.builds(
        lambda role, c: {"role": role, "content": c},
        role=_ROLES,
        c=content,
    )


def _message_with_plain_content() -> st.SearchStrategy[dict[str, object]]:
    """Generate a message with plain text content (no structured blocks)."""
    return st.builds(
        lambda role, c: {"role": role, "content": c},
        role=_ROLES,
        c=st.text(min_size=0, max_size=300),
    )


def _message_list() -> st.SearchStrategy[list[dict[str, object]]]:
    """Generate a list of messages mixing structured and plain content."""
    return st.lists(
        st.one_of(
            _message_with_structured_content(),
            _message_with_plain_content(),
        ),
        min_size=1,
        max_size=10,
    )


class TestProperty7MessageCountAndRolePreservation:
    """For any list of messages, compress() preserves the message count
    and all role assignments.

    **Validates: Requirement 4.4**
    """

    @given(messages=_message_list())
    @settings(max_examples=30)
    def test_message_count_preserved(self, messages: list[dict[str, object]]) -> None:
        """compress() preserves the number of messages.

        **Validates: Requirement 4.4**
        """
        engine = _make_engine()
        compressed = engine.compress(messages)

        assert len(compressed) == len(messages), (
            f"Message count changed: input={len(messages)}, output={len(compressed)}"
        )

    @given(messages=_message_list())
    @settings(max_examples=30)
    def test_role_assignments_preserved(self, messages: list[dict[str, object]]) -> None:
        """compress() preserves all role assignments in order.

        **Validates: Requirement 4.4**
        """
        engine = _make_engine()
        compressed = engine.compress(messages)

        original_roles = [m.get("role") for m in messages]
        compressed_roles = [m.get("role") for m in compressed]

        assert original_roles == compressed_roles, (
            f"Roles changed: input={original_roles}, output={compressed_roles}"
        )


# --- Property 8: Short Message Identity Under Compression ---


def _short_message(min_block_size: int = 64) -> st.SearchStrategy[dict[str, object]]:
    """Generate a message with content shorter than toon_min_block_size bytes."""
    # Generate content that is guaranteed to be shorter than min_block_size in UTF-8
    # Use ASCII-only to ensure 1 byte per char, and limit length to min_block_size - 1
    return st.builds(
        lambda role, c: {"role": role, "content": c},
        role=_ROLES,
        c=st.text(
            alphabet=st.characters(codec="ascii", categories=("L", "N", "P", "S", "Z")),
            min_size=0,
            max_size=min_block_size - 1,
        ),
    )


def _short_message_list() -> st.SearchStrategy[list[dict[str, object]]]:
    """Generate a list of messages all shorter than toon_min_block_size."""
    return st.lists(
        _short_message(min_block_size=64),
        min_size=1,
        max_size=10,
    )


class TestProperty8ShortMessageIdentity:
    """For any message with content shorter than toon_min_block_size,
    compress() leaves it unchanged.

    **Validates: Requirement 4.5**
    """

    @given(messages=_short_message_list())
    @settings(max_examples=30)
    def test_short_messages_unchanged(self, messages: list[dict[str, object]]) -> None:
        """Messages shorter than toon_min_block_size are left unchanged by compress().

        **Validates: Requirement 4.5**
        """
        engine = _make_engine()
        compressed = engine.compress(messages)

        assert compressed == messages, (
            "Short messages were modified by compress() but should be unchanged"
        )

    @given(
        short_msgs=_short_message_list(),
        long_msgs=st.lists(_message_with_structured_content(), min_size=1, max_size=3),
    )
    @settings(max_examples=30)
    def test_short_messages_unchanged_in_mixed_list(
        self,
        short_msgs: list[dict[str, object]],
        long_msgs: list[dict[str, object]],
    ) -> None:
        """Short messages remain unchanged even when mixed with compressible messages.

        **Validates: Requirement 4.5**
        """
        engine = _make_engine()
        # Interleave short and long messages
        messages = []
        for i in range(max(len(short_msgs), len(long_msgs))):
            if i < len(short_msgs):
                messages.append(short_msgs[i])
            if i < len(long_msgs):
                messages.append(long_msgs[i])

        compressed = engine.compress(messages)

        # Check that each short message is unchanged at its position
        original_idx = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content.encode("utf-8")) < 64:
                assert compressed[original_idx] == msg, (
                    f"Short message at index {original_idx} was modified by compress()"
                )
            original_idx += 1


# --- Property 5: TOON Compression Round-Trip ---


class TestProperty5TOONRoundTrip:
    """For any valid file tree or symbol map JSON array, compressing to TOON
    format then re-hydrating produces content equivalent to the original
    (round-trip property).

    The round-trip preserves all data values:
    - File trees: path, type, size
    - Symbol maps: name, kind, location

    **Validates: Requirements 5.1, 5.2**
    """

    @given(entries=st.lists(_file_tree_entry(), min_size=3, max_size=20))
    @settings(max_examples=30)
    def test_file_tree_round_trip(self, entries: list[dict[str, object]]) -> None:
        """Compressing a file tree to TOON then re-hydrating produces
        data equivalent to the original.

        **Validates: Requirements 5.1, 5.2**
        """
        engine = _make_engine()
        original_json = json.dumps(entries)

        # Detect the block
        blocks = engine.detect_structured_blocks(original_json)
        assert len(blocks) == 1, (
            f"Expected 1 file_tree block, got {len(blocks)}"
        )
        block = blocks[0]
        assert block.block_type == "file_tree"

        # Compress to TOON
        toon_output = engine.to_toon(block)

        # Re-hydrate back
        rehydrated = engine.rehydrate(toon_output)

        # Parse both and compare data values
        original_data = json.loads(original_json)
        rehydrated_data = json.loads(rehydrated)

        assert len(rehydrated_data) == len(original_data), (
            f"Entry count mismatch: original={len(original_data)}, "
            f"rehydrated={len(rehydrated_data)}"
        )

        for i, (orig, rehyd) in enumerate(zip(original_data, rehydrated_data)):
            assert rehyd["path"] == orig["path"], (
                f"Entry {i} path mismatch: {orig['path']!r} vs {rehyd['path']!r}"
            )
            assert rehyd["type"] == orig["type"], (
                f"Entry {i} type mismatch: {orig['type']!r} vs {rehyd['type']!r}"
            )
            assert rehyd["size"] == orig["size"], (
                f"Entry {i} size mismatch: {orig['size']!r} vs {rehyd['size']!r}"
            )

    @given(entries=st.lists(_symbol_map_entry(), min_size=3, max_size=20))
    @settings(max_examples=30)
    def test_symbol_map_round_trip(self, entries: list[dict[str, object]]) -> None:
        """Compressing a symbol map to TOON then re-hydrating produces
        data equivalent to the original.

        **Validates: Requirements 5.1, 5.2**
        """
        engine = _make_engine()
        original_json = json.dumps(entries)

        # Detect the block
        blocks = engine.detect_structured_blocks(original_json)
        assert len(blocks) == 1, (
            f"Expected 1 symbol_map block, got {len(blocks)}"
        )
        block = blocks[0]
        assert block.block_type == "symbol_map"

        # Compress to TOON
        toon_output = engine.to_toon(block)

        # Re-hydrate back
        rehydrated = engine.rehydrate(toon_output)

        # Parse both and compare data values
        original_data = json.loads(original_json)
        rehydrated_data = json.loads(rehydrated)

        assert len(rehydrated_data) == len(original_data), (
            f"Entry count mismatch: original={len(original_data)}, "
            f"rehydrated={len(rehydrated_data)}"
        )

        for i, (orig, rehyd) in enumerate(zip(original_data, rehydrated_data)):
            assert rehyd["name"] == orig["name"], (
                f"Entry {i} name mismatch: {orig['name']!r} vs {rehyd['name']!r}"
            )
            assert rehyd["kind"] == orig["kind"], (
                f"Entry {i} kind mismatch: {orig['kind']!r} vs {rehyd['kind']!r}"
            )
            assert rehyd["location"] == orig["location"], (
                f"Entry {i} location mismatch: {orig['location']!r} vs {rehyd['location']!r}"
            )

    @given(entries=st.lists(_file_tree_entry(), min_size=3, max_size=20))
    @settings(max_examples=30)
    def test_file_tree_round_trip_preserves_all_values(
        self, entries: list[dict[str, object]]
    ) -> None:
        """The round-trip preserves all path, type, and size values
        for every entry in the file tree.

        **Validates: Requirements 5.1, 5.2**
        """
        engine = _make_engine()
        original_json = json.dumps(entries)

        blocks = engine.detect_structured_blocks(original_json)
        if not blocks:
            return  # Skip if no block detected (shouldn't happen with min_size=3)

        block = blocks[0]
        toon_output = engine.to_toon(block)
        rehydrated = engine.rehydrate(toon_output)
        rehydrated_data = json.loads(rehydrated)

        # Collect all original values
        original_paths = [e["path"] for e in entries]
        original_types = [e["type"] for e in entries]
        original_sizes = [e["size"] for e in entries]

        # Collect all rehydrated values
        rehydrated_paths = [e["path"] for e in rehydrated_data]
        rehydrated_types = [e["type"] for e in rehydrated_data]
        rehydrated_sizes = [e["size"] for e in rehydrated_data]

        assert rehydrated_paths == original_paths, "Paths not preserved"
        assert rehydrated_types == original_types, "Types not preserved"
        assert rehydrated_sizes == original_sizes, "Sizes not preserved"

    @given(entries=st.lists(_symbol_map_entry(), min_size=3, max_size=20))
    @settings(max_examples=30)
    def test_symbol_map_round_trip_preserves_all_values(
        self, entries: list[dict[str, object]]
    ) -> None:
        """The round-trip preserves all name, kind, and location values
        for every entry in the symbol map.

        **Validates: Requirements 5.1, 5.2**
        """
        engine = _make_engine()
        original_json = json.dumps(entries)

        blocks = engine.detect_structured_blocks(original_json)
        if not blocks:
            return  # Skip if no block detected (shouldn't happen with min_size=3)

        block = blocks[0]
        toon_output = engine.to_toon(block)
        rehydrated = engine.rehydrate(toon_output)
        rehydrated_data = json.loads(rehydrated)

        # Collect all original values
        original_names = [e["name"] for e in entries]
        original_kinds = [e["kind"] for e in entries]
        original_locations = [e["location"] for e in entries]

        # Collect all rehydrated values
        rehydrated_names = [e["name"] for e in rehydrated_data]
        rehydrated_kinds = [e["kind"] for e in rehydrated_data]
        rehydrated_locations = [e["location"] for e in rehydrated_data]

        assert rehydrated_names == original_names, "Names not preserved"
        assert rehydrated_kinds == original_kinds, "Kinds not preserved"
        assert rehydrated_locations == original_locations, "Locations not preserved"
