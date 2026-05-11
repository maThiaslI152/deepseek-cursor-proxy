"""Unit tests for the TOON Engine — structured block detection.

Tests detection of file trees, symbol maps, and multi-file diffs,
non-overlapping guarantees, minimum block size filtering, and offset sorting.

Requirements: 4.1, 4.2, 4.6
"""

import json

from deepseek_cursor_proxy.rap.config import RAPConfig
from deepseek_cursor_proxy.rap.toon import TOONEngine


class TestStructuredBlockDetection:
    """Tests for TOONEngine.detect_structured_blocks()."""

    def setup_method(self) -> None:
        """Set up a TOONEngine with a small min_block_size for testing."""
        # Use min_block_size=64 (the minimum allowed) for easier testing
        self.config = RAPConfig(toon_min_block_size=64)
        self.engine = TOONEngine(self.config)

    def test_detects_file_tree(self) -> None:
        """Requirement 4.2: Detects file trees (JSON arrays with path/type/size)."""
        file_tree = json.dumps([
            {"path": "src/main.py", "type": "file", "size": 1234},
            {"path": "src/utils.py", "type": "file", "size": 567},
            {"path": "src/models/", "type": "directory", "size": 0},
        ])
        blocks = self.engine.detect_structured_blocks(file_tree)
        assert len(blocks) == 1
        assert blocks[0].block_type == "file_tree"
        assert blocks[0].raw_content == file_tree

    def test_detects_symbol_map(self) -> None:
        """Requirement 4.2: Detects symbol maps (JSON arrays with name/kind/location)."""
        symbol_map = json.dumps([
            {"name": "MyClass", "kind": "class", "location": "src/main.py:10"},
            {"name": "helper_func", "kind": "function", "location": "src/utils.py:5"},
            {"name": "CONFIG", "kind": "constant", "location": "src/config.py:1"},
        ])
        blocks = self.engine.detect_structured_blocks(symbol_map)
        assert len(blocks) == 1
        assert blocks[0].block_type == "symbol_map"
        assert blocks[0].raw_content == symbol_map

    def test_detects_multi_file_diff_git_format(self) -> None:
        """Requirement 4.2: Detects multi-file diffs starting with 'diff --git'."""
        diff_content = (
            "diff --git a/src/main.py b/src/main.py\n"
            "index abc1234..def5678 100644\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,5 +1,6 @@\n"
            " import os\n"
            " import sys\n"
            "+import json\n"
            " \n"
            " def main():\n"
            "     pass\n"
        )
        # Ensure it meets min block size
        assert len(diff_content.encode("utf-8")) >= 64
        blocks = self.engine.detect_structured_blocks(diff_content)
        assert len(blocks) == 1
        assert blocks[0].block_type == "multi_file_diff"

    def test_detects_unified_diff_format(self) -> None:
        """Requirement 4.2: Detects unified diff format (--- / +++ pairs)."""
        diff_content = (
            "--- a/src/utils.py\n"
            "+++ b/src/utils.py\n"
            "@@ -10,3 +10,4 @@\n"
            " def helper():\n"
            "     return True\n"
            "+    # Added comment for testing purposes to meet size\n"
            "+    # Another line to ensure minimum block size is met\n"
        )
        # Ensure it meets min block size
        assert len(diff_content.encode("utf-8")) >= 64
        blocks = self.engine.detect_structured_blocks(diff_content)
        assert len(blocks) == 1
        assert blocks[0].block_type == "multi_file_diff"

    def test_returns_sorted_by_offset(self) -> None:
        """Requirement 4.6: Blocks are sorted by start_offset."""
        # Place a diff first, then a file tree
        diff_content = (
            "diff --git a/src/main.py b/src/main.py\n"
            "index abc1234..def5678 100644\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,3 +1,4 @@\n"
            " import os\n"
            "+import json\n"
            " def main():\n"
            "     pass\n"
        )
        file_tree = json.dumps([
            {"path": "src/main.py", "type": "file", "size": 1234},
            {"path": "src/utils.py", "type": "file", "size": 567},
            {"path": "src/config.py", "type": "file", "size": 890},
        ])
        content = diff_content + "\n\nHere is the file tree:\n" + file_tree
        blocks = self.engine.detect_structured_blocks(content)

        # Verify sorted by start_offset
        for i in range(len(blocks) - 1):
            assert blocks[i].start_offset < blocks[i + 1].start_offset

    def test_non_overlapping_blocks(self) -> None:
        """Requirement 4.6: Detected blocks do not overlap in character range."""
        file_tree = json.dumps([
            {"path": "src/main.py", "type": "file", "size": 1234},
            {"path": "src/utils.py", "type": "file", "size": 567},
            {"path": "src/config.py", "type": "file", "size": 890},
        ])
        symbol_map = json.dumps([
            {"name": "MyClass", "kind": "class", "location": "src/main.py:10"},
            {"name": "helper", "kind": "function", "location": "src/utils.py:5"},
            {"name": "CONFIG", "kind": "constant", "location": "src/config.py:1"},
        ])
        content = file_tree + "\n\nSymbols:\n" + symbol_map
        blocks = self.engine.detect_structured_blocks(content)

        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                # No overlap: one must end before the other starts
                assert (
                    blocks[i].end_offset <= blocks[j].start_offset
                    or blocks[j].end_offset <= blocks[i].start_offset
                )

    def test_ignores_blocks_below_min_size(self) -> None:
        """Requirement 4.1: Only detects blocks >= toon_min_block_size bytes."""
        # Use a larger min_block_size
        config = RAPConfig(toon_min_block_size=500)
        engine = TOONEngine(config)

        # This small file tree should be below 500 bytes
        small_tree = json.dumps([
            {"path": "a.py", "type": "file", "size": 1},
        ])
        assert len(small_tree.encode("utf-8")) < 500

        blocks = engine.detect_structured_blocks(small_tree)
        assert len(blocks) == 0

    def test_empty_content_returns_empty(self) -> None:
        """Edge case: empty content returns no blocks."""
        blocks = self.engine.detect_structured_blocks("")
        assert blocks == []

    def test_plain_text_returns_empty(self) -> None:
        """Edge case: plain text without structured data returns no blocks."""
        content = "This is just a regular message with no structured data at all."
        blocks = self.engine.detect_structured_blocks(content)
        assert blocks == []

    def test_invalid_json_array_ignored(self) -> None:
        """Edge case: malformed JSON arrays are not detected."""
        content = '[{"path": "a.py", "type": "file", "size": 1}, INVALID]'
        blocks = self.engine.detect_structured_blocks(content)
        assert blocks == []

    def test_json_array_without_required_keys_ignored(self) -> None:
        """Edge case: JSON arrays without matching keys are not detected."""
        # Array of objects but not matching file_tree or symbol_map patterns
        data = json.dumps([
            {"foo": "bar", "baz": 123, "extra_key": "value_to_pad_size"},
            {"foo": "qux", "baz": 456, "extra_key": "another_value_padding"},
        ])
        blocks = self.engine.detect_structured_blocks(data)
        assert blocks == []

    def test_mixed_entries_not_detected(self) -> None:
        """Edge case: arrays with mixed entry types are not detected."""
        # Some entries match file_tree, some don't
        data = json.dumps([
            {"path": "src/main.py", "type": "file", "size": 1234},
            {"name": "MyClass", "kind": "class", "location": "src/main.py:10"},
        ])
        blocks = self.engine.detect_structured_blocks(data)
        assert blocks == []

    def test_correct_offsets_for_embedded_block(self) -> None:
        """Offsets correctly reflect position within larger content."""
        prefix = "Here is the project structure:\n\n"
        file_tree = json.dumps([
            {"path": "src/main.py", "type": "file", "size": 1234},
            {"path": "src/utils.py", "type": "file", "size": 567},
            {"path": "src/config.py", "type": "file", "size": 890},
        ])
        suffix = "\n\nPlease review the code."
        content = prefix + file_tree + suffix

        blocks = self.engine.detect_structured_blocks(content)
        assert len(blocks) == 1
        assert blocks[0].start_offset == len(prefix)
        assert blocks[0].end_offset == len(prefix) + len(file_tree)
        assert blocks[0].raw_content == file_tree

    def test_multi_file_diff_with_multiple_files(self) -> None:
        """Detects a diff block spanning multiple files as a single block."""
        diff_content = (
            "diff --git a/src/main.py b/src/main.py\n"
            "index abc1234..def5678 100644\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,3 +1,4 @@\n"
            " import os\n"
            "+import json\n"
            " def main():\n"
            "diff --git a/src/utils.py b/src/utils.py\n"
            "index 1111111..2222222 100644\n"
            "--- a/src/utils.py\n"
            "+++ b/src/utils.py\n"
            "@@ -5,3 +5,4 @@\n"
            " def helper():\n"
            "+    # new comment\n"
            "     return True\n"
        )
        blocks = self.engine.detect_structured_blocks(diff_content)
        assert len(blocks) == 1
        assert blocks[0].block_type == "multi_file_diff"

    def test_default_config_min_block_size(self) -> None:
        """Default config uses 256 bytes as min_block_size."""
        engine = TOONEngine()
        # A small file tree below 256 bytes should not be detected
        small_tree = json.dumps([
            {"path": "a.py", "type": "file", "size": 1},
            {"path": "b.py", "type": "file", "size": 2},
        ])
        assert len(small_tree.encode("utf-8")) < 256
        blocks = engine.detect_structured_blocks(small_tree)
        assert blocks == []

    def test_file_tree_with_extra_keys_detected(self) -> None:
        """File trees with extra keys beyond path/type/size are still detected."""
        file_tree = json.dumps([
            {"path": "src/main.py", "type": "file", "size": 1234, "modified": "2024-01-01"},
            {"path": "src/utils.py", "type": "file", "size": 567, "modified": "2024-01-02"},
            {"path": "src/config.py", "type": "file", "size": 890, "modified": "2024-01-03"},
        ])
        blocks = self.engine.detect_structured_blocks(file_tree)
        assert len(blocks) == 1
        assert blocks[0].block_type == "file_tree"


class TestToToon:
    """Tests for TOONEngine.to_toon() — converting structured blocks to TOON format."""

    def setup_method(self) -> None:
        self.config = RAPConfig(toon_min_block_size=64)
        self.engine = TOONEngine(self.config)

    def test_file_tree_to_toon_format(self) -> None:
        """Requirement 4.1, 4.3: File tree converted to pipe-delimited TOON format."""
        raw = json.dumps([
            {"path": "src/main.py", "type": "file", "size": 1234},
            {"path": "src/utils.py", "type": "file", "size": 567},
        ])
        from deepseek_cursor_proxy.rap.toon import StructuredBlock

        block = StructuredBlock(
            block_type="file_tree",
            start_offset=0,
            end_offset=len(raw),
            raw_content=raw,
        )
        result = self.engine.to_toon(block)

        assert result.startswith("@TOON:file_tree\n")
        assert result.endswith("\n@END")
        lines = result.split("\n")
        # Header + 2 entries + footer
        assert lines[0] == "@TOON:file_tree"
        assert lines[1] == "src/main.py|file|1234"
        assert lines[2] == "src/utils.py|file|567"
        assert lines[3] == "@END"

    def test_symbol_map_to_toon_format(self) -> None:
        """Requirement 4.1, 4.3: Symbol map converted to pipe-delimited TOON format."""
        raw = json.dumps([
            {"name": "MyClass", "kind": "class", "location": "src/main.py:10"},
            {"name": "helper_func", "kind": "function", "location": "src/utils.py:5"},
        ])
        from deepseek_cursor_proxy.rap.toon import StructuredBlock

        block = StructuredBlock(
            block_type="symbol_map",
            start_offset=0,
            end_offset=len(raw),
            raw_content=raw,
        )
        result = self.engine.to_toon(block)

        assert result.startswith("@TOON:symbol_map\n")
        assert result.endswith("\n@END")
        lines = result.split("\n")
        assert lines[0] == "@TOON:symbol_map"
        assert lines[1] == "MyClass|class|src/main.py:10"
        assert lines[2] == "helper_func|function|src/utils.py:5"
        assert lines[3] == "@END"

    def test_multi_file_diff_to_toon_format(self) -> None:
        """Requirement 4.1, 4.3: Diff compressed by removing redundant lines."""
        raw = (
            "diff --git a/src/main.py b/src/main.py\n"
            "index abc1234..def5678 100644\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,5 +1,6 @@\n"
            " import os\n"
            " import sys\n"
            "+import json\n"
            " \n"
            " def main():\n"
            "     pass\n"
        )
        from deepseek_cursor_proxy.rap.toon import StructuredBlock

        block = StructuredBlock(
            block_type="multi_file_diff",
            start_offset=0,
            end_offset=len(raw),
            raw_content=raw,
        )
        result = self.engine.to_toon(block)

        assert result.startswith("@TOON:multi_file_diff\n")
        assert result.endswith("\n@END")
        # Should contain the file path and changed lines
        assert "D|src/main.py" in result
        assert "+import json" in result
        # Should NOT contain index line, --- or +++ lines, or context lines
        assert "index abc1234" not in result
        assert "--- a/src/main.py" not in result
        assert "+++ b/src/main.py" not in result
        assert " import os" not in result

    def test_toon_output_smaller_than_original(self) -> None:
        """Requirement 4.3: TOON output is at most 70% the size of original."""
        # Create a reasonably sized file tree
        entries = [
            {"path": f"src/module_{i}/file_{j}.py", "type": "file", "size": 1000 + i * 100 + j}
            for i in range(5)
            for j in range(5)
        ]
        raw = json.dumps(entries)
        from deepseek_cursor_proxy.rap.toon import StructuredBlock

        block = StructuredBlock(
            block_type="file_tree",
            start_offset=0,
            end_offset=len(raw),
            raw_content=raw,
        )
        result = self.engine.to_toon(block)

        ratio = len(result.encode("utf-8")) / len(raw.encode("utf-8"))
        assert ratio <= 0.70, f"Compression ratio {ratio:.2f} exceeds 0.70"

    def test_symbol_map_compression_ratio(self) -> None:
        """Requirement 4.3: Symbol map TOON output is at most 70% of original."""
        entries = [
            {"name": f"ClassName{i}", "kind": "class", "location": f"src/module_{i}.py:{i * 10}"}
            for i in range(20)
        ]
        raw = json.dumps(entries)
        from deepseek_cursor_proxy.rap.toon import StructuredBlock

        block = StructuredBlock(
            block_type="symbol_map",
            start_offset=0,
            end_offset=len(raw),
            raw_content=raw,
        )
        result = self.engine.to_toon(block)

        ratio = len(result.encode("utf-8")) / len(raw.encode("utf-8"))
        assert ratio <= 0.70, f"Compression ratio {ratio:.2f} exceeds 0.70"


class TestCompress:
    """Tests for TOONEngine.compress() — processing message lists."""

    def setup_method(self) -> None:
        self.config = RAPConfig(toon_min_block_size=64)
        self.engine = TOONEngine(self.config)

    def test_preserves_message_count(self) -> None:
        """Requirement 4.4: Message count is preserved after compression."""
        file_tree = json.dumps([
            {"path": "src/main.py", "type": "file", "size": 1234},
            {"path": "src/utils.py", "type": "file", "size": 567},
            {"path": "src/config.py", "type": "file", "size": 890},
        ])
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Here is the file tree:\n{file_tree}"},
            {"role": "assistant", "content": "I see the files."},
        ]
        result = self.engine.compress(messages)
        assert len(result) == len(messages)

    def test_preserves_role_assignments(self) -> None:
        """Requirement 4.4: Role assignments are preserved after compression."""
        file_tree = json.dumps([
            {"path": "src/main.py", "type": "file", "size": 1234},
            {"path": "src/utils.py", "type": "file", "size": 567},
            {"path": "src/config.py", "type": "file", "size": 890},
        ])
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Here is the file tree:\n{file_tree}"},
            {"role": "assistant", "content": "I see the files."},
        ]
        result = self.engine.compress(messages)
        for orig, comp in zip(messages, result):
            assert orig["role"] == comp["role"]

    def test_short_messages_unchanged(self) -> None:
        """Requirement 4.5: Messages shorter than min_block_size are unchanged."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = self.engine.compress(messages)
        assert result == messages

    def test_messages_without_structured_blocks_unchanged(self) -> None:
        """Messages without structured data are left unchanged."""
        long_text = "This is a regular message. " * 20  # Long but no structured data
        messages = [
            {"role": "user", "content": long_text},
        ]
        result = self.engine.compress(messages)
        assert result[0]["content"] == long_text

    def test_compress_replaces_file_tree_with_toon(self) -> None:
        """Requirement 4.1: Structured blocks are replaced with TOON format."""
        file_tree = json.dumps([
            {"path": "src/main.py", "type": "file", "size": 1234},
            {"path": "src/utils.py", "type": "file", "size": 567},
            {"path": "src/config.py", "type": "file", "size": 890},
        ])
        messages = [
            {"role": "user", "content": f"Here is the file tree:\n{file_tree}\nPlease review."},
        ]
        result = self.engine.compress(messages)

        content = result[0]["content"]
        assert "@TOON:file_tree" in content
        assert "@END" in content
        assert "src/main.py|file|1234" in content
        # The surrounding text should be preserved
        assert content.startswith("Here is the file tree:\n")
        assert content.endswith("\nPlease review.")

    def test_compress_handles_non_string_content(self) -> None:
        """Messages with non-string content (e.g., tool_calls) are unchanged."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "user", "content": "Hello"},
        ]
        result = self.engine.compress(messages)
        assert result == messages

    def test_compress_empty_messages_list(self) -> None:
        """Empty message list returns empty list."""
        result = self.engine.compress([])
        assert result == []

    def test_compress_preserves_other_message_fields(self) -> None:
        """Other message fields (name, tool_call_id, etc.) are preserved."""
        file_tree = json.dumps([
            {"path": "src/main.py", "type": "file", "size": 1234},
            {"path": "src/utils.py", "type": "file", "size": 567},
            {"path": "src/config.py", "type": "file", "size": 890},
        ])
        messages = [
            {"role": "user", "content": file_tree, "name": "developer"},
        ]
        result = self.engine.compress(messages)
        assert result[0]["role"] == "user"
        assert result[0]["name"] == "developer"
        assert "@TOON:file_tree" in result[0]["content"]

    def test_compress_multiple_blocks_in_one_message(self) -> None:
        """Multiple structured blocks in one message are all compressed."""
        file_tree = json.dumps([
            {"path": "src/main.py", "type": "file", "size": 1234},
            {"path": "src/utils.py", "type": "file", "size": 567},
            {"path": "src/config.py", "type": "file", "size": 890},
        ])
        symbol_map = json.dumps([
            {"name": "MyClass", "kind": "class", "location": "src/main.py:10"},
            {"name": "helper", "kind": "function", "location": "src/utils.py:5"},
            {"name": "CONFIG", "kind": "constant", "location": "src/config.py:1"},
        ])
        content = f"Files:\n{file_tree}\n\nSymbols:\n{symbol_map}"
        messages = [{"role": "user", "content": content}]
        result = self.engine.compress(messages)

        compressed_content = result[0]["content"]
        assert "@TOON:file_tree" in compressed_content
        assert "@TOON:symbol_map" in compressed_content

    def test_compression_ratio_method(self) -> None:
        """compression_ratio() correctly calculates the ratio."""
        original = "Hello, World! This is a test string."
        compressed = "Hello"
        ratio = self.engine.compression_ratio(original, compressed)
        expected = len(compressed.encode("utf-8")) / len(original.encode("utf-8"))
        assert abs(ratio - expected) < 0.001

    def test_compression_ratio_empty_original(self) -> None:
        """compression_ratio() returns 1.0 for empty original."""
        ratio = self.engine.compression_ratio("", "anything")
        assert ratio == 1.0


class TestRehydrate:
    """Tests for TOONEngine.rehydrate() — converting TOON format back to JSON.

    Requirements: 5.1, 5.2, 5.3
    """

    def setup_method(self) -> None:
        self.config = RAPConfig(toon_min_block_size=64)
        self.engine = TOONEngine(self.config)

    def test_rehydrate_file_tree(self) -> None:
        """Requirement 5.1: Re-hydrates file_tree TOON block back to JSON array."""
        toon_content = (
            "@TOON:file_tree\n"
            "src/main.py|file|1234\n"
            "src/utils.py|file|567\n"
            "@END"
        )
        result = self.engine.rehydrate(toon_content)
        data = json.loads(result)
        assert len(data) == 2
        assert data[0] == {"path": "src/main.py", "type": "file", "size": 1234}
        assert data[1] == {"path": "src/utils.py", "type": "file", "size": 567}

    def test_rehydrate_symbol_map(self) -> None:
        """Requirement 5.1: Re-hydrates symbol_map TOON block back to JSON array."""
        toon_content = (
            "@TOON:symbol_map\n"
            "MyClass|class|src/main.py:10\n"
            "helper_func|function|src/utils.py:5\n"
            "@END"
        )
        result = self.engine.rehydrate(toon_content)
        data = json.loads(result)
        assert len(data) == 2
        assert data[0] == {"name": "MyClass", "kind": "class", "location": "src/main.py:10"}
        assert data[1] == {"name": "helper_func", "kind": "function", "location": "src/utils.py:5"}

    def test_rehydrate_multi_file_diff(self) -> None:
        """Requirement 5.1: Re-hydrates multi_file_diff TOON block back to diff format."""
        toon_content = (
            "@TOON:multi_file_diff\n"
            "D|src/main.py\n"
            "@@ -1,5 +1,6 @@\n"
            "+import json\n"
            "@END"
        )
        result = self.engine.rehydrate(toon_content)
        assert "diff --git a/src/main.py b/src/main.py" in result
        assert "--- a/src/main.py" in result
        assert "+++ b/src/main.py" in result
        assert "+import json" in result

    def test_rehydrate_preserves_surrounding_text(self) -> None:
        """Requirement 5.1: Surrounding text is preserved during re-hydration."""
        content = (
            "Here is the file tree:\n"
            "@TOON:file_tree\n"
            "src/main.py|file|1234\n"
            "@END\n"
            "Please review."
        )
        result = self.engine.rehydrate(content)
        assert result.startswith("Here is the file tree:\n")
        assert result.endswith("\nPlease review.")
        # The middle should be valid JSON
        json_part = result[len("Here is the file tree:\n"):-len("\nPlease review.")]
        data = json.loads(json_part)
        assert data == [{"path": "src/main.py", "type": "file", "size": 1234}]

    def test_rehydrate_multiple_blocks(self) -> None:
        """Requirement 5.1: Multiple TOON blocks in one content are all re-hydrated."""
        content = (
            "Files:\n"
            "@TOON:file_tree\n"
            "src/main.py|file|1234\n"
            "@END\n"
            "\nSymbols:\n"
            "@TOON:symbol_map\n"
            "MyClass|class|src/main.py:10\n"
            "@END"
        )
        result = self.engine.rehydrate(content)
        assert "@TOON:" not in result
        assert "@END" not in result
        # Both blocks should be re-hydrated to JSON
        assert '"path"' in result
        assert '"name"' in result

    def test_rehydrate_failure_leaves_block_unchanged(self) -> None:
        """Requirement 5.3: Failed re-hydration leaves the block unchanged."""
        # Invalid file_tree: wrong number of pipe-separated fields
        content = (
            "@TOON:file_tree\n"
            "src/main.py|file|1234|extra_field\n"
            "@END"
        )
        result = self.engine.rehydrate(content)
        # Block should be left unchanged
        assert result == content

    def test_rehydrate_unknown_block_type_unchanged(self) -> None:
        """Requirement 5.3: Unknown block types are left unchanged."""
        content = (
            "@TOON:unknown_type\n"
            "some|data|here\n"
            "@END"
        )
        result = self.engine.rehydrate(content)
        assert result == content

    def test_rehydrate_empty_content(self) -> None:
        """Edge case: empty content returns empty string."""
        result = self.engine.rehydrate("")
        assert result == ""

    def test_rehydrate_no_toon_blocks(self) -> None:
        """Edge case: content without TOON blocks is returned unchanged."""
        content = "This is just regular text without any TOON blocks."
        result = self.engine.rehydrate(content)
        assert result == content

    def test_rehydrate_file_tree_round_trip(self) -> None:
        """Requirement 5.2: Compress then re-hydrate produces equivalent content."""
        original_data = [
            {"path": "src/main.py", "type": "file", "size": 1234},
            {"path": "src/utils.py", "type": "file", "size": 567},
            {"path": "src/config.py", "type": "file", "size": 890},
        ]
        original_json = json.dumps(original_data)

        from deepseek_cursor_proxy.rap.toon import StructuredBlock

        block = StructuredBlock(
            block_type="file_tree",
            start_offset=0,
            end_offset=len(original_json),
            raw_content=original_json,
        )
        toon_repr = self.engine.to_toon(block)
        rehydrated = self.engine.rehydrate(toon_repr)

        # Parse both and compare
        rehydrated_data = json.loads(rehydrated)
        assert rehydrated_data == original_data

    def test_rehydrate_symbol_map_round_trip(self) -> None:
        """Requirement 5.2: Compress then re-hydrate produces equivalent content."""
        original_data = [
            {"name": "MyClass", "kind": "class", "location": "src/main.py:10"},
            {"name": "helper_func", "kind": "function", "location": "src/utils.py:5"},
            {"name": "CONFIG", "kind": "constant", "location": "src/config.py:1"},
        ]
        original_json = json.dumps(original_data)

        from deepseek_cursor_proxy.rap.toon import StructuredBlock

        block = StructuredBlock(
            block_type="symbol_map",
            start_offset=0,
            end_offset=len(original_json),
            raw_content=original_json,
        )
        toon_repr = self.engine.to_toon(block)
        rehydrated = self.engine.rehydrate(toon_repr)

        rehydrated_data = json.loads(rehydrated)
        assert rehydrated_data == original_data

    def test_rehydrate_partial_failure_other_blocks_still_processed(self) -> None:
        """Requirement 5.3: If one block fails, others are still re-hydrated."""
        content = (
            "@TOON:file_tree\n"
            "src/main.py|file|1234\n"
            "@END\n"
            "middle text\n"
            "@TOON:file_tree\n"
            "invalid|too|many|fields\n"
            "@END"
        )
        result = self.engine.rehydrate(content)
        # First block should be re-hydrated
        assert '"path": "src/main.py"' in result
        # Second block should remain unchanged due to failure
        assert "@TOON:file_tree\ninvalid|too|many|fields\n@END" in result

    def test_rehydrate_diff_with_multiple_files(self) -> None:
        """Re-hydrates a diff with multiple file changes."""
        toon_content = (
            "@TOON:multi_file_diff\n"
            "D|src/main.py\n"
            "@@ -1,3 +1,4 @@\n"
            "+import json\n"
            "D|src/utils.py\n"
            "@@ -5,3 +5,4 @@\n"
            "+    # new comment\n"
            "@END"
        )
        result = self.engine.rehydrate(toon_content)
        assert "diff --git a/src/main.py b/src/main.py" in result
        assert "diff --git a/src/utils.py b/src/utils.py" in result
        assert "--- a/src/main.py" in result
        assert "+++ b/src/main.py" in result
        assert "--- a/src/utils.py" in result
        assert "+++ b/src/utils.py" in result
        assert "+import json" in result
        assert "+    # new comment" in result

    def test_rehydrate_file_tree_with_zero_size(self) -> None:
        """Re-hydrates file tree entries with size 0 (directories)."""
        toon_content = (
            "@TOON:file_tree\n"
            "src/|directory|0\n"
            "@END"
        )
        result = self.engine.rehydrate(toon_content)
        data = json.loads(result)
        assert data == [{"path": "src/", "type": "directory", "size": 0}]
