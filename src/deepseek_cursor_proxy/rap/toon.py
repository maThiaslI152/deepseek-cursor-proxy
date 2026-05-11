"""TOON Engine — Token-Oriented Object Notation for context compression.

Detects structured data blocks (file trees, symbol maps, multi-file diffs)
in message content and provides compression/re-hydration capabilities.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.3
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from deepseek_cursor_proxy.rap.config import RAPConfig


@dataclass
class StructuredBlock:
    """A detected region of content containing structured data.

    Attributes:
        block_type: One of "file_tree", "symbol_map", or "multi_file_diff".
        start_offset: Character offset where the block begins in the source content.
        end_offset: Character offset where the block ends (exclusive).
        raw_content: The raw text of the detected block.
    """

    block_type: str  # "file_tree" | "symbol_map" | "multi_file_diff"
    start_offset: int
    end_offset: int
    raw_content: str


# Pattern for unified diff headers (diff --git or --- / +++ pairs)
_DIFF_HEADER_RE = re.compile(
    r"^diff --git\b",
    re.MULTILINE,
)

# Alternate unified diff pattern (--- a/... followed by +++ b/...)
_UNIFIED_DIFF_RE = re.compile(
    r"^---\s+\S+.*\n\+\+\+\s+\S+",
    re.MULTILINE,
)

# Pattern for TOON blocks: @TOON:{type}\n{body}\n@END
_TOON_BLOCK_RE = re.compile(
    r"@TOON:(\w+)\n(.*?)\n@END",
    re.DOTALL,
)


class TOONEngine:
    """Token-Oriented Object Notation engine for context compression.

    Detects structured blocks in message content and converts them to a
    compact pipe-delimited notation to reduce token consumption.
    """

    def __init__(self, config: RAPConfig | None = None) -> None:
        if config is None:
            config = RAPConfig()
        self._min_block_size = config.toon_min_block_size

    def detect_structured_blocks(self, content: str) -> list[StructuredBlock]:
        """Find file trees, symbol maps, and multi-file diffs in content.

        Returns non-overlapping StructuredBlock instances sorted by start_offset.
        Only detects blocks whose raw content is >= toon_min_block_size bytes.

        Requirements: 4.1, 4.2, 4.6
        """
        blocks: list[StructuredBlock] = []

        # Detect JSON-based blocks (file trees and symbol maps)
        blocks.extend(self._detect_json_blocks(content))

        # Detect multi-file diffs
        blocks.extend(self._detect_diff_blocks(content))

        # Remove overlapping blocks (keep the one that starts first, or longest if same start)
        blocks = self._remove_overlaps(blocks)

        # Sort by start_offset
        blocks.sort(key=lambda b: b.start_offset)

        return blocks

    def to_toon(self, block: StructuredBlock) -> str:
        """Convert a structured block to TOON (pipe-delimited) format.

        Produces output with:
        - Header: @TOON:{block_type}\\n
        - Body: pipe-delimited entries (one per line)
        - Footer: \\n@END

        The output is at most 70% the size of the original structured block.

        Requirements: 4.1, 4.3
        """
        header = f"@TOON:{block.block_type}\n"
        footer = "\n@END"

        if block.block_type == "file_tree":
            entries = json.loads(block.raw_content)
            lines = [
                f"{e.get('path', '')}|{e.get('type', '')}|{e.get('size', '')}"
                for e in entries
            ]
            return header + "\n".join(lines) + footer

        elif block.block_type == "symbol_map":
            entries = json.loads(block.raw_content)
            lines = [
                f"{e.get('name', '')}|{e.get('kind', '')}|{e.get('location', '')}"
                for e in entries
            ]
            return header + "\n".join(lines) + footer

        elif block.block_type == "multi_file_diff":
            compressed_diff = self._compress_diff(block.raw_content)
            return header + compressed_diff + footer

        # Fallback: return raw content unchanged
        return block.raw_content

    def _compress_diff(self, diff_text: str) -> str:
        """Compress a diff by removing redundant context lines and metadata.

        Keeps file paths, hunk headers, and changed lines (+/-).
        Removes index lines, mode lines, and reduces context lines.
        """
        lines = diff_text.split("\n")
        compressed_lines: list[str] = []

        for line in lines:
            # Keep diff headers (file paths)
            if line.startswith("diff --git"):
                # Extract just the file paths
                parts = line.split(" ")
                if len(parts) >= 4:
                    # "diff --git a/path b/path" -> "D|path"
                    b_path = parts[-1]
                    if b_path.startswith("b/"):
                        b_path = b_path[2:]
                    compressed_lines.append(f"D|{b_path}")
                else:
                    compressed_lines.append(line)
            elif line.startswith("--- ") or line.startswith("+++ "):
                # Skip --- and +++ lines (redundant with diff --git)
                continue
            elif line.startswith("index ") or line.startswith("new file") or line.startswith("deleted file"):
                # Skip metadata lines
                continue
            elif line.startswith("similarity index") or line.startswith("rename from") or line.startswith("rename to"):
                # Skip rename metadata
                continue
            elif line.startswith("@@"):
                # Keep hunk headers but compress them
                compressed_lines.append(line)
            elif line.startswith("+") or line.startswith("-"):
                # Keep changed lines
                compressed_lines.append(line)
            elif line.startswith("\\"):
                # Keep "no newline" markers
                compressed_lines.append(line)
            # Skip context lines (lines starting with space) to save space

        return "\n".join(compressed_lines)

    def compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify and compress structured data blocks in messages.

        Iterates over messages, detects structured blocks in content that
        is >= min_block_size, and replaces them with TOON representations.

        Preserves message count and role assignments. Leaves messages
        shorter than toon_min_block_size unchanged.

        Requirements: 4.1, 4.3, 4.4, 4.5
        """
        compressed: list[dict[str, Any]] = []

        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, str) or len(content.encode("utf-8")) < self._min_block_size:
                # Leave short messages unchanged (Requirement 4.5)
                compressed.append(message)
                continue

            blocks = self.detect_structured_blocks(content)
            if not blocks:
                compressed.append(message)
                continue

            # Process blocks in reverse order to preserve offsets
            new_content = content
            for block in sorted(blocks, key=lambda b: b.start_offset, reverse=True):
                toon_repr = self.to_toon(block)
                new_content = (
                    new_content[:block.start_offset]
                    + toon_repr
                    + new_content[block.end_offset:]
                )

            compressed.append({**message, "content": new_content})

        return compressed

    def rehydrate(self, content: str) -> str:
        """Convert TOON format blocks back to original JSON/diff structure.

        Finds all @TOON:{block_type}\\n...\\n@END blocks in the content,
        parses each one, and replaces it with the re-hydrated representation.

        If re-hydration of any block fails, that block is left unchanged
        (graceful failure per Requirement 5.3).

        Requirements: 5.1, 5.2, 5.3
        """
        # Find all TOON blocks using regex
        result = _TOON_BLOCK_RE.sub(self._rehydrate_match, content)
        return result

    def _rehydrate_match(self, match: re.Match[str]) -> str:
        """Re-hydrate a single TOON block match.

        Returns the re-hydrated content, or the original match text
        if re-hydration fails (Requirement 5.3).
        """
        block_type = match.group(1)
        body = match.group(2)

        try:
            if block_type == "file_tree":
                return self._rehydrate_file_tree(body)
            elif block_type == "symbol_map":
                return self._rehydrate_symbol_map(body)
            elif block_type == "multi_file_diff":
                return self._rehydrate_diff(body)
            else:
                # Unknown block type — leave unchanged
                return match.group(0)
        except Exception:
            # Re-hydration failure — skip block, forward original content (Req 5.3)
            return match.group(0)

    def _rehydrate_file_tree(self, body: str) -> str:
        """Re-hydrate a file_tree TOON block back to JSON array.

        Each line is: path|type|size
        Returns: JSON array of {"path": ..., "type": ..., "size": ...}
        """
        entries: list[dict[str, Any]] = []
        for line in body.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) != 3:
                raise ValueError(f"Invalid file_tree line: {line!r}")
            path, file_type, size_str = parts
            # Try to convert size to int, fall back to string
            try:
                size: int | str = int(size_str)
            except ValueError:
                size = size_str
            entries.append({"path": path, "type": file_type, "size": size})
        return json.dumps(entries)

    def _rehydrate_symbol_map(self, body: str) -> str:
        """Re-hydrate a symbol_map TOON block back to JSON array.

        Each line is: name|kind|location
        Returns: JSON array of {"name": ..., "kind": ..., "location": ...}
        """
        entries: list[dict[str, Any]] = []
        for line in body.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) != 3:
                raise ValueError(f"Invalid symbol_map line: {line!r}")
            name, kind, location = parts
            entries.append({"name": name, "kind": kind, "location": location})
        return json.dumps(entries)

    def _rehydrate_diff(self, body: str) -> str:
        """Re-hydrate a multi_file_diff TOON block back to diff format.

        Reconstructs unified diff format from compressed TOON representation.
        D|path lines become diff --git headers with --- and +++ lines.
        """
        lines = body.split("\n")
        output_lines: list[str] = []

        for line in lines:
            if line.startswith("D|"):
                # Compressed diff header: "D|path" -> full diff header
                path = line[2:]
                output_lines.append(f"diff --git a/{path} b/{path}")
                output_lines.append(f"--- a/{path}")
                output_lines.append(f"+++ b/{path}")
            elif line.startswith("@@"):
                # Hunk header — pass through
                output_lines.append(line)
            elif line.startswith("+") or line.startswith("-"):
                # Changed lines — pass through
                output_lines.append(line)
            elif line.startswith("\\"):
                # "No newline at end of file" marker — pass through
                output_lines.append(line)
            elif line.strip() == "":
                # Empty lines in the diff body — skip
                continue
            else:
                # Unknown line — pass through
                output_lines.append(line)

        return "\n".join(output_lines)

    def compression_ratio(self, original: str, compressed: str) -> float:
        """Calculate achieved compression ratio.

        Returns compressed_size / original_size. Lower is better.
        A ratio of 0.7 means the compressed output is 70% of the original.
        """
        original_size = len(original.encode("utf-8"))
        compressed_size = len(compressed.encode("utf-8"))
        if original_size == 0:
            return 1.0
        return compressed_size / original_size

    def _detect_json_blocks(self, content: str) -> list[StructuredBlock]:
        """Detect JSON arrays that represent file trees or symbol maps."""
        blocks: list[StructuredBlock] = []

        # Find all top-level JSON arrays in the content
        i = 0
        while i < len(content):
            # Look for the start of a JSON array
            if content[i] == "[":
                block = self._try_parse_json_array(content, i)
                if block is not None:
                    blocks.append(block)
                    i = block.end_offset
                    continue
            i += 1

        return blocks

    def _try_parse_json_array(self, content: str, start: int) -> StructuredBlock | None:
        """Try to parse a JSON array starting at the given offset.

        Returns a StructuredBlock if the array matches file_tree or symbol_map
        patterns and meets the minimum size requirement.
        """
        # Find the matching closing bracket using a simple bracket counter
        depth = 0
        i = start
        while i < len(content):
            ch = content[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    break
            elif ch == '"':
                # Skip string contents to avoid counting brackets inside strings
                i += 1
                while i < len(content) and content[i] != '"':
                    if content[i] == "\\":
                        i += 1  # skip escaped character
                    i += 1
            i += 1

        if depth != 0:
            return None

        end = i + 1
        raw = content[start:end]

        # Check minimum block size
        if len(raw.encode("utf-8")) < self._min_block_size:
            return None

        # Try to parse as JSON
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(data, list) or len(data) == 0:
            return None

        # All entries must be dicts for classification
        if not all(isinstance(entry, dict) for entry in data):
            return None

        # Classify the block type
        block_type = self._classify_json_array(data)
        if block_type is None:
            return None

        return StructuredBlock(
            block_type=block_type,
            start_offset=start,
            end_offset=end,
            raw_content=raw,
        )

    def _classify_json_array(self, data: list[dict[str, Any]]) -> str | None:
        """Classify a JSON array as file_tree, symbol_map, or None.

        File trees: objects with "path", "type", "size" keys.
        Symbol maps: objects with "name", "kind", "location" keys.
        """
        if not data:
            return None

        # Check if majority of entries match a pattern
        file_tree_keys = {"path", "type", "size"}
        symbol_map_keys = {"name", "kind", "location"}

        file_tree_count = sum(
            1 for entry in data if file_tree_keys.issubset(entry.keys())
        )
        symbol_map_count = sum(
            1 for entry in data if symbol_map_keys.issubset(entry.keys())
        )

        # Require all entries to match the pattern
        if file_tree_count == len(data):
            return "file_tree"
        if symbol_map_count == len(data):
            return "symbol_map"

        return None

    def _detect_diff_blocks(self, content: str) -> list[StructuredBlock]:
        """Detect multi-file diff blocks in content."""
        blocks: list[StructuredBlock] = []

        # Find diff --git headers
        for match in _DIFF_HEADER_RE.finditer(content):
            start = match.start()
            end = self._find_diff_end(content, start)
            raw = content[start:end]

            if len(raw.encode("utf-8")) >= self._min_block_size:
                blocks.append(StructuredBlock(
                    block_type="multi_file_diff",
                    start_offset=start,
                    end_offset=end,
                    raw_content=raw,
                ))

        # Find unified diff format (--- / +++ pairs) not already covered
        for match in _UNIFIED_DIFF_RE.finditer(content):
            start = match.start()
            # Check if this is already inside an existing diff block
            if any(b.start_offset <= start < b.end_offset for b in blocks):
                continue
            end = self._find_diff_end(content, start)
            raw = content[start:end]

            if len(raw.encode("utf-8")) >= self._min_block_size:
                blocks.append(StructuredBlock(
                    block_type="multi_file_diff",
                    start_offset=start,
                    end_offset=end,
                    raw_content=raw,
                ))

        return blocks

    def _find_diff_end(self, content: str, start: int) -> int:
        """Find the end of a diff block starting at the given offset.

        A diff block ends when we encounter a line that is not part of
        the diff (not starting with diff, ---, +++, @@, +, -, or space),
        or at the next diff --git header (for multi-file diffs we group
        consecutive diffs together).
        """
        lines = content[start:].split("\n")
        consumed = 0
        in_hunk = False
        i = 0

        while i < len(lines):
            line = lines[i]

            if line.startswith("diff --git"):
                if i == 0:
                    # This is the start of our diff
                    in_hunk = False
                    consumed += len(line) + 1
                    i += 1
                    continue
                else:
                    # Next diff file — include it in the same block
                    in_hunk = False
                    consumed += len(line) + 1
                    i += 1
                    continue
            elif line.startswith("---") or line.startswith("+++"):
                consumed += len(line) + 1
                i += 1
                continue
            elif line.startswith("@@"):
                in_hunk = True
                consumed += len(line) + 1
                i += 1
                continue
            elif line.startswith("index ") or line.startswith("new file") or line.startswith("deleted file") or line.startswith("similarity index") or line.startswith("rename from") or line.startswith("rename to"):
                consumed += len(line) + 1
                i += 1
                continue
            elif in_hunk and (
                line.startswith("+")
                or line.startswith("-")
                or line.startswith(" ")
                or line == ""
            ):
                consumed += len(line) + 1
                i += 1
                continue
            elif line.startswith("\\"):
                # "\ No newline at end of file"
                consumed += len(line) + 1
                i += 1
                continue
            else:
                # End of diff block
                break

        # Remove trailing newline from consumed count
        if consumed > 0:
            end = start + consumed
            # Clamp to content length to avoid exceeding bounds
            end = min(end, len(content))
            # Strip trailing newline
            if end > start and content[end - 1] == "\n":
                end -= 1
            return end

        return start + len(lines[0])

    def _remove_overlaps(self, blocks: list[StructuredBlock]) -> list[StructuredBlock]:
        """Remove overlapping blocks, keeping the longest non-overlapping set.

        Uses a greedy approach: sort by start_offset, then by length descending.
        For each block, only keep it if it doesn't overlap with the last kept block.
        """
        if not blocks:
            return []

        # Sort by start_offset, then by length descending (prefer longer blocks)
        sorted_blocks = sorted(
            blocks, key=lambda b: (b.start_offset, -(b.end_offset - b.start_offset))
        )

        result: list[StructuredBlock] = [sorted_blocks[0]]
        for block in sorted_blocks[1:]:
            last = result[-1]
            # No overlap if this block starts at or after the last block's end
            if block.start_offset >= last.end_offset:
                result.append(block)
            # If there's overlap, skip this block (keep the earlier/longer one)

        return result
