from __future__ import annotations

from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest

from deepseek_cursor_proxy.reasoning_store import ReasoningStore, conversation_scope


class ReasoningStoreTests(unittest.TestCase):
    def test_file_store_creates_private_database_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            reasoning_content_path = (
                Path(temp_dir) / "nested" / "reasoning_content.sqlite3"
            )

            store = ReasoningStore(reasoning_content_path)
            store.close()

            self.assertTrue(reasoning_content_path.exists())
            self.assertEqual(stat.S_IMODE(reasoning_content_path.stat().st_mode), 0o600)

    def test_store_prunes_to_max_rows_and_can_clear(self) -> None:
        store = ReasoningStore(":memory:", max_rows=2)
        try:
            store.put("a", "reasoning a", {"role": "assistant"})
            store.put("b", "reasoning b", {"role": "assistant"})
            store.put("c", "reasoning c", {"role": "assistant"})

            self.assertIsNone(store.get("a"))
            self.assertEqual(store.get("b"), "reasoning b")
            self.assertEqual(store.get("c"), "reasoning c")
            self.assertEqual(store.clear(), 2)
            self.assertIsNone(store.get("b"))
            self.assertIsNone(store.get("c"))
        finally:
            store.close()

    def test_empty_reasoning_content_is_stored_as_present_value(self) -> None:
        store = ReasoningStore(":memory:")
        try:
            scope = conversation_scope([{"role": "user", "content": "lookup"}])
            tool_call = {
                "id": "call_empty",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
            message = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [tool_call],
            }

            self.assertGreater(store.store_assistant_message(message, scope), 0)
            self.assertEqual(store.get(f"scope:{scope}:tool_call:call_empty"), "")
            self.assertEqual(
                store.lookup_for_message(
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                    scope,
                ),
                "",
            )
        finally:
            store.close()

    def test_dedup_reasoning_text_normalized_across_keys(self) -> None:
        """Same reasoning text stored under N keys produces 1 reasoning_texts row."""
        store = ReasoningStore(":memory:")
        try:
            scope = conversation_scope([{"role": "user", "content": "hi"}])
            message = {
                "role": "assistant",
                "content": "Hello",
                "reasoning_content": "I am thinking about this.",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f1", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "f2", "arguments": "{}"}},
                ],
            }
            count = store.store_assistant_message(message, scope)
            self.assertGreater(count, 0)

            with store._lock:
                rows = store._conn.execute(
                    "SELECT COUNT(*) FROM reasoning_texts"
                ).fetchone()
                self.assertEqual(
                    rows[0], 1,
                    "Multiple cache keys must share one reasoning_texts row",
                )
        finally:
            store.close()

    def test_batch_lookup_hits_first_matching_key(self) -> None:
        """batch_lookup returns the first match in priority order."""
        store = ReasoningStore(":memory:")
        try:
            store.put("z_last", "last val")
            store.put("a_first", "first val")
            store.put("m_mid", "mid val")

            matched, reasoning = store.batch_lookup(
                ["a_first", "m_mid", "z_last"]
            )
            self.assertEqual(matched, "a_first")
            self.assertEqual(reasoning, "first val")

            matched, reasoning = store.batch_lookup(
                ["missing1", "z_last", "missing2"]
            )
            self.assertEqual(matched, "z_last")
            self.assertEqual(reasoning, "last val")

            matched, reasoning = store.batch_lookup(["nope"])
            self.assertIsNone(matched)
            self.assertIsNone(reasoning)

            matched, reasoning = store.batch_lookup([])
            self.assertIsNone(matched)
            self.assertIsNone(reasoning)
        finally:
            store.close()

    def test_warm_cache_adds_scope_and_portable_keys(self) -> None:
        """warm_cache stores scope+portable keys so next lookup is Priority-1."""
        store = ReasoningStore(":memory:")
        try:
            scope = conversation_scope([{"role": "user", "content": "ask"}])
            ns = "test_namespace"
            prior = [{"role": "user", "content": "ask"}]

            msg = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "My reasoning.",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            }

            warmed = store.warm_cache(msg, "My reasoning.", scope, ns, prior)
            self.assertGreater(warmed, 0)

            # scope key should be present
            from deepseek_cursor_proxy.reasoning_store import message_signature
            scope_key = f"scope:{scope}:signature:{message_signature(msg)}"
            self.assertEqual(store.get(scope_key), "My reasoning.")

            # portable key should be present
            from deepseek_cursor_proxy.reasoning_store import (
                portable_reasoning_keys,
            )
            portable = portable_reasoning_keys(msg, ns, prior)
            self.assertTrue(any(store.get(k) == "My reasoning." for k in portable))
        finally:
            store.close()

    def test_warm_cache_skips_missing_reasoning(self) -> None:
        """warm_cache returns 0 when reasoning is not a string."""
        store = ReasoningStore(":memory:")
        try:
            result = store.warm_cache(
                {"role": "assistant", "content": ""},
                None,
                "scope",
            )
            self.assertEqual(result, 0)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
