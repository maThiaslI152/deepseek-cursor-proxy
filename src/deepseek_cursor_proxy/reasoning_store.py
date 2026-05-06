from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


def normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        function = {}

    arguments = function.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)

    normalized: dict[str, Any] = {
        "id": tool_call.get("id"),
        "type": tool_call.get("type") or "function",
        "function": {
            "name": function.get("name") or "",
            "arguments": arguments,
        },
    }
    return normalized


def tool_call_signature(tool_call: dict[str, Any]) -> str:
    normalized = normalize_tool_call(tool_call)
    normalized.pop("id", None)
    canonical = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tool_call_ids(message: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for tool_call in message.get("tool_calls") or []:
        if isinstance(tool_call, dict) and tool_call.get("id"):
            ids.append(str(tool_call["id"]))
    return ids


def tool_call_names(message: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
    return names


def message_signature(message: dict[str, Any]) -> str:
    tool_calls = [
        normalize_tool_call(tool_call)
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    ]
    payload = {
        "content": message.get("content") or "",
        "tool_calls": tool_calls,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_scope_message(message: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {"role": message.get("role")}
    for key in ("content", "name", "tool_call_id", "prefix"):
        if key in message:
            canonical[key] = message[key]
    if message.get("tool_calls"):
        canonical["tool_calls"] = [
            normalize_tool_call(tool_call)
            for tool_call in message.get("tool_calls") or []
            if isinstance(tool_call, dict)
        ]
    return canonical


def conversation_scope(messages: list[dict[str, Any]], namespace: str = "") -> str:
    scope_messages = [canonical_scope_message(message) for message in messages]
    payload: Any = scope_messages
    if namespace:
        payload = {"namespace": namespace, "messages": scope_messages}
    return _sha256_json(payload)


def turn_context_signature(prior_messages: list[dict[str, Any]]) -> str:
    last_user_index = next(
        (
            index
            for index in range(len(prior_messages) - 1, -1, -1)
            if prior_messages[index].get("role") == "user"
        ),
        -1,
    )
    start_index = 0
    if last_user_index != -1:
        start_index = last_user_index
        while start_index > 0 and prior_messages[start_index - 1].get("role") == "user":
            start_index -= 1

    context_messages = [
        canonical_scope_message(message)
        for message in prior_messages[start_index:]
        if message.get("role") != "system"
    ]
    return _sha256_json(context_messages)


def scoped_reasoning_keys(message: dict[str, Any], scope: str) -> list[str]:
    keys = [f"scope:{scope}:signature:{message_signature(message)}"]
    keys.extend(
        f"scope:{scope}:tool_call:{tool_call_id}"
        for tool_call_id in tool_call_ids(message)
    )
    keys.extend(
        f"scope:{scope}:tool_call_signature:{tool_call_signature(tool_call)}"
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    )
    # Recovery-of-last-resort key. Catches the case where a streaming response
    # was interrupted (user pressed Stop) before the tool_call.id chunk arrived,
    # so neither tool_call_id nor tool_call_signature (which canonicalizes
    # arguments) survives the round-trip through Cursor's transcript.
    keys.extend(
        f"scope:{scope}:tool_name:{tool_name}" for tool_name in tool_call_names(message)
    )
    return keys


def portable_reasoning_keys(
    message: dict[str, Any],
    cache_namespace: str,
    prior_messages: list[dict[str, Any]],
) -> list[str]:
    if not cache_namespace:
        return []

    turn_signature = turn_context_signature(prior_messages)
    keys = [
        f"namespace:{cache_namespace}:turn:{turn_signature}:"
        f"signature:{message_signature(message)}"
    ]
    keys.extend(
        f"namespace:{cache_namespace}:turn:{turn_signature}:"
        f"tool_call:{tool_call_id}"
        for tool_call_id in tool_call_ids(message)
    )
    keys.extend(
        f"namespace:{cache_namespace}:turn:{turn_signature}:"
        f"tool_call_signature:{tool_call_signature(tool_call)}"
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    )
    keys.extend(
        f"namespace:{cache_namespace}:turn:{turn_signature}:" f"tool_name:{tool_name}"
        for tool_name in tool_call_names(message)
    )
    return keys


def namespace_reasoning_keys(
    message: dict[str, Any],
    cache_namespace: str,
) -> list[str]:
    """Broad namespace-scoped keys that survive conversation prefix changes.

    Unlike scope keys (which incorporate the full message prefix hash) and
    portable keys (which incorporate the turn-context tail hash), these keys
    only depend on the cache_namespace (API config + auth hash) plus the
    message content itself. This means two different conversations with the
    same API key and same tool call signature share cached reasoning,
    providing recall across Cursor context resets.

    Lookup should try scope keys -> portable keys -> namespace keys, in order.
    """
    if not cache_namespace:
        return []
    keys = [f"namespace:{cache_namespace}:signature:{message_signature(message)}"]
    keys.extend(
        f"namespace:{cache_namespace}:tool_call:{tool_call_id}"
        for tool_call_id in tool_call_ids(message)
    )
    keys.extend(
        f"namespace:{cache_namespace}:tool_call_signature:{tool_call_signature(tool_call)}"
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    )
    return keys


SCHEMA_VERSION = 2
SCHEMA_META_TABLE = "reasoning_cache_meta"


class ReasoningStore:
    def __init__(
        self,
        reasoning_content_path: str | Path,
        max_age_seconds: int | None = None,
        max_rows: int | None = None,
    ) -> None:
        self.max_age_seconds = max_age_seconds
        self.max_rows = max_rows
        if str(reasoning_content_path) == ":memory:":
            self.reasoning_content_path: str | Path = ":memory:"
        else:
            self.reasoning_content_path = Path(reasoning_content_path).expanduser()
            self.reasoning_content_path.parent.mkdir(
                mode=0o700, parents=True, exist_ok=True
            )
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.reasoning_content_path, check_same_thread=False
        )
        if isinstance(self.reasoning_content_path, Path):
            self.reasoning_content_path.chmod(0o600)

        # Performance pragmas: WAL for concurrent reads, mmap for fast access
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA mmap_size=268435456")
        self._conn.execute("PRAGMA cache_size=-64000")

        # Create v2 data tables (IF NOT EXISTS so existing v1 tables are
        # preserved for migration; _migrate handles replacement).
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_texts (
                hash TEXT PRIMARY KEY,
                reasoning TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_cache (
                key TEXT PRIMARY KEY,
                reasoning_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reasoning_cache_created
            ON reasoning_cache(created_at)
        """)

        self._init_schema()
        self._migrate()
        self.prune()

    @property
    def schema_version(self) -> int:
        with self._lock:
            row = self._conn.execute(
                f"SELECT value FROM {SCHEMA_META_TABLE} WHERE key = 'schema_version'"
            ).fetchone()
            return int(row[0]) if row else 0

    def _init_schema(self) -> None:
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_META_TABLE} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """)
        # Only insert version on fresh DB — do NOT overwrite existing versions
        # so that _migrate can detect the old version and upgrade.
        self._conn.execute(
            f"""
            INSERT INTO {SCHEMA_META_TABLE}(key, value)
            VALUES ('schema_version', ?)
            ON CONFLICT(key) DO NOTHING
            """,
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    def _migrate(self) -> None:
        version = self.schema_version
        if version >= SCHEMA_VERSION:
            return

        if version == 1:
            # v1→v2: Normalize reasoning_text into separate table to
            # eliminate row-count blowup (each assistant message was stored
            # under N keys with full reasoning text duplicated N times).
            columns = [
                row[1]
                for row in self._conn.execute(
                    "PRAGMA table_info(reasoning_cache)"
                ).fetchall()
            ]
            if "reasoning" in columns:
                old_rows = self._conn.execute(
                    "SELECT key, reasoning, created_at FROM reasoning_cache"
                ).fetchall()

                self._conn.execute("DROP TABLE IF EXISTS reasoning_cache")
                self._conn.execute("""
                    CREATE TABLE reasoning_cache (
                        key TEXT PRIMARY KEY,
                        reasoning_hash TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                """)

                seen: set[str] = set()
                for key, reasoning, created_at in old_rows:
                    hash_val = hashlib.sha256(
                        reasoning.encode("utf-8")
                    ).hexdigest()
                    if hash_val not in seen:
                        self._conn.execute(
                            "INSERT OR IGNORE INTO reasoning_texts"
                            "(hash, reasoning, created_at) VALUES (?, ?, ?)",
                            (hash_val, reasoning, created_at),
                        )
                        seen.add(hash_val)
                    self._conn.execute(
                        "INSERT OR REPLACE INTO reasoning_cache"
                        "(key, reasoning_hash, created_at) VALUES (?, ?, ?)",
                        (key, hash_val, created_at),
                    )

            self._set_meta("schema_version", str(SCHEMA_VERSION))
            self._conn.commit()
            return

        if version == 0:
            # v0→v1: No-op, bump version to enable v1→v2 migration
            self._set_meta("schema_version", "1")
            self._conn.commit()
            return self._migrate()

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            f"""
            INSERT INTO {SCHEMA_META_TABLE}(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _reasoning_hash(self, reasoning: str) -> str:
        return hashlib.sha256(reasoning.encode("utf-8")).hexdigest()

    def put(self, key: str, reasoning: str, message: dict[str, Any] | None = None) -> None:
        """Store a single key→reasoning mapping (backward-compatible convenience)."""
        if not isinstance(reasoning, str):
            return
        now = time.time()
        hash_val = self._reasoning_hash(reasoning)
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO reasoning_texts"
                "(hash, reasoning, created_at) VALUES (?, ?, ?)",
                (hash_val, reasoning, now),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO reasoning_cache"
                "(key, reasoning_hash, created_at) VALUES (?, ?, ?)",
                (key, hash_val, now),
            )
            self._prune_locked()
            self._conn.commit()

    def get(self, key: str) -> str | None:
        """Retrieve reasoning text by cache key (single-key lookup)."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT rt.reasoning
                FROM reasoning_cache rc
                JOIN reasoning_texts rt ON rc.reasoning_hash = rt.hash
                WHERE rc.key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def batch_lookup(self, keys: list[str]) -> tuple[str | None, str | None]:
        """Try all keys in priority order, return (matched_key, reasoning).

        Uses a single query with ORDER BY CASE to maintain caller priority
        ordering, eliminating up to ~15 individual SELECT round-trips per
        message lookup.
        """
        if not keys:
            return (None, None)
        placeholders = ",".join("?" for _ in keys)
        case_expr = (
            "CASE key "
            + " ".join(f"WHEN ? THEN {i}" for i in range(len(keys)))
            + " END"
        )
        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT rc.key, rt.reasoning
                FROM reasoning_cache rc
                JOIN reasoning_texts rt ON rc.reasoning_hash = rt.hash
                WHERE rc.key IN ({placeholders})
                ORDER BY {case_expr}
                LIMIT 1
                """,
                keys + keys,
            ).fetchone()
        if row is None:
            return (None, None)
        return (str(row[0]), str(row[1]))

    def store_assistant_message(
        self,
        message: dict[str, Any],
        scope: str,
        cache_namespace: str = "",
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> int:
        """Store reasoning for an assistant message under all relevant key types.

        Uses a single batched INSERT instead of N individual put() calls,
        with deduplicated reasoning text via the reasoning_texts table.
        """
        if message.get("role") != "assistant":
            return 0
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            return 0

        keys = scoped_reasoning_keys(message, scope)
        if cache_namespace:
            keys.extend(namespace_reasoning_keys(message, cache_namespace))
        if prior_messages is not None:
            keys.extend(
                portable_reasoning_keys(message, cache_namespace, prior_messages)
            )
        keys = list(dict.fromkeys(keys))
        if not keys:
            return 0

        now = time.time()
        hash_val = self._reasoning_hash(reasoning)
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO reasoning_texts"
                "(hash, reasoning, created_at) VALUES (?, ?, ?)",
                (hash_val, reasoning, now),
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO reasoning_cache"
                "(key, reasoning_hash, created_at) VALUES (?, ?, ?)",
                [(key, hash_val, now) for key in keys],
            )
            self._prune_locked()
            self._conn.commit()
        return len(keys)

    def lookup_for_message(
        self,
        message: dict[str, Any],
        scope: str,
        cache_namespace: str = "",
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Priority-ordered cache lookup using the batched query."""
        keys: list[str] = []
        # Priority 1: exact conversation scope keys
        keys.extend(scoped_reasoning_keys(message, scope))
        # Priority 2: portable turn-context keys (same tail, different prefix)
        if prior_messages is not None:
            keys.extend(
                portable_reasoning_keys(message, cache_namespace, prior_messages)
            )
        # Priority 3: broad namespace keys (any conversation, same API config)
        if cache_namespace:
            keys.extend(namespace_reasoning_keys(message, cache_namespace))

        _, reasoning = self.batch_lookup(keys)
        return reasoning

    def warm_cache(
        self,
        message: dict[str, Any],
        reasoning: str,
        scope: str,
        cache_namespace: str = "",
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> int:
        """Warm cache by pre-writing scope + portable keys on a namespace hit.

        After a context reset triggers a Priority 3 (namespace) hit, this
        ensures the *next* turn in the same conversation hits at Priority 1
        (scope) — zero wasted lookups.
        """
        if not isinstance(reasoning, str):
            return 0
        message_with_reasoning = dict(message)
        message_with_reasoning["reasoning_content"] = reasoning

        keys = list(
            dict.fromkeys(
                scoped_reasoning_keys(message_with_reasoning, scope)
                + (
                    portable_reasoning_keys(
                        message_with_reasoning, cache_namespace, prior_messages
                    )
                    if prior_messages is not None
                    else []
                )
            )
        )
        if not keys:
            return 0

        now = time.time()
        hash_val = self._reasoning_hash(reasoning)
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO reasoning_texts"
                "(hash, reasoning, created_at) VALUES (?, ?, ?)",
                (hash_val, reasoning, now),
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO reasoning_cache"
                "(key, reasoning_hash, created_at) VALUES (?, ?, ?)",
                [(key, hash_val, now) for key in keys],
            )
            self._conn.commit()
        return len(keys)

    def backfill_portable_aliases(
        self,
        message: dict[str, Any],
        reasoning: str,
        cache_namespace: str,
        prior_messages: list[dict[str, Any]],
    ) -> int:
        if not isinstance(reasoning, str):
            return 0
        keys = portable_reasoning_keys(message, cache_namespace, prior_messages)
        if not keys:
            return 0
        message_with_reasoning = dict(message)
        message_with_reasoning["reasoning_content"] = reasoning
        keys = list(dict.fromkeys(keys))

        now = time.time()
        hash_val = self._reasoning_hash(reasoning)
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO reasoning_texts"
                "(hash, reasoning, created_at) VALUES (?, ?, ?)",
                (hash_val, reasoning, now),
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO reasoning_cache"
                "(key, reasoning_hash, created_at) VALUES (?, ?, ?)",
                [(key, hash_val, now) for key in keys],
            )
            self._conn.commit()
        return len(keys)

    def clear(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM reasoning_cache"
            ).fetchone()
            count = int(row[0] if row else 0)
            self._conn.execute("DELETE FROM reasoning_cache")
            self._conn.execute("DELETE FROM reasoning_texts")
            self._conn.commit()
        return count

    def prune(self) -> int:
        with self._lock:
            deleted = self._prune_locked()
            self._conn.commit()
        return deleted

    def _prune_locked(self) -> int:
        deleted = 0
        if self.max_age_seconds is not None and self.max_age_seconds > 0:
            cutoff = time.time() - self.max_age_seconds
            cursor = self._conn.execute(
                "DELETE FROM reasoning_cache WHERE created_at < ?",
                (cutoff,),
            )
            deleted += cursor.rowcount if cursor.rowcount != -1 else 0

        if self.max_rows is not None and self.max_rows > 0:
            cursor = self._conn.execute(
                """
                DELETE FROM reasoning_cache
                WHERE key NOT IN (
                    SELECT key
                    FROM reasoning_cache
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                """,
                (self.max_rows,),
            )
            deleted += cursor.rowcount if cursor.rowcount != -1 else 0

        # Clean up orphaned reasoning_texts entries (no remaining cache refs)
        cursor = self._conn.execute("""
            DELETE FROM reasoning_texts
            WHERE hash NOT IN (
                SELECT DISTINCT reasoning_hash FROM reasoning_cache
            )
        """)
        deleted += cursor.rowcount if cursor.rowcount != -1 else 0
        return deleted
