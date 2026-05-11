"""Property-based tests for the Security Gateway outbound redaction.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4**

Property 15: Redaction Completeness
For any payload containing known secret patterns, after scan_outbound()
the result contains no unredacted secrets (all patterns are replaced
with [REDACTED]).

Property 16: Redaction Immutability
For any payload, scan_outbound() never mutates the original payload
(the input dict is unchanged after the call).
"""

from __future__ import annotations

import copy
import string

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from deepseek_cursor_proxy.rap.security import (
    SECRET_PATTERNS,
    SecurityConfig,
    SecurityGateway,
)


# --- Strategies ---

# Generate realistic API key patterns (sk-/pk-/api_key- followed by 20+ alphanumeric chars)
api_key_strategy = st.builds(
    lambda prefix, suffix: f"{prefix}_{suffix}",
    prefix=st.sampled_from(["sk", "pk", "api_key", "api-key"]),
    suffix=st.text(
        alphabet=string.ascii_letters + string.digits,
        min_size=20,
        max_size=40,
    ),
)

# Generate AWS key patterns (AKIA followed by 16 uppercase alphanumeric chars)
aws_key_strategy = st.builds(
    lambda suffix: f"AKIA{suffix}",
    suffix=st.text(
        alphabet=string.ascii_uppercase + string.digits,
        min_size=16,
        max_size=16,
    ),
)

# Generate GitHub token patterns (ghp_/ghs_ followed by 36+ alphanumeric chars)
github_token_strategy = st.builds(
    lambda prefix, suffix: f"{prefix}_{suffix}",
    prefix=st.sampled_from(["ghp", "ghs"]),
    suffix=st.text(
        alphabet=string.ascii_letters + string.digits,
        min_size=36,
        max_size=50,
    ),
)

# Generate SSH key header patterns
ssh_key_strategy = st.sampled_from([
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN EC PRIVATE KEY-----",
    "-----BEGIN DSA PRIVATE KEY-----",
])

# Generate environment variable patterns (UPPERCASE_NAME='value')
env_var_strategy = st.builds(
    lambda name, value: f"{name}='{value}'",
    name=st.text(
        alphabet=string.ascii_uppercase + "_",
        min_size=2,
        max_size=20,
    ).filter(lambda s: len(s) >= 2 and s[0] != "_"),
    value=st.text(
        alphabet=string.ascii_letters + string.digits + "/+=",
        min_size=8,
        max_size=30,
    ),
)

# Combined secret strategy
secret_strategy = st.one_of(
    api_key_strategy,
    aws_key_strategy,
    github_token_strategy,
    ssh_key_strategy,
    env_var_strategy,
)

# Generate surrounding text (non-secret content)
surrounding_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        whitelist_characters=" \n\t.,;:!?()[]{}",
    ),
    min_size=0,
    max_size=100,
)

# Generate a message content string that embeds one or more secrets
def _build_content_with_secrets(prefix: str, secret: str, suffix: str) -> str:
    return f"{prefix} {secret} {suffix}"


content_with_secret = st.builds(
    _build_content_with_secrets,
    prefix=surrounding_text,
    secret=secret_strategy,
    suffix=surrounding_text,
)

# Generate a payload with messages containing secrets
message_with_secret = st.builds(
    lambda role, content: {"role": role, "content": content},
    role=st.sampled_from(["user", "assistant", "system"]),
    content=content_with_secret,
)

payload_with_secrets = st.builds(
    lambda messages: {"messages": messages},
    messages=st.lists(message_with_secret, min_size=1, max_size=5),
)

# Generate arbitrary payloads (may or may not contain secrets)
arbitrary_content = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=0,
    max_size=200,
)

arbitrary_message = st.builds(
    lambda role, content: {"role": role, "content": content},
    role=st.sampled_from(["user", "assistant", "system"]),
    content=arbitrary_content,
)

arbitrary_payload = st.builds(
    lambda messages: {"messages": messages},
    messages=st.lists(arbitrary_message, min_size=0, max_size=5),
)


# --- Fixtures ---

def _make_gateway() -> SecurityGateway:
    """Create a SecurityGateway with default config."""
    return SecurityGateway(SecurityConfig())


# --- Property 15: Redaction Completeness ---


class TestProperty15RedactionCompleteness:
    """For any payload containing known secret patterns, after scan_outbound()
    the result contains no unredacted secrets (all patterns are replaced
    with [REDACTED]).

    **Validates: Requirements 8.1, 8.2**
    """

    @given(payload=payload_with_secrets)
    @settings(max_examples=30)
    def test_no_api_keys_survive_redaction(self, payload: dict) -> None:
        """After scan_outbound, no API key patterns remain in the output.

        **Validates: Requirements 8.1, 8.2**
        """
        gateway = _make_gateway()
        result, _redactions = gateway.scan_outbound(payload)

        for message in result.get("messages", []):
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            # Check that no API key pattern matches in the result
            pattern_name, pattern = SECRET_PATTERNS[0]  # api_key pattern
            assert not pattern.search(content), (
                f"API key pattern still found in redacted content: {content!r}"
            )

    @given(payload=payload_with_secrets)
    @settings(max_examples=30)
    def test_no_aws_keys_survive_redaction(self, payload: dict) -> None:
        """After scan_outbound, no AWS key patterns remain in the output.

        **Validates: Requirements 8.1, 8.2**
        """
        gateway = _make_gateway()
        result, _redactions = gateway.scan_outbound(payload)

        for message in result.get("messages", []):
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            pattern_name, pattern = SECRET_PATTERNS[1]  # aws_key pattern
            assert not pattern.search(content), (
                f"AWS key pattern still found in redacted content: {content!r}"
            )

    @given(payload=payload_with_secrets)
    @settings(max_examples=30)
    def test_no_ssh_keys_survive_redaction(self, payload: dict) -> None:
        """After scan_outbound, no SSH key patterns remain in the output.

        **Validates: Requirements 8.1, 8.2**
        """
        gateway = _make_gateway()
        result, _redactions = gateway.scan_outbound(payload)

        for message in result.get("messages", []):
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            pattern_name, pattern = SECRET_PATTERNS[2]  # ssh_key pattern
            assert not pattern.search(content), (
                f"SSH key pattern still found in redacted content: {content!r}"
            )

    @given(payload=payload_with_secrets)
    @settings(max_examples=30)
    def test_no_github_tokens_survive_redaction(self, payload: dict) -> None:
        """After scan_outbound, no GitHub token patterns remain in the output.

        **Validates: Requirements 8.1, 8.2**
        """
        gateway = _make_gateway()
        result, _redactions = gateway.scan_outbound(payload)

        for message in result.get("messages", []):
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            pattern_name, pattern = SECRET_PATTERNS[5]  # github_token pattern
            assert not pattern.search(content), (
                f"GitHub token pattern still found in redacted content: {content!r}"
            )

    @given(payload=payload_with_secrets)
    @settings(max_examples=30)
    def test_all_known_patterns_redacted(self, payload: dict) -> None:
        """After scan_outbound, no known secret pattern matches in the output.

        **Validates: Requirements 8.1, 8.2**
        """
        gateway = _make_gateway()
        result, _redactions = gateway.scan_outbound(payload)

        for message in result.get("messages", []):
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            for pattern_name, pattern in SECRET_PATTERNS:
                assert not pattern.search(content), (
                    f"Pattern '{pattern_name}' still found in redacted content: "
                    f"{content!r}"
                )

    @given(payload=payload_with_secrets)
    @settings(max_examples=30)
    def test_redaction_produces_redacted_marker(self, payload: dict) -> None:
        """When secrets are present, the output contains [REDACTED] markers.

        **Validates: Requirements 8.2**
        """
        gateway = _make_gateway()
        result, redactions = gateway.scan_outbound(payload)

        # If redactions were made, [REDACTED] should appear in the output
        if redactions:
            all_content = " ".join(
                msg.get("content", "")
                for msg in result.get("messages", [])
                if isinstance(msg.get("content"), str)
            )
            assert "[REDACTED]" in all_content


# --- Property 16: Redaction Immutability ---


class TestProperty16RedactionImmutability:
    """For any payload, scan_outbound() never mutates the original payload
    (the input dict is unchanged after the call).

    **Validates: Requirements 8.4**
    """

    @given(payload=arbitrary_payload)
    @settings(max_examples=30)
    def test_original_payload_unchanged(self, payload: dict) -> None:
        """The original payload dict is identical before and after scan_outbound.

        **Validates: Requirements 8.4**
        """
        gateway = _make_gateway()
        original_copy = copy.deepcopy(payload)
        gateway.scan_outbound(payload)
        assert payload == original_copy, (
            "scan_outbound mutated the original payload"
        )

    @given(payload=payload_with_secrets)
    @settings(max_examples=30)
    def test_original_payload_unchanged_with_secrets(self, payload: dict) -> None:
        """Even when secrets are found and redacted, the original is unchanged.

        **Validates: Requirements 8.4**
        """
        gateway = _make_gateway()
        original_copy = copy.deepcopy(payload)
        gateway.scan_outbound(payload)
        assert payload == original_copy, (
            "scan_outbound mutated the original payload when secrets were present"
        )

    @given(payload=arbitrary_payload)
    @settings(max_examples=30)
    def test_result_is_independent_copy(self, payload: dict) -> None:
        """The returned result is a separate object from the input.

        **Validates: Requirements 8.4**
        """
        gateway = _make_gateway()
        result, _ = gateway.scan_outbound(payload)
        # Mutating the result should not affect the original
        if result.get("messages"):
            result["messages"].append({"role": "test", "content": "injected"})
            assert payload != result or not payload.get("messages") or \
                {"role": "test", "content": "injected"} not in payload.get("messages", [])
