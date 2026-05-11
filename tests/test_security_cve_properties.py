"""Property-based tests for CVE Finding structural completeness.

**Validates: Requirement 9.3**

Property 17: CVE Finding Structural Completeness
For any CVE_Finding produced by scan_inbound(), it SHALL have a non-empty
cve_type, a severity in {low, medium, high, critical}, a non-empty
code_snippet, a valid line_range tuple, and a non-empty recommendation.
"""

from __future__ import annotations

import json
import string
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from deepseek_cursor_proxy.rap.security import (
    CVEFinding,
    SecurityConfig,
    SecurityGateway,
)


# --- Strategies ---

# Valid CVE types that the LM Studio model might return
VALID_CVE_TYPES = [
    "buffer_overflow",
    "hardcoded_credential",
    "sql_injection",
    "xss",
    "path_traversal",
    "command_injection",
    "insecure_deserialization",
    "use_after_free",
    "integer_overflow",
    "race_condition",
]

# Valid severity levels
VALID_SEVERITIES = ["low", "medium", "high", "critical"]

# Strategy for CVE type strings (including arbitrary non-empty strings)
cve_type_strategy = st.one_of(
    st.sampled_from(VALID_CVE_TYPES),
    st.text(
        alphabet=string.ascii_lowercase + "_",
        min_size=1,
        max_size=30,
    ).filter(lambda s: s.strip()),
)

# Strategy for severity (mix of valid and invalid to test normalization)
severity_strategy = st.one_of(
    st.sampled_from(VALID_SEVERITIES),
    st.text(
        alphabet=string.ascii_lowercase + "_",
        min_size=1,
        max_size=20,
    ),
)

# Strategy for recommendation strings
recommendation_strategy = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=100,
).filter(lambda s: s.strip())

# Strategy for line numbers
line_start_strategy = st.integers(min_value=0, max_value=50)
line_end_strategy = st.integers(min_value=0, max_value=100)

# Strategy for a single vulnerability dict as returned by LM Studio
vulnerability_strategy = st.fixed_dictionaries({
    "cve_type": cve_type_strategy,
    "severity": severity_strategy,
    "line_start": line_start_strategy,
    "line_end": line_end_strategy,
    "recommendation": recommendation_strategy,
})

# Strategy for a list of vulnerabilities (1 to 5)
vulnerabilities_list_strategy = st.lists(
    vulnerability_strategy,
    min_size=1,
    max_size=5,
)

# Strategy for code blocks (multi-line code snippets)
code_line_strategy = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=60,
).filter(lambda s: s.strip() and "```" not in s)

code_block_strategy = st.lists(
    code_line_strategy,
    min_size=1,
    max_size=20,
).map(lambda lines: "\n".join(lines))


# --- Helpers ---

def _make_response(content: str) -> dict:
    """Create a chat completion response with given content."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _lm_studio_response(vulnerabilities: list[dict]) -> dict:
    """Create a mock LM Studio response."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(vulnerabilities),
                }
            }
        ]
    }


def _make_gateway() -> SecurityGateway:
    """Create a SecurityGateway with CVE scanning enabled."""
    return SecurityGateway(SecurityConfig(
        cve_scanning_enabled=True,
        local_security_model_url="http://localhost:1234/v1/chat/completions",
    ))


# --- Property 17: CVE Finding Structural Completeness ---


class TestProperty17CVEFindingStructuralCompleteness:
    """For any CVE_Finding produced by scan_inbound(), it SHALL have a non-empty
    cve_type, a severity in {low, medium, high, critical}, a non-empty
    code_snippet, a valid line_range tuple, and a non-empty recommendation.

    **Validates: Requirement 9.3**
    """

    @given(
        vulnerabilities=vulnerabilities_list_strategy,
        code=code_block_strategy,
    )
    @settings(max_examples=30)
    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_every_finding_has_non_empty_cve_type(
        self, mock_client_cls: MagicMock, vulnerabilities: list[dict], code: str
    ) -> None:
        """Every CVEFinding produced has a non-empty cve_type string.

        **Validates: Requirement 9.3**
        """
        assume(len(code.strip()) > 0)

        mock_response = MagicMock()
        mock_response.json.return_value = _lm_studio_response(vulnerabilities)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        response = _make_response(f"```python\n{code}\n```")
        gateway = _make_gateway()

        _, findings = gateway.scan_inbound(response)

        for finding in findings:
            assert isinstance(finding.cve_type, str), (
                f"cve_type is not a string: {finding.cve_type!r}"
            )
            assert len(finding.cve_type) > 0, (
                f"cve_type is empty for finding: {finding}"
            )

    @given(
        vulnerabilities=vulnerabilities_list_strategy,
        code=code_block_strategy,
    )
    @settings(max_examples=30)
    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_every_finding_has_valid_severity(
        self, mock_client_cls: MagicMock, vulnerabilities: list[dict], code: str
    ) -> None:
        """Every CVEFinding produced has severity in {low, medium, high, critical}.

        **Validates: Requirement 9.3**
        """
        assume(len(code.strip()) > 0)

        mock_response = MagicMock()
        mock_response.json.return_value = _lm_studio_response(vulnerabilities)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        response = _make_response(f"```python\n{code}\n```")
        gateway = _make_gateway()

        _, findings = gateway.scan_inbound(response)

        valid_severities = {"low", "medium", "high", "critical"}
        for finding in findings:
            assert finding.severity in valid_severities, (
                f"severity '{finding.severity}' not in {valid_severities}"
            )

    @given(
        vulnerabilities=vulnerabilities_list_strategy,
        code=code_block_strategy,
    )
    @settings(max_examples=30)
    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_every_finding_has_non_empty_code_snippet(
        self, mock_client_cls: MagicMock, vulnerabilities: list[dict], code: str
    ) -> None:
        """Every CVEFinding produced has a non-empty code_snippet string.

        **Validates: Requirement 9.3**
        """
        assume(len(code.strip()) > 0)

        mock_response = MagicMock()
        mock_response.json.return_value = _lm_studio_response(vulnerabilities)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        response = _make_response(f"```python\n{code}\n```")
        gateway = _make_gateway()

        _, findings = gateway.scan_inbound(response)

        for finding in findings:
            assert isinstance(finding.code_snippet, str), (
                f"code_snippet is not a string: {finding.code_snippet!r}"
            )
            assert len(finding.code_snippet) > 0, (
                f"code_snippet is empty for finding: {finding}"
            )

    @given(
        vulnerabilities=vulnerabilities_list_strategy,
        code=code_block_strategy,
    )
    @settings(max_examples=30)
    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_every_finding_has_valid_line_range(
        self, mock_client_cls: MagicMock, vulnerabilities: list[dict], code: str
    ) -> None:
        """Every CVEFinding produced has a valid line_range tuple (start, end)
        where start >= 1 and start <= end.

        **Validates: Requirement 9.3**
        """
        assume(len(code.strip()) > 0)

        mock_response = MagicMock()
        mock_response.json.return_value = _lm_studio_response(vulnerabilities)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        response = _make_response(f"```python\n{code}\n```")
        gateway = _make_gateway()

        _, findings = gateway.scan_inbound(response)

        for finding in findings:
            assert isinstance(finding.line_range, tuple), (
                f"line_range is not a tuple: {finding.line_range!r}"
            )
            assert len(finding.line_range) == 2, (
                f"line_range does not have 2 elements: {finding.line_range}"
            )
            start, end = finding.line_range
            assert isinstance(start, int) and isinstance(end, int), (
                f"line_range elements are not ints: {finding.line_range}"
            )
            assert start >= 1, (
                f"line_range start {start} < 1"
            )
            assert start <= end, (
                f"line_range start {start} > end {end}"
            )

    @given(
        vulnerabilities=vulnerabilities_list_strategy,
        code=code_block_strategy,
    )
    @settings(max_examples=30)
    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_every_finding_has_non_empty_recommendation(
        self, mock_client_cls: MagicMock, vulnerabilities: list[dict], code: str
    ) -> None:
        """Every CVEFinding produced has a non-empty recommendation string.

        **Validates: Requirement 9.3**
        """
        assume(len(code.strip()) > 0)

        mock_response = MagicMock()
        mock_response.json.return_value = _lm_studio_response(vulnerabilities)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        response = _make_response(f"```python\n{code}\n```")
        gateway = _make_gateway()

        _, findings = gateway.scan_inbound(response)

        for finding in findings:
            assert isinstance(finding.recommendation, str), (
                f"recommendation is not a string: {finding.recommendation!r}"
            )
            assert len(finding.recommendation) > 0, (
                f"recommendation is empty for finding: {finding}"
            )

    @given(
        vulnerabilities=vulnerabilities_list_strategy,
        code=code_block_strategy,
    )
    @settings(max_examples=30)
    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_all_structural_fields_complete(
        self, mock_client_cls: MagicMock, vulnerabilities: list[dict], code: str
    ) -> None:
        """Combined check: every CVEFinding has all required fields valid.

        **Validates: Requirement 9.3**
        """
        assume(len(code.strip()) > 0)

        mock_response = MagicMock()
        mock_response.json.return_value = _lm_studio_response(vulnerabilities)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        response = _make_response(f"```python\n{code}\n```")
        gateway = _make_gateway()

        _, findings = gateway.scan_inbound(response)

        valid_severities = {"low", "medium", "high", "critical"}

        for finding in findings:
            # cve_type: non-empty string
            assert isinstance(finding.cve_type, str) and finding.cve_type, (
                f"Invalid cve_type: {finding.cve_type!r}"
            )
            # severity: in valid set
            assert finding.severity in valid_severities, (
                f"Invalid severity: {finding.severity!r}"
            )
            # code_snippet: non-empty string
            assert isinstance(finding.code_snippet, str) and finding.code_snippet, (
                f"Invalid code_snippet: {finding.code_snippet!r}"
            )
            # line_range: valid tuple
            assert isinstance(finding.line_range, tuple) and len(finding.line_range) == 2, (
                f"Invalid line_range: {finding.line_range!r}"
            )
            start, end = finding.line_range
            assert isinstance(start, int) and isinstance(end, int), (
                f"line_range elements not ints: {finding.line_range}"
            )
            assert 1 <= start <= end, (
                f"Invalid line_range values: start={start}, end={end}"
            )
            # recommendation: non-empty string
            assert isinstance(finding.recommendation, str) and finding.recommendation, (
                f"Invalid recommendation: {finding.recommendation!r}"
            )
