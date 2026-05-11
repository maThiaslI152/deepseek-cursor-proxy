"""Unit tests for the Security Gateway — inbound CVE scanning.

Tests code block extraction, LM Studio integration (mocked),
CVEFinding production, response annotation, and graceful degradation.

Requirements: 9.1, 9.2, 9.3, 9.4
"""

import copy
import json
from unittest.mock import patch, MagicMock

import httpx

from deepseek_cursor_proxy.rap.security import (
    CVEFinding,
    SecurityConfig,
    SecurityGateway,
)


def _make_response(content: str) -> dict:
    """Helper to create a chat completion response with given content."""
    return {
        "id": "chatcmpl-123",
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
    """Helper to create a mock LM Studio response."""
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


class TestCodeBlockExtraction:
    """Tests for extracting code blocks from response content."""

    def setup_method(self) -> None:
        self.config = SecurityConfig(cve_scanning_enabled=True)
        self.gateway = SecurityGateway(self.config)

    def test_extracts_single_code_block(self) -> None:
        """Requirement 9.1: Extracts code blocks from response."""
        content = "Here's some code:\n```python\nprint('hello')\n```"
        blocks = self.gateway._extract_code_blocks(content)
        assert len(blocks) == 1
        assert "print('hello')" in blocks[0]

    def test_extracts_multiple_code_blocks(self) -> None:
        """Requirement 9.1: Extracts all code blocks from response."""
        content = (
            "First:\n```python\nx = 1\n```\n"
            "Second:\n```javascript\nlet y = 2;\n```"
        )
        blocks = self.gateway._extract_code_blocks(content)
        assert len(blocks) == 2
        assert "x = 1" in blocks[0]
        assert "let y = 2;" in blocks[1]

    def test_extracts_code_block_without_language(self) -> None:
        """Requirement 9.1: Extracts code blocks without language identifier."""
        content = "Code:\n```\nsome code here\n```"
        blocks = self.gateway._extract_code_blocks(content)
        assert len(blocks) == 1
        assert "some code here" in blocks[0]

    def test_no_code_blocks_returns_empty(self) -> None:
        """No code blocks in content returns empty list."""
        content = "Just some text without any code blocks."
        blocks = self.gateway._extract_code_blocks(content)
        assert blocks == []

    def test_empty_content_returns_empty(self) -> None:
        """Empty content returns empty list."""
        blocks = self.gateway._extract_code_blocks("")
        assert blocks == []


class TestScanInboundWithMockedLMStudio:
    """Tests for scan_inbound with mocked LM Studio calls."""

    def setup_method(self) -> None:
        self.config = SecurityConfig(
            cve_scanning_enabled=True,
            local_security_model_url="http://localhost:1234/v1/chat/completions",
        )
        self.gateway = SecurityGateway(self.config)

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_produces_cve_finding_with_all_fields(self, mock_client_cls) -> None:
        """Requirement 9.3: Produces CVEFinding with type, severity, snippet, line_range, recommendation."""
        vuln_response = _lm_studio_response([
            {
                "cve_type": "sql_injection",
                "severity": "high",
                "line_start": 2,
                "line_end": 3,
                "recommendation": "Use parameterized queries.",
            }
        ])

        mock_response = MagicMock()
        mock_response.json.return_value = vuln_response
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        code = "import sqlite3\nquery = f\"SELECT * FROM users WHERE id={user_id}\"\ncursor.execute(query)"
        response = _make_response(f"Here's the code:\n```python\n{code}\n```")

        result, findings = self.gateway.scan_inbound(response)

        assert len(findings) == 1
        finding = findings[0]
        assert finding.cve_type == "sql_injection"
        assert finding.severity == "high"
        assert finding.line_range == (2, 3)
        assert finding.recommendation == "Use parameterized queries."
        assert "SELECT * FROM users" in finding.code_snippet

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_annotates_response_with_findings(self, mock_client_cls) -> None:
        """Requirement 9.4: Annotates response with findings for developer visibility."""
        vuln_response = _lm_studio_response([
            {
                "cve_type": "hardcoded_credential",
                "severity": "critical",
                "line_start": 1,
                "line_end": 1,
                "recommendation": "Use environment variables for credentials.",
            }
        ])

        mock_response = MagicMock()
        mock_response.json.return_value = vuln_response
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        code = 'password = "admin123"'
        response = _make_response(f"```python\n{code}\n```")

        result, findings = self.gateway.scan_inbound(response)

        # Response should be annotated
        annotated_content = result["choices"][0]["message"]["content"]
        assert "Security Scan Results" in annotated_content
        assert "hardcoded_credential" in annotated_content
        assert "CRITICAL" in annotated_content

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_no_vulnerabilities_returns_unchanged(self, mock_client_cls) -> None:
        """No vulnerabilities found returns response unchanged (except copy)."""
        vuln_response = _lm_studio_response([])

        mock_response = MagicMock()
        mock_response.json.return_value = vuln_response
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        code = "x = 1 + 2"
        response = _make_response(f"```python\n{code}\n```")

        result, findings = self.gateway.scan_inbound(response)

        assert len(findings) == 0
        # Content should not have annotation
        assert "Security Scan Results" not in result["choices"][0]["message"]["content"]

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_multiple_vulnerabilities(self, mock_client_cls) -> None:
        """Multiple vulnerabilities produce multiple CVEFindings."""
        vuln_response = _lm_studio_response([
            {
                "cve_type": "sql_injection",
                "severity": "high",
                "line_start": 1,
                "line_end": 1,
                "recommendation": "Use parameterized queries.",
            },
            {
                "cve_type": "xss",
                "severity": "medium",
                "line_start": 3,
                "line_end": 3,
                "recommendation": "Sanitize user input.",
            },
        ])

        mock_response = MagicMock()
        mock_response.json.return_value = vuln_response
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        code = "query = f'SELECT * FROM users WHERE id={uid}'\nresult = db.execute(query)\nhtml = f'<div>{user_input}</div>'"
        response = _make_response(f"```python\n{code}\n```")

        result, findings = self.gateway.scan_inbound(response)

        assert len(findings) == 2
        assert findings[0].cve_type == "sql_injection"
        assert findings[1].cve_type == "xss"


class TestScanInboundGracefulDegradation:
    """Tests for graceful degradation when LM Studio is unavailable."""

    def setup_method(self) -> None:
        self.config = SecurityConfig(
            cve_scanning_enabled=True,
            local_security_model_url="http://localhost:1234/v1/chat/completions",
        )
        self.gateway = SecurityGateway(self.config)

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_lm_studio_connection_error_returns_unchanged(self, mock_client_cls) -> None:
        """If LM Studio is unavailable, skip scanning gracefully."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        code = "password = 'secret'"
        response = _make_response(f"```python\n{code}\n```")
        original = copy.deepcopy(response)

        result, findings = self.gateway.scan_inbound(response)

        assert len(findings) == 0
        # Content should be unchanged (no annotation)
        assert result["choices"][0]["message"]["content"] == original["choices"][0]["message"]["content"]

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_lm_studio_timeout_returns_unchanged(self, mock_client_cls) -> None:
        """If LM Studio times out, skip scanning gracefully."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("Timeout")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        code = "eval(user_input)"
        response = _make_response(f"```python\n{code}\n```")

        result, findings = self.gateway.scan_inbound(response)

        assert len(findings) == 0

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_lm_studio_invalid_json_response(self, mock_client_cls) -> None:
        """If LM Studio returns invalid JSON, skip gracefully."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "not valid json"}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        code = "x = 1"
        response = _make_response(f"```python\n{code}\n```")

        result, findings = self.gateway.scan_inbound(response)

        assert len(findings) == 0

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_lm_studio_http_error(self, mock_client_cls) -> None:
        """If LM Studio returns HTTP error, skip gracefully."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        code = "x = 1"
        response = _make_response(f"```python\n{code}\n```")

        result, findings = self.gateway.scan_inbound(response)

        assert len(findings) == 0


class TestScanInboundDisabled:
    """Tests for when CVE scanning is disabled."""

    def test_disabled_scanning_returns_copy_unchanged(self) -> None:
        """When cve_scanning_enabled is False, skip scanning entirely."""
        config = SecurityConfig(cve_scanning_enabled=False)
        gateway = SecurityGateway(config)

        code = "password = 'admin123'"
        response = _make_response(f"```python\n{code}\n```")

        result, findings = gateway.scan_inbound(response)

        assert len(findings) == 0
        # Content unchanged
        assert result["choices"][0]["message"]["content"] == response["choices"][0]["message"]["content"]
        # But it's a copy
        assert result is not response


class TestScanInboundImmutability:
    """Tests that scan_inbound never mutates the original response."""

    def setup_method(self) -> None:
        self.config = SecurityConfig(cve_scanning_enabled=True)
        self.gateway = SecurityGateway(self.config)

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_original_response_unchanged(self, mock_client_cls) -> None:
        """Original response dict is not mutated."""
        vuln_response = _lm_studio_response([
            {
                "cve_type": "xss",
                "severity": "medium",
                "line_start": 1,
                "line_end": 1,
                "recommendation": "Sanitize input.",
            }
        ])

        mock_response = MagicMock()
        mock_response.json.return_value = vuln_response
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        code = "html = f'<div>{user_input}</div>'"
        response = _make_response(f"```python\n{code}\n```")
        original = copy.deepcopy(response)

        self.gateway.scan_inbound(response)

        # Original should be completely unchanged
        assert response == original


class TestScanInboundEdgeCases:
    """Tests for edge cases in scan_inbound."""

    def setup_method(self) -> None:
        self.config = SecurityConfig(cve_scanning_enabled=True)
        self.gateway = SecurityGateway(self.config)

    def test_response_without_choices(self) -> None:
        """Response without choices key is handled gracefully."""
        response = {"id": "chatcmpl-123", "object": "chat.completion"}
        result, findings = self.gateway.scan_inbound(response)
        assert len(findings) == 0

    def test_response_with_empty_choices(self) -> None:
        """Response with empty choices list is handled gracefully."""
        response = {"choices": []}
        result, findings = self.gateway.scan_inbound(response)
        assert len(findings) == 0

    def test_response_without_code_blocks(self) -> None:
        """Response without code blocks skips scanning."""
        response = _make_response("Just some text without code blocks.")
        result, findings = self.gateway.scan_inbound(response)
        assert len(findings) == 0
        # No LM Studio call should be made (no code blocks to scan)

    def test_response_with_non_string_content(self) -> None:
        """Response with non-string content is handled gracefully."""
        response = {
            "choices": [
                {"message": {"role": "assistant", "content": None}}
            ]
        }
        result, findings = self.gateway.scan_inbound(response)
        assert len(findings) == 0

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_invalid_severity_normalized(self, mock_client_cls) -> None:
        """Invalid severity values are normalized to 'medium'."""
        vuln_response = _lm_studio_response([
            {
                "cve_type": "buffer_overflow",
                "severity": "super_critical",  # invalid
                "line_start": 1,
                "line_end": 1,
                "recommendation": "Fix it.",
            }
        ])

        mock_response = MagicMock()
        mock_response.json.return_value = vuln_response
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        code = "char buf[10];\nstrcpy(buf, user_input);"
        response = _make_response(f"```c\n{code}\n```")

        result, findings = self.gateway.scan_inbound(response)

        assert len(findings) == 1
        assert findings[0].severity == "medium"

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_line_range_clamped_to_code_bounds(self, mock_client_cls) -> None:
        """Line ranges are clamped to actual code block bounds."""
        vuln_response = _lm_studio_response([
            {
                "cve_type": "sql_injection",
                "severity": "high",
                "line_start": 0,   # below minimum
                "line_end": 100,   # above actual lines
                "recommendation": "Fix it.",
            }
        ])

        mock_response = MagicMock()
        mock_response.json.return_value = vuln_response
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        code = "line1\nline2\nline3"
        response = _make_response(f"```\n{code}\n```")

        result, findings = self.gateway.scan_inbound(response)

        assert len(findings) == 1
        assert findings[0].line_range == (1, 3)  # clamped to valid range
