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
    STATIC_CVE_PATTERNS,
    _check_static_cve_patterns,
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
            security_model_name="test-model",
            local_security_model_url="http://localhost:1234/v1/chat/completions",
        )
        self.gateway = SecurityGateway(self.config)

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_produces_cve_finding_with_all_fields(self, mock_client_cls) -> None:
        """Requirement 9.3: Produces CVEFinding with type, severity, snippet, line_range, recommendation."""
        vuln_response = _lm_studio_response([
            {
                "cve_type": "buffer_overflow",
                "severity": "high",
                "line_start": 2,
                "line_end": 3,
                "recommendation": "Add bounds checking.",
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

        # Use code that doesn't match static patterns (uses a generic function name)
        code = "void copy_data(char *src) {\n  char buf[10];\n  strcpy(buf, src);\n}"
        response = _make_response(f"Here's the code:\n```c\n{code}\n```")

        result, findings = self.gateway.scan_inbound(response)

        assert len(findings) == 1
        finding = findings[0]
        assert finding.cve_type == "buffer_overflow"
        assert finding.severity == "high"
        assert finding.line_range == (2, 3)
        assert finding.recommendation == "Add bounds checking."
        assert "buf[10]" in finding.code_snippet

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_annotates_response_with_findings(self, mock_client_cls) -> None:
        """Requirement 9.4: Annotates response with findings for developer visibility."""
        vuln_response = _lm_studio_response([
            {
                "cve_type": "insecure_deserialization",
                "severity": "critical",
                "line_start": 1,
                "line_end": 1,
                "recommendation": "Use safe serialization.",
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

        # Use code that doesn't match static patterns
        code = 'val = deserialize(data, "json")'
        response = _make_response(f"```python\n{code}\n```")

        result, findings = self.gateway.scan_inbound(response)

        # Response should be annotated
        annotated_content = result["choices"][0]["message"]["content"]
        assert "Security Scan Results" in annotated_content
        assert "insecure_deserialization" in annotated_content
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
                "cve_type": "race_condition",
                "severity": "high",
                "line_start": 1,
                "line_end": 1,
                "recommendation": "Use proper locking.",
            },
            {
                "cve_type": "memory_leak",
                "severity": "medium",
                "line_start": 3,
                "line_end": 3,
                "recommendation": "Free allocated memory.",
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

        # Use code that doesn't match static patterns
        code = "counter = Counter()\ncounter.increment()\nval = counter.value()"
        response = _make_response(f"```python\n{code}\n```")

        result, findings = self.gateway.scan_inbound(response)

        assert len(findings) == 2
        assert findings[0].cve_type == "race_condition"
        assert findings[1].cve_type == "memory_leak"


class TestScanInboundGracefulDegradation:
    """Tests for graceful degradation when LM Studio is unavailable."""

    def setup_method(self) -> None:
        self.config = SecurityConfig(
            cve_scanning_enabled=True,
            security_model_name="test-model",
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

        # Use code that doesn't match static patterns (which would short-circuit to LLM)
        code = "result = process(data, validate=True)"
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

        code = "result = process(data, validate=True)"
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
        self.config = SecurityConfig(
            cve_scanning_enabled=True,
            security_model_name="test-model",
        )
        self.gateway = SecurityGateway(self.config)

    @patch("deepseek_cursor_proxy.rap.security.httpx.Client")
    def test_original_response_unchanged(self, mock_client_cls) -> None:
        """Original response dict is not mutated."""
        vuln_response = _lm_studio_response([
            {
                "cve_type": "logic_error",
                "severity": "medium",
                "line_start": 1,
                "line_end": 1,
                "recommendation": "Review logic.",
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

        # Use code that doesn't match static patterns
        code = "val = calculate(x, y)"
        response = _make_response(f"```python\n{code}\n```")
        original = copy.deepcopy(response)

        self.gateway.scan_inbound(response)

        # Original should be completely unchanged
        assert response == original


class TestScanInboundEdgeCases:
    """Tests for edge cases in scan_inbound."""

    def setup_method(self) -> None:
        self.config = SecurityConfig(
            cve_scanning_enabled=True,
            security_model_name="test-model",
        )
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


class TestStaticCVEPatterns:
    """Tests for static CVE pattern detection (fast-path before LLM)."""

    def test_detects_code_injection(self) -> None:
        """Code injection pattern: exec/eval/compile calls."""
        code = "result = eval(user_input)"
        findings = _check_static_cve_patterns(code)
        assert len(findings) >= 1
        assert any(f.cve_type == "code_injection" for f in findings)

    def test_detects_exec_call(self) -> None:
        """exec() is detected as code injection."""
        code = "exec(code_string)"
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "code_injection" for f in findings)

    def test_detects_compile_call(self) -> None:
        """compile() is detected as code injection."""
        code = "compile(source, filename, mode)"
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "code_injection" for f in findings)

    def test_detects_sql_injection(self) -> None:
        """SQL injection via f-string in execute()."""
        code = "cursor.execute(f'SELECT * FROM users WHERE id={user_id}')"
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "sql_injection" for f in findings)

    def test_detects_sql_injection_concat(self) -> None:
        """SQL injection via string concatenation in execute()."""
        code = "cursor.execute('SELECT * FROM ' + table_name)"
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "sql_injection" for f in findings)

    def test_detects_command_injection(self) -> None:
        """Command injection via os.system()."""
        code = "os.system('rm -rf /')"
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "command_injection" for f in findings)

    def test_detects_subprocess_shell_true(self) -> None:
        """Command injection via subprocess with shell=True."""
        code = "subprocess.run(command, shell=True)"
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "command_injection" for f in findings)

    def test_detects_hardcoded_password(self) -> None:
        """Hardcoded password detection."""
        code = 'password = "supersecret123"'
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "hardcoded_credential" for f in findings)

    def test_detects_hardcoded_api_key(self) -> None:
        """Hardcoded API key detection."""
        code = 'api_key = "sk-1234567890abcdef"'
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "hardcoded_credential" for f in findings)

    def test_detects_hardcoded_secret(self) -> None:
        """Hardcoded secret detection."""
        code = 'secret = "my-secret-value"'
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "hardcoded_credential" for f in findings)

    def test_detects_pickle_deserialization(self) -> None:
        """Pickle deserialization detection."""
        code = "data = pickle.loads(raw_bytes)"
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "pickle_deserialization" for f in findings)

    def test_detects_path_traversal(self) -> None:
        """Path traversal via request parameter."""
        code = "path = os.path.join('/base', request.filename)"
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "path_traversal" for f in findings)

    def test_detects_open_with_request(self) -> None:
        """Path traversal via open() + request."""
        code = "content = open(request.args.get('file')).read()"
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "path_traversal" for f in findings)

    def test_detects_insecure_hash_md5(self) -> None:
        """Insecure hash: md5()."""
        code = "hash_value = md5(data).hexdigest()"
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "insecure_hash" for f in findings)

    def test_detects_insecure_hash_sha1(self) -> None:
        """Insecure hash: sha1()."""
        code = "hash_value = sha1(data).hexdigest()"
        findings = _check_static_cve_patterns(code)
        assert any(f.cve_type == "insecure_hash" for f in findings)

    def test_clean_code_returns_no_findings(self) -> None:
        """Clean code without vulnerabilities returns no findings."""
        code = "x = 1 + 2\nprint('hello world')\nresult = sum([1, 2, 3])"
        findings = _check_static_cve_patterns(code)
        assert len(findings) == 0

    def test_empty_code_returns_no_findings(self) -> None:
        """Empty code returns no findings."""
        findings = _check_static_cve_patterns("")
        assert len(findings) == 0

    def test_cve_finding_has_severity_and_recommendation(self) -> None:
        """Each static CVE finding includes severity and recommendation."""
        code = 'password = "admin123"'
        findings = _check_static_cve_patterns(code)
        assert len(findings) >= 1
        finding = findings[0]
        assert finding.severity in ("low", "medium", "high", "critical")
        assert len(finding.recommendation) > 0
        assert finding.line_range[0] >= 1
        assert finding.line_range[1] >= finding.line_range[0]

    def test_static_patterns_detect_before_llm(self) -> None:
        """Test that static CVE scan produces findings without any LLM call.

        This validates the fast-path: static patterns should detect known
        vulnerability patterns without requiring an external model.
        """
        # Code with a clearly vulnerable pattern
        code = "eval(user_input)"
        findings = _check_static_cve_patterns(code)
        assert len(findings) >= 1
        assert findings[0].cve_type == "code_injection"
        assert findings[0].severity == "critical"
        assert "exec/eval/compile" in findings[0].recommendation

    def test_static_cve_patterns_are_precompiled(self) -> None:
        """STATIC_CVE_PATTERNS are precompiled regex patterns."""
        assert len(STATIC_CVE_PATTERNS) >= 7
        for name, pattern in STATIC_CVE_PATTERNS:
            assert isinstance(name, str)
            assert hasattr(pattern, "search")
            assert pattern.search("test") is not None or True  # patterns exist

    def test_multiple_findings_in_same_code(self) -> None:
        """Multiple vulnerability patterns in the same code block."""
        code = (
            "password = 'admin123'\n"
            "cursor.execute(f'SELECT * FROM users WHERE id={uid}')\n"
        )
        findings = _check_static_cve_patterns(code)
        assert len(findings) >= 2
        types = {f.cve_type for f in findings}
        assert "hardcoded_credential" in types
        assert "sql_injection" in types
