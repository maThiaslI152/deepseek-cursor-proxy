"""Property-based tests for the Fidelity Module header injection.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

Property 1: Header Injection Completeness
For any dict of headers, after injection, the result contains both
X-Cursor-Plan: pro and X-Cursor-Tier: unlimited, AND all original
headers are preserved.

Property 2: Header Injection Idempotency
For any dict of headers, applying inject_spoof_headers twice produces
the same result as applying it once (idempotency).
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from deepseek_cursor_proxy.rap.fidelity import FidelityConfig, FidelityModule


# --- Strategies ---

# Generate arbitrary HTTP header dicts with printable ASCII keys and values
header_keys = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters="-_",
    ),
    min_size=1,
    max_size=50,
)

header_values = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=0,
    max_size=200,
)

arbitrary_headers = st.dictionaries(
    keys=header_keys,
    values=header_values,
    min_size=0,
    max_size=20,
)


# --- Fixtures ---

def _make_module() -> FidelityModule:
    """Create a FidelityModule with default config."""
    return FidelityModule(FidelityConfig())


# --- Property 1: Header Injection Completeness ---


class TestProperty1HeaderInjectionCompleteness:
    """For any dict of headers, after injection the result contains the
    required spoof headers AND all original headers are preserved.

    **Validates: Requirements 1.1, 1.2, 1.3**
    """

    @given(headers=arbitrary_headers)
    def test_result_contains_cursor_plan_pro(self, headers: dict[str, str]) -> None:
        """After injection, X-Cursor-Plan is always 'pro'.

        **Validates: Requirements 1.1**
        """
        module = _make_module()
        result = module.intercept_request(headers, {})
        assert result["X-Cursor-Plan"] == "pro"

    @given(headers=arbitrary_headers)
    def test_result_contains_cursor_tier_unlimited(self, headers: dict[str, str]) -> None:
        """After injection, X-Cursor-Tier is always 'unlimited'.

        **Validates: Requirements 1.2**
        """
        module = _make_module()
        result = module.intercept_request(headers, {})
        assert result["X-Cursor-Tier"] == "unlimited"

    @given(headers=arbitrary_headers)
    def test_all_original_headers_preserved(self, headers: dict[str, str]) -> None:
        """After injection, every original header key-value pair is present in the result.

        **Validates: Requirements 1.3**
        """
        module = _make_module()
        result = module.intercept_request(headers, {})

        for key, value in headers.items():
            # Original headers are preserved unless they conflict with spoof keys
            if key not in module.config.spoof_headers:
                assert result[key] == value

    @given(headers=arbitrary_headers)
    def test_original_dict_not_mutated(self, headers: dict[str, str]) -> None:
        """The original headers dict is never mutated by injection.

        **Validates: Requirements 1.3**
        """
        module = _make_module()
        original_copy = dict(headers)
        module.intercept_request(headers, {})
        assert headers == original_copy


# --- Property 2: Header Injection Idempotency ---


class TestProperty2HeaderInjectionIdempotency:
    """For any dict of headers, applying inject_spoof_headers twice produces
    the same result as applying it once.

    **Validates: Requirements 1.4**
    """

    @given(headers=arbitrary_headers)
    def test_double_application_equals_single(self, headers: dict[str, str]) -> None:
        """Applying intercept_request twice yields the same result as once.

        **Validates: Requirements 1.4**
        """
        module = _make_module()
        first_result = module.intercept_request(headers, {})
        second_result = module.intercept_request(first_result, {})
        assert first_result == second_result

    @given(headers=arbitrary_headers)
    def test_triple_application_equals_single(self, headers: dict[str, str]) -> None:
        """Applying intercept_request three times yields the same result as once.

        **Validates: Requirements 1.4**
        """
        module = _make_module()
        first_result = module.intercept_request(headers, {})
        second_result = module.intercept_request(first_result, {})
        third_result = module.intercept_request(second_result, {})
        assert first_result == third_result
