"""
Unit tests for CodeArtifact Pydantic model validation.

Validates:
- Empty source_code raises ValidationError
- Whitespace-only source_code raises ValidationError
- Unsupported language raises ValidationError
- Valid construction succeeds and fields are accessible
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from coder_buddy.models import CodeArtifact


VALID_KWARGS = {
    "source_code": "print('hello')",
    "file_name": "hello.py",
    "dependencies": [],
    "language": "python",
}


class TestSourceCodeValidation:
    def test_empty_source_code_raises(self):
        with pytest.raises(ValidationError):
            CodeArtifact(**{**VALID_KWARGS, "source_code": ""})

    def test_whitespace_only_spaces_raises(self):
        with pytest.raises(ValidationError):
            CodeArtifact(**{**VALID_KWARGS, "source_code": "   "})

    def test_whitespace_only_newline_tab_raises(self):
        with pytest.raises(ValidationError):
            CodeArtifact(**{**VALID_KWARGS, "source_code": "\n\t"})

    def test_whitespace_only_mixed_raises(self):
        with pytest.raises(ValidationError):
            CodeArtifact(**{**VALID_KWARGS, "source_code": "  \n  \t  "})


class TestLanguageValidation:
    def test_javascript_raises(self):
        with pytest.raises(ValidationError):
            CodeArtifact(**{**VALID_KWARGS, "language": "javascript"})

    def test_rust_raises(self):
        with pytest.raises(ValidationError):
            CodeArtifact(**{**VALID_KWARGS, "language": "rust"})

    def test_empty_language_raises(self):
        with pytest.raises(ValidationError):
            CodeArtifact(**{**VALID_KWARGS, "language": ""})


class TestValidConstruction:
    def test_valid_minimal(self):
        artifact = CodeArtifact(
            source_code="x = 1",
            file_name="script.py",
            dependencies=[],
        )
        assert artifact.source_code == "x = 1"
        assert artifact.file_name == "script.py"
        assert artifact.dependencies == []
        assert artifact.language == "python"

    def test_valid_with_dependencies(self):
        artifact = CodeArtifact(
            source_code="import numpy as np",
            file_name="compute.py",
            dependencies=["numpy"],
        )
        assert artifact.dependencies == ["numpy"]

    def test_language_defaults_to_python(self):
        artifact = CodeArtifact(
            source_code="pass",
            file_name="noop.py",
            dependencies=[],
        )
        assert artifact.language == "python"

    def test_language_case_insensitive_normalised(self):
        """Validator lowercases the language value."""
        artifact = CodeArtifact(**{**VALID_KWARGS, "language": "Python"})
        assert artifact.language == "python"

    def test_source_code_with_leading_trailing_whitespace_is_accepted(self):
        """Non-blank source_code with surrounding whitespace is valid."""
        artifact = CodeArtifact(**{**VALID_KWARGS, "source_code": "  x = 1  "})
        assert artifact.source_code == "  x = 1  "


# Feature: coder-buddy, Property 3: CodeArtifact rejects empty or whitespace-only source_code

# Strategy: generate empty string OR strings composed entirely of characters that
# Python's str.strip() considers whitespace (the same check used by the validator).
# str.strip() removes: space, tab, newline, carriage return, form feed, vertical tab,
# and Unicode whitespace (category Zs). We restrict to exactly these characters so
# every generated string satisfies s.strip() == "".
_PYTHON_STRIP_WHITESPACE = " \t\n\r\f\v\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
_whitespace_only = st.text(
    alphabet=st.sampled_from(list(_PYTHON_STRIP_WHITESPACE)),
    min_size=1,
).filter(lambda s: s.strip() == "")
_empty_or_whitespace = st.one_of(st.just(""), _whitespace_only)


@given(s=_empty_or_whitespace)
@settings(max_examples=200)
def test_property3_code_artifact_rejects_empty_or_whitespace_source_code(s: str):
    """
    **Validates: Requirements 2.6**

    Property 3: For any string s that is empty or composed entirely of
    whitespace characters, constructing CodeArtifact(source_code=s, ...)
    SHALL raise a ValidationError.
    """
    with pytest.raises(ValidationError):
        CodeArtifact(
            source_code=s,
            file_name="main.py",
            dependencies=[],
            language="python",
        )


# Feature: coder-buddy, Property 4: valid CodeArtifact instances always have all four required fields present and correctly typed

_non_empty_source = st.text(min_size=1).filter(lambda s: s.strip() != "")


@given(
    source_code=_non_empty_source,
    file_name=st.text(),
    dependencies=st.lists(st.text()),
)
@settings(max_examples=200)
def test_property4_valid_code_artifact_has_all_required_fields(
    source_code: str,
    file_name: str,
    dependencies: list[str],
):
    """
    **Validates: Requirements 2.2**

    Property 4: For any valid CodeArtifact instance, all four required fields
    (source_code, file_name, dependencies, language) SHALL be present and
    correctly typed:
    - source_code: non-empty string
    - file_name: string
    - dependencies: list of strings
    - language: string equal to "python"
    """
    artifact = CodeArtifact(
        source_code=source_code,
        file_name=file_name,
        dependencies=dependencies,
        language="python",
    )

    assert isinstance(artifact.source_code, str)
    assert len(artifact.source_code.strip()) > 0

    assert isinstance(artifact.file_name, str)

    assert isinstance(artifact.dependencies, list)
    assert all(isinstance(dep, str) for dep in artifact.dependencies)

    assert artifact.language == "python"


# Feature: coder-buddy, Property 15: CodeArtifact rejects any language value other than "python"

_non_python_language = st.text().filter(lambda s: s.lower() != "python")


@given(language=_non_python_language)
@settings(max_examples=200)
def test_property15_code_artifact_rejects_non_python_language(language: str):
    """
    **Validates: Requirements 2.1, 8.1**

    Property 15: For any string `language` that is not equal to "python"
    (case-insensitive), constructing CodeArtifact(language=language, ...)
    SHALL raise a ValidationError.
    """
    with pytest.raises(ValidationError):
        CodeArtifact(
            source_code="print('hello')",
            file_name="main.py",
            dependencies=[],
            language=language,
        )
