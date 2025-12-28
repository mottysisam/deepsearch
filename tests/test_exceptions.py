"""Tests for custom exceptions.

Tests cover:
- Exception creation and message formatting
- Context binding
- Error message helpfulness
"""

import pytest

from gemini_deepsearch_mcp.exceptions import (
    APIError,
    AnswerGenerationError,
    ConfigurationError,
    DeepSearchError,
    ModelNotFoundError,
    QueryGenerationError,
    ReflectionError,
    ResearchTimeoutError,
    SearchError,
)


class TestConfigurationErrorRaised:
    """Test ConfigurationError is raised correctly."""

    def test_configuration_error_message(self):
        """ConfigurationError should include error message."""
        error = ConfigurationError("GEMINI_API_KEY not set")
        assert "Configuration error" in str(error)
        assert "GEMINI_API_KEY not set" in str(error)

    def test_configuration_error_with_context(self):
        """ConfigurationError should include context."""
        error = ConfigurationError(
            "Invalid value", variable_name="LOG_LEVEL", expected="DEBUG|INFO|WARNING"
        )
        assert "variable_name=LOG_LEVEL" in str(error)
        assert "expected=DEBUG|INFO|WARNING" in str(error)

    def test_configuration_error_to_dict(self):
        """ConfigurationError.to_dict should return structured data."""
        error = ConfigurationError("Missing key", key="API_KEY")
        result = error.to_dict()

        assert result["error_type"] == "ConfigurationError"
        assert "Missing key" in result["message"]
        assert result["context"]["key"] == "API_KEY"


class TestModelNotFoundError:
    """Test ModelNotFoundError is raised correctly."""

    def test_model_not_found_basic(self):
        """ModelNotFoundError should include model ID."""
        error = ModelNotFoundError("invalid-model")
        assert "invalid-model" in str(error)
        assert "not found" in str(error)

    def test_model_not_found_with_suggestions(self):
        """ModelNotFoundError should suggest similar models."""
        available = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-thinking"]
        error = ModelNotFoundError("gemini-2.5-fast", available_models=available)

        # Should suggest models with similar prefix
        assert "gemini-2.5-flash" in str(error) or "Did you mean" in str(error)

    def test_model_not_found_context(self):
        """ModelNotFoundError should have model_id in context."""
        error = ModelNotFoundError("bad-model")
        assert error.context["model_id"] == "bad-model"


class TestSearchErrorContainsContext:
    """Test SearchError contains relevant context."""

    def test_search_error_basic(self):
        """SearchError should include basic message."""
        error = SearchError("Network timeout")
        assert "Search failed" in str(error)
        assert "Network timeout" in str(error)

    def test_search_error_with_query(self):
        """SearchError should include query context."""
        error = SearchError("Rate limited", query="What is AI?")
        assert "query=What is AI?" in str(error)

    def test_search_error_truncates_long_query(self):
        """SearchError should truncate long queries."""
        long_query = "x" * 100
        error = SearchError("Failed", query=long_query)

        # Query should be truncated
        assert "..." in str(error)
        assert len(error.context["query"]) < 60

    def test_search_error_with_status_code(self):
        """SearchError should include HTTP status code."""
        error = SearchError("Server error", status_code=500)
        assert "status_code=500" in str(error)


class TestErrorMessagesAreHelpful:
    """Test that error messages are helpful and actionable."""

    def test_api_error_includes_api_name(self):
        """APIError should include which API failed."""
        error = APIError("Authentication failed", api="Gemini")
        assert "Gemini" in str(error)
        assert "API error" in str(error)

    def test_api_error_with_status(self):
        """APIError should include status code for debugging."""
        error = APIError("Quota exceeded", status_code=429)
        assert "429" in str(error)

    def test_query_generation_error_includes_query(self):
        """QueryGenerationError should include original query."""
        error = QueryGenerationError("Invalid format", original_query="Tell me about AI")
        assert "Tell me about AI" in str(error)

    def test_reflection_error_includes_iteration(self):
        """ReflectionError should include iteration number."""
        error = ReflectionError("Analysis failed", iteration=3)
        assert "iteration=3" in str(error)

    def test_answer_generation_error_includes_sources(self):
        """AnswerGenerationError should include sources count."""
        error = AnswerGenerationError("Synthesis failed", sources_count=5)
        assert "sources_count=5" in str(error)

    def test_research_timeout_includes_loop_info(self):
        """ResearchTimeoutError should include loop information."""
        error = ResearchTimeoutError(
            "Exceeded max iterations", loops_completed=3, max_loops=5
        )
        assert "loops_completed=3" in str(error)
        assert "max_loops=5" in str(error)


class TestBaseDeepSearchError:
    """Test base DeepSearchError functionality."""

    def test_base_error_message(self):
        """DeepSearchError should store message."""
        error = DeepSearchError("Something went wrong")
        assert error.message == "Something went wrong"
        assert str(error) == "Something went wrong"

    def test_base_error_with_context(self):
        """DeepSearchError should include context in string."""
        error = DeepSearchError("Failed", key="value", num=42)
        result = str(error)
        assert "key=value" in result
        assert "num=42" in result

    def test_base_error_to_dict(self):
        """DeepSearchError.to_dict should return structured data."""
        error = DeepSearchError("Test error", foo="bar")
        result = error.to_dict()

        assert result["error_type"] == "DeepSearchError"
        assert result["message"] == "Test error"
        assert result["context"]["foo"] == "bar"

    def test_error_is_exception(self):
        """DeepSearchError should be a proper Exception."""
        error = DeepSearchError("Test")
        assert isinstance(error, Exception)

    def test_error_can_be_raised(self):
        """DeepSearchError should be raisable."""
        with pytest.raises(DeepSearchError) as exc_info:
            raise DeepSearchError("Test error", context="value")

        assert "Test error" in str(exc_info.value)

    def test_subclass_inheritance(self):
        """Subclasses should inherit from DeepSearchError."""
        assert issubclass(ConfigurationError, DeepSearchError)
        assert issubclass(ModelNotFoundError, DeepSearchError)
        assert issubclass(SearchError, DeepSearchError)
        assert issubclass(APIError, DeepSearchError)
        assert issubclass(ResearchTimeoutError, DeepSearchError)
        assert issubclass(QueryGenerationError, DeepSearchError)
        assert issubclass(ReflectionError, DeepSearchError)
        assert issubclass(AnswerGenerationError, DeepSearchError)


class TestExceptionCatching:
    """Test that exceptions can be caught appropriately."""

    def test_catch_specific_error(self):
        """Should be able to catch specific exception types."""
        try:
            raise ConfigurationError("Test")
        except ConfigurationError as e:
            assert "Configuration error" in str(e)

    def test_catch_base_error(self):
        """Should be able to catch all errors with base class."""
        try:
            raise ModelNotFoundError("test-model")
        except DeepSearchError as e:
            assert "not found" in str(e)

    def test_catch_as_exception(self):
        """Should be catchable as generic Exception."""
        try:
            raise SearchError("Network error")
        except Exception as e:
            assert "Search failed" in str(e)
