"""Tests for response models.

Tests cover:
- Source model creation and conversion
- ModelsUsed model
- ResearchMetadata model
- SearchResult model with factory method
- ErrorResult model
- File format conversion
"""

import pytest

from gemini_deepsearch_mcp.response_models import (
    ErrorResult,
    ModelsUsed,
    ResearchMetadata,
    SearchResult,
    Source,
)


class TestSourceModel:
    """Test Source model creation and methods."""

    def test_source_basic_creation(self):
        """Source should accept url, title, snippet."""
        source = Source(url="https://example.com", title="Example", snippet="Some text")
        assert source.url == "https://example.com"
        assert source.title == "Example"
        assert source.snippet == "Some text"

    def test_source_minimal_creation(self):
        """Source should work with just url."""
        source = Source(url="https://example.com")
        assert source.url == "https://example.com"
        assert source.title is None
        assert source.snippet is None

    def test_source_from_url(self):
        """from_url should create Source with just URL."""
        source = Source.from_url("https://example.com")
        assert source.url == "https://example.com"
        assert source.title is None

    def test_source_from_dict_with_all_fields(self):
        """from_dict should handle full dictionaries."""
        data = {
            "url": "https://example.com",
            "title": "Example Page",
            "snippet": "This is a snippet",
        }
        source = Source.from_dict(data)
        assert source.url == "https://example.com"
        assert source.title == "Example Page"
        assert source.snippet == "This is a snippet"

    def test_source_from_dict_minimal(self):
        """from_dict should handle minimal dictionaries."""
        data = {"url": "https://example.com"}
        source = Source.from_dict(data)
        assert source.url == "https://example.com"
        assert source.title is None

    def test_source_from_string(self):
        """from_dict should handle string input."""
        source = Source.from_dict("https://example.com")
        assert source.url == "https://example.com"

    def test_source_serialization(self):
        """Source should serialize to dict correctly."""
        source = Source(url="https://example.com", title="Test")
        data = source.model_dump()
        assert data["url"] == "https://example.com"
        assert data["title"] == "Test"


class TestModelsUsed:
    """Test ModelsUsed model."""

    def test_models_used_creation(self):
        """ModelsUsed should store all model names."""
        models = ModelsUsed(
            query_generator="gemini-2.5-flash",
            web_search="gemini-2.5-flash-lite",
            reflection="gemini-2.5-flash",
            answer="gemini-2.5-pro",
        )
        assert models.query_generator == "gemini-2.5-flash"
        assert models.web_search == "gemini-2.5-flash-lite"
        assert models.reflection == "gemini-2.5-flash"
        assert models.answer == "gemini-2.5-pro"

    def test_models_used_serialization(self):
        """ModelsUsed should serialize correctly."""
        models = ModelsUsed(
            query_generator="flash",
            web_search="flash-lite",
            reflection="flash",
            answer="pro",
        )
        data = models.model_dump()
        assert "query_generator" in data
        assert "answer" in data


class TestResearchMetadata:
    """Test ResearchMetadata model."""

    def test_metadata_creation(self):
        """ResearchMetadata should store all fields."""
        models = ModelsUsed(
            query_generator="flash",
            web_search="flash-lite",
            reflection="flash",
            answer="pro",
        )
        metadata = ResearchMetadata(
            query="What is AI?",
            effort="medium",
            models_used=models,
            research_loops=2,
            sources_count=5,
            duration_seconds=10.5,
        )
        assert metadata.query == "What is AI?"
        assert metadata.effort == "medium"
        assert metadata.research_loops == 2
        assert metadata.sources_count == 5
        assert metadata.duration_seconds == 10.5

    def test_metadata_models_nested(self):
        """ResearchMetadata should contain ModelsUsed."""
        models = ModelsUsed(
            query_generator="flash",
            web_search="flash-lite",
            reflection="flash",
            answer="pro",
        )
        metadata = ResearchMetadata(
            query="Test",
            effort="low",
            models_used=models,
            research_loops=1,
            sources_count=3,
            duration_seconds=5.0,
        )
        assert metadata.models_used.query_generator == "flash"
        assert metadata.models_used.answer == "pro"


class TestSearchResult:
    """Test SearchResult model."""

    def test_search_result_creation(self):
        """SearchResult should accept all fields."""
        source = Source(url="https://example.com")
        models = ModelsUsed(
            query_generator="flash",
            web_search="flash-lite",
            reflection="flash",
            answer="pro",
        )
        metadata = ResearchMetadata(
            query="Test query",
            effort="medium",
            models_used=models,
            research_loops=2,
            sources_count=1,
            duration_seconds=5.5,
        )
        result = SearchResult(
            answer="This is the answer",
            sources=[source],
            metadata=metadata,
        )
        assert result.answer == "This is the answer"
        assert len(result.sources) == 1
        assert result.metadata.query == "Test query"

    def test_search_result_create_factory(self):
        """SearchResult.create should build from raw data."""
        result = SearchResult.create(
            answer="The answer",
            sources=["https://example.com", "https://test.com"],
            query="What is this?",
            effort="high",
            models_used={
                "query_generator_model": "gemini-2.5-flash",
                "web_search_model": "gemini-2.5-flash-lite",
                "reflection_model": "gemini-2.5-flash",
                "answer_model": "gemini-2.5-pro",
            },
            research_loops=3,
            duration_seconds=15.123,
        )
        assert result.answer == "The answer"
        assert len(result.sources) == 2
        assert result.sources[0].url == "https://example.com"
        assert result.metadata.query == "What is this?"
        assert result.metadata.effort == "high"
        assert result.metadata.research_loops == 3
        assert result.metadata.sources_count == 2
        assert result.metadata.duration_seconds == 15.12  # Rounded

    def test_search_result_create_with_dict_sources(self):
        """SearchResult.create should handle dict sources."""
        result = SearchResult.create(
            answer="Answer",
            sources=[
                {"url": "https://example.com", "title": "Example"},
                {"url": "https://test.com", "snippet": "Test snippet"},
            ],
            query="Query",
            effort="low",
            models_used={
                "query_generator_model": "flash",
                "web_search_model": "flash-lite",
                "reflection_model": "flash",
                "answer_model": "flash",
            },
            research_loops=1,
            duration_seconds=3.0,
        )
        assert result.sources[0].title == "Example"
        assert result.sources[1].snippet == "Test snippet"

    def test_search_result_create_with_source_objects(self):
        """SearchResult.create should handle Source objects."""
        sources = [
            Source(url="https://example.com", title="Example"),
        ]
        result = SearchResult.create(
            answer="Answer",
            sources=sources,
            query="Query",
            effort="low",
            models_used={
                "query_generator_model": "flash",
                "web_search_model": "flash-lite",
                "reflection_model": "flash",
                "answer_model": "flash",
            },
            research_loops=1,
            duration_seconds=3.0,
        )
        assert result.sources[0].title == "Example"

    def test_search_result_to_file_format(self):
        """to_file_format should produce correct output structure."""
        result = SearchResult.create(
            answer="The answer",
            sources=["https://example.com"],
            query="What?",
            effort="medium",
            models_used={
                "query_generator_model": "flash",
                "web_search_model": "flash-lite",
                "reflection_model": "flash",
                "answer_model": "pro",
            },
            research_loops=2,
            duration_seconds=10.0,
        )
        file_format = result.to_file_format()

        assert file_format["answer"] == "The answer"
        assert file_format["sources"] == ["https://example.com"]
        assert "metadata" in file_format
        assert file_format["metadata"]["query"] == "What?"
        assert file_format["metadata"]["effort"] == "medium"
        assert file_format["metadata"]["research_loops"] == 2
        assert file_format["metadata"]["models_used"]["answer"] == "pro"


class TestErrorResult:
    """Test ErrorResult model."""

    def test_error_result_basic(self):
        """ErrorResult should store error information."""
        error = ErrorResult(error="Something went wrong")
        assert error.error == "Something went wrong"
        assert error.error_type == "DeepSearchError"

    def test_error_result_with_type(self):
        """ErrorResult should accept custom error type."""
        error = ErrorResult(
            error="Config error",
            error_type="ConfigurationError",
            context={"key": "GEMINI_API_KEY"},
        )
        assert error.error_type == "ConfigurationError"
        assert error.context["key"] == "GEMINI_API_KEY"

    def test_error_result_from_exception(self):
        """from_exception should create from standard exception."""
        exc = ValueError("Invalid value")
        error = ErrorResult.from_exception(exc)
        assert error.error == "Invalid value"
        assert error.error_type == "ValueError"

    def test_error_result_from_deep_search_error(self):
        """from_exception should handle DeepSearchError with context."""
        # Create a mock exception with message and context attributes
        class MockDeepSearchError(Exception):
            def __init__(self, message, **context):
                self.message = message
                self.context = context
                super().__init__(message)

        exc = MockDeepSearchError("Test error", key="value", num=42)
        error = ErrorResult.from_exception(exc)

        assert error.error == "Test error"
        assert error.error_type == "MockDeepSearchError"
        assert error.context["key"] == "value"
        assert error.context["num"] == 42

    def test_error_result_serialization(self):
        """ErrorResult should serialize correctly."""
        error = ErrorResult(
            error="Failed",
            error_type="APIError",
            context={"status": 500},
        )
        data = error.model_dump()
        assert data["error"] == "Failed"
        assert data["error_type"] == "APIError"
        assert data["context"]["status"] == 500


class TestResponseModelIntegration:
    """Integration tests for response models."""

    def test_full_workflow(self):
        """Test complete workflow from creation to file format."""
        # Simulate what main.py would do
        result = SearchResult.create(
            answer="Quantum computing uses qubits...",
            sources=[
                "https://en.wikipedia.org/wiki/Quantum_computing",
                "https://research.ibm.com/quantum",
            ],
            query="What is quantum computing?",
            effort="high",
            models_used={
                "query_generator_model": "gemini-2.5-flash",
                "web_search_model": "gemini-2.5-flash-lite-preview-06-17",
                "reflection_model": "gemini-2.5-flash",
                "answer_model": "gemini-2.5-pro",
            },
            research_loops=3,
            duration_seconds=25.5,
        )

        # Convert to file format
        file_data = result.to_file_format()

        # Verify structure
        assert "answer" in file_data
        assert "sources" in file_data
        assert "metadata" in file_data
        assert len(file_data["sources"]) == 2
        assert file_data["metadata"]["effort"] == "high"
        assert file_data["metadata"]["research_loops"] == 3

    def test_empty_sources(self):
        """SearchResult should handle empty sources list."""
        result = SearchResult.create(
            answer="No sources found",
            sources=[],
            query="Test",
            effort="low",
            models_used={
                "query_generator_model": "flash",
                "web_search_model": "flash-lite",
                "reflection_model": "flash",
                "answer_model": "flash",
            },
            research_loops=1,
            duration_seconds=1.0,
        )
        assert len(result.sources) == 0
        assert result.metadata.sources_count == 0
