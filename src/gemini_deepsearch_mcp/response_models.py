"""Response models for DeepSearch MCP.

Provides Pydantic models for structured responses with metadata.
"""

from typing import Any

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A source reference from web research.

    Attributes:
        url: The URL of the source.
        title: The title of the source page.
        snippet: A brief excerpt or snippet from the source.
    """

    url: str = Field(description="The URL of the source")
    title: str | None = Field(default=None, description="The title of the source page")
    snippet: str | None = Field(default=None, description="A brief excerpt from the source")

    @classmethod
    def from_url(cls, url: str) -> "Source":
        """Create a Source from just a URL.

        Args:
            url: The source URL.

        Returns:
            A Source instance with the URL set.
        """
        return cls(url=url)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Source":
        """Create a Source from a dictionary.

        Args:
            data: Dictionary with url, title, and/or snippet.

        Returns:
            A Source instance.
        """
        if isinstance(data, str):
            return cls(url=data)
        return cls(
            url=data.get("url", ""),
            title=data.get("title"),
            snippet=data.get("snippet"),
        )


class ModelsUsed(BaseModel):
    """Models used during research.

    Attributes:
        query_generator: Model used for query generation.
        web_search: Model used for web search.
        reflection: Model used for reflection.
        answer: Model used for answer generation.
    """

    query_generator: str = Field(description="Model for query generation")
    web_search: str = Field(description="Model for web search")
    reflection: str = Field(description="Model for reflection")
    answer: str = Field(description="Model for answer generation")


class ResearchMetadata(BaseModel):
    """Metadata about the research process.

    Attributes:
        query: The original user query.
        effort: The effort level used (low, medium, high).
        models_used: Models used for each stage.
        research_loops: Number of research loops completed.
        sources_count: Number of sources gathered.
        duration_seconds: Total time taken in seconds.
    """

    query: str = Field(description="The original user query")
    effort: str = Field(description="Research effort level (low, medium, high)")
    models_used: ModelsUsed = Field(description="Models used for each stage")
    research_loops: int = Field(description="Number of research loops completed")
    sources_count: int = Field(description="Number of sources gathered")
    duration_seconds: float = Field(description="Total duration in seconds")


class SearchResult(BaseModel):
    """Complete search result with answer, sources, and metadata.

    Attributes:
        answer: The synthesized answer to the query.
        sources: List of sources used to generate the answer.
        metadata: Metadata about the research process.
    """

    answer: str = Field(description="The synthesized answer")
    sources: list[Source] = Field(default_factory=list, description="Sources used")
    metadata: ResearchMetadata = Field(description="Research process metadata")

    @classmethod
    def create(
        cls,
        answer: str,
        sources: list[str | dict[str, Any]],
        query: str,
        effort: str,
        models_used: dict[str, str],
        research_loops: int,
        duration_seconds: float,
    ) -> "SearchResult":
        """Create a SearchResult from raw data.

        Args:
            answer: The synthesized answer.
            sources: List of source URLs or dicts with url/title/snippet.
            query: The original user query.
            effort: The effort level used.
            models_used: Dict mapping stage names to model names.
            research_loops: Number of research loops completed.
            duration_seconds: Total time taken.

        Returns:
            A SearchResult instance.
        """
        # Convert sources to Source objects
        source_objects = []
        for src in sources:
            if isinstance(src, str):
                source_objects.append(Source(url=src))
            elif isinstance(src, dict):
                source_objects.append(Source.from_dict(src))
            elif isinstance(src, Source):
                source_objects.append(src)

        # Create ModelsUsed
        models = ModelsUsed(
            query_generator=models_used.get("query_generator_model", "unknown"),
            web_search=models_used.get("web_search_model", "unknown"),
            reflection=models_used.get("reflection_model", "unknown"),
            answer=models_used.get("answer_model", "unknown"),
        )

        # Create metadata
        metadata = ResearchMetadata(
            query=query,
            effort=effort,
            models_used=models,
            research_loops=research_loops,
            sources_count=len(source_objects),
            duration_seconds=round(duration_seconds, 2),
        )

        return cls(answer=answer, sources=source_objects, metadata=metadata)

    def to_file_format(self) -> dict[str, Any]:
        """Convert to the format used for file output.

        Returns:
            Dict with answer, sources as URLs, and metadata.
        """
        return {
            "answer": self.answer,
            "sources": [src.url for src in self.sources],
            "metadata": {
                "query": self.metadata.query,
                "effort": self.metadata.effort,
                "models_used": {
                    "query_generator": self.metadata.models_used.query_generator,
                    "web_search": self.metadata.models_used.web_search,
                    "reflection": self.metadata.models_used.reflection,
                    "answer": self.metadata.models_used.answer,
                },
                "research_loops": self.metadata.research_loops,
                "sources_count": self.metadata.sources_count,
                "duration_seconds": self.metadata.duration_seconds,
            },
        }


class ErrorResult(BaseModel):
    """Error result when research fails.

    Attributes:
        error: The error message.
        error_type: The type of error that occurred.
        context: Additional context about the error.
    """

    error: str = Field(description="The error message")
    error_type: str = Field(default="DeepSearchError", description="The type of error")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")

    @classmethod
    def from_exception(cls, exc: Exception) -> "ErrorResult":
        """Create an ErrorResult from an exception.

        Args:
            exc: The exception that occurred.

        Returns:
            An ErrorResult instance.
        """
        # Check if it's a DeepSearchError with context
        if hasattr(exc, "context") and hasattr(exc, "message"):
            return cls(
                error=exc.message,  # type: ignore[attr-defined]
                error_type=exc.__class__.__name__,
                context=exc.context,  # type: ignore[attr-defined]
            )
        return cls(
            error=str(exc),
            error_type=exc.__class__.__name__,
        )
