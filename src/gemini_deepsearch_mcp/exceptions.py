"""Custom exceptions for DeepSearch MCP.

Provides meaningful error types for different failure scenarios.
"""

from typing import Any


class DeepSearchError(Exception):
    """Base exception for all DeepSearch errors.

    Attributes:
        message: Human-readable error message.
        context: Additional context about the error.
    """

    def __init__(self, message: str, **context: Any) -> None:
        """Initialize with message and optional context.

        Args:
            message: Human-readable error description.
            **context: Additional key-value context for debugging.
        """
        self.message = message
        self.context = context
        super().__init__(message)

    def __str__(self) -> str:
        """Return formatted error message with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses.

        Returns:
            Dict with error type, message, and context.
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
        }


class ConfigurationError(DeepSearchError):
    """Configuration-related errors.

    Raised when:
    - Required environment variables are missing
    - Configuration values are invalid
    - Configuration file cannot be loaded
    """

    def __init__(self, message: str, **context: Any) -> None:
        """Initialize configuration error.

        Args:
            message: Error description.
            **context: Additional context (e.g., variable_name, expected_type).
        """
        super().__init__(f"Configuration error: {message}", **context)


class ModelNotFoundError(DeepSearchError):
    """Requested model not available in registry.

    Raised when:
    - Invalid model ID is specified
    - Model preset doesn't exist
    """

    def __init__(self, model_id: str, available_models: list[str] | None = None) -> None:
        """Initialize model not found error.

        Args:
            model_id: The model ID that was not found.
            available_models: Optional list of valid model IDs for suggestions.
        """
        message = f"Model '{model_id}' not found in registry"
        context: dict[str, Any] = {"model_id": model_id}

        if available_models:
            # Suggest similar models
            suggestions = [m for m in available_models if model_id.split("-")[0] in m][:3]
            if suggestions:
                context["suggestions"] = suggestions
                message += f". Did you mean: {', '.join(suggestions)}?"

        super().__init__(message, **context)


class SearchError(DeepSearchError):
    """Web search execution errors.

    Raised when:
    - Google Search API fails
    - Network errors during search
    - Rate limiting or quota exceeded
    """

    def __init__(
        self,
        message: str,
        query: str | None = None,
        status_code: int | None = None,
        **context: Any,
    ) -> None:
        """Initialize search error.

        Args:
            message: Error description.
            query: The search query that failed.
            status_code: HTTP status code if applicable.
            **context: Additional context.
        """
        if query:
            context["query"] = query[:50] + "..." if len(query) > 50 else query
        if status_code:
            context["status_code"] = status_code

        super().__init__(f"Search failed: {message}", **context)


class ResearchTimeoutError(DeepSearchError):
    """Research loop timeout.

    Raised when:
    - Research loops exceed maximum iterations
    - Individual operation times out
    """

    def __init__(
        self,
        message: str,
        loops_completed: int | None = None,
        max_loops: int | None = None,
        **context: Any,
    ) -> None:
        """Initialize research timeout error.

        Args:
            message: Error description.
            loops_completed: Number of loops completed before timeout.
            max_loops: Maximum loops configured.
            **context: Additional context.
        """
        if loops_completed is not None:
            context["loops_completed"] = loops_completed
        if max_loops is not None:
            context["max_loops"] = max_loops

        super().__init__(f"Research timeout: {message}", **context)


class APIError(DeepSearchError):
    """API-related errors.

    Raised when:
    - Gemini API returns an error
    - Authentication fails
    - Rate limiting
    """

    def __init__(
        self,
        message: str,
        api: str = "Gemini",
        status_code: int | None = None,
        **context: Any,
    ) -> None:
        """Initialize API error.

        Args:
            message: Error description.
            api: API name (default: Gemini).
            status_code: HTTP status code if applicable.
            **context: Additional context.
        """
        context["api"] = api
        if status_code:
            context["status_code"] = status_code

        super().__init__(f"{api} API error: {message}", **context)


class QueryGenerationError(DeepSearchError):
    """Query generation failed.

    Raised when:
    - Failed to generate search queries from user input
    - LLM response parsing fails
    """

    def __init__(self, message: str, original_query: str | None = None, **context: Any) -> None:
        """Initialize query generation error.

        Args:
            message: Error description.
            original_query: The user's original query.
            **context: Additional context.
        """
        if original_query:
            context["original_query"] = (
                original_query[:50] + "..." if len(original_query) > 50 else original_query
            )

        super().__init__(f"Query generation failed: {message}", **context)


class ReflectionError(DeepSearchError):
    """Reflection/analysis failed.

    Raised when:
    - Failed to analyze search results
    - Knowledge gap analysis fails
    """

    def __init__(self, message: str, iteration: int | None = None, **context: Any) -> None:
        """Initialize reflection error.

        Args:
            message: Error description.
            iteration: Current research loop iteration.
            **context: Additional context.
        """
        if iteration is not None:
            context["iteration"] = iteration

        super().__init__(f"Reflection failed: {message}", **context)


class AnswerGenerationError(DeepSearchError):
    """Answer generation failed.

    Raised when:
    - Failed to synthesize final answer
    - Citation generation fails
    """

    def __init__(
        self,
        message: str,
        sources_count: int | None = None,
        **context: Any,
    ) -> None:
        """Initialize answer generation error.

        Args:
            message: Error description.
            sources_count: Number of sources gathered.
            **context: Additional context.
        """
        if sources_count is not None:
            context["sources_count"] = sources_count

        super().__init__(f"Answer generation failed: {message}", **context)
