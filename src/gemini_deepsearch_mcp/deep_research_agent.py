"""Google Deep Research Agent integration using the Interactions API.

This module provides integration with Google's Deep Research Agent,
which autonomously plans, executes, and synthesizes multi-step research tasks.

The Deep Research Agent:
- Uses the Interactions API (not standard generate_content)
- Runs in background mode (tasks can take minutes)
- Has built-in Google Search and URL context tools
- Produces detailed, cited reports
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Generator

from google import genai

from .config import load_config
from .logging import get_logger

# Agent model name
DEEP_RESEARCH_AGENT = "deep-research-pro-preview-12-2025"

# Default polling interval in seconds
DEFAULT_POLL_INTERVAL = 10

# Maximum research time in seconds (60 minutes per Google docs)
MAX_RESEARCH_TIME = 3600


@dataclass
class ResearchProgress:
    """Progress update during research."""

    status: str
    poll_count: int
    elapsed_seconds: float
    message: str | None = None
    thought_summary: str | None = None


@dataclass
class ResearchResult:
    """Result from a deep research task."""

    answer: str
    sources: list[dict[str, Any]]
    interaction_id: str
    status: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "interaction_id": self.interaction_id,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
        }


class DeepResearchAgent:
    """Wrapper for Google's Deep Research Agent.

    This agent uses the Interactions API to perform autonomous research tasks.
    It handles:
    - Starting background research tasks
    - Polling for completion
    - Extracting results and sources
    - Optional streaming for progress updates
    """

    def __init__(self, api_key: str | None = None):
        """Initialize the Deep Research Agent.

        Args:
            api_key: Optional Gemini API key. If not provided, uses config.
        """
        if api_key is None:
            config = load_config()
            api_key = config.gemini_api_key

        self.client = genai.Client(api_key=api_key)
        self.logger = get_logger("deepsearch.deep_research_agent")

    def research(
        self,
        query: str,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        max_time: int = MAX_RESEARCH_TIME,
        progress_callback: Callable | None = None,
    ) -> ResearchResult:
        """Perform a deep research task.

        Args:
            query: The research question or topic.
            poll_interval: Seconds between status polls.
            max_time: Maximum time to wait for research completion.
            progress_callback: Optional callback for progress updates.
                               Called with ResearchProgress on each poll.

        Returns:
            ResearchResult with answer, sources, and metadata.

        Raises:
            TimeoutError: If research exceeds max_time.
            RuntimeError: If research fails.
        """
        log = self.logger.bind(query=query[:50] + "..." if len(query) > 50 else query)
        log.info("deep_research_started")

        start_time = time.time()

        # Start the research task in background mode
        interaction = self.client.interactions.create(
            input=query,
            agent=DEEP_RESEARCH_AGENT,
            background=True,
        )

        log.info("research_task_created", interaction_id=interaction.id)

        poll_count = 0
        while True:
            elapsed = time.time() - start_time

            if elapsed > max_time:
                log.error("research_timeout", elapsed_seconds=elapsed)
                raise TimeoutError(
                    f"Research exceeded maximum time of {max_time} seconds. "
                    f"Interaction ID: {interaction.id}"
                )

            # Get current status
            interaction = self.client.interactions.get(interaction.id)
            poll_count += 1

            log.debug(
                "research_poll",
                poll_count=poll_count,
                status=interaction.status,
                elapsed_seconds=round(elapsed, 1),
            )

            # Report progress if callback provided
            if progress_callback:
                progress = ResearchProgress(
                    status=interaction.status,
                    poll_count=poll_count,
                    elapsed_seconds=elapsed,
                    message=f"Research in progress ({interaction.status})",
                )
                progress_callback(progress)

            if interaction.status == "completed":
                log.info(
                    "research_completed",
                    elapsed_seconds=round(elapsed, 1),
                    poll_count=poll_count,
                )
                return self._extract_result(interaction, elapsed)

            elif interaction.status == "failed":
                error_msg = getattr(interaction, "error", "Unknown error")
                log.error("research_failed", error=error_msg)
                raise RuntimeError(f"Research failed: {error_msg}")

            elif interaction.status == "cancelled":
                log.warning("research_cancelled")
                raise RuntimeError("Research was cancelled")

            time.sleep(poll_interval)

    def research_streaming(
        self,
        query: str,
        max_time: int = MAX_RESEARCH_TIME,
    ) -> Generator[ResearchProgress | ResearchResult, None, None]:
        """Perform deep research with streaming progress updates.

        Yields progress updates as the research runs, then yields
        the final result.

        Args:
            query: The research question or topic.
            max_time: Maximum time to wait for research completion.

        Yields:
            ResearchProgress objects during research, then ResearchResult.
        """
        log = self.logger.bind(query=query[:50] + "..." if len(query) > 50 else query)
        log.info("deep_research_streaming_started")

        start_time = time.time()

        try:
            # Start streaming research
            stream = self.client.interactions.create(
                input=query,
                agent=DEEP_RESEARCH_AGENT,
                background=True,
                stream=True,
                agent_config={
                    "type": "deep-research",
                    "thinking_summaries": "auto",
                },
            )

            interaction_id = None
            last_event_id = None

            for chunk in stream:
                elapsed = time.time() - start_time

                if elapsed > max_time:
                    log.error("research_streaming_timeout", elapsed_seconds=elapsed)
                    raise TimeoutError(f"Research exceeded maximum time of {max_time} seconds")

                # Capture interaction ID
                if chunk.event_type == "interaction.start":
                    interaction_id = chunk.interaction.id
                    log.info("research_streaming_id", interaction_id=interaction_id)

                # Track event ID for reconnection
                if hasattr(chunk, "event_id") and chunk.event_id:
                    last_event_id = chunk.event_id

                # Handle different event types
                if chunk.event_type == "content.delta":
                    if hasattr(chunk.delta, "type"):
                        if chunk.delta.type == "thought_summary":
                            yield ResearchProgress(
                                status="in_progress",
                                poll_count=0,
                                elapsed_seconds=elapsed,
                                thought_summary=chunk.delta.content.text,
                            )
                        elif chunk.delta.type == "text":
                            # Partial text during research
                            pass

                elif chunk.event_type == "interaction.complete":
                    log.info("research_streaming_completed", elapsed_seconds=round(elapsed, 1))
                    # Get the final interaction to extract results
                    if interaction_id:
                        final_interaction = self.client.interactions.get(interaction_id)
                        yield self._extract_result(final_interaction, elapsed)
                    return

        except Exception as e:
            log.exception("research_streaming_error", error=str(e))
            raise

    def _extract_result(self, interaction: Any, elapsed: float) -> ResearchResult:
        """Extract result from completed interaction.

        Args:
            interaction: The completed interaction object.
            elapsed: Total elapsed time in seconds.

        Returns:
            ResearchResult with answer and sources.
        """
        # Extract the answer text
        answer = "No answer generated."
        if interaction.outputs:
            for output in reversed(interaction.outputs):
                if hasattr(output, "text") and output.text:
                    answer = output.text
                    break

        # Extract sources/citations if available
        sources = []
        if hasattr(interaction, "citations"):
            for citation in interaction.citations:
                sources.append({
                    "url": getattr(citation, "url", ""),
                    "title": getattr(citation, "title", ""),
                })

        return ResearchResult(
            answer=answer,
            sources=sources,
            interaction_id=interaction.id,
            status=interaction.status,
            elapsed_seconds=round(elapsed, 2),
        )

    def get_interaction(self, interaction_id: str) -> Any:
        """Get an interaction by ID.

        Useful for checking status of long-running research tasks.

        Args:
            interaction_id: The interaction ID.

        Returns:
            The Interaction object.
        """
        return self.client.interactions.get(interaction_id)

    def cancel_research(self, interaction_id: str) -> None:
        """Cancel an in-progress research task.

        Args:
            interaction_id: The interaction ID to cancel.
        """
        self.logger.info("cancelling_research", interaction_id=interaction_id)
        self.client.interactions.cancel(interaction_id)


def run_deep_research(
    query: str,
    api_key: str | None = None,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    max_time: int = MAX_RESEARCH_TIME,
) -> dict[str, Any]:
    """Convenience function to run deep research.

    Args:
        query: The research question or topic.
        api_key: Optional Gemini API key.
        poll_interval: Seconds between status polls.
        max_time: Maximum time to wait.

    Returns:
        Dictionary with answer, sources, and metadata.
    """
    agent = DeepResearchAgent(api_key=api_key)
    result = agent.research(query, poll_interval=poll_interval, max_time=max_time)
    return result.to_dict()
