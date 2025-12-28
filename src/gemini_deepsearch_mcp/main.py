"""Main entry point for the stdio MCP server."""

import json
import os
import re
import sys
import tempfile
import time
from typing import Annotated, Literal

from fastmcp import FastMCP
from langchain_core.messages import HumanMessage
from pydantic import Field

from .config import load_config
from .logging import configure_logging, get_logger
from .model_selection import is_deep_research_mode, resolve_models
from .response_models import SearchResult

# Create MCP server
mcp = FastMCP("DeepSearch")

# Lazy-loaded global state
_graph = None
_logger = None
_logging_configured = False


def _ensure_logging():
    """Ensure logging is configured (lazy initialization).

    This allows CLI commands like --status and --models to work
    without requiring GEMINI_API_KEY.
    """
    global _logging_configured, _logger
    if not _logging_configured:
        try:
            config = load_config()
            configure_logging(config.log_level)
        except Exception:
            # If config fails, use default logging
            configure_logging("INFO")
        _logger = get_logger("deepsearch.main")
        _logging_configured = True
    return _logger


def _get_logger():
    """Get the logger, ensuring logging is configured first."""
    return _ensure_logging()


def _get_graph():
    """Lazy load the LangGraph graph.

    This allows CLI commands like --status and --models to work
    without requiring GEMINI_API_KEY.

    Returns:
        The compiled LangGraph graph.
    """
    global _graph
    if _graph is None:
        from .agent.graph import graph
        _graph = graph
    return _graph


def _run_deep_research(query: str, log) -> dict:
    """Run research using Google's Deep Research Agent.

    This uses the Interactions API instead of the LangGraph workflow.

    Args:
        query: The research question.
        log: Bound logger for structured logging.

    Returns:
        Dictionary with file_path to the results JSON file.
    """
    from .deep_research_agent import DeepResearchAgent

    start_time = time.time()

    try:
        log.info("deep_research_agent_starting")
        agent = DeepResearchAgent()
        result = agent.research(query)
        log.info(
            "deep_research_agent_completed",
            elapsed_seconds=result.elapsed_seconds,
            interaction_id=result.interaction_id,
        )
    except TimeoutError as e:
        log.error("deep_research_agent_timeout", error=str(e))
        return {"error": str(e)}
    except RuntimeError as e:
        log.error("deep_research_agent_failed", error=str(e))
        return {"error": str(e)}
    except Exception as e:
        log.exception("deep_research_agent_error", error=str(e))
        return {"error": f"Deep research failed: {str(e)}"}

    duration = time.time() - start_time

    # Create structured SearchResult with metadata
    # Use special models_used structure for Deep Research
    models_used = {
        "agent_model": "deep-research-pro-preview-12-2025",
        "mode": "google-deep-research",
    }

    search_result = SearchResult.create(
        answer=result.answer,
        sources=result.sources,
        query=query,
        effort="high",  # Deep research is always high effort
        models_used=models_used,
        research_loops=0,  # Agent manages its own loops
        duration_seconds=duration,
    )

    # Create filename from first few characters of query
    sanitized_query = re.sub(r'[^\w\s-]', '', query)[:20]
    filename = re.sub(r'\s+', '_', sanitized_query.strip()) + '.json'
    file_path = os.path.join(tempfile.gettempdir(), filename)

    # Write answer, sources, and metadata to JSON file
    result_data = search_result.to_file_format()

    # Add interaction_id for debugging/resumption
    result_data["metadata"]["interaction_id"] = result.interaction_id

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    log.info(
        "deep_research_completed",
        duration_seconds=round(duration, 2),
        sources_count=len(result.sources),
        answer_length=len(result.answer),
        output_file=file_path,
        interaction_id=result.interaction_id,
    )

    return {"file_path": file_path}


@mcp.tool()
def deep_search(
    query: Annotated[str, Field(description="Search query string")],
    effort: Annotated[
        Literal["low", "medium", "high"],
        Field(description="Research effort level: low (1 query, 1 loop), "
                         "medium (3 queries, 2 loops), high (5 queries, 3 loops)")
    ] = "low",
    model: Annotated[
        Literal["flash", "pro", "thinking", "deep-research"] | None,
        Field(description="Model preset: flash (fast), pro (capable), "
                         "thinking (extended reasoning), deep-research (Google's "
                         "autonomous research agent). Overrides config defaults.")
    ] = None,
    query_model: Annotated[
        str | None,
        Field(description="Specific model for query generation. Overrides preset.")
    ] = None,
    search_model: Annotated[
        str | None,
        Field(description="Specific model for web search. Overrides preset.")
    ] = None,
    reflection_model: Annotated[
        str | None,
        Field(description="Specific model for reflection. Overrides preset.")
    ] = None,
    answer_model: Annotated[
        str | None,
        Field(description="Specific model for answer generation. Overrides preset.")
    ] = None,
) -> dict:
    """Perform a deep search on a given query using an advanced web research agent.

    Args:
        query: The research question or topic to investigate.
        effort: Research effort level (low, medium, high). Higher effort means
                more queries and research loops.
        model: Model preset (flash, pro, thinking). Uses optimized model
               configurations for each stage.
        query_model: Override the query generation model.
        search_model: Override the web search model.
        reflection_model: Override the reflection model.
        answer_model: Override the answer generation model.

    Returns:
        A dictionary containing the file path to a JSON file with the answer and sources.
    """
    start_time = time.time()
    logger = _get_logger()
    log = logger.bind(
        query=query[:50] + "..." if len(query) > 50 else query,
        effort=effort,
        preset=model,
    )
    log.info("deep_search_started")

    # Resolve models with priority: individual > preset > config defaults
    try:
        models = resolve_models(
            model=model,
            query_model=query_model,
            search_model=search_model,
            reflection_model=reflection_model,
            answer_model=answer_model,
        )
    except ValueError as e:
        log.error("model_resolution_failed", error=str(e))
        return {"error": str(e)}

    # Check if using Google's Deep Research Agent (Interactions API)
    if is_deep_research_mode(models):
        log.info(
            "using_deep_research_agent",
            agent_model=models.get("agent_model"),
        )
        return _run_deep_research(query, log)

    log.debug(
        "models_resolved",
        query_model=models["query_generator_model"],
        search_model=models["web_search_model"],
        reflection_model=models["reflection_model"],
        answer_model=models["answer_model"],
    )

    # Load config for effort-based settings
    app_config = load_config()

    # Set search query count and research loops based on effort level
    if effort == "low":
        initial_search_query_count = 1
        max_research_loops = 1
    elif effort == "medium":
        initial_search_query_count = app_config.initial_query_count
        max_research_loops = app_config.max_research_loops
    else:  # high effort
        initial_search_query_count = 5
        max_research_loops = 3

    log.info(
        "research_config",
        initial_queries=initial_search_query_count,
        max_loops=max_research_loops,
    )

    # Prepare the input state with the user's query
    input_state = {
        "messages": [HumanMessage(content=query)],
        "search_query": [],
        "web_research_result": [],
        "sources_gathered": [],
        "initial_search_query_count": initial_search_query_count,
        "max_research_loops": max_research_loops,
        "reasoning_model": models["answer_model"],  # Used for reasoning steps
    }

    # Configuration for the agent
    config = {
        "configurable": {
            "query_generator_model": models["query_generator_model"],
            "web_search_model": models["web_search_model"],
            "reflection_model": models["reflection_model"],
            "answer_model": models["answer_model"],
        }
    }

    # Run the agent graph to process the query
    try:
        log.info("agent_invocation_started")
        graph = _get_graph()
        result = graph.invoke(input_state, config)
        log.info("agent_invocation_completed")
    except Exception as e:
        log.exception("agent_invocation_failed", error=str(e))
        return {"error": f"Research failed: {str(e)}"}

    # Extract the final answer and sources from the result
    answer = (
        result["messages"][-1].content if result["messages"] else "No answer generated."
    )
    sources = result["sources_gathered"]
    duration = time.time() - start_time

    # Create structured SearchResult with metadata
    search_result = SearchResult.create(
        answer=answer,
        sources=sources,
        query=query,
        effort=effort,
        models_used=models,
        research_loops=max_research_loops,
        duration_seconds=duration,
    )

    # Create filename from first few characters of query (spaces replaced with underscores)
    sanitized_query = re.sub(r'[^\w\s-]', '', query)[:20]
    filename = re.sub(r'\s+', '_', sanitized_query.strip()) + '.json'
    file_path = os.path.join(tempfile.gettempdir(), filename)

    # Write answer, sources, and metadata to JSON file
    result_data = search_result.to_file_format()
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    log.info(
        "deep_search_completed",
        duration_seconds=round(duration, 2),
        sources_count=len(sources),
        answer_length=len(answer),
        output_file=file_path,
    )

    return {"file_path": file_path}


def main():
    """Main entry point for the MCP server.

    Handles CLI commands (--verify, --status, --models, --demo, --version)
    or starts the MCP stdio server if no command is specified.
    """
    from .cli import run_cli

    # Run CLI and check if a command was executed
    exit_code = run_cli()

    if exit_code == -1:
        # No CLI command specified, start MCP server
        logger = _get_logger()
        logger.info("mcp_server_starting", transport="stdio")
        mcp.run(transport="stdio")
    else:
        # CLI command was executed, exit with its return code
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
