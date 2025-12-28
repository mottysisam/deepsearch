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

from .agent.graph import graph
from .config import load_config
from .logging import configure_logging, get_logger
from .model_selection import resolve_models

# Initialize logging and create logger
_config = load_config()
configure_logging(_config.log_level)
logger = get_logger("deepsearch.main")

# Create MCP server
mcp = FastMCP("DeepSearch")


@mcp.tool()
def deep_search(
    query: Annotated[str, Field(description="Search query string")],
    effort: Annotated[
        Literal["low", "medium", "high"],
        Field(description="Research effort level: low (1 query, 1 loop), "
                         "medium (3 queries, 2 loops), high (5 queries, 3 loops)")
    ] = "low",
    model: Annotated[
        Literal["flash", "pro", "thinking"] | None,
        Field(description="Model preset: flash (fast), pro (capable), "
                         "thinking (extended reasoning). Overrides config defaults.")
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
        log.debug(
            "models_resolved",
            query_model=models["query_generator_model"],
            search_model=models["web_search_model"],
            reflection_model=models["reflection_model"],
            answer_model=models["answer_model"],
        )
    except ValueError as e:
        log.error("model_resolution_failed", error=str(e))
        return {"error": str(e)}

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

    # Create filename from first few characters of query (spaces replaced with underscores)
    sanitized_query = re.sub(r'[^\w\s-]', '', query)[:20]
    filename = re.sub(r'\s+', '_', sanitized_query.strip()) + '.json'
    file_path = os.path.join(tempfile.gettempdir(), filename)

    # Write answer and sources to JSON file
    result_data = {"answer": answer, "sources": sources}
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    duration = time.time() - start_time
    log.info(
        "deep_search_completed",
        duration_seconds=round(duration, 2),
        sources_count=len(sources),
        answer_length=len(answer),
        output_file=file_path,
    )

    return {"file_path": file_path}


def main():
    """Main entry point for the MCP server."""
    logger.info("mcp_server_starting", transport="stdio")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
