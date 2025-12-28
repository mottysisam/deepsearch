"""Main entry point for the stdio MCP server."""

import json
import os
import re
import sys
import tempfile
from typing import Annotated, Literal

from fastmcp import FastMCP
from langchain_core.messages import HumanMessage
from pydantic import Field

from .agent.graph import graph

# Create MCP server
mcp = FastMCP("DeepSearch")


@mcp.tool()
def deep_search(
    query: Annotated[str, Field(description="Search query string")],
    effort: Annotated[
        Literal["low", "medium", "high"], Field(description="Search effort")
    ] = "low",
) -> dict:
    """Perform a deep search on a given query using an advanced web research agent.

    Args:
        query: The research question or topic to investigate.
        effort: The amount of effect for the research, low, medium or hight (default: low).

    Returns:
        A dictionary containing the file path to a JSON file with the answer and sources.
    """
    # Set search query count, research loops and reasoning model based on effort level
    if effort == "low":
        initial_search_query_count = 1
        max_research_loops = 1
        reasoning_model = "gemini-2.5-flash-preview-05-20"
    elif effort == "medium":
        initial_search_query_count = 3
        max_research_loops = 2
        reasoning_model = "gemini-2.5-flash-preview-05-20"
    else:  # high effort
        initial_search_query_count = 5
        max_research_loops = 3
        reasoning_model = "gemini-2.5-pro-preview-06-05"

    # Prepare the input state with the user's query
    input_state = {
        "messages": [HumanMessage(content=query)],
        "search_query": [],
        "web_research_result": [],
        "sources_gathered": [],
        "initial_search_query_count": initial_search_query_count,
        "max_research_loops": max_research_loops,
        "reasoning_model": reasoning_model,
    }

    query_generator_model: str = "gemini-2.5-flash-preview-05-20"
    reflection_model: str = "gemini-2.5-flash-preview-05-20"
    answer_model: str = "gemini-2.5-pro-preview-06-05"

    # Configuration for the agent
    config = {
        "configurable": {
            "query_generator_model": query_generator_model,
            "reflection_model": reflection_model,
            "answer_model": answer_model,
        }
    }

    # Run the agent graph to process the query in a separate thread to avoid blocking
    result = graph.invoke(input_state, config)

    # Extract the final answer and sources from the result
    answer = (
        result["messages"][-1].content if result["messages"] else "No answer generated."
    )
    sources = result["sources_gathered"]

    # Create filename from first few characters of query (spaces replaced with underscores)
    sanitized_query = re.sub(r'[^\w\s-]', '', query)[:20]  # Remove special chars, keep first 20 chars
    filename = re.sub(r'\s+', '_', sanitized_query.strip()) + '.json'
    file_path = os.path.join(tempfile.gettempdir(), filename)
    
    # Write answer and sources to JSON file
    result_data = {"answer": answer, "sources": sources}
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    return {"file_path": file_path}


def main():
    """Main entry point for the MCP server."""
    sys.stderr.write("Starting MCP stdio server...\n")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
