"""HTTP MCP server for development and API access."""

import asyncio
from typing import Annotated, Literal

from fastapi import FastAPI
from fastmcp import FastMCP
from langchain_core.messages import HumanMessage
from pydantic import Field
from starlette.routing import Mount

from .agent.graph import graph
from .config import load_config
from .model_selection import resolve_models

mcp = FastMCP("DeepSearch")


@mcp.tool()
async def deep_search(
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
        A dictionary containing the answer to the query and a list of sources used.
    """
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

    # Run the agent graph to process the query in a separate thread to avoid blocking
    result = await asyncio.to_thread(graph.invoke, input_state, config)

    # Extract the final answer and sources from the result
    answer = (
        result["messages"][-1].content if result["messages"] else "No answer generated."
    )
    sources = result["sources_gathered"]

    return {"answer": answer, "sources": sources}


# Create the ASGI app
mcp_app = mcp.http_app(path="/mcp")

# Create a FastAPI app and mount the MCP server
app = FastAPI(lifespan=mcp_app.lifespan)
app.mount("/mcp-server", mcp_app)
