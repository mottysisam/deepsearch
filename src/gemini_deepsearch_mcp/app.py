import asyncio
from typing import Annotated, Literal

from fastapi import FastAPI
from fastmcp import FastMCP
from langchain_core.messages import HumanMessage
from pydantic import Field
from starlette.routing import Mount

from .agent.graph import graph

mcp = FastMCP("DeepSearch")


@mcp.tool()
async def deep_search(
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
        A dictionary containing the answer to the query and a list of sources used.
    """
    # Set search query count, research loops and reasoning model based on effort level
    if effort == "low":
        initial_search_query_count = 1
        max_research_loops = 1
        reasoning_model = "gemini-2.5-flash"
    elif effort == "medium":
        initial_search_query_count = 3
        max_research_loops = 2
        reasoning_model = "gemini-2.5-flash"
    else:  # high effort
        initial_search_query_count = 5
        max_research_loops = 3
        reasoning_model = "gemini-2.5-pro"

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

    query_generator_model: str = "gemini-2.5-flash"
    web_search_model: str = "gemini-2.5-flash-lite-preview-06-17"
    reflection_model: str = "gemini-2.5-flash"
    answer_model: str = "gemini-2.5-pro"

    # Configuration for the agent
    config = {
        "configurable": {
            "query_generator_model": query_generator_model,
            "web_search_model": web_search_model,
            "reflection_model": reflection_model,
            "answer_model": answer_model,
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
