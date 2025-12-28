# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

- **Start development server**: `make dev` (uses LangGraph dev server with HTTP and Studio UI)
- **Start stdio MCP server**: `make local` (starts LangGraph server + stdio MCP server)
- **Run tests**: `make test` (runs pytest on tests/ directory, excludes trio backend)
- **Run specific test file**: `make test TEST_FILE=path/to/test`
- **Test MCP stdio server**: `make test_mcp` (tests stdio MCP functionality)
- **Watch tests**: `make test_watch`
- **Lint code**: `make lint` (runs ruff, mypy with strict mode)
- **Format code**: `make format` (runs ruff format and import sorting)
- **Check spelling**: `make spell_check`

## Architecture Overview

This is a LangGraph-based web research agent that uses Google Gemini models and Google Search API to perform multi-step research. The system supports dual deployment modes: LangGraph development server (HTTP + Studio UI) and stdio MCP server for client integration.

### Core Components

- **LangGraph Agent** (`src/gemini_deepsearch_mcp/agent/graph.py`): State-driven research workflow with nodes for query generation, web research, reflection, and answer synthesis
- **FastMCP HTTP Server** (`src/gemini_deepsearch_mcp/app.py`): HTTP API that exposes the `deep_search` tool with configurable effort levels - returns full answer and sources data
- **FastMCP stdio Server** (`src/gemini_deepsearch_mcp/main.py`): Core stdio MCP server implementation with deep_search tool - optimized for token usage by writing results to JSON files and returning file paths
- **Root stdio Entry Point** (`main.py`): Wrapper script that imports and runs the stdio MCP server from the main package
- **State Management** (`src/gemini_deepsearch_mcp/agent/state.py`): TypedDict-based states for different workflow stages

### Research Flow

1. **Query Generation**: Generates multiple search queries from user input
2. **Web Research**: Parallel execution of searches using Google Search API
3. **Reflection**: Analyzes results to identify knowledge gaps
4. **Iteration**: Continues research loops based on effort level and sufficiency
5. **Answer Synthesis**: Produces final citation-rich response

### Configuration

- LangGraph configuration in `langgraph.json` defines graph location and HTTP app
- Environment variables required: `GEMINI_API_KEY`
- Effort levels control research depth:
  - Low: 1 query, 1 loop, Flash model
  - Medium: 3 queries, 2 loops, Flash model  
  - High: 5 queries, 3 loops, Pro model

### Output Formats

**HTTP Server** (`app.py` - Development mode):
- Returns complete research data: `{"answer": str, "sources": list}`
- Suitable for web APIs and development workflows

**stdio Server** (`main.py` - Production/Claude Desktop):
- Returns file path: `{"file_path": str}` 
- Research data written to JSON file in system temp directory
- Optimizes token usage for MCP client integrations
- Filename format: `{sanitized_query_prefix}.json` (e.g., `What_is_artificial_i.json`)
- Cross-platform temp directory handling via `tempfile.gettempdir()`

### Deployment Modes

1. **Development Mode** (`make dev`): LangGraph server with HTTP API and Studio UI for development
   - Uses FastMCP HTTP server (`app.py`)
   - Returns full `{"answer": str, "sources": list}` data structure
   - Suitable for web-based development and testing

2. **stdio MCP Mode** (`make local`): Programmatically starts LangGraph server + stdio MCP server for client integration
   - Uses FastMCP stdio server (`main.py`) 
   - Returns `{"file_path": str}` pointing to JSON file with results
   - Optimized for token usage in Claude Desktop integration
   - JSON files are written to cross-platform system temp directory

### Key Files

- `src/gemini_deepsearch_mcp/agent/graph.py`: Main LangGraph workflow definition
- `src/gemini_deepsearch_mcp/app.py`: FastMCP HTTP server with deep_search tool (returns full data)
- `src/gemini_deepsearch_mcp/main.py`: Core stdio MCP server implementation (returns file paths for token optimization)
- `main.py`: Root entry point wrapper for stdio MCP server (imports from src/gemini_deepsearch_mcp/main.py)
- `src/gemini_deepsearch_mcp/agent/configuration.py`: Agent configuration schema
- `src/gemini_deepsearch_mcp/agent/prompts.py`: Prompt templates for different workflow stages
- `src/gemini_deepsearch_mcp/agent/tools_and_schemas.py`: Pydantic schemas for structured outputs
- `tests/test_app.py`: Unit tests for FastMCP HTTP server and deep_search tool (expects full data format)
- `tests/test_simple_mcp.py`: Basic tests for stdio MCP server import and startup
- `tests/test_stdio_client.py`: Integration test client for MCP stdio protocol (expects file path format)

### Testing

The test suite includes comprehensive coverage of the MCP server functionality:

- **HTTP Server Tests** (`tests/test_app.py`): Test deep_search tool with different effort levels, error handling, and FastAPI integration. Expects traditional `{"answer": str, "sources": list}` return format.
- **stdio Server Tests** (`tests/test_simple_mcp.py`, `tests/test_stdio_client.py`): Test MCP stdio server startup, protocol compliance, and file-based output functionality. Expects `{"file_path": str}` return format with JSON file validation.
- **Test Configuration**: Tests use asyncio backend only (trio excluded due to missing dependencies)
- **Mock Support**: Tests use proper mocking for graph.invoke calls to avoid requiring GEMINI_API_KEY during testing
- **Cross-platform**: stdio tests verify temp directory handling works across Windows, macOS, and Linux

### Project Structure

```
├── main.py                                    # Root stdio MCP server entry point
├── src/gemini_deepsearch_mcp/                # Main package
│   ├── __init__.py
│   ├── main.py                               # Core stdio MCP server implementation
│   ├── app.py                                # FastMCP HTTP server
│   └── agent/                                # LangGraph agent components
│       ├── __init__.py
│       ├── graph.py                          # Main workflow definition
│       ├── configuration.py                 # Configuration schema
│       ├── prompts.py                        # Prompt templates
│       ├── state.py                          # State management
│       ├── tools_and_schemas.py              # Pydantic schemas
│       └── utils.py                          # Utility functions
├── tests/                                    # Test suite
│   ├── test_app.py                           # HTTP server tests
│   ├── test_simple_mcp.py                    # Basic MCP tests
│   └── test_stdio_client.py                 # Integration tests
├── langgraph.json                            # LangGraph configuration
└── pyproject.toml                            # Project configuration
```