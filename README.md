# Gemini DeepSearch MCP

Gemini DeepSearch MCP is an automated research agent that leverages Google Gemini models and Google Search to perform deep, multi-step web research. It generates sophisticated queries, synthesizes information from search results, identifies knowledge gaps, and produces high-quality, citation-rich answers.

## Features

- **Automated multi-step research** using Gemini models and Google Search
- **FastMCP integration** for both HTTP API and stdio deployment
- **Configurable effort levels** (low, medium, high) for research depth
- **Citation-rich responses** with source tracking
- **LangGraph-powered workflow** with state management

## Usage

### Development Server (HTTP + Studio UI)
Start the LangGraph development server with Studio UI:
```bash
make dev
```

### Local MCP Server (stdio)
Start the MCP server with stdio transport for integration with MCP clients:
```bash
make local
```

### Testing
Run the test suite:
```bash
make test
```

Test the MCP stdio server:
```bash
make test_mcp
```

Use MCP inspector
```bash
make inspect
```

With Langsmith tracing
```bash
GEMINI_API_KEY=AI******* LANGSMITH_API_KEY=ls******* LANGSMITH_TRACING=true make inspect
```

## API

The `deep_search` tool accepts:
- **query** (string): The research question or topic to investigate
- **effort** (string): Research effort level - "low", "medium", or "high"
  - **Low**: 1 query, 1 loop, Flash model
  - **Medium**: 3 queries, 2 loops, Flash model
  - **High**: 5 queries, 3 loops, Pro model

### Return Format

**HTTP MCP Server** (Development mode):
- **answer**: Comprehensive research response with citations
- **sources**: List of source URLs used in research

**Stdio MCP Server** (Claude Desktop integration):
- **file_path**: Path to a JSON file containing the research results

The stdio MCP server writes results to a JSON file in the system temp directory to optimize token usage. The JSON file contains the same `answer` and `sources` data as the HTTP version, but is accessed via file path rather than returned directly.

## Requirements

- Python 3.12+
- `GEMINI_API_KEY` environment variable

## Installation

Install directly using uvx:

```bash
uvx install gemini-deepsearch-mcp
```

## Claude Desktop Integration

To use the MCP server with Claude Desktop, add this configuration to your Claude Desktop config file:

### macOS
Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gemini-deepsearch": {
      "command": "uvx",
      "args": ["gemini-deepsearch-mcp"],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here"
      },
      "timeout": 180000
    }
  }
}
```

### Windows
Edit `%APPDATA%/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gemini-deepsearch": {
      "command": "uvx",
      "args": ["gemini-deepsearch-mcp"],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here"
      },
      "timeout": 180000
    }
  }
}
```

### Linux
Edit `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gemini-deepsearch": {
      "command": "uvx",
      "args": ["gemini-deepsearch-mcp"],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here"
      },
      "timeout": 180000
    }
  }
}
```

**Important:**
- Replace `your-gemini-api-key-here` with your actual Gemini API key
- Restart Claude Desktop after updating the configuration
- Set ample timeout to avoid `MCP error -32001: Request timed out`

### Alternative: Local Development Setup

For development or if you prefer to run from source:

```json
{
  "mcpServers": {
    "gemini-deepsearch": {
      "command": "uv",
      "args": ["run", "python", "main.py"],
      "cwd": "/path/to/gemini-deepsearch-mcp",
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here"
      }
    }
  }
}
```

Replace `/path/to/gemini-deepsearch-mcp` with the actual absolute path to your project directory.

Once configured, you can use the `deep_search` tool in Claude Desktop by asking questions like:
- "Use deep_search to research the latest developments in quantum computing"
- "Search for information about renewable energy trends with high effort"

## Agent Source
The deep search agent is from the [Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) repository.


## License
MIT
