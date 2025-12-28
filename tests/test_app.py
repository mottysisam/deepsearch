"""Tests for the FastMCP server in src/app.py."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from gemini_deepsearch_mcp.app import app, deep_search

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="session")
def anyio_backend():
    """Use only asyncio backend for tests."""
    return "asyncio"


@pytest.fixture
def mock_graph():
    """Create a mock graph with invoke method."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_graph_result():
    """Mock graph result with answer and sources."""
    return {
        "messages": [
            HumanMessage(content="What is climate change?"),
            AIMessage(
                content="Climate change refers to long-term shifts in global temperatures and weather patterns."
            ),
        ],
        "sources_gathered": [
            {"url": "https://example.com/climate", "title": "Climate Change Overview"},
            {"url": "https://example.com/science", "title": "Climate Science"},
        ],
    }


@pytest.fixture
def mock_models():
    """Mock resolved models configuration."""
    return {
        "query_generator_model": "gemini-2.5-flash",
        "web_search_model": "gemini-2.5-flash-lite-preview-06-17",
        "reflection_model": "gemini-2.5-flash",
        "answer_model": "gemini-2.5-pro",
    }


@pytest.fixture
def mock_config():
    """Mock configuration object."""
    mock = MagicMock()
    mock.initial_query_count = 3
    mock.max_research_loops = 2
    return mock


class TestDeepSearchTool:
    """Test cases for the deep_search tool function."""

    async def test_deep_search_low_effort(self, mock_graph_result, mock_models, mock_config):
        """Test deep_search with low effort level."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_graph_result

        with (
            patch("gemini_deepsearch_mcp.app._get_graph", return_value=mock_graph),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
            patch("gemini_deepsearch_mcp.app.resolve_models", return_value=mock_models),
            patch("gemini_deepsearch_mcp.app.load_config", return_value=mock_config),
        ):
            mock_to_thread.return_value = mock_graph_result

            # Access underlying function via .fn attribute (FunctionTool wrapper)
            result = await deep_search.fn("What is climate change?", "low")

            # Verify result structure
            assert "answer" in result
            assert "sources" in result
            assert (
                result["answer"]
                == "Climate change refers to long-term shifts in global temperatures and weather patterns."
            )
            assert len(result["sources"]) == 2

            # Verify asyncio.to_thread was called
            mock_to_thread.assert_called_once()

            # Get the actual arguments passed to graph.invoke via asyncio.to_thread
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Verify low effort configuration
            assert input_state["initial_search_query_count"] == 1
            assert input_state["max_research_loops"] == 1
            assert len(input_state["messages"]) == 1
            assert input_state["messages"][0].content == "What is climate change?"

    async def test_deep_search_medium_effort(self, mock_graph_result, mock_models, mock_config):
        """Test deep_search with medium effort level."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_graph_result

        with (
            patch("gemini_deepsearch_mcp.app._get_graph", return_value=mock_graph),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
            patch("gemini_deepsearch_mcp.app.resolve_models", return_value=mock_models),
            patch("gemini_deepsearch_mcp.app.load_config", return_value=mock_config),
        ):
            mock_to_thread.return_value = mock_graph_result

            result = await deep_search.fn("What is artificial intelligence?", "medium")

            # Verify result structure
            assert "answer" in result
            assert "sources" in result

            # Get the actual arguments passed to graph.invoke via asyncio.to_thread
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Verify medium effort configuration (uses mock_config values)
            assert input_state["initial_search_query_count"] == 3
            assert input_state["max_research_loops"] == 2

    async def test_deep_search_high_effort(self, mock_graph_result, mock_models, mock_config):
        """Test deep_search with high effort level."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_graph_result

        with (
            patch("gemini_deepsearch_mcp.app._get_graph", return_value=mock_graph),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
            patch("gemini_deepsearch_mcp.app.resolve_models", return_value=mock_models),
            patch("gemini_deepsearch_mcp.app.load_config", return_value=mock_config),
        ):
            mock_to_thread.return_value = mock_graph_result

            result = await deep_search.fn("Explain quantum computing", "high")

            # Verify result structure
            assert "answer" in result
            assert "sources" in result

            # Get the actual arguments passed to graph.invoke via asyncio.to_thread
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Verify high effort configuration
            assert input_state["initial_search_query_count"] == 5
            assert input_state["max_research_loops"] == 3

    async def test_deep_search_default_effort(self, mock_graph_result, mock_models, mock_config):
        """Test deep_search with default effort level (should be low)."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_graph_result

        with (
            patch("gemini_deepsearch_mcp.app._get_graph", return_value=mock_graph),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
            patch("gemini_deepsearch_mcp.app.resolve_models", return_value=mock_models),
            patch("gemini_deepsearch_mcp.app.load_config", return_value=mock_config),
        ):
            mock_to_thread.return_value = mock_graph_result

            await deep_search.fn("What is machine learning?")

            # Get the actual arguments passed to graph.invoke via asyncio.to_thread
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Verify default (low) effort configuration
            assert input_state["initial_search_query_count"] == 1
            assert input_state["max_research_loops"] == 1

    async def test_deep_search_empty_messages(self, mock_models, mock_config):
        """Test deep_search when graph returns empty messages."""
        mock_result = {"messages": [], "sources_gathered": []}
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_result

        with (
            patch("gemini_deepsearch_mcp.app._get_graph", return_value=mock_graph),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
            patch("gemini_deepsearch_mcp.app.resolve_models", return_value=mock_models),
            patch("gemini_deepsearch_mcp.app.load_config", return_value=mock_config),
        ):
            mock_to_thread.return_value = mock_result

            result = await deep_search.fn("Test query", "low")

            assert result["answer"] == "No answer generated."
            assert result["sources"] == []

    async def test_deep_search_config_models(self, mock_graph_result, mock_models, mock_config):
        """Test that deep_search passes correct model configuration."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_graph_result

        with (
            patch("gemini_deepsearch_mcp.app._get_graph", return_value=mock_graph),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
            patch("gemini_deepsearch_mcp.app.resolve_models", return_value=mock_models),
            patch("gemini_deepsearch_mcp.app.load_config", return_value=mock_config),
        ):
            mock_to_thread.return_value = mock_graph_result

            await deep_search.fn("Test query", "low")

            # Get the config passed to graph.invoke
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Verify model configuration contains expected keys
            assert "configurable" in config
            assert "query_generator_model" in config["configurable"]
            assert "reflection_model" in config["configurable"]
            assert "answer_model" in config["configurable"]

    async def test_deep_search_input_state_structure(self, mock_graph_result, mock_models, mock_config):
        """Test that deep_search creates correct input state structure."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_graph_result

        with (
            patch("gemini_deepsearch_mcp.app._get_graph", return_value=mock_graph),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
            patch("gemini_deepsearch_mcp.app.resolve_models", return_value=mock_models),
            patch("gemini_deepsearch_mcp.app.load_config", return_value=mock_config),
        ):
            mock_to_thread.return_value = mock_graph_result

            query = "What is renewable energy?"
            await deep_search.fn(query, "medium")

            # Get the input state passed to graph.invoke
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Verify input state structure
            assert "messages" in input_state
            assert "search_query" in input_state
            assert "web_research_result" in input_state
            assert "sources_gathered" in input_state
            assert "initial_search_query_count" in input_state
            assert "max_research_loops" in input_state
            assert "reasoning_model" in input_state

            # Verify initial values
            assert len(input_state["messages"]) == 1
            assert isinstance(input_state["messages"][0], HumanMessage)
            assert input_state["messages"][0].content == query
            assert input_state["search_query"] == []
            assert input_state["web_research_result"] == []
            assert input_state["sources_gathered"] == []


class TestFastAPIApp:
    """Test cases for the FastAPI application."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_app_creation(self, client):
        """Test that the FastAPI app is created correctly."""
        # Test that the app responds (even if MCP endpoints aren't directly accessible)
        # The app should at least be instantiated without errors
        assert client.app is not None

    def test_mcp_mount(self, client):
        """Test that MCP server is mounted correctly."""
        # The MCP server should be mounted at /mcp-server
        # We can't easily test the MCP endpoints without full integration,
        # but we can verify the mount exists
        # Look for the mounted MCP server path
        mcp_mounted = any("/mcp-server" in str(route) for route in client.app.routes)
        assert mcp_mounted


class TestErrorHandling:
    """Test error handling scenarios."""

    async def test_deep_search_graph_exception(self, mock_models, mock_config):
        """Test deep_search when graph.invoke raises an exception."""
        mock_graph = MagicMock()

        with (
            patch("gemini_deepsearch_mcp.app._get_graph", return_value=mock_graph),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
            patch("gemini_deepsearch_mcp.app.resolve_models", return_value=mock_models),
            patch("gemini_deepsearch_mcp.app.load_config", return_value=mock_config),
        ):
            mock_to_thread.side_effect = Exception("Graph execution failed")

            with pytest.raises(Exception, match="Graph execution failed"):
                await deep_search.fn("Test query", "low")

    async def test_deep_search_invalid_effort_level(self, mock_graph_result, mock_models, mock_config):
        """Test deep_search with invalid effort level (should default to high)."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_graph_result

        with (
            patch("gemini_deepsearch_mcp.app._get_graph", return_value=mock_graph),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
            patch("gemini_deepsearch_mcp.app.resolve_models", return_value=mock_models),
            patch("gemini_deepsearch_mcp.app.load_config", return_value=mock_config),
        ):
            mock_to_thread.return_value = mock_graph_result

            # Pass an invalid effort level - should default to high effort
            await deep_search.fn("Test query", "invalid")

            # Get the actual arguments passed to graph.invoke via asyncio.to_thread
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Should default to high effort configuration
            assert input_state["initial_search_query_count"] == 5
            assert input_state["max_research_loops"] == 3


class TestDeepResearchMode:
    """Test cases for Google Deep Research Agent integration."""

    @pytest.fixture
    def mock_deep_research_result(self):
        """Mock result from DeepResearchAgent.research()."""
        # Create a simple mock object with the expected attributes
        mock_result = MagicMock()
        mock_result.answer = "This is a comprehensive research report about AI."
        mock_result.sources = [
            {"url": "https://example.com/ai", "title": "AI Overview"},
            {"url": "https://example.com/ml", "title": "Machine Learning Guide"},
        ]
        mock_result.interaction_id = "test-interaction-123"
        mock_result.status = "completed"
        mock_result.elapsed_seconds = 120.5
        return mock_result

    @pytest.fixture
    def mock_deep_research_models(self):
        """Mock models configuration for deep-research mode."""
        return {
            "use_interactions_api": True,
            "agent_model": "deep-research-pro-preview-12-2025",
        }

    async def test_deep_search_deep_research_mode(
        self, mock_deep_research_result, mock_deep_research_models
    ):
        """Test deep_search with deep-research model preset."""
        with (
            patch(
                "gemini_deepsearch_mcp.app.resolve_models",
                return_value=mock_deep_research_models,
            ),
            patch(
                "gemini_deepsearch_mcp.app.is_deep_research_mode",
                return_value=True,
            ),
            patch(
                "gemini_deepsearch_mcp.app._run_deep_research_async"
            ) as mock_run_deep_research,
        ):
            mock_run_deep_research.return_value = {
                "answer": mock_deep_research_result.answer,
                "sources": mock_deep_research_result.sources,
                "metadata": {
                    "mode": "google-deep-research",
                    "agent_model": "deep-research-pro-preview-12-2025",
                    "interaction_id": mock_deep_research_result.interaction_id,
                    "elapsed_seconds": mock_deep_research_result.elapsed_seconds,
                },
            }

            result = await deep_search.fn("What is AI?", "low", model="deep-research")

            # Verify result structure
            assert "answer" in result
            assert "sources" in result
            assert "metadata" in result
            assert result["answer"] == "This is a comprehensive research report about AI."
            assert len(result["sources"]) == 2
            assert result["metadata"]["mode"] == "google-deep-research"
            assert result["metadata"]["interaction_id"] == "test-interaction-123"

            # Verify _run_deep_research_async was called
            mock_run_deep_research.assert_called_once_with("What is AI?")

    async def test_deep_search_deep_research_timeout_error(self, mock_deep_research_models):
        """Test deep_search when Deep Research times out."""
        with (
            patch(
                "gemini_deepsearch_mcp.app.resolve_models",
                return_value=mock_deep_research_models,
            ),
            patch(
                "gemini_deepsearch_mcp.app.is_deep_research_mode",
                return_value=True,
            ),
            patch(
                "gemini_deepsearch_mcp.app._run_deep_research_async"
            ) as mock_run_deep_research,
        ):
            mock_run_deep_research.return_value = {
                "error": "Research exceeded maximum time of 3600 seconds."
            }

            result = await deep_search.fn("Complex research query", model="deep-research")

            assert "error" in result
            assert "exceeded maximum time" in result["error"]

    async def test_deep_search_deep_research_runtime_error(self, mock_deep_research_models):
        """Test deep_search when Deep Research fails."""
        with (
            patch(
                "gemini_deepsearch_mcp.app.resolve_models",
                return_value=mock_deep_research_models,
            ),
            patch(
                "gemini_deepsearch_mcp.app.is_deep_research_mode",
                return_value=True,
            ),
            patch(
                "gemini_deepsearch_mcp.app._run_deep_research_async"
            ) as mock_run_deep_research,
        ):
            mock_run_deep_research.return_value = {"error": "Research failed: API error"}

            result = await deep_search.fn("Test query", model="deep-research")

            assert "error" in result
            assert "Research failed" in result["error"]

    async def test_deep_search_uses_langgraph_for_non_deep_research(
        self, mock_graph_result, mock_models, mock_config
    ):
        """Test that non-deep-research presets still use LangGraph workflow."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_graph_result

        with (
            patch("gemini_deepsearch_mcp.app._get_graph", return_value=mock_graph),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
            patch("gemini_deepsearch_mcp.app.resolve_models", return_value=mock_models),
            patch("gemini_deepsearch_mcp.app.is_deep_research_mode", return_value=False),
            patch("gemini_deepsearch_mcp.app.load_config", return_value=mock_config),
        ):
            mock_to_thread.return_value = mock_graph_result

            result = await deep_search.fn("Test query", "low", model="pro")

            # Verify result comes from LangGraph (has answer and sources, no metadata)
            assert "answer" in result
            assert "sources" in result
            # LangGraph path doesn't add metadata
            assert "metadata" not in result

            # Verify asyncio.to_thread was called (LangGraph path)
            mock_to_thread.assert_called_once()
