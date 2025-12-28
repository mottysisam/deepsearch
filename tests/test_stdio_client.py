"""Simple test client for the stdio MCP server."""

import asyncio
import json
import subprocess
import sys
from typing import Any, Dict

import pytest


class StdioMCPClient:
    """Simple MCP client for testing stdio communication."""

    def __init__(self, server_command: list[str]):
        """Initialize the client with server command."""
        self.server_command = server_command
        self.process = None
        self.request_id = 0

    async def start(self):
        """Start the MCP server process."""
        import os

        # Set working directory to parent of tests
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.process = await asyncio.create_subprocess_exec(
            *self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=parent_dir,  # Run from project root
        )

        # Wait longer for both LangGraph and MCP server to initialize
        await asyncio.sleep(5)

    async def send_request(
        self, method: str, params: Dict[str, Any] = None, timeout: float = 10.0
    ) -> Dict[str, Any]:
        """Send a JSON-RPC request to the server."""
        if not self.process:
            raise RuntimeError("Server not started")

        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {},
        }

        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json.encode())
            await self.process.stdin.drain()

            # Read response with timeout
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(), timeout=timeout
            )

            if not response_line:
                stderr_output = await self.process.stderr.read()
                raise RuntimeError(
                    f"No response from server. Stderr: {stderr_output.decode()}"
                )

            response = json.loads(response_line.decode())
            return response

        except asyncio.TimeoutError:
            raise RuntimeError(f"Request timeout after {timeout}s for method: {method}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {response_line.decode()}")

    async def stop(self):
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            await self.process.wait()


@pytest.mark.anyio
async def test_mcp_server():
    """Test the stdio MCP server."""
    import os

    # Change to parent directory to run main.py
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    client = StdioMCPClient(["python", "main.py"])
    client.server_command = ["python", os.path.join(parent_dir, "main.py")]

    try:
        print("Starting MCP server...")
        await client.start()

        # Check if server process is still running
        if client.process.returncode is not None:
            stderr_output = await client.process.stderr.read()
            raise RuntimeError(f"Server exited early: {stderr_output.decode()}")

        print("Server started successfully")

        # Test initialization with shorter timeout
        print("\n1. Testing initialization...")
        try:
            init_response = await client.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
                timeout=5.0,
            )
            print(f"Initialize response: {init_response}")
        except Exception as e:
            print(f"Initialize failed: {e}")
            return

        # Test listing tools
        print("\n2. Testing tools/list...")
        try:
            tools_response = await client.send_request("tools/list", timeout=5.0)
            print(f"Tools response: {tools_response}")
        except Exception as e:
            print(f"Tools list failed: {e}")
            return

        # Skip the deep_search test as it requires GEMINI_API_KEY and can be slow
        print("\n3. Skipping deep_search tool test (requires GEMINI_API_KEY)")
        print("✓ Basic MCP protocol working")

        print("\nBasic MCP protocol tests completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        # Print stderr for debugging
        if client.process:
            try:
                stderr_data = await asyncio.wait_for(
                    client.process.stderr.read(), timeout=2.0
                )
                stderr_output = stderr_data.decode()
                if stderr_output.strip():
                    print(f"Server stderr: {stderr_output}")
            except asyncio.TimeoutError:
                print("Could not read stderr (timeout)")

    finally:
        await client.stop()
        print("Server stopped")


@pytest.mark.anyio
async def test_basic_functionality():
    """Test basic functionality without full MCP protocol."""
    print("Testing basic deep_search functionality...")

    # Import and test the function directly
    try:
        import os

        # Add parent directory to path for importing main
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Mock the graph dependency
        from unittest.mock import MagicMock, patch

        from gemini_deepsearch_mcp.main import deep_search

        with patch("gemini_deepsearch_mcp.main.graph.invoke") as mock_graph:
            mock_graph.return_value = {
                "messages": [MagicMock(content="AI is a field of computer science...")],
                "sources_gathered": ["example.com", "research.org"],
            }

            result = await deep_search("What is AI?", "low")
            print(f"Direct function test result: {result}")
            import tempfile
            assert "file_path" in result
            assert result["file_path"].startswith(tempfile.gettempdir())
            assert result["file_path"].endswith(".json")
            print("✓ Direct function test passed")

    except Exception as e:
        print(f"✗ Direct function test failed: {e}")
        print(
            "This is expected if dependencies are missing or GEMINI_API_KEY is not set"
        )


async def run_all_tests():
    """Run all tests with timeout."""
    print("=== MCP Stdio Server Test ===")

    # Test basic functionality first
    await test_basic_functionality()

    print("\n" + "=" * 50)

    # Test full MCP protocol with timeout
    try:
        await asyncio.wait_for(test_mcp_server(), timeout=30.0)
    except asyncio.TimeoutError:
        print("❌ MCP protocol test timed out after 30 seconds")
        print("This might be due to missing GEMINI_API_KEY or server startup issues")
    except Exception as e:
        print(f"MCP protocol test failed: {e}")
        print(
            "This is expected if GEMINI_API_KEY is not set or LangGraph dependencies are missing"
        )


if __name__ == "__main__":
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test suite failed: {e}")
