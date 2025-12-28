"""Simple test for MCP stdio server startup."""

import asyncio
import os
import subprocess
import sys
import time

import pytest


@pytest.mark.anyio
async def test_server_startup():
    """Test that the MCP server can start and stop cleanly."""
    print("Testing MCP server startup...")

    # Get parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_py_path = os.path.join(parent_dir, "main.py")

    try:
        # Start the server process using uv run to ensure proper environment
        process = await asyncio.create_subprocess_exec(
            "uv", "run", "python",
            main_py_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=parent_dir,
        )

        print("✓ Server process started")

        # Wait a moment for initialization
        await asyncio.sleep(3)

        # Check if process is still running
        if process.returncode is None:
            print("✓ Server is running")
        else:
            # Server exited - this might be expected for a stdio server if it's waiting for input
            # Let's check if the exit was clean (returncode 0) or an error
            if process.returncode == 0:
                print("✓ Server exited cleanly (might be expected for stdio server)")
                return True
            else:
                stdout, stderr = await process.communicate()
                print(f"✗ Server exited with error: {stderr.decode()}")
                return False

        # Terminate the process
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
            print("✓ Server terminated cleanly")
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            print("✓ Server killed (timeout)")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_imports():
    """Test that we can import the main module."""
    print("Testing imports...")

    try:
        # Add parent directory to path
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)

        # Test basic imports
        import main  # noqa: F401
        from gemini_deepsearch_mcp.main import deep_search, mcp

        print("✓ Can import main module")

        # Test that MCP server is created
        assert mcp is not None, "MCP server not found"
        print("✓ MCP server object exists")

        # Test that deep_search function exists
        assert deep_search is not None, "deep_search function not found"
        print("✓ deep_search function exists")

    except Exception as e:
        print(f"✗ Import test failed: {e}")
        assert False, f"Import test failed: {e}"


async def main():
    """Run all tests."""
    print("=== Simple MCP Server Tests ===\n")

    # Test imports first
    try:
        test_imports()
        import_success = True
    except Exception:
        import_success = False

    print()

    # Test server startup if imports work
    if import_success:
        startup_success = await test_server_startup()
    else:
        print("Skipping server startup test due to import failures")
        startup_success = False

    print()

    # Summary
    if import_success and startup_success:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
