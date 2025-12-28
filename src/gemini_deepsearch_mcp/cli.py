"""CLI commands for DeepSearch MCP.

Provides utility commands for verification, status, demo, and model listing.
"""

import argparse
import os
import sys
from typing import NoReturn

from .config import Config, load_config, reset_config
from .exceptions import ConfigurationError
from .models import AVAILABLE_MODELS, MODEL_PRESETS, list_models


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="gemini-deepsearch-mcp",
        description="DeepSearch MCP - Automated research agent with Gemini models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gemini-deepsearch-mcp           Start MCP stdio server (default)
  gemini-deepsearch-mcp --verify  Test Gemini API connection
  gemini-deepsearch-mcp --status  Show current configuration
  gemini-deepsearch-mcp --models  List available models
  gemini-deepsearch-mcp --demo    Run sample search
        """,
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify Gemini API connection and show account info",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current configuration and defaults",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="List available Gemini models with capabilities",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help='Run a demo search with query "What is quantum computing?"',
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )

    return parser


def verify_connection() -> int:
    """Verify Gemini API connection.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    print("Verifying Gemini API connection...")
    print()

    try:
        config = load_config()
        api_key = config.gemini_api_key

        # Mask API key for display
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "****"
        print(f"API Key: {masked_key}")
        print()

        # Try to import and use Gemini
        try:
            import google.genai as genai

            client = genai.Client(api_key=api_key)

            # Try a simple operation to verify the key works
            models = client.models.list()
            model_count = sum(1 for _ in models)

            print(f"Connection: SUCCESS")
            print(f"Available Models: {model_count}")
            print()
            print("Gemini API is configured correctly.")
            return 0

        except ImportError:
            print("ERROR: google-genai package not installed")
            return 1
        except Exception as e:
            print(f"Connection: FAILED")
            print(f"Error: {e}")
            return 1

    except Exception as e:
        print(f"Configuration Error: {e}")
        print()
        print("Make sure GEMINI_API_KEY is set in your environment or .env file.")
        return 1


def show_status() -> int:
    """Show current configuration status.

    Returns:
        Exit code (0 for success).
    """
    print("DeepSearch MCP Configuration Status")
    print("=" * 40)
    print()

    # Check environment variables
    print("Environment Variables:")
    env_vars = [
        ("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY")),
        ("QUERY_GENERATOR_MODEL", os.getenv("QUERY_GENERATOR_MODEL")),
        ("WEB_SEARCH_MODEL", os.getenv("WEB_SEARCH_MODEL")),
        ("REFLECTION_MODEL", os.getenv("REFLECTION_MODEL")),
        ("ANSWER_MODEL", os.getenv("ANSWER_MODEL")),
        ("DEFAULT_EFFORT", os.getenv("DEFAULT_EFFORT")),
        ("MAX_RESEARCH_LOOPS", os.getenv("MAX_RESEARCH_LOOPS")),
        ("INITIAL_QUERY_COUNT", os.getenv("INITIAL_QUERY_COUNT")),
        ("LOG_LEVEL", os.getenv("LOG_LEVEL")),
    ]

    for name, value in env_vars:
        if value:
            if name == "GEMINI_API_KEY":
                # Mask API key
                value = value[:8] + "..." + value[-4:] if len(value) > 12 else "****"
            print(f"  {name}: {value}")
        else:
            print(f"  {name}: (not set)")

    print()

    # Show loaded config
    try:
        reset_config()
        config = load_config()

        print("Loaded Configuration:")
        print(f"  Query Generator Model: {config.query_generator_model}")
        print(f"  Web Search Model: {config.web_search_model}")
        print(f"  Reflection Model: {config.reflection_model}")
        print(f"  Answer Model: {config.answer_model}")
        print(f"  Default Effort: {config.default_effort}")
        print(f"  Max Research Loops: {config.max_research_loops}")
        print(f"  Initial Query Count: {config.initial_query_count}")
        print(f"  Log Level: {config.log_level}")

    except Exception as e:
        print(f"  ERROR loading config: {e}")

    print()

    # Show presets
    print("Available Presets:")
    for name, preset in MODEL_PRESETS.items():
        print(f"  {name}:")
        if "agent_model" in preset:
            # Deep Research preset uses Interactions API
            print(f"    agent_model: {preset['agent_model']}")
            print(f"    (uses Google Deep Research Agent)")
        else:
            print(f"    answer_model: {preset['answer_model']}")

    return 0


def list_available_models() -> int:
    """List available Gemini models.

    Returns:
        Exit code (0 for success).
    """
    print("Available Gemini Models")
    print("=" * 60)
    print()

    # Group by category
    categories = ["flash", "pro", "thinking"]

    for category in categories:
        models = list_models(category=category)
        if models:
            print(f"{category.upper()} Models:")
            print("-" * 40)

            for model in models:
                print(f"  {model.id}")
                print(f"    Name: {model.name}")
                print(f"    Description: {model.description}")
                if model.capabilities:
                    print(f"    Capabilities: {', '.join(model.capabilities)}")
                print(f"    Supports Grounding: {model.supports_grounding}")
                print()

    # Show presets
    print("Model Presets:")
    print("-" * 40)
    for name, preset in MODEL_PRESETS.items():
        print(f"  {name}:")
        for key, value in preset.items():
            print(f"    {key}: {value}")
        print()

    return 0


def run_demo() -> int:
    """Run a demo search.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    print("Running DeepSearch Demo")
    print("=" * 40)
    print()

    query = "What is quantum computing?"
    print(f"Query: {query}")
    print(f"Effort: low")
    print()

    try:
        # This will import the graph which requires GEMINI_API_KEY
        from .main import deep_search

        print("Executing search...")
        print()

        result = deep_search(
            query=query,
            effort="low",
            model="flash",
        )

        if "error" in result:
            print(f"ERROR: {result['error']}")
            return 1

        if "file_path" in result:
            import json

            print(f"Results saved to: {result['file_path']}")
            print()

            # Read and display results
            with open(result["file_path"], "r") as f:
                data = json.load(f)

            print("Answer (first 500 chars):")
            print("-" * 40)
            answer = data.get("answer", "")
            print(answer[:500] + "..." if len(answer) > 500 else answer)
            print()

            sources = data.get("sources", [])
            print(f"Sources: {len(sources)}")
            for i, source in enumerate(sources[:5], 1):
                print(f"  {i}. {source}")

        return 0

    except Exception as e:
        print(f"Demo failed: {e}")
        return 1


def show_version() -> int:
    """Show version information.

    Returns:
        Exit code (0 for success).
    """
    try:
        from importlib.metadata import version

        ver = version("gemini-deepsearch-mcp")
    except Exception:
        ver = "unknown"

    print(f"gemini-deepsearch-mcp version {ver}")
    print()
    print("DeepSearch MCP - Automated research agent with Gemini models")
    print("https://github.com/alexcong/gemini-deepsearch-mcp")

    return 0


def run_cli(args: list[str] | None = None) -> int:
    """Run CLI with given arguments.

    Args:
        args: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code.
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.version:
        return show_version()
    elif parsed_args.verify:
        return verify_connection()
    elif parsed_args.status:
        return show_status()
    elif parsed_args.models:
        return list_available_models()
    elif parsed_args.demo:
        return run_demo()
    else:
        # No command specified - return None to indicate server should start
        return -1  # Special value meaning "run server"
