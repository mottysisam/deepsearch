"""Tests for CLI commands.

Tests cover:
- CLI argument parsing
- verify command (success and failure)
- status command output
- models command listing
- version command
- default behavior (no args runs server)
"""

import os
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from gemini_deepsearch_mcp.cli import (
    create_parser,
    list_available_models,
    run_cli,
    show_status,
    show_version,
    verify_connection,
)


class TestCreateParser:
    """Test argument parser creation."""

    def test_parser_created(self):
        """create_parser should return ArgumentParser."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "gemini-deepsearch-mcp"

    def test_parser_has_verify_flag(self):
        """Parser should have --verify flag."""
        parser = create_parser()
        args = parser.parse_args(["--verify"])
        assert args.verify is True

    def test_parser_has_status_flag(self):
        """Parser should have --status flag."""
        parser = create_parser()
        args = parser.parse_args(["--status"])
        assert args.status is True

    def test_parser_has_models_flag(self):
        """Parser should have --models flag."""
        parser = create_parser()
        args = parser.parse_args(["--models"])
        assert args.models is True

    def test_parser_has_demo_flag(self):
        """Parser should have --demo flag."""
        parser = create_parser()
        args = parser.parse_args(["--demo"])
        assert args.demo is True

    def test_parser_has_version_flag(self):
        """Parser should have --version flag."""
        parser = create_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_parser_no_args_all_false(self):
        """No args should have all flags False."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.verify is False
        assert args.status is False
        assert args.models is False
        assert args.demo is False
        assert args.version is False


class TestVerifyConnectionSuccess:
    """Test verify_connection when API is configured."""

    @patch("gemini_deepsearch_mcp.cli.load_config")
    @patch("google.genai.Client")
    def test_verify_success(self, mock_client_class, mock_load_config):
        """verify_connection should return 0 on success."""
        # Mock config
        mock_config = MagicMock()
        mock_config.gemini_api_key = "test-api-key-12345678"
        mock_load_config.return_value = mock_config

        # Mock client
        mock_client = MagicMock()
        mock_client.models.list.return_value = [MagicMock(), MagicMock()]
        mock_client_class.return_value = mock_client

        result = verify_connection()
        assert result == 0

    @patch("gemini_deepsearch_mcp.cli.load_config")
    @patch("google.genai.Client")
    def test_verify_prints_success(self, mock_client_class, mock_load_config, capsys):
        """verify_connection should print success message."""
        mock_config = MagicMock()
        mock_config.gemini_api_key = "test-api-key-12345678"
        mock_load_config.return_value = mock_config

        mock_client = MagicMock()
        mock_client.models.list.return_value = [MagicMock(), MagicMock()]
        mock_client_class.return_value = mock_client

        verify_connection()
        captured = capsys.readouterr()
        assert "SUCCESS" in captured.out


class TestVerifyConnectionFailure:
    """Test verify_connection when API fails."""

    @patch("gemini_deepsearch_mcp.cli.load_config")
    def test_verify_failure_no_key(self, mock_load_config):
        """verify_connection should return 1 when no API key."""
        mock_load_config.side_effect = Exception("GEMINI_API_KEY not set")

        result = verify_connection()
        assert result == 1

    @patch("gemini_deepsearch_mcp.cli.load_config")
    @patch("google.genai.Client")
    def test_verify_failure_api_error(self, mock_client_class, mock_load_config):
        """verify_connection should return 1 on API error."""
        mock_config = MagicMock()
        mock_config.gemini_api_key = "test-api-key-12345678"
        mock_load_config.return_value = mock_config

        mock_client_class.side_effect = Exception("Invalid API key")

        result = verify_connection()
        assert result == 1

    @patch("gemini_deepsearch_mcp.cli.load_config")
    def test_verify_failure_prints_error(self, mock_load_config, capsys):
        """verify_connection should print error message on failure."""
        mock_load_config.side_effect = Exception("Config error")

        verify_connection()
        captured = capsys.readouterr()
        assert "Error" in captured.out or "error" in captured.out.lower()


class TestShowStatusOutput:
    """Test show_status command output."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-12345678"}, clear=False)
    @patch("gemini_deepsearch_mcp.cli.reset_config")
    @patch("gemini_deepsearch_mcp.cli.load_config")
    def test_status_returns_zero(self, mock_load_config, mock_reset):
        """show_status should return 0."""
        mock_config = MagicMock()
        mock_config.query_generator_model = "gemini-2.5-flash"
        mock_config.web_search_model = "gemini-2.5-flash"
        mock_config.reflection_model = "gemini-2.5-flash"
        mock_config.answer_model = "gemini-2.5-pro"
        mock_config.default_effort = "medium"
        mock_config.max_research_loops = 2
        mock_config.initial_query_count = 3
        mock_config.log_level = "INFO"
        mock_load_config.return_value = mock_config

        result = show_status()
        assert result == 0

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-12345678"}, clear=False)
    @patch("gemini_deepsearch_mcp.cli.reset_config")
    @patch("gemini_deepsearch_mcp.cli.load_config")
    def test_status_shows_environment_vars(self, mock_load_config, mock_reset, capsys):
        """show_status should display environment variables."""
        mock_config = MagicMock()
        mock_config.query_generator_model = "gemini-2.5-flash"
        mock_config.web_search_model = "gemini-2.5-flash"
        mock_config.reflection_model = "gemini-2.5-flash"
        mock_config.answer_model = "gemini-2.5-pro"
        mock_config.default_effort = "medium"
        mock_config.max_research_loops = 2
        mock_config.initial_query_count = 3
        mock_config.log_level = "INFO"
        mock_load_config.return_value = mock_config

        show_status()
        captured = capsys.readouterr()
        assert "GEMINI_API_KEY" in captured.out
        assert "Environment Variables" in captured.out

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-12345678"}, clear=False)
    @patch("gemini_deepsearch_mcp.cli.reset_config")
    @patch("gemini_deepsearch_mcp.cli.load_config")
    def test_status_shows_loaded_config(self, mock_load_config, mock_reset, capsys):
        """show_status should display loaded configuration."""
        mock_config = MagicMock()
        mock_config.query_generator_model = "gemini-2.5-flash"
        mock_config.web_search_model = "gemini-2.5-flash"
        mock_config.reflection_model = "gemini-2.5-flash"
        mock_config.answer_model = "gemini-2.5-pro"
        mock_config.default_effort = "medium"
        mock_config.max_research_loops = 2
        mock_config.initial_query_count = 3
        mock_config.log_level = "INFO"
        mock_load_config.return_value = mock_config

        show_status()
        captured = capsys.readouterr()
        assert "Loaded Configuration" in captured.out

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-12345678"}, clear=False)
    @patch("gemini_deepsearch_mcp.cli.reset_config")
    @patch("gemini_deepsearch_mcp.cli.load_config")
    def test_status_shows_presets(self, mock_load_config, mock_reset, capsys):
        """show_status should display available presets."""
        mock_config = MagicMock()
        mock_config.query_generator_model = "gemini-2.5-flash"
        mock_config.web_search_model = "gemini-2.5-flash"
        mock_config.reflection_model = "gemini-2.5-flash"
        mock_config.answer_model = "gemini-2.5-pro"
        mock_config.default_effort = "medium"
        mock_config.max_research_loops = 2
        mock_config.initial_query_count = 3
        mock_config.log_level = "INFO"
        mock_load_config.return_value = mock_config

        show_status()
        captured = capsys.readouterr()
        assert "Presets" in captured.out


class TestListAvailableModels:
    """Test list_available_models command."""

    def test_models_returns_zero(self):
        """list_available_models should return 0."""
        result = list_available_models()
        assert result == 0

    def test_models_lists_flash(self, capsys):
        """list_available_models should list Flash models."""
        list_available_models()
        captured = capsys.readouterr()
        assert "FLASH" in captured.out.upper()

    def test_models_lists_pro(self, capsys):
        """list_available_models should list Pro models."""
        list_available_models()
        captured = capsys.readouterr()
        assert "PRO" in captured.out.upper()

    def test_models_lists_thinking(self, capsys):
        """list_available_models should list Thinking models."""
        list_available_models()
        captured = capsys.readouterr()
        assert "THINKING" in captured.out.upper()

    def test_models_lists_presets(self, capsys):
        """list_available_models should list presets."""
        list_available_models()
        captured = capsys.readouterr()
        assert "Presets" in captured.out


class TestShowVersion:
    """Test show_version command."""

    def test_version_returns_zero(self):
        """show_version should return 0."""
        result = show_version()
        assert result == 0

    def test_version_prints_version(self, capsys):
        """show_version should print version number."""
        show_version()
        captured = capsys.readouterr()
        assert "gemini-deepsearch-mcp" in captured.out
        assert "version" in captured.out.lower()


class TestRunCliNoArgs:
    """Test run_cli with no arguments returns server signal."""

    def test_no_args_returns_minus_one(self):
        """run_cli with no args should return -1 (run server signal)."""
        result = run_cli([])
        assert result == -1

    def test_no_args_does_not_start_server(self):
        """run_cli with no args should not start server (just returns signal)."""
        # This test ensures run_cli just returns -1 without side effects
        result = run_cli([])
        assert result == -1


class TestRunCliWithCommands:
    """Test run_cli dispatches to correct commands."""

    def test_version_command(self, capsys):
        """run_cli --version should show version."""
        result = run_cli(["--version"])
        assert result == 0
        captured = capsys.readouterr()
        assert "gemini-deepsearch-mcp" in captured.out

    def test_models_command(self, capsys):
        """run_cli --models should list models."""
        result = run_cli(["--models"])
        assert result == 0
        captured = capsys.readouterr()
        assert "FLASH" in captured.out.upper() or "flash" in captured.out.lower()

    @patch("gemini_deepsearch_mcp.cli.load_config")
    @patch("gemini_deepsearch_mcp.cli.reset_config")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-12345678"}, clear=False)
    def test_status_command(self, mock_reset, mock_load_config, capsys):
        """run_cli --status should show status."""
        mock_config = MagicMock()
        mock_config.query_generator_model = "gemini-2.5-flash"
        mock_config.web_search_model = "gemini-2.5-flash"
        mock_config.reflection_model = "gemini-2.5-flash"
        mock_config.answer_model = "gemini-2.5-pro"
        mock_config.default_effort = "medium"
        mock_config.max_research_loops = 2
        mock_config.initial_query_count = 3
        mock_config.log_level = "INFO"
        mock_load_config.return_value = mock_config

        result = run_cli(["--status"])
        assert result == 0

    @patch("gemini_deepsearch_mcp.cli.load_config")
    @patch("google.genai.Client")
    def test_verify_command_success(self, mock_client_class, mock_load_config):
        """run_cli --verify should verify connection."""
        mock_config = MagicMock()
        mock_config.gemini_api_key = "test-api-key-12345678"
        mock_load_config.return_value = mock_config

        mock_client = MagicMock()
        mock_client.models.list.return_value = [MagicMock()]
        mock_client_class.return_value = mock_client

        result = run_cli(["--verify"])
        assert result == 0


class TestRunCliDemo:
    """Test run_cli demo command."""

    @patch("gemini_deepsearch_mcp.main.deep_search")
    def test_demo_success(self, mock_deep_search, tmp_path, capsys):
        """run_cli --demo should run demo search."""
        import json

        # Create a temp file with mock results
        result_file = tmp_path / "demo_result.json"
        result_data = {"answer": "Test answer", "sources": ["http://example.com"]}
        result_file.write_text(json.dumps(result_data))

        mock_deep_search.return_value = {"file_path": str(result_file)}

        result = run_cli(["--demo"])
        assert result == 0

    @patch("gemini_deepsearch_mcp.main.deep_search")
    def test_demo_failure(self, mock_deep_search, capsys):
        """run_cli --demo should handle errors gracefully."""
        mock_deep_search.return_value = {"error": "Test error"}

        result = run_cli(["--demo"])
        assert result == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.out


class TestApiKeyMasking:
    """Test that API key is masked in output."""

    @patch("gemini_deepsearch_mcp.cli.load_config")
    @patch("google.genai.Client")
    def test_verify_masks_api_key(self, mock_client_class, mock_load_config, capsys):
        """verify_connection should mask the API key."""
        mock_config = MagicMock()
        mock_config.gemini_api_key = "sk-1234567890abcdefghij"
        mock_load_config.return_value = mock_config

        mock_client = MagicMock()
        mock_client.models.list.return_value = [MagicMock()]
        mock_client_class.return_value = mock_client

        verify_connection()
        captured = capsys.readouterr()

        # Full key should not appear
        assert "sk-1234567890abcdefghij" not in captured.out
        # Masked version should appear
        assert "..." in captured.out

    @patch.dict(os.environ, {"GEMINI_API_KEY": "sk-abcdefghij1234567890"}, clear=False)
    @patch("gemini_deepsearch_mcp.cli.reset_config")
    @patch("gemini_deepsearch_mcp.cli.load_config")
    def test_status_masks_api_key(self, mock_load_config, mock_reset, capsys):
        """show_status should mask the API key."""
        mock_config = MagicMock()
        mock_config.query_generator_model = "gemini-2.5-flash"
        mock_config.web_search_model = "gemini-2.5-flash"
        mock_config.reflection_model = "gemini-2.5-flash"
        mock_config.answer_model = "gemini-2.5-pro"
        mock_config.default_effort = "medium"
        mock_config.max_research_loops = 2
        mock_config.initial_query_count = 3
        mock_config.log_level = "INFO"
        mock_load_config.return_value = mock_config

        show_status()
        captured = capsys.readouterr()

        # Full key should not appear
        assert "sk-abcdefghij1234567890" not in captured.out
        # Masked version should appear (first 8 chars + ... + last 4 chars)
        assert "..." in captured.out
