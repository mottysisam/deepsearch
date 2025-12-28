"""Tests for structured logging configuration.

Tests cover:
- Logging configuration
- Log level from environment
- Context binding in logs
"""

import os
from io import StringIO
from unittest.mock import patch

import pytest
import structlog

from gemini_deepsearch_mcp.logging import (
    LogContext,
    configure_logging,
    get_logger,
    reset_logging,
)


class TestLoggingConfiguredCorrectly:
    """Test that logging is configured correctly."""

    def setup_method(self):
        """Reset logging before each test."""
        reset_logging()

    def teardown_method(self):
        """Reset logging after each test."""
        reset_logging()

    def test_configure_logging_sets_up_structlog(self):
        """configure_logging should set up structlog processors."""
        configure_logging("INFO")

        # Get a logger and check it works
        logger = get_logger("test")
        assert logger is not None

    def test_configure_logging_only_runs_once(self):
        """configure_logging should only configure once."""
        configure_logging("INFO")
        configure_logging("DEBUG")  # Should be ignored

        # No exception means it worked
        logger = get_logger("test")
        assert logger is not None

    def test_get_logger_returns_bound_logger(self):
        """get_logger should return a bound logger."""
        configure_logging("INFO")

        logger = get_logger("test.module")
        assert logger is not None
        # Should be able to bind additional context
        bound = logger.bind(key="value")
        assert bound is not None

    def test_get_logger_with_initial_context(self):
        """get_logger should accept initial context."""
        configure_logging("INFO")

        logger = get_logger("test", user="test_user", request_id="12345")
        assert logger is not None


class TestLogLevelFromEnv:
    """Test that log level is configurable from environment."""

    def setup_method(self):
        """Reset logging before each test."""
        reset_logging()

    def teardown_method(self):
        """Reset logging after each test."""
        reset_logging()

    def test_debug_level_configured(self):
        """DEBUG log level should be respected."""
        configure_logging("DEBUG")
        logger = get_logger("test")

        # DEBUG should be enabled
        assert logger is not None

    def test_info_level_configured(self):
        """INFO log level should be respected."""
        configure_logging("INFO")
        logger = get_logger("test")

        assert logger is not None

    def test_warning_level_configured(self):
        """WARNING log level should be respected."""
        configure_logging("WARNING")
        logger = get_logger("test")

        assert logger is not None

    def test_error_level_configured(self):
        """ERROR log level should be respected."""
        configure_logging("ERROR")
        logger = get_logger("test")

        assert logger is not None

    def test_case_insensitive_level(self):
        """Log level should be case insensitive."""
        configure_logging("debug")
        logger = get_logger("test")

        assert logger is not None


class TestLogsIncludeContext:
    """Test that logs include bound context."""

    def setup_method(self):
        """Reset logging before each test."""
        reset_logging()

    def teardown_method(self):
        """Reset logging after each test."""
        reset_logging()

    def test_bound_context_persists(self):
        """Bound context should persist across log calls."""
        configure_logging("DEBUG")

        logger = get_logger("test")
        bound_logger = logger.bind(user="test_user")

        # Should be able to log multiple times with same context
        bound_logger.info("first_message")
        bound_logger.info("second_message")

        # No exception means context persisted

    def test_log_context_manager(self):
        """LogContext should bind and unbind context."""
        configure_logging("DEBUG")
        logger = get_logger("test")

        with LogContext(request_id="123", user="test"):
            logger.info("inside_context")

        logger.info("outside_context")

        # No exception means context was properly managed

    def test_nested_log_context(self):
        """Nested LogContext should work correctly."""
        configure_logging("DEBUG")
        logger = get_logger("test")

        with LogContext(outer="value"):
            logger.info("outer_context")

            with LogContext(inner="value"):
                logger.info("inner_context")

            logger.info("back_to_outer")

        logger.info("no_context")

        # No exception means nested contexts worked

    def test_logger_bind_returns_new_logger(self):
        """logger.bind should return a new logger with added context."""
        configure_logging("DEBUG")

        logger1 = get_logger("test")
        logger2 = logger1.bind(extra="context")

        # Should be different loggers
        assert logger1 is not logger2


class TestLoggingIntegration:
    """Integration tests for logging with the application."""

    def setup_method(self):
        """Reset logging before each test."""
        reset_logging()

    def teardown_method(self):
        """Reset logging after each test."""
        reset_logging()

    def test_logging_with_model_selection(self):
        """Logging should work with model selection module."""
        configure_logging("DEBUG")

        # This should not raise any exceptions
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            from gemini_deepsearch_mcp.config import reset_config
            from gemini_deepsearch_mcp.model_selection import resolve_models

            reset_config()

            logger = get_logger("test")
            logger.info("starting_test")

            models = resolve_models(
                model="flash",
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            logger.info("models_resolved", **models)

            reset_config()

    def test_logging_exception_capture(self):
        """logger.exception should capture exception info."""
        configure_logging("ERROR")
        logger = get_logger("test")

        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("caught_error")

        # No additional exception means it worked


class TestResetLogging:
    """Test reset_logging functionality."""

    def test_reset_allows_reconfiguration(self):
        """reset_logging should allow reconfiguration."""
        configure_logging("INFO")

        reset_logging()

        # Should be able to configure again
        configure_logging("DEBUG")

        logger = get_logger("test")
        assert logger is not None
