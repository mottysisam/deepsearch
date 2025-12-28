"""Structured logging configuration for DeepSearch MCP.

Uses structlog for consistent, queryable logs with context binding.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import FilteringBoundLogger

# Global configured flag
_configured = False


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog with console-friendly output.

    Args:
        log_level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    global _configured

    if _configured:
        return

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str | None = None, **initial_context: Any) -> FilteringBoundLogger:
    """Get a structlog logger with optional initial context.

    Args:
        name: Logger name (module name typically).
        **initial_context: Initial context key-value pairs to bind.

    Returns:
        Configured structlog logger with bound context.
    """
    # Ensure logging is configured
    if not _configured:
        configure_logging()

    logger = structlog.get_logger(name)

    if initial_context:
        logger = logger.bind(**initial_context)

    return logger


def reset_logging() -> None:
    """Reset logging configuration (useful for testing)."""
    global _configured
    _configured = False
    structlog.reset_defaults()


class LogContext:
    """Context manager for temporarily binding log context.

    Example:
        with LogContext(query="test", effort="high"):
            logger.info("Processing")
    """

    def __init__(self, **context: Any) -> None:
        """Initialize with context to bind.

        Args:
            **context: Key-value pairs to bind to log context.
        """
        self.context = context
        self._token: Any = None

    def __enter__(self) -> "LogContext":
        """Enter context and bind variables."""
        structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and unbind variables."""
        structlog.contextvars.unbind_contextvars(*self.context.keys())
