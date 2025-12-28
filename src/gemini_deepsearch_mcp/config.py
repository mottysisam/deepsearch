"""
Configuration management for the DeepSearch MCP Server.

Supports loading from environment variables and .env files using pydantic-settings.
"""

import os
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Settings
    gemini_api_key: str = Field(
        ...,
        alias="GEMINI_API_KEY",
        description="Google Gemini API key",
    )

    # Model Defaults (configurable per stage)
    query_generator_model: str = Field(
        default="gemini-2.5-flash",
        alias="QUERY_GENERATOR_MODEL",
        description="Model for generating search queries",
    )
    web_search_model: str = Field(
        default="gemini-2.5-flash-lite-preview-06-17",
        alias="WEB_SEARCH_MODEL",
        description="Model for web search with grounding",
    )
    reflection_model: str = Field(
        default="gemini-2.5-flash",
        alias="REFLECTION_MODEL",
        description="Model for reflection and gap analysis",
    )
    answer_model: str = Field(
        default="gemini-2.5-pro",
        alias="ANSWER_MODEL",
        description="Model for final answer generation",
    )

    # Research Defaults
    default_effort: str = Field(
        default="medium",
        alias="DEFAULT_EFFORT",
        description="Default research effort level (low, medium, high)",
    )
    max_research_loops: int = Field(
        default=2,
        alias="MAX_RESEARCH_LOOPS",
        description="Maximum number of research iterations",
    )
    initial_query_count: int = Field(
        default=3,
        alias="INITIAL_QUERY_COUNT",
        description="Number of initial search queries to generate",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    @field_validator("default_effort")
    @classmethod
    def validate_effort(cls, v: str) -> str:
        """Ensure valid effort level."""
        valid_levels = {"low", "medium", "high"}
        v = v.lower()
        if v not in valid_levels:
            raise ValueError(f"Effort must be one of: {valid_levels}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure valid log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v

    @field_validator("max_research_loops")
    @classmethod
    def validate_max_loops(cls, v: int) -> int:
        """Ensure reasonable loop count."""
        if v < 1:
            raise ValueError("max_research_loops must be at least 1")
        if v > 10:
            raise ValueError("max_research_loops cannot exceed 10")
        return v

    @field_validator("initial_query_count")
    @classmethod
    def validate_query_count(cls, v: int) -> int:
        """Ensure reasonable query count."""
        if v < 1:
            raise ValueError("initial_query_count must be at least 1")
        if v > 10:
            raise ValueError("initial_query_count cannot exceed 10")
        return v


# Global config instance (lazy loaded)
_config: Config | None = None


def load_config() -> Config:
    """Load configuration from environment variables.

    Returns:
        Config object with all settings.

    Raises:
        ValueError: If required settings are missing or invalid.
    """
    global _config
    if _config is None:
        _config = Config()  # type: ignore[call-arg]
    return _config


def reset_config() -> None:
    """Reset the global config instance (useful for testing)."""
    global _config
    _config = None


def get_env_or_raise(name: str) -> str:
    """Get environment variable or raise error.

    Args:
        name: Environment variable name.

    Returns:
        Environment variable value.

    Raises:
        ValueError: If variable is not set.
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable '{name}' is not set")
    return value
