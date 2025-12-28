"""Tests for configuration management and model registry.

Tests cover:
- Config loading from environment variables
- Config defaults
- Config validation errors
- Model registry completeness
"""

import os
from unittest.mock import patch

import pytest

from gemini_deepsearch_mcp.config import Config, load_config, reset_config
from gemini_deepsearch_mcp.models import (
    AVAILABLE_MODELS,
    MODEL_PRESETS,
    get_model,
    get_preset,
    is_valid_model,
    list_models,
)


class TestConfigLoadsFromEnv:
    """Test that configuration loads correctly from environment variables."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_config_loads_api_key_from_env(self):
        """Config should load GEMINI_API_KEY from environment."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key-12345"}):
            config = Config()
            assert config.gemini_api_key == "test-api-key-12345"

    def test_config_loads_model_settings_from_env(self):
        """Config should load model configuration from environment."""
        env_vars = {
            "GEMINI_API_KEY": "test-key",
            "QUERY_GENERATOR_MODEL": "gemini-2.5-pro",
            "WEB_SEARCH_MODEL": "gemini-2.5-flash",
            "REFLECTION_MODEL": "gemini-2.5-pro",
            "ANSWER_MODEL": "gemini-2.5-flash-thinking",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = Config()
            assert config.query_generator_model == "gemini-2.5-pro"
            assert config.web_search_model == "gemini-2.5-flash"
            assert config.reflection_model == "gemini-2.5-pro"
            assert config.answer_model == "gemini-2.5-flash-thinking"

    def test_config_loads_research_settings_from_env(self):
        """Config should load research configuration from environment."""
        env_vars = {
            "GEMINI_API_KEY": "test-key",
            "DEFAULT_EFFORT": "high",
            "MAX_RESEARCH_LOOPS": "5",
            "INITIAL_QUERY_COUNT": "7",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = Config()
            assert config.default_effort == "high"
            assert config.max_research_loops == 5
            assert config.initial_query_count == 7

    def test_config_loads_log_level_from_env(self):
        """Config should load log level from environment."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key", "LOG_LEVEL": "DEBUG"}
        ):
            config = Config()
            assert config.log_level == "DEBUG"


class TestConfigDefaults:
    """Test that configuration has correct default values."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_default_model_settings(self):
        """Config should have correct default model settings."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            config = Config()
            assert config.query_generator_model == "gemini-2.5-flash"
            assert config.web_search_model == "gemini-2.5-flash-lite-preview-06-17"
            assert config.reflection_model == "gemini-2.5-flash"
            assert config.answer_model == "gemini-2.5-pro"

    def test_default_research_settings(self):
        """Config should have correct default research settings."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            config = Config()
            assert config.default_effort == "medium"
            assert config.max_research_loops == 2
            assert config.initial_query_count == 3

    def test_default_log_level(self):
        """Config should default to INFO log level."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            config = Config()
            assert config.log_level == "INFO"


class TestConfigValidationErrors:
    """Test that configuration validation catches invalid values."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_missing_api_key_raises_error(self):
        """Config should raise error when GEMINI_API_KEY is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception):  # ValidationError
                Config()

    def test_invalid_effort_level_raises_error(self):
        """Config should raise error for invalid effort level."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key", "DEFAULT_EFFORT": "extreme"}
        ):
            with pytest.raises(ValueError, match="Effort must be one of"):
                Config()

    def test_invalid_log_level_raises_error(self):
        """Config should raise error for invalid log level."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key", "LOG_LEVEL": "VERBOSE"}
        ):
            with pytest.raises(ValueError, match="Log level must be one of"):
                Config()

    def test_max_research_loops_too_low_raises_error(self):
        """Config should raise error when max_research_loops < 1."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key", "MAX_RESEARCH_LOOPS": "0"}
        ):
            with pytest.raises(ValueError, match="must be at least 1"):
                Config()

    def test_max_research_loops_too_high_raises_error(self):
        """Config should raise error when max_research_loops > 10."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key", "MAX_RESEARCH_LOOPS": "15"}
        ):
            with pytest.raises(ValueError, match="cannot exceed 10"):
                Config()

    def test_initial_query_count_too_low_raises_error(self):
        """Config should raise error when initial_query_count < 1."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key", "INITIAL_QUERY_COUNT": "0"}
        ):
            with pytest.raises(ValueError, match="must be at least 1"):
                Config()

    def test_initial_query_count_too_high_raises_error(self):
        """Config should raise error when initial_query_count > 10."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key", "INITIAL_QUERY_COUNT": "20"}
        ):
            with pytest.raises(ValueError, match="cannot exceed 10"):
                Config()

    def test_effort_level_case_insensitive(self):
        """Config should accept effort levels in any case."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key", "DEFAULT_EFFORT": "HIGH"}
        ):
            config = Config()
            assert config.default_effort == "high"

    def test_log_level_case_insensitive(self):
        """Config should accept log levels in any case."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key", "LOG_LEVEL": "debug"}
        ):
            config = Config()
            assert config.log_level == "DEBUG"


class TestModelRegistryContainsAllModels:
    """Test that model registry contains all expected models."""

    def test_registry_contains_flash_models(self):
        """Registry should contain all Flash model variants."""
        flash_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-preview-05-20",
            "gemini-2.5-flash-lite-preview-06-17",
        ]
        for model_id in flash_models:
            assert model_id in AVAILABLE_MODELS, f"Missing flash model: {model_id}"
            assert AVAILABLE_MODELS[model_id].category == "flash"

    def test_registry_contains_pro_models(self):
        """Registry should contain all Pro model variants."""
        pro_models = ["gemini-2.5-pro", "gemini-2.5-pro-preview-06-05"]
        for model_id in pro_models:
            assert model_id in AVAILABLE_MODELS, f"Missing pro model: {model_id}"
            assert AVAILABLE_MODELS[model_id].category == "pro"

    def test_registry_contains_thinking_models(self):
        """Registry should contain all Thinking model variants."""
        thinking_models = ["gemini-2.5-flash-thinking"]
        for model_id in thinking_models:
            assert model_id in AVAILABLE_MODELS, f"Missing thinking model: {model_id}"
            assert AVAILABLE_MODELS[model_id].category == "thinking"

    def test_all_models_have_required_fields(self):
        """All models should have id, name, category, and description."""
        for model_id, model in AVAILABLE_MODELS.items():
            assert model.id == model_id
            assert model.name, f"Model {model_id} missing name"
            assert model.category in ["flash", "pro", "thinking"]
            assert model.description, f"Model {model_id} missing description"

    def test_grounding_support_correctly_set(self):
        """Models should have correct grounding support flags."""
        # Thinking models don't support grounding
        thinking_model = AVAILABLE_MODELS["gemini-2.5-flash-thinking"]
        assert thinking_model.supports_grounding is False

        # Flash and Pro models support grounding
        assert AVAILABLE_MODELS["gemini-2.5-flash"].supports_grounding is True
        assert AVAILABLE_MODELS["gemini-2.5-pro"].supports_grounding is True


class TestModelPresets:
    """Test model preset configurations."""

    def test_flash_preset_exists(self):
        """Flash preset should exist with correct model mappings."""
        preset = get_preset("flash")
        assert preset is not None
        assert preset["query_generator_model"] == "gemini-2.5-flash"
        assert preset["answer_model"] == "gemini-2.5-flash"

    def test_pro_preset_exists(self):
        """Pro preset should exist with correct model mappings."""
        preset = get_preset("pro")
        assert preset is not None
        assert preset["answer_model"] == "gemini-2.5-pro"

    def test_thinking_preset_exists(self):
        """Thinking preset should exist with correct model mappings."""
        preset = get_preset("thinking")
        assert preset is not None
        assert preset["query_generator_model"] == "gemini-2.5-flash-thinking"
        assert preset["reflection_model"] == "gemini-2.5-flash-thinking"

    def test_invalid_preset_returns_none(self):
        """Invalid preset name should return None."""
        assert get_preset("invalid") is None


class TestModelRegistryHelpers:
    """Test model registry helper functions."""

    def test_get_model_returns_model(self):
        """get_model should return model for valid ID."""
        model = get_model("gemini-2.5-flash")
        assert model is not None
        assert model.id == "gemini-2.5-flash"

    def test_get_model_returns_none_for_invalid(self):
        """get_model should return None for invalid ID."""
        assert get_model("invalid-model") is None

    def test_is_valid_model(self):
        """is_valid_model should return correct boolean."""
        assert is_valid_model("gemini-2.5-flash") is True
        assert is_valid_model("gemini-2.5-pro") is True
        assert is_valid_model("invalid-model") is False

    def test_list_models_returns_all(self):
        """list_models should return all models when no filter."""
        models = list_models()
        assert len(models) == len(AVAILABLE_MODELS)

    def test_list_models_filters_by_category(self):
        """list_models should filter by category."""
        flash_models = list_models(category="flash")
        assert all(m.category == "flash" for m in flash_models)
        assert len(flash_models) == 3  # 3 flash variants

        pro_models = list_models(category="pro")
        assert all(m.category == "pro" for m in pro_models)
        assert len(pro_models) == 2  # 2 pro variants

        thinking_models = list_models(category="thinking")
        assert all(m.category == "thinking" for m in thinking_models)
        assert len(thinking_models) == 1  # 1 thinking variant
