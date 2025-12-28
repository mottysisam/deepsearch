"""Tests for model selection in deep_search tool.

Tests cover:
- Preset model selection (flash, pro, thinking)
- Individual model overrides
- Invalid model error handling
- Default model usage
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from gemini_deepsearch_mcp.config import reset_config
from gemini_deepsearch_mcp.model_selection import resolve_models
from gemini_deepsearch_mcp.models import MODEL_PRESETS


class TestPresetFlashUsesCorrectModels:
    """Test that flash preset uses correct model configuration."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_flash_preset_applies_flash_models(self):
        """Flash preset should use flash models for all stages."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="flash",
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            expected = MODEL_PRESETS["flash"]
            assert models["query_generator_model"] == expected["query_generator_model"]
            assert models["web_search_model"] == expected["web_search_model"]
            assert models["reflection_model"] == expected["reflection_model"]
            assert models["answer_model"] == expected["answer_model"]

    def test_flash_preset_uses_flash_for_answer(self):
        """Flash preset should use gemini-2.5-flash for answer generation."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="flash",
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            assert models["answer_model"] == "gemini-2.5-flash"


class TestPresetProUsesCorrectModels:
    """Test that pro preset uses correct model configuration."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_pro_preset_applies_pro_models(self):
        """Pro preset should use pro model for answer generation."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="pro",
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            expected = MODEL_PRESETS["pro"]
            assert models["query_generator_model"] == expected["query_generator_model"]
            assert models["answer_model"] == expected["answer_model"]

    def test_pro_preset_uses_pro_for_answer(self):
        """Pro preset should use gemini-2.5-pro for answer generation."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="pro",
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            assert models["answer_model"] == "gemini-2.5-pro"


class TestPresetThinkingUsesCorrectModels:
    """Test that thinking preset uses correct model configuration."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_thinking_preset_applies_thinking_models(self):
        """Thinking preset should use thinking models for query and reflection."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="thinking",
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            expected = MODEL_PRESETS["thinking"]
            assert models["query_generator_model"] == expected["query_generator_model"]
            assert models["reflection_model"] == expected["reflection_model"]

    def test_thinking_preset_uses_thinking_for_query_and_reflection(self):
        """Thinking preset should use flash-thinking for query and reflection."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="thinking",
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            assert models["query_generator_model"] == "gemini-2.5-flash-thinking"
            assert models["reflection_model"] == "gemini-2.5-flash-thinking"


class TestIndividualModelOverride:
    """Test that individual model parameters override presets and defaults."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_individual_query_model_override(self):
        """Individual query_model should override preset."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="flash",
                query_model="gemini-2.5-pro",
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            assert models["query_generator_model"] == "gemini-2.5-pro"
            # Other models should still use flash preset
            assert models["answer_model"] == "gemini-2.5-flash"

    def test_individual_answer_model_override(self):
        """Individual answer_model should override preset."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="flash",
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model="gemini-2.5-pro",
            )

            assert models["answer_model"] == "gemini-2.5-pro"

    def test_individual_search_model_override(self):
        """Individual search_model should override preset."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="flash",
                query_model=None,
                search_model="gemini-2.5-flash",
                reflection_model=None,
                answer_model=None,
            )

            assert models["web_search_model"] == "gemini-2.5-flash"

    def test_individual_reflection_model_override(self):
        """Individual reflection_model should override preset."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="flash",
                query_model=None,
                search_model=None,
                reflection_model="gemini-2.5-pro",
                answer_model=None,
            )

            assert models["reflection_model"] == "gemini-2.5-pro"

    def test_multiple_individual_overrides(self):
        """Multiple individual overrides should all apply."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="flash",
                query_model="gemini-2.5-pro",
                search_model="gemini-2.5-flash",
                reflection_model="gemini-2.5-flash-thinking",
                answer_model="gemini-2.5-pro",
            )

            assert models["query_generator_model"] == "gemini-2.5-pro"
            assert models["web_search_model"] == "gemini-2.5-flash"
            assert models["reflection_model"] == "gemini-2.5-flash-thinking"
            assert models["answer_model"] == "gemini-2.5-pro"

    def test_individual_overrides_without_preset(self):
        """Individual overrides should work without a preset."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model=None,
                query_model="gemini-2.5-pro",
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            assert models["query_generator_model"] == "gemini-2.5-pro"


class TestInvalidModelReturnsError:
    """Test that invalid model names return helpful errors."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_invalid_query_model_raises_error(self):
        """Invalid query_model should raise ValueError."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with pytest.raises(ValueError) as exc_info:
                resolve_models(
                    model=None,
                    query_model="invalid-model",
                    search_model=None,
                    reflection_model=None,
                    answer_model=None,
                )

            assert "Invalid model 'invalid-model'" in str(exc_info.value)
            assert "query_generator_model" in str(exc_info.value)

    def test_invalid_search_model_raises_error(self):
        """Invalid search_model should raise ValueError."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with pytest.raises(ValueError) as exc_info:
                resolve_models(
                    model=None,
                    query_model=None,
                    search_model="nonexistent-model",
                    reflection_model=None,
                    answer_model=None,
                )

            assert "Invalid model 'nonexistent-model'" in str(exc_info.value)

    def test_invalid_reflection_model_raises_error(self):
        """Invalid reflection_model should raise ValueError."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with pytest.raises(ValueError) as exc_info:
                resolve_models(
                    model=None,
                    query_model=None,
                    search_model=None,
                    reflection_model="bad-model",
                    answer_model=None,
                )

            assert "Invalid model 'bad-model'" in str(exc_info.value)

    def test_invalid_answer_model_raises_error(self):
        """Invalid answer_model should raise ValueError."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with pytest.raises(ValueError) as exc_info:
                resolve_models(
                    model=None,
                    query_model=None,
                    search_model=None,
                    reflection_model=None,
                    answer_model="fake-model",
                )

            assert "Invalid model 'fake-model'" in str(exc_info.value)


class TestDefaultsUsedWhenNoParams:
    """Test that config defaults are used when no parameters provided."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_defaults_from_config(self):
        """When no params provided, should use config defaults."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model=None,
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            # These should match the config defaults
            assert models["query_generator_model"] == "gemini-2.5-flash"
            assert models["web_search_model"] == "gemini-2.5-flash-lite-preview-06-17"
            assert models["reflection_model"] == "gemini-2.5-flash"
            assert models["answer_model"] == "gemini-2.5-pro"

    def test_custom_defaults_from_env(self):
        """Config defaults from environment should be used."""
        env_vars = {
            "GEMINI_API_KEY": "test-key",
            "QUERY_GENERATOR_MODEL": "gemini-2.5-pro",
            "WEB_SEARCH_MODEL": "gemini-2.5-flash",
            "REFLECTION_MODEL": "gemini-2.5-pro",
            "ANSWER_MODEL": "gemini-2.5-flash-thinking",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            models = resolve_models(
                model=None,
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            assert models["query_generator_model"] == "gemini-2.5-pro"
            assert models["web_search_model"] == "gemini-2.5-flash"
            assert models["reflection_model"] == "gemini-2.5-pro"
            assert models["answer_model"] == "gemini-2.5-flash-thinking"


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY required for integration tests"
)
class TestDeepSearchToolModelSelection:
    """Integration tests for deep_search tool with model selection.

    These tests require GEMINI_API_KEY to be set because they import
    from main.py which loads the graph module.
    """

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_deep_search_with_flash_preset(self):
        """deep_search should work with flash preset."""
        from gemini_deepsearch_mcp.main import deep_search

        # Mock the graph.invoke to avoid actual API calls
        with patch("gemini_deepsearch_mcp.main.graph") as mock_graph:
            mock_graph.invoke.return_value = {
                "messages": [MagicMock(content="Test answer")],
                "sources_gathered": ["http://example.com"],
            }

            result = deep_search(
                query="test query",
                effort="low",
                model="flash",
            )

            # Should return file path, not error
            assert "file_path" in result or "error" not in result
            # Verify graph was called with flash models
            call_args = mock_graph.invoke.call_args
            config = call_args[0][1]
            assert config["configurable"]["answer_model"] == "gemini-2.5-flash"

    def test_deep_search_with_invalid_model_returns_error(self):
        """deep_search should return error for invalid model."""
        from gemini_deepsearch_mcp.main import deep_search

        result = deep_search(
            query="test query",
            effort="low",
            model=None,
            answer_model="invalid-model-id",
        )

        assert "error" in result
        assert "Invalid model" in result["error"]

    def test_deep_search_with_individual_override(self):
        """deep_search should use individual model override."""
        from gemini_deepsearch_mcp.main import deep_search

        with patch("gemini_deepsearch_mcp.main.graph") as mock_graph:
            mock_graph.invoke.return_value = {
                "messages": [MagicMock(content="Test answer")],
                "sources_gathered": [],
            }

            result = deep_search(
                query="test query",
                effort="low",
                model="flash",
                answer_model="gemini-2.5-pro",  # Override flash preset
            )

            # Should not have error
            assert "error" not in result
            # Verify the override was applied
            call_args = mock_graph.invoke.call_args
            config = call_args[0][1]
            assert config["configurable"]["answer_model"] == "gemini-2.5-pro"


class TestDeepResearchPreset:
    """Test cases for deep-research preset using Interactions API."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_deep_research_preset_returns_interactions_api_flag(self):
        """Deep research preset should return use_interactions_api flag."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="deep-research",
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            assert models["use_interactions_api"] is True
            assert models["agent_model"] == "deep-research-pro-preview-12-2025"

    def test_deep_research_preset_does_not_include_standard_models(self):
        """Deep research preset should not include standard model keys."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="deep-research",
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            # Should NOT have LangGraph model keys
            assert "query_generator_model" not in models
            assert "web_search_model" not in models
            assert "reflection_model" not in models
            assert "answer_model" not in models

    def test_is_deep_research_mode_returns_true_for_deep_research(self):
        """is_deep_research_mode should return True for deep-research preset."""
        from gemini_deepsearch_mcp.model_selection import is_deep_research_mode

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="deep-research",
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            assert is_deep_research_mode(models) is True

    def test_is_deep_research_mode_returns_false_for_standard_presets(self):
        """is_deep_research_mode should return False for standard presets."""
        from gemini_deepsearch_mcp.model_selection import is_deep_research_mode

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            for preset in ["flash", "pro", "thinking"]:
                models = resolve_models(
                    model=preset,
                    query_model=None,
                    search_model=None,
                    reflection_model=None,
                    answer_model=None,
                )

                assert is_deep_research_mode(models) is False

    def test_is_deep_research_mode_returns_false_for_no_preset(self):
        """is_deep_research_mode should return False when no preset specified."""
        from gemini_deepsearch_mcp.model_selection import is_deep_research_mode

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model=None,
                query_model=None,
                search_model=None,
                reflection_model=None,
                answer_model=None,
            )

            assert is_deep_research_mode(models) is False

    def test_deep_research_ignores_individual_model_overrides(self):
        """Deep research preset should ignore individual model overrides.

        When deep-research is selected, it uses the Interactions API which
        handles everything internally, so individual model overrides don't apply.
        """
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            models = resolve_models(
                model="deep-research",
                query_model="gemini-2.5-pro",  # Should be ignored
                search_model="gemini-2.5-flash",  # Should be ignored
                reflection_model="gemini-2.5-pro",  # Should be ignored
                answer_model="gemini-2.5-pro",  # Should be ignored
            )

            # Should still return Interactions API config
            assert models["use_interactions_api"] is True
            assert models["agent_model"] == "deep-research-pro-preview-12-2025"
            # Individual overrides should not be present
            assert "query_generator_model" not in models
