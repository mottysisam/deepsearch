"""
Data models and model registry for the DeepSearch MCP Server.

Contains Pydantic models for responses and a registry of available Gemini models.
"""

from typing import Literal

from pydantic import BaseModel, Field


class GeminiModel(BaseModel):
    """Metadata for an available Gemini model."""

    id: str = Field(..., description="Model identifier for API calls")
    name: str = Field(..., description="Human-readable model name")
    category: Literal["flash", "pro", "thinking"] = Field(
        ..., description="Model category"
    )
    description: str = Field(..., description="Model description")
    capabilities: list[str] = Field(
        default_factory=list, description="Model capabilities"
    )
    supports_grounding: bool = Field(
        default=False, description="Whether model supports Google Search grounding"
    )


# Registry of available Gemini models
AVAILABLE_MODELS: dict[str, GeminiModel] = {
    # Flash models - Fast and efficient
    "gemini-2.5-flash": GeminiModel(
        id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        category="flash",
        description="Fast, efficient model for most tasks",
        capabilities=["text-generation", "reasoning", "structured-output"],
        supports_grounding=True,
    ),
    "gemini-2.5-flash-preview-05-20": GeminiModel(
        id="gemini-2.5-flash-preview-05-20",
        name="Gemini 2.5 Flash Preview (05-20)",
        category="flash",
        description="Preview version of Flash with latest improvements",
        capabilities=["text-generation", "reasoning", "structured-output"],
        supports_grounding=True,
    ),
    "gemini-2.5-flash-lite-preview-06-17": GeminiModel(
        id="gemini-2.5-flash-lite-preview-06-17",
        name="Gemini 2.5 Flash Lite Preview",
        category="flash",
        description="Lightweight Flash variant optimized for speed",
        capabilities=["text-generation", "web-search"],
        supports_grounding=True,
    ),
    # Pro models - High capability
    "gemini-2.5-pro": GeminiModel(
        id="gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        category="pro",
        description="Most capable model for complex reasoning",
        capabilities=[
            "text-generation",
            "advanced-reasoning",
            "structured-output",
            "long-context",
        ],
        supports_grounding=True,
    ),
    "gemini-2.5-pro-preview-06-05": GeminiModel(
        id="gemini-2.5-pro-preview-06-05",
        name="Gemini 2.5 Pro Preview (06-05)",
        category="pro",
        description="Preview version of Pro with latest capabilities",
        capabilities=[
            "text-generation",
            "advanced-reasoning",
            "structured-output",
            "long-context",
        ],
        supports_grounding=True,
    ),
    # Thinking models - Extended reasoning
    "gemini-2.5-flash-thinking": GeminiModel(
        id="gemini-2.5-flash-thinking",
        name="Gemini 2.5 Flash Thinking",
        category="thinking",
        description="Flash model with extended thinking capabilities",
        capabilities=[
            "text-generation",
            "extended-reasoning",
            "structured-output",
            "chain-of-thought",
        ],
        supports_grounding=False,
    ),
}


# Model presets for easy selection
MODEL_PRESETS: dict[str, dict[str, str]] = {
    "flash": {
        "query_generator_model": "gemini-2.5-flash",
        "web_search_model": "gemini-2.5-flash-lite-preview-06-17",
        "reflection_model": "gemini-2.5-flash",
        "answer_model": "gemini-2.5-flash",
    },
    "pro": {
        "query_generator_model": "gemini-2.5-flash",
        "web_search_model": "gemini-2.5-flash-lite-preview-06-17",
        "reflection_model": "gemini-2.5-flash",
        "answer_model": "gemini-2.5-pro",
    },
    "thinking": {
        "query_generator_model": "gemini-2.5-flash-thinking",
        "web_search_model": "gemini-2.5-flash-lite-preview-06-17",
        "reflection_model": "gemini-2.5-flash-thinking",
        "answer_model": "gemini-2.5-pro",
    },
}


def get_model(model_id: str) -> GeminiModel | None:
    """Get model metadata by ID.

    Args:
        model_id: The model identifier.

    Returns:
        GeminiModel if found, None otherwise.
    """
    return AVAILABLE_MODELS.get(model_id)


def get_preset(preset_name: str) -> dict[str, str] | None:
    """Get model configuration for a preset.

    Args:
        preset_name: The preset name (flash, pro, thinking).

    Returns:
        Dict of model configurations if found, None otherwise.
    """
    return MODEL_PRESETS.get(preset_name)


def list_models(category: str | None = None) -> list[GeminiModel]:
    """List available models, optionally filtered by category.

    Args:
        category: Optional category filter (flash, pro, thinking).

    Returns:
        List of matching GeminiModel objects.
    """
    models = list(AVAILABLE_MODELS.values())
    if category:
        models = [m for m in models if m.category == category]
    return models


def is_valid_model(model_id: str) -> bool:
    """Check if a model ID is valid.

    Args:
        model_id: The model identifier to check.

    Returns:
        True if model exists in registry.
    """
    return model_id in AVAILABLE_MODELS
