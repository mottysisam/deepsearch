"""Model selection utilities for deep_search tool.

Handles model resolution with priority: individual params > preset > config defaults.
"""

from .config import load_config
from .models import MODEL_PRESETS, is_valid_model


def resolve_models(
    model: str | None,
    query_model: str | None,
    search_model: str | None,
    reflection_model: str | None,
    answer_model: str | None,
) -> dict[str, str]:
    """Resolve model selections with priority: individual > preset > config defaults.

    Args:
        model: Model preset (flash, pro, thinking) or None.
        query_model: Specific model for query generation or None.
        search_model: Specific model for web search or None.
        reflection_model: Specific model for reflection or None.
        answer_model: Specific model for answer generation or None.

    Returns:
        Dict with resolved model IDs for each stage.

    Raises:
        ValueError: If an invalid model ID is provided.
    """
    config = load_config()

    # Start with config defaults
    resolved = {
        "query_generator_model": config.query_generator_model,
        "web_search_model": config.web_search_model,
        "reflection_model": config.reflection_model,
        "answer_model": config.answer_model,
    }

    # Apply preset if specified
    if model and model in MODEL_PRESETS:
        preset = MODEL_PRESETS[model]
        resolved.update(preset)

    # Apply individual overrides (highest priority)
    overrides = {
        "query_generator_model": query_model,
        "web_search_model": search_model,
        "reflection_model": reflection_model,
        "answer_model": answer_model,
    }

    for key, value in overrides.items():
        if value is not None:
            if not is_valid_model(value):
                raise ValueError(
                    f"Invalid model '{value}' for {key}. "
                    f"Use a valid model ID from the registry."
                )
            resolved[key] = value

    return resolved
