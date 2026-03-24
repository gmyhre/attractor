"""Model catalog - known models across providers."""
from .types import ModelInfo

MODELS: list[ModelInfo] = [
    # Anthropic
    ModelInfo(
        id="claude-opus-4-6",
        provider="anthropic",
        display_name="Claude Opus 4.6",
        context_window=200000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["opus", "claude-opus"],
    ),
    ModelInfo(
        id="claude-sonnet-4-6",
        provider="anthropic",
        display_name="Claude Sonnet 4.6",
        context_window=200000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["sonnet", "claude-sonnet"],
    ),
    ModelInfo(
        id="claude-sonnet-4-5",
        provider="anthropic",
        display_name="Claude Sonnet 4.5",
        context_window=200000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
    ModelInfo(
        id="claude-haiku-4-5-20251001",
        provider="anthropic",
        display_name="Claude Haiku 4.5",
        context_window=200000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=False,
        aliases=["haiku"],
    ),
    # OpenAI
    ModelInfo(
        id="gpt-5.2",
        provider="openai",
        display_name="GPT-5.2",
        context_window=1047576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
    ModelInfo(
        id="gpt-5.2-mini",
        provider="openai",
        display_name="GPT-5.2 Mini",
        context_window=1047576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
    ModelInfo(
        id="gpt-5.2-codex",
        provider="openai",
        display_name="GPT-5.2 Codex",
        context_window=1047576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["codex"],
    ),
    ModelInfo(
        id="gpt-4o",
        provider="openai",
        display_name="GPT-4o",
        context_window=128000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=False,
    ),
    # Gemini
    ModelInfo(
        id="gemini-3-pro-preview",
        provider="gemini",
        display_name="Gemini 3 Pro (Preview)",
        context_window=1048576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
    ModelInfo(
        id="gemini-3-flash-preview",
        provider="gemini",
        display_name="Gemini 3 Flash (Preview)",
        context_window=1048576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gemini-flash"],
    ),
    ModelInfo(
        id="gemini-2.5-pro",
        provider="gemini",
        display_name="Gemini 2.5 Pro",
        context_window=1048576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
    ModelInfo(
        id="gemini-2.5-flash",
        provider="gemini",
        display_name="Gemini 2.5 Flash",
        context_window=1048576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
]

_by_id: dict[str, ModelInfo] = {m.id: m for m in MODELS}
_by_alias: dict[str, ModelInfo] = {}
for _m in MODELS:
    for _alias in _m.aliases:
        _by_alias[_alias] = _m


def get_model_info(model_id: str) -> ModelInfo | None:
    return _by_id.get(model_id) or _by_alias.get(model_id)


def list_models(provider: str | None = None) -> list[ModelInfo]:
    if provider:
        return [m for m in MODELS if m.provider == provider]
    return list(MODELS)


def get_latest_model(provider: str, capability: str | None = None) -> ModelInfo | None:
    candidates = [m for m in MODELS if m.provider == provider]
    if capability == "reasoning":
        candidates = [m for m in candidates if m.supports_reasoning]
    elif capability == "vision":
        candidates = [m for m in candidates if m.supports_vision]
    elif capability == "tools":
        candidates = [m for m in candidates if m.supports_tools]
    return candidates[0] if candidates else None


def infer_provider(model_id: str) -> str | None:
    """Infer provider from model ID string."""
    info = get_model_info(model_id)
    if info:
        return info.provider
    model_lower = model_id.lower()
    if "claude" in model_lower:
        return "anthropic"
    if "gpt" in model_lower or "o1" in model_lower or "codex" in model_lower:
        return "openai"
    if "gemini" in model_lower or "bard" in model_lower:
        return "gemini"
    return None
