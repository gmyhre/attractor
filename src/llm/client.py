"""Unified LLM Client - Layer 3: Core Client."""
from __future__ import annotations
import os
from typing import AsyncIterator, Callable, Any

from .types import Request, Response, StreamEvent, ConfigurationError
from .catalog import infer_provider


class Client:
    """Provider-agnostic LLM client."""

    def __init__(
        self,
        providers: dict | None = None,
        default_provider: str | None = None,
        middleware: list[Callable] | None = None,
    ):
        self._providers: dict = providers or {}
        self._default_provider = default_provider
        self._middleware: list[Callable] = middleware or []

    @classmethod
    def from_env(cls) -> "Client":
        """Initialize from environment variables."""
        providers = {}
        default_provider = None

        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            from .providers.anthropic import AnthropicAdapter
            providers["anthropic"] = AnthropicAdapter(
                api_key=anthropic_key,
                base_url=os.getenv("ANTHROPIC_BASE_URL"),
            )
            if default_provider is None:
                default_provider = "anthropic"

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            from .providers.openai import OpenAIAdapter
            providers["openai"] = OpenAIAdapter(
                api_key=openai_key,
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
            if default_provider is None:
                default_provider = "openai"

        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            from .providers.gemini import GeminiAdapter
            providers["gemini"] = GeminiAdapter(api_key=gemini_key)
            if default_provider is None:
                default_provider = "gemini"

        return cls(providers=providers, default_provider=default_provider)

    def register(self, name: str, adapter) -> None:
        self._providers[name] = adapter
        if self._default_provider is None:
            self._default_provider = name

    def _resolve_provider(self, request: Request) -> tuple[str, Any]:
        provider_name = request.provider
        if not provider_name:
            provider_name = infer_provider(request.model)
        if not provider_name:
            provider_name = self._default_provider
        if not provider_name:
            raise ConfigurationError("No provider specified and no default configured")
        adapter = self._providers.get(provider_name)
        if not adapter:
            raise ConfigurationError(f"Provider '{provider_name}' not registered")
        return provider_name, adapter

    def complete(self, request: Request) -> Response:
        """Blocking completion. No automatic retry."""
        provider_name, adapter = self._resolve_provider(request)
        # Ensure provider is set on request
        if not request.provider:
            request = Request(**{**request.__dict__, "provider": provider_name})

        # Apply middleware (request phase)
        def call(req: Request) -> Response:
            return adapter.complete(req)

        fn = call
        for mw in reversed(self._middleware):
            fn = self._wrap_middleware(mw, fn)

        return fn(request)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Streaming. No automatic retry."""
        provider_name, adapter = self._resolve_provider(request)
        if not request.provider:
            request = Request(**{**request.__dict__, "provider": provider_name})
        async for event in adapter.stream(request):
            yield event

    @staticmethod
    def _wrap_middleware(mw: Callable, next_fn: Callable) -> Callable:
        def wrapped(req: Request) -> Response:
            return mw(req, next_fn)
        return wrapped

    def add_middleware(self, mw: Callable) -> None:
        self._middleware.append(mw)


# Module-level default client (lazy initialized)
_default_client: Client | None = None


def get_default_client() -> Client:
    global _default_client
    if _default_client is None:
        _default_client = Client.from_env()
    return _default_client


def set_default_client(client: Client) -> None:
    global _default_client
    _default_client = client


def generate(
    model: str,
    prompt: str | None = None,
    messages: list | None = None,
    system: str | None = None,
    tools: list | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    provider_options: dict | None = None,
    client: Client | None = None,
    **kwargs,
) -> Response:
    """High-level blocking generation."""
    from .types import Message, ToolDefinition

    c = client or get_default_client()

    if prompt and messages:
        raise ValueError("Provide either prompt or messages, not both")

    msgs: list[Message] = []
    if system:
        msgs.append(Message.system(system))
    if prompt:
        msgs.append(Message.user(prompt))
    elif messages:
        msgs.extend(messages)

    tool_defs = None
    if tools:
        tool_defs = [
            ToolDefinition(name=t["name"], description=t["description"],
                           parameters=t["parameters"])
            if isinstance(t, dict) else t
            for t in tools
        ]

    request = Request(
        model=model,
        messages=msgs,
        provider=provider,
        tools=tool_defs,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        provider_options=provider_options,
    )
    return c.complete(request)
