"""CodergenBackend implementations for the Attractor pipeline engine."""
from __future__ import annotations
import concurrent.futures
import os
from typing import Any

from .types import Node, Outcome, StageStatus
from .context import Context


class AgentLoopBackend:
    """Uses the coding agent session loop for LLM execution."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        working_dir: str | None = None,
        llm_client=None,
    ):
        self.model = model
        self.provider = provider
        self.working_dir = working_dir
        self._llm_client = llm_client

    def run(self, node: Node, prompt: str, context: Context) -> str | Outcome:
        from ..agent.session import Session, SessionConfig
        from ..agent.environment import LocalExecutionEnvironment
        from ..llm.client import Client

        model = node.attrs.llm_model or self.model or "claude-sonnet-4-6"
        provider = node.attrs.llm_provider or self.provider or "anthropic"
        working_dir = self.working_dir or os.getcwd()

        # Validate provider is available before starting
        if self._llm_client is None:
            client = Client.from_env()
            if provider not in client._providers:
                key_var = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}.get(provider, f"{provider.upper()}_API_KEY")
                raise RuntimeError(
                    f"Provider '{provider}' is not registered. "
                    f"Set the {key_var} environment variable and try again."
                )
        else:
            client = self._llm_client

        config = SessionConfig(
            model=model,
            provider=provider,
            reasoning_effort=node.attrs.reasoning_effort if node.attrs.reasoning_effort != "high" else None,
        )
        env = LocalExecutionEnvironment(working_dir)

        events = []
        def on_event(event):
            events.append(event)

        session = Session(
            config=config,
            execution_env=env,
            llm_client=client,
            on_event=on_event,
        )

        timeout_s = None
        if node.attrs.timeout:
            from .handlers import _parse_duration_s
            timeout_s = _parse_duration_s(node.attrs.timeout)

        if timeout_s:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(session.submit, prompt)
                try:
                    result = future.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    session.abort()
                    return Outcome(
                        status=StageStatus.FAIL,
                        failure_reason=f"Node '{node.id}' timed out after {timeout_s}s",
                    )
        else:
            result = session.submit(prompt)

        return result or "(no output)"


class DirectLLMBackend:
    """Calls LLM directly (single shot, no agent loop) - for simple stages."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        llm_client=None,
    ):
        self.model = model
        self.provider = provider
        self._llm_client = llm_client

    def run(self, node: Node, prompt: str, context: Context) -> str | Outcome:
        from ..llm.client import get_default_client
        from ..llm.types import Message, Request

        client = self._llm_client or get_default_client()
        model = node.attrs.llm_model or self.model or "claude-sonnet-4-6"
        provider = node.attrs.llm_provider or self.provider

        goal = context.get_string("graph.goal", "")
        system = f"You are an AI assistant helping with: {goal}" if goal else "You are a helpful AI assistant."

        request = Request(
            model=model,
            provider=provider,
            messages=[
                Message.system(system),
                Message.user(prompt),
            ],
            max_tokens=4096,
            reasoning_effort=node.attrs.reasoning_effort if node.attrs.reasoning_effort != "high" else None,
        )

        response = client.complete(request)
        return response.text
