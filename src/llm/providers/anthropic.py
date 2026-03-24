"""Anthropic provider adapter - uses native Messages API."""
from __future__ import annotations
import json
import uuid
from typing import AsyncIterator

from ..types import (
    Request, Response, Message, ContentPart, ContentKind,
    FinishReason, Usage, StreamEvent, StreamEventType,
    ToolCall, ToolCallData, ThinkingData, Role,
    ProviderError, AuthenticationError, RateLimitError,
    ServerError, ContentFilterError, ContextLengthError,
    InvalidRequestError, _make_http_error,
)


class AnthropicAdapter:
    name = "anthropic"

    def __init__(self, api_key: str, base_url: str | None = None,
                 default_headers: dict | None = None, timeout: float = 120.0):
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=api_key,
                base_url=base_url,
                default_headers=default_headers or {},
                timeout=timeout,
            )
            self._async_client = anthropic.AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
                default_headers=default_headers or {},
                timeout=timeout,
            )
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def complete(self, request: Request) -> Response:
        """Blocking completion via Anthropic Messages API."""
        import anthropic
        try:
            kwargs = self._build_kwargs(request)
            result = self._client.messages.create(**kwargs)
            return self._convert_response(result, request.model)
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), provider="anthropic") from e
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e), provider="anthropic") from e
        except anthropic.APIStatusError as e:
            raise _make_http_error(e.status_code, str(e), "anthropic") from e
        except anthropic.APIConnectionError as e:
            from ..types import NetworkError
            raise NetworkError(str(e)) from e

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Streaming via Anthropic Messages API."""
        import anthropic
        kwargs = self._build_kwargs(request)
        try:
            async with self._async_client.messages.stream(**kwargs) as stream:
                yield StreamEvent(type=StreamEventType.STREAM_START)
                async for event in stream:
                    evt_type = getattr(event, "type", None)
                    if evt_type == "content_block_start":
                        block = event.content_block
                        if block.type == "text":
                            yield StreamEvent(type=StreamEventType.TEXT_START, text_id=str(event.index))
                        elif block.type == "thinking":
                            yield StreamEvent(type=StreamEventType.REASONING_START)
                        elif block.type == "tool_use":
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                tool_call=ToolCall(id=block.id, name=block.name, arguments={}),
                            )
                    elif evt_type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield StreamEvent(
                                type=StreamEventType.TEXT_DELTA,
                                delta=delta.text,
                                text_id=str(event.index),
                            )
                        elif delta.type == "thinking_delta":
                            yield StreamEvent(
                                type=StreamEventType.REASONING_DELTA,
                                reasoning_delta=delta.thinking,
                            )
                        elif delta.type == "input_json_delta":
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                delta=delta.partial_json,
                            )
                    elif evt_type == "content_block_stop":
                        block_type = getattr(getattr(stream, "_current_block", None), "type", None)
                        if block_type == "text":
                            yield StreamEvent(type=StreamEventType.TEXT_END, text_id=str(event.index))
                        elif block_type == "thinking":
                            yield StreamEvent(type=StreamEventType.REASONING_END)
                    elif evt_type == "message_stop":
                        final_msg = await stream.get_final_message()
                        response = self._convert_response(final_msg, request.model)
                        yield StreamEvent(
                            type=StreamEventType.FINISH,
                            finish_reason=response.finish_reason,
                            usage=response.usage,
                            response=response,
                        )
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), provider="anthropic") from e
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e), provider="anthropic") from e
        except anthropic.APIStatusError as e:
            raise _make_http_error(e.status_code, str(e), "anthropic") from e

    def _build_kwargs(self, request: Request) -> dict:
        """Build kwargs for Anthropic API call."""
        # Separate system messages from conversation
        system_parts = []
        messages = []
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                system_parts.append(msg.text)
            elif msg.role == Role.DEVELOPER:
                system_parts.append(msg.text)
            else:
                messages.append(self._convert_message(msg))

        kwargs: dict = {
            "model": request.model,
            "max_tokens": request.max_tokens or 8192,
            "messages": messages,
        }
        if system_parts:
            system_text = "\n\n".join(system_parts)
            # Add cache_control to system for prompt caching
            kwargs["system"] = [{"type": "text", "text": system_text,
                                  "cache_control": {"type": "ephemeral"}}]

        if request.tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description,
                 "input_schema": t.parameters}
                for t in request.tools
            ]
            if request.tool_choice:
                mode = request.tool_choice.mode
                if mode == "auto":
                    kwargs["tool_choice"] = {"type": "auto"}
                elif mode == "required":
                    kwargs["tool_choice"] = {"type": "any"}
                elif mode == "named" and request.tool_choice.tool_name:
                    kwargs["tool_choice"] = {"type": "tool", "name": request.tool_choice.tool_name}
                # "none" mode: omit tools entirely (handled by not setting tool_choice with tools)
            else:
                kwargs["tool_choice"] = {"type": "auto"}

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences

        # Provider-specific options
        if request.provider_options:
            opts = request.provider_options.get("anthropic", {})
            if "thinking" in opts:
                kwargs["thinking"] = opts["thinking"]
            if "beta_headers" in opts or "beta_features" in opts:
                headers = opts.get("beta_headers") or opts.get("beta_features") or []
                if isinstance(headers, list):
                    kwargs["extra_headers"] = {"anthropic-beta": ",".join(headers)}

        # Reasoning effort -> thinking budget
        if request.reasoning_effort and "thinking" not in kwargs:
            budget_map = {"low": 1000, "medium": 5000, "high": 10000}
            budget = budget_map.get(request.reasoning_effort)
            if budget:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        return kwargs

    def _convert_message(self, msg: Message) -> dict:
        """Convert SDK Message to Anthropic format."""
        content = []
        for part in msg.content:
            if part.kind == ContentKind.TEXT:
                content.append({"type": "text", "text": part.text or ""})
            elif part.kind == ContentKind.TOOL_CALL and part.tool_call:
                tc = part.tool_call
                args = tc.arguments if isinstance(tc.arguments, dict) else {}
                content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": args})
            elif part.kind == ContentKind.TOOL_RESULT and part.tool_result:
                tr = part.tool_result
                content_val = tr.content if isinstance(tr.content, str) else json.dumps(tr.content)
                content.append({
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": content_val,
                    "is_error": tr.is_error,
                })
            elif part.kind == ContentKind.THINKING and part.thinking:
                t = part.thinking
                if t.redacted:
                    content.append({"type": "redacted_thinking", "data": t.signature or t.text})
                else:
                    block: dict = {"type": "thinking", "thinking": t.text}
                    if t.signature:
                        block["signature"] = t.signature
                    content.append(block)
            elif part.kind == ContentKind.IMAGE and part.image:
                img = part.image
                if img.url:
                    content.append({"type": "image", "source": {"type": "url", "url": img.url}})
                elif img.data:
                    import base64
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.media_type or "image/png",
                            "data": base64.b64encode(img.data).decode(),
                        }
                    })

        if msg.role == Role.TOOL:
            # Tool results go into a user message as tool_result blocks
            return {"role": "user", "content": content}

        role_map = {Role.USER: "user", Role.ASSISTANT: "assistant"}
        return {"role": role_map.get(msg.role, "user"), "content": content}

    def _convert_response(self, result, model: str) -> Response:
        """Convert Anthropic response to SDK Response."""
        content_parts = []
        for block in result.content:
            if block.type == "text":
                content_parts.append(ContentPart.text_part(block.text))
            elif block.type == "tool_use":
                content_parts.append(ContentPart.tool_call_part(
                    ToolCallData(id=block.id, name=block.name, arguments=block.input or {})
                ))
            elif block.type == "thinking":
                sig = getattr(block, "signature", None)
                content_parts.append(ContentPart.thinking_part(
                    ThinkingData(text=block.thinking, signature=sig)
                ))
            elif block.type == "redacted_thinking":
                content_parts.append(ContentPart.thinking_part(
                    ThinkingData(text="", signature=getattr(block, "data", None), redacted=True)
                ))

        finish_map = {
            "end_turn": "stop", "stop_sequence": "stop",
            "max_tokens": "length", "tool_use": "tool_calls",
        }
        raw_stop = result.stop_reason or "end_turn"
        finish = FinishReason(reason=finish_map.get(raw_stop, "other"), raw=raw_stop)

        usage_data = result.usage
        usage = Usage(
            input_tokens=usage_data.input_tokens,
            output_tokens=usage_data.output_tokens,
            total_tokens=usage_data.input_tokens + usage_data.output_tokens,
            cache_read_tokens=getattr(usage_data, "cache_read_input_tokens", None),
            cache_write_tokens=getattr(usage_data, "cache_creation_input_tokens", None),
        )

        return Response(
            id=result.id,
            model=result.model,
            provider="anthropic",
            message=Message(role=Role.ASSISTANT, content=content_parts),
            finish_reason=finish,
            usage=usage,
        )
