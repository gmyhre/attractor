"""OpenAI provider adapter - uses native Responses API."""
from __future__ import annotations
import json
from typing import AsyncIterator

from ..types import (
    Request, Response, Message, ContentPart, ContentKind,
    FinishReason, Usage, StreamEvent, StreamEventType,
    ToolCall, ToolCallData, Role,
    AuthenticationError, RateLimitError, _make_http_error, NetworkError,
)


class OpenAIAdapter:
    name = "openai"

    def __init__(self, api_key: str, base_url: str | None = None,
                 default_headers: dict | None = None, timeout: float = 120.0):
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=default_headers or {},
                timeout=timeout,
            )
            self._async_client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=default_headers or {},
                timeout=timeout,
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def complete(self, request: Request) -> Response:
        import openai
        try:
            kwargs = self._build_kwargs(request)
            # Use Responses API for reasoning models, Chat Completions for others
            if self._is_responses_model(request.model):
                result = self._client.responses.create(**kwargs)
                return self._convert_responses_response(result, request.model)
            else:
                result = self._client.chat.completions.create(**kwargs)
                return self._convert_chat_response(result, request.model)
        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), provider="openai") from e
        except openai.RateLimitError as e:
            raise RateLimitError(str(e), provider="openai") from e
        except openai.APIStatusError as e:
            raise _make_http_error(e.status_code, str(e), "openai") from e
        except openai.APIConnectionError as e:
            raise NetworkError(str(e)) from e

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        import openai
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True
        try:
            if self._is_responses_model(request.model):
                async for event in await self._async_client.responses.create(**kwargs):
                    yield self._convert_responses_stream_event(event)
            else:
                async for chunk in await self._async_client.chat.completions.create(**kwargs):
                    for event in self._convert_chat_stream_chunk(chunk):
                        yield event
        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), provider="openai") from e
        except openai.RateLimitError as e:
            raise RateLimitError(str(e), provider="openai") from e
        except openai.APIStatusError as e:
            raise _make_http_error(e.status_code, str(e), "openai") from e

    def _is_responses_model(self, model: str) -> bool:
        """Check if model should use Responses API."""
        # GPT-5.2 series and o-series use Responses API
        return any(x in model for x in ["gpt-5", "o1", "o3", "codex"])

    def _build_kwargs(self, request: Request) -> dict:
        messages = []
        for msg in request.messages:
            messages.append(self._convert_message(msg))

        kwargs: dict = {
            "model": request.model,
            "messages": messages,
        }
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences
        if request.tools:
            kwargs["tools"] = [
                {"type": "function", "function": {
                    "name": t.name, "description": t.description, "parameters": t.parameters
                }}
                for t in request.tools
            ]
            if request.tool_choice:
                mode = request.tool_choice.mode
                if mode == "auto":
                    kwargs["tool_choice"] = "auto"
                elif mode == "none":
                    kwargs["tool_choice"] = "none"
                elif mode == "required":
                    kwargs["tool_choice"] = "required"
                elif mode == "named" and request.tool_choice.tool_name:
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": request.tool_choice.tool_name}
                    }
        if request.reasoning_effort:
            kwargs["reasoning_effort"] = request.reasoning_effort
        if request.response_format:
            if request.response_format.type == "json":
                kwargs["response_format"] = {"type": "json_object"}
            elif request.response_format.type == "json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"schema": request.response_format.json_schema,
                                    "strict": request.response_format.strict},
                }
        return kwargs

    def _convert_message(self, msg: Message) -> dict:
        if msg.role == Role.SYSTEM:
            return {"role": "system", "content": msg.text}
        if msg.role == Role.DEVELOPER:
            return {"role": "developer", "content": msg.text}
        if msg.role == Role.USER:
            content = []
            for part in msg.content:
                if part.kind == ContentKind.TEXT:
                    content.append({"type": "text", "text": part.text or ""})
                elif part.kind == ContentKind.IMAGE and part.image:
                    img = part.image
                    if img.url:
                        item: dict = {"type": "image_url", "image_url": {"url": img.url}}
                        if img.detail:
                            item["image_url"]["detail"] = img.detail
                        content.append(item)
            return {"role": "user", "content": content if len(content) > 1 else msg.text}
        if msg.role == Role.ASSISTANT:
            # Check for tool calls
            tool_calls = [p for p in msg.content if p.kind == ContentKind.TOOL_CALL]
            if tool_calls:
                tc_list = []
                for p in tool_calls:
                    if p.tool_call:
                        tc = p.tool_call
                        args = tc.arguments if isinstance(tc.arguments, str) else json.dumps(tc.arguments)
                        tc_list.append({
                            "id": tc.id, "type": "function",
                            "function": {"name": tc.name, "arguments": args}
                        })
                text_parts = [p.text for p in msg.content if p.kind == ContentKind.TEXT and p.text]
                result: dict = {"role": "assistant", "tool_calls": tc_list}
                if text_parts:
                    result["content"] = "".join(text_parts)
                return result
            return {"role": "assistant", "content": msg.text}
        if msg.role == Role.TOOL:
            # Find tool result
            for part in msg.content:
                if part.kind == ContentKind.TOOL_RESULT and part.tool_result:
                    tr = part.tool_result
                    content = tr.content if isinstance(tr.content, str) else json.dumps(tr.content)
                    return {"role": "tool", "tool_call_id": tr.tool_call_id, "content": content}
        return {"role": "user", "content": msg.text}

    def _convert_chat_response(self, result, model: str) -> Response:
        choice = result.choices[0]
        msg = choice.message
        content_parts = []
        if msg.content:
            content_parts.append(ContentPart.text_part(msg.content))
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = tc.function.arguments
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {}
                content_parts.append(ContentPart.tool_call_part(
                    ToolCallData(id=tc.id, name=tc.function.name, arguments=args,
                                 type="function")
                ))

        finish_map = {"stop": "stop", "length": "length", "tool_calls": "tool_calls",
                      "content_filter": "content_filter"}
        raw_finish = choice.finish_reason or "stop"
        finish = FinishReason(reason=finish_map.get(raw_finish, "other"), raw=raw_finish)

        u = result.usage
        usage = Usage(
            input_tokens=u.prompt_tokens,
            output_tokens=u.completion_tokens,
            total_tokens=u.total_tokens,
        )

        return Response(
            id=result.id,
            model=result.model,
            provider="openai",
            message=Message(role=Role.ASSISTANT, content=content_parts),
            finish_reason=finish,
            usage=usage,
        )

    def _convert_responses_response(self, result, model: str) -> Response:
        """Convert OpenAI Responses API result."""
        content_parts = []
        # Responses API has output list
        for item in getattr(result, "output", []):
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for block in getattr(item, "content", []):
                    if getattr(block, "type", None) == "output_text":
                        content_parts.append(ContentPart.text_part(block.text))
            elif item_type == "function_call":
                try:
                    args = json.loads(item.arguments)
                except Exception:
                    args = {}
                content_parts.append(ContentPart.tool_call_part(
                    ToolCallData(id=item.call_id, name=item.name, arguments=args)
                ))

        raw_finish = getattr(result, "status", "completed")
        finish = FinishReason(reason="stop" if raw_finish == "completed" else raw_finish, raw=raw_finish)

        u = getattr(result, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) if u else 0,
            output_tokens=getattr(u, "output_tokens", 0) if u else 0,
            total_tokens=getattr(u, "total_tokens", 0) if u else 0,
        )
        if u:
            details = getattr(u, "output_tokens_details", None)
            if details:
                usage.reasoning_tokens = getattr(details, "reasoning_tokens", None)
            prompt_details = getattr(u, "prompt_tokens_details", None)
            if prompt_details:
                usage.cache_read_tokens = getattr(prompt_details, "cached_tokens", None)

        return Response(
            id=getattr(result, "id", ""),
            model=model,
            provider="openai",
            message=Message(role=Role.ASSISTANT, content=content_parts),
            finish_reason=finish,
            usage=usage,
        )

    def _convert_responses_stream_event(self, event) -> StreamEvent:
        evt_type = getattr(event, "type", "")
        if "text.delta" in evt_type:
            return StreamEvent(type=StreamEventType.TEXT_DELTA, delta=getattr(event, "delta", ""))
        if "text.done" in evt_type:
            return StreamEvent(type=StreamEventType.TEXT_END)
        if "completed" in evt_type:
            return StreamEvent(type=StreamEventType.FINISH)
        return StreamEvent(type=StreamEventType.PROVIDER_EVENT, raw={"type": evt_type})

    def _convert_chat_stream_chunk(self, chunk) -> list[StreamEvent]:
        events = []
        for choice in chunk.choices:
            delta = choice.delta
            if delta.content:
                events.append(StreamEvent(type=StreamEventType.TEXT_DELTA, delta=delta.content))
            if choice.finish_reason:
                events.append(StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason(reason=choice.finish_reason, raw=choice.finish_reason),
                ))
        return events
