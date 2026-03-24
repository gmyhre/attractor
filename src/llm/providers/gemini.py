"""Gemini provider adapter - uses native Gemini API."""
from __future__ import annotations
import json
import uuid
from typing import AsyncIterator

from ..types import (
    Request, Response, Message, ContentPart, ContentKind,
    FinishReason, Usage, StreamEvent, StreamEventType,
    ToolCall, ToolCallData, Role,
    AuthenticationError, RateLimitError, _make_http_error,
)


class GeminiAdapter:
    name = "gemini"

    def __init__(self, api_key: str, base_url: str | None = None,
                 default_headers: dict | None = None, timeout: float = 120.0):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
        except ImportError:
            raise ImportError("google-generativeai package required: pip install google-generativeai")

    def complete(self, request: Request) -> Response:
        model_name = request.model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        gen_config = self._build_generation_config(request)
        tools = self._build_tools(request)

        model = self._genai.GenerativeModel(
            model_name=model_name,
            generation_config=gen_config,
            tools=tools if tools else None,
            system_instruction=self._extract_system(request),
        )

        contents = self._convert_messages(request.messages)
        response = model.generate_content(contents)
        return self._convert_response(response, request.model)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        model_name = request.model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        gen_config = self._build_generation_config(request)
        tools = self._build_tools(request)

        model = self._genai.GenerativeModel(
            model_name=model_name,
            generation_config=gen_config,
            tools=tools if tools else None,
            system_instruction=self._extract_system(request),
        )

        contents = self._convert_messages(request.messages)
        yield StreamEvent(type=StreamEventType.STREAM_START)

        response = model.generate_content(contents, stream=True)
        for chunk in response:
            for part in chunk.parts:
                if hasattr(part, "text") and part.text:
                    yield StreamEvent(type=StreamEventType.TEXT_DELTA, delta=part.text)
                elif hasattr(part, "function_call"):
                    fc = part.function_call
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_END,
                        tool_call=ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=fc.name,
                            arguments=dict(fc.args),
                        ),
                    )

        full_response = self._convert_response(response, request.model)
        yield StreamEvent(
            type=StreamEventType.FINISH,
            finish_reason=full_response.finish_reason,
            usage=full_response.usage,
            response=full_response,
        )

    def _extract_system(self, request: Request) -> str | None:
        parts = [msg.text for msg in request.messages
                 if msg.role in (Role.SYSTEM, Role.DEVELOPER)]
        return "\n\n".join(parts) if parts else None

    def _build_generation_config(self, request: Request) -> dict:
        config: dict = {}
        if request.max_tokens:
            config["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.top_p is not None:
            config["top_p"] = request.top_p
        if request.stop_sequences:
            config["stop_sequences"] = request.stop_sequences
        if request.response_format and request.response_format.type == "json":
            config["response_mime_type"] = "application/json"
        return config

    def _build_tools(self, request: Request) -> list | None:
        if not request.tools:
            return None
        function_declarations = []
        for t in request.tools:
            function_declarations.append({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            })
        return [{"function_declarations": function_declarations}]

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        contents = []
        for msg in messages:
            if msg.role in (Role.SYSTEM, Role.DEVELOPER):
                continue  # handled via system_instruction
            parts = []
            for part in msg.content:
                if part.kind == ContentKind.TEXT:
                    parts.append({"text": part.text or ""})
                elif part.kind == ContentKind.TOOL_CALL and part.tool_call:
                    tc = part.tool_call
                    args = tc.arguments if isinstance(tc.arguments, dict) else {}
                    parts.append({"function_call": {"name": tc.name, "args": args}})
                elif part.kind == ContentKind.TOOL_RESULT and part.tool_result:
                    tr = part.tool_result
                    content_val = tr.content if isinstance(tr.content, str) else json.dumps(tr.content)
                    parts.append({
                        "function_response": {
                            "name": "tool",  # Gemini needs function name
                            "response": {"result": content_val},
                        }
                    })
                elif part.kind == ContentKind.IMAGE and part.image:
                    img = part.image
                    if img.data:
                        import base64
                        parts.append({
                            "inline_data": {
                                "mime_type": img.media_type or "image/png",
                                "data": base64.b64encode(img.data).decode(),
                            }
                        })
                    elif img.url:
                        parts.append({"file_data": {"file_uri": img.url}})

            role_map = {Role.USER: "user", Role.ASSISTANT: "model", Role.TOOL: "user"}
            role = role_map.get(msg.role, "user")
            contents.append({"role": role, "parts": parts})
        return contents

    def _convert_response(self, response, model: str) -> Response:
        content_parts = []
        try:
            for part in response.parts:
                if hasattr(part, "text") and part.text:
                    content_parts.append(ContentPart.text_part(part.text))
                elif hasattr(part, "function_call"):
                    fc = part.function_call
                    content_parts.append(ContentPart.tool_call_part(
                        ToolCallData(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=fc.name,
                            arguments=dict(fc.args),
                        )
                    ))
        except Exception:
            pass

        # Check finish reason
        finish_reason_map = {
            "STOP": "stop", "MAX_TOKENS": "length",
            "SAFETY": "content_filter", "RECITATION": "content_filter",
        }
        raw_finish = "STOP"
        try:
            if response.candidates:
                raw_finish = response.candidates[0].finish_reason.name
        except Exception:
            pass

        has_tool_calls = any(p.kind == ContentKind.TOOL_CALL for p in content_parts)
        if has_tool_calls:
            finish_reason = FinishReason(reason="tool_calls", raw=raw_finish)
        else:
            finish_reason = FinishReason(
                reason=finish_reason_map.get(raw_finish, "other"),
                raw=raw_finish,
            )

        usage = Usage()
        try:
            meta = response.usage_metadata
            usage = Usage(
                input_tokens=meta.prompt_token_count,
                output_tokens=meta.candidates_token_count,
                total_tokens=meta.total_token_count,
                reasoning_tokens=getattr(meta, "thoughts_token_count", None),
                cache_read_tokens=getattr(meta, "cached_content_token_count", None),
            )
        except Exception:
            pass

        return Response(
            id=f"gemini-{uuid.uuid4().hex[:8]}",
            model=model,
            provider="gemini",
            message=Message(role=Role.ASSISTANT, content=content_parts),
            finish_reason=finish_reason,
            usage=usage,
        )
