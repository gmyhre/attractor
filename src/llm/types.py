"""Unified LLM Client - Data Model (Section 3 of unified-llm-spec.md)"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class ContentKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"


@dataclass
class ImageData:
    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    detail: str | None = None


@dataclass
class AudioData:
    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None


@dataclass
class DocumentData:
    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    file_name: str | None = None


@dataclass
class ToolCallData:
    id: str
    name: str
    arguments: dict | str
    type: str = "function"


@dataclass
class ToolResultData:
    tool_call_id: str
    content: str | dict
    is_error: bool = False
    image_data: bytes | None = None
    image_media_type: str | None = None


@dataclass
class ThinkingData:
    text: str
    signature: str | None = None
    redacted: bool = False


@dataclass
class ContentPart:
    kind: ContentKind | str
    text: str | None = None
    image: ImageData | None = None
    audio: AudioData | None = None
    document: DocumentData | None = None
    tool_call: ToolCallData | None = None
    tool_result: ToolResultData | None = None
    thinking: ThinkingData | None = None

    @classmethod
    def text_part(cls, text: str) -> "ContentPart":
        return cls(kind=ContentKind.TEXT, text=text)

    @classmethod
    def tool_call_part(cls, data: ToolCallData) -> "ContentPart":
        return cls(kind=ContentKind.TOOL_CALL, tool_call=data)

    @classmethod
    def tool_result_part(cls, data: ToolResultData) -> "ContentPart":
        return cls(kind=ContentKind.TOOL_RESULT, tool_result=data)

    @classmethod
    def thinking_part(cls, data: ThinkingData) -> "ContentPart":
        return cls(kind=ContentKind.THINKING, thinking=data)


@dataclass
class Message:
    role: Role
    content: list[ContentPart]
    name: str | None = None
    tool_call_id: str | None = None

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls(role=Role.SYSTEM, content=[ContentPart.text_part(text)])

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role=Role.USER, content=[ContentPart.text_part(text)])

    @classmethod
    def assistant(cls, text: str) -> "Message":
        return cls(role=Role.ASSISTANT, content=[ContentPart.text_part(text)])

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str, is_error: bool = False) -> "Message":
        return cls(
            role=Role.TOOL,
            content=[ContentPart.tool_result_part(
                ToolResultData(tool_call_id=tool_call_id, content=content, is_error=is_error)
            )],
            tool_call_id=tool_call_id,
        )

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.content if p.kind == ContentKind.TEXT and p.text)


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict


@dataclass
class ToolChoice:
    mode: str  # "auto", "none", "required", "named"
    tool_name: str | None = None


@dataclass
class ResponseFormat:
    type: str  # "text", "json", "json_schema"
    json_schema: dict | None = None
    strict: bool = False


@dataclass
class Request:
    model: str
    messages: list[Message]
    provider: str | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None
    response_format: ResponseFormat | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    reasoning_effort: str | None = None
    metadata: dict[str, str] | None = None
    provider_options: dict | None = None


@dataclass
class FinishReason:
    reason: str  # "stop", "length", "tool_calls", "content_filter", "error", "other"
    raw: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    raw: dict | None = None

    def __add__(self, other: "Usage") -> "Usage":
        def add_opt(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=add_opt(self.reasoning_tokens, other.reasoning_tokens),
            cache_read_tokens=add_opt(self.cache_read_tokens, other.cache_read_tokens),
            cache_write_tokens=add_opt(self.cache_write_tokens, other.cache_write_tokens),
        )


@dataclass
class Warning:
    message: str
    code: str | None = None


@dataclass
class RateLimitInfo:
    requests_remaining: int | None = None
    requests_limit: int | None = None
    tokens_remaining: int | None = None
    tokens_limit: int | None = None
    reset_at: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str | None = None


@dataclass
class ToolResult:
    tool_call_id: str
    content: str | dict | list
    is_error: bool = False


@dataclass
class Response:
    id: str
    model: str
    provider: str
    message: Message
    finish_reason: FinishReason
    usage: Usage
    raw: dict | None = None
    warnings: list[Warning] = field(default_factory=list)
    rate_limit: RateLimitInfo | None = None

    @property
    def text(self) -> str:
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCall]:
        calls = []
        for part in self.message.content:
            if part.kind == ContentKind.TOOL_CALL and part.tool_call:
                tc = part.tool_call
                args = tc.arguments if isinstance(tc.arguments, dict) else {}
                raw_args = tc.arguments if isinstance(tc.arguments, str) else None
                calls.append(ToolCall(id=tc.id, name=tc.name, arguments=args, raw_arguments=raw_args))
        return calls

    @property
    def reasoning(self) -> str | None:
        parts = [p.thinking.text for p in self.message.content
                 if p.kind == ContentKind.THINKING and p.thinking]
        return "\n".join(parts) if parts else None


class StreamEventType(str, Enum):
    STREAM_START = "stream_start"
    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_END = "text_end"
    REASONING_START = "reasoning_start"
    REASONING_DELTA = "reasoning_delta"
    REASONING_END = "reasoning_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    FINISH = "finish"
    ERROR = "error"
    PROVIDER_EVENT = "provider_event"


@dataclass
class StreamEvent:
    type: StreamEventType | str
    delta: str | None = None
    text_id: str | None = None
    reasoning_delta: str | None = None
    tool_call: ToolCall | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    response: Response | None = None
    error: Exception | None = None
    raw: dict | None = None


# Error hierarchy
class SDKError(Exception):
    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


class ProviderError(SDKError):
    def __init__(self, message: str, provider: str = "", status_code: int | None = None,
                 error_code: str | None = None, retryable: bool = True,
                 retry_after: float | None = None, raw: dict | None = None,
                 cause: Exception | None = None):
        super().__init__(message, cause)
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.raw = raw


class AuthenticationError(ProviderError):
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class AccessDeniedError(ProviderError):
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class NotFoundError(ProviderError):
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class InvalidRequestError(ProviderError):
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class RateLimitError(ProviderError):
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=True, **kwargs)


class ServerError(ProviderError):
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=True, **kwargs)


class ContentFilterError(ProviderError):
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class ContextLengthError(ProviderError):
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class QuotaExceededError(ProviderError):
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class RequestTimeoutError(SDKError):
    retryable = True


class AbortError(SDKError):
    retryable = False


class NetworkError(SDKError):
    retryable = True


class StreamError(SDKError):
    retryable = True


class InvalidToolCallError(SDKError):
    retryable = False


class NoObjectGeneratedError(SDKError):
    retryable = False


class ConfigurationError(SDKError):
    retryable = False


@dataclass
class ModelInfo:
    id: str
    provider: str
    display_name: str
    context_window: int
    max_output: int | None = None
    supports_tools: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    aliases: list[str] = field(default_factory=list)


def _make_http_error(status_code: int, message: str, provider: str,
                     raw: dict | None = None) -> ProviderError:
    """Map HTTP status codes to error types."""
    mapping = {
        400: InvalidRequestError,
        401: AuthenticationError,
        403: AccessDeniedError,
        404: NotFoundError,
        408: RequestTimeoutError,
        413: ContextLengthError,
        422: InvalidRequestError,
        429: RateLimitError,
    }
    if status_code in mapping:
        cls = mapping[status_code]
        if cls == RequestTimeoutError:
            return cls(message, cause=None)  # type: ignore
        return cls(message, provider=provider, status_code=status_code, raw=raw)
    if 500 <= status_code < 600:
        return ServerError(message, provider=provider, status_code=status_code, raw=raw)
    # classify by message content
    msg_lower = message.lower()
    if "not found" in msg_lower or "does not exist" in msg_lower:
        return NotFoundError(message, provider=provider, status_code=status_code, raw=raw)
    if "unauthorized" in msg_lower or "invalid key" in msg_lower:
        return AuthenticationError(message, provider=provider, status_code=status_code, raw=raw)
    if "context length" in msg_lower or "too many tokens" in msg_lower:
        return ContextLengthError(message, provider=provider, status_code=status_code, raw=raw)
    if "content filter" in msg_lower or "safety" in msg_lower:
        return ContentFilterError(message, provider=provider, status_code=status_code, raw=raw)
    return ProviderError(message, provider=provider, status_code=status_code, raw=raw)
