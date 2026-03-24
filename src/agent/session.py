"""Coding Agent Session - the agentic loop (Section 2 of coding-agent-loop-spec.md)."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from typing import Any, Callable, Iterator

from ..llm.types import (
    Message, Request, Response, Role, ContentKind, ContentPart,
    ToolCallData, ToolResultData, ToolDefinition, ToolCall, ToolResult,
)
from .environment import LocalExecutionEnvironment, ExecutionEnvironment
from .truncation import truncate_tool_output
from .tools.core import ToolRegistry, ANTHROPIC_TOOLS, OPENAI_TOOLS, GEMINI_TOOLS


class SessionState(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    AWAITING_INPUT = "awaiting_input"
    CLOSED = "closed"


@dataclass
class SessionConfig:
    max_turns: int = 0  # 0 = unlimited
    max_tool_rounds_per_input: int = 0  # 0 = unlimited
    default_command_timeout_ms: int = 10000
    max_command_timeout_ms: int = 600000
    reasoning_effort: str | None = None
    tool_output_limits: dict[str, int] = field(default_factory=dict)
    enable_loop_detection: bool = True
    loop_detection_window: int = 10
    max_subagent_depth: int = 1
    model: str = "claude-sonnet-4-6"
    provider: str = "anthropic"


class EventKind(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_INPUT = "user_input"
    ASSISTANT_TEXT_START = "assistant_text_start"
    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    ASSISTANT_TEXT_END = "assistant_text_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_OUTPUT_DELTA = "tool_call_output_delta"
    TOOL_CALL_END = "tool_call_end"
    STEERING_INJECTED = "steering_injected"
    TURN_LIMIT = "turn_limit"
    LOOP_DETECTION = "loop_detection"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class SessionEvent:
    kind: EventKind
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserTurn:
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None
    usage: Any = None
    response_id: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResultsTurn:
    results: list[ToolResult]
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemTurn:
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class SteeringTurn:
    content: str
    timestamp: float = field(default_factory=time.time)


Turn = UserTurn | AssistantTurn | ToolResultsTurn | SystemTurn | SteeringTurn


class Session:
    """Coding agent session with agentic loop."""

    def __init__(
        self,
        config: SessionConfig | None = None,
        execution_env: ExecutionEnvironment | None = None,
        llm_client=None,
        on_event: Callable[[SessionEvent], None] | None = None,
        depth: int = 0,
    ):
        self.id = uuid.uuid4().hex[:8]
        self.config = config or SessionConfig()
        self.execution_env = execution_env or LocalExecutionEnvironment()
        self._llm_client = llm_client
        self._on_event = on_event
        self.history: list[Turn] = []
        self.state = SessionState.IDLE
        self._steering_queue: Queue[str] = Queue()
        self._followup_queue: Queue[str] = Queue()
        self._depth = depth
        self._abort = threading.Event()
        self._subagents: dict[str, "Session"] = {}

        # Build tool registry
        self._tool_registry = self._build_tool_registry()

    def _build_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        provider = self.config.provider.lower()
        tools = OPENAI_TOOLS if provider == "openai" else (
            GEMINI_TOOLS if provider == "gemini" else ANTHROPIC_TOOLS
        )
        for tool in tools:
            registry.register(tool)

        # Register subagent tools if depth allows
        if self._depth < self.config.max_subagent_depth:
            from .tools.core import RegisteredTool
            registry.register(RegisteredTool(
                name="spawn_agent",
                description="Spawn a subagent to handle a scoped task autonomously.",
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "working_dir": {"type": "string"},
                        "model": {"type": "string"},
                        "max_turns": {"type": "integer"},
                    },
                    "required": ["task"],
                },
                executor=self._exec_spawn_agent,
            ))
            registry.register(RegisteredTool(
                name="send_input",
                description="Send a message to a running subagent.",
                parameters={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["agent_id", "message"],
                },
                executor=self._exec_send_input,
            ))
            registry.register(RegisteredTool(
                name="wait_agent",
                description="Wait for a subagent to complete and return its result.",
                parameters={
                    "type": "object",
                    "properties": {"agent_id": {"type": "string"}},
                    "required": ["agent_id"],
                },
                executor=self._exec_wait_agent,
            ))
            registry.register(RegisteredTool(
                name="close_agent",
                description="Terminate a subagent.",
                parameters={
                    "type": "object",
                    "properties": {"agent_id": {"type": "string"}},
                    "required": ["agent_id"],
                },
                executor=self._exec_close_agent,
            ))
        return registry

    def _emit(self, kind: EventKind, **data) -> None:
        if self._on_event:
            self._on_event(SessionEvent(kind=kind, session_id=self.id, data=data))

    def steer(self, message: str) -> None:
        """Inject a steering message after the current tool round."""
        self._steering_queue.put(message)

    def follow_up(self, message: str) -> None:
        """Queue a message to process after current input completes."""
        self._followup_queue.put(message)

    def abort(self) -> None:
        self._abort.set()
        self.state = SessionState.CLOSED

    def submit(self, user_input: str) -> str:
        """Process user input and return final assistant text."""
        self._emit(EventKind.SESSION_START)
        result = self._process_input(user_input)
        return result

    def _process_input(self, user_input: str) -> str:
        self.state = SessionState.PROCESSING
        self.history.append(UserTurn(content=user_input))
        self._emit(EventKind.USER_INPUT, content=user_input)

        self._drain_steering()

        round_count = 0
        final_text = ""

        while True:
            if self._abort.is_set():
                break

            # Check limits
            if (self.config.max_tool_rounds_per_input > 0 and
                    round_count >= self.config.max_tool_rounds_per_input):
                self._emit(EventKind.TURN_LIMIT, round=round_count)
                break

            if (self.config.max_turns > 0 and
                    len(self.history) >= self.config.max_turns):
                self._emit(EventKind.TURN_LIMIT, total_turns=len(self.history))
                break

            # Build request
            llm_client = self._get_client()
            system_prompt = self._build_system_prompt()
            messages = self._history_to_messages()
            tool_defs = [
                ToolDefinition(name=d["name"], description=d["description"],
                                parameters=d["parameters"])
                for d in self._tool_registry.definitions()
            ]

            request = Request(
                model=self.config.model,
                provider=self.config.provider,
                messages=[Message.system(system_prompt)] + messages,
                tools=tool_defs if tool_defs else None,
                tool_choice=None,
                reasoning_effort=self.config.reasoning_effort,
                max_tokens=8192,
            )

            try:
                response = llm_client.complete(request)
            except Exception as e:
                self._emit(EventKind.ERROR, error=str(e))
                self.state = SessionState.CLOSED
                return f"Error: {e}"

            # Record assistant turn
            tool_calls = response.tool_calls
            assistant_turn = AssistantTurn(
                content=response.text,
                tool_calls=tool_calls,
                reasoning=response.reasoning,
                usage=response.usage,
                response_id=response.id,
            )
            self.history.append(assistant_turn)
            final_text = response.text

            self._emit(EventKind.ASSISTANT_TEXT_END,
                       text=response.text, reasoning=response.reasoning)

            # Natural completion (no tool calls)
            if not tool_calls:
                break

            # Execute tool calls
            round_count += 1
            results = self._execute_tool_calls(tool_calls)
            self.history.append(ToolResultsTurn(results=results))

            # Drain steering
            self._drain_steering()

            # Loop detection
            if self.config.enable_loop_detection:
                if self._detect_loop():
                    warning = (
                        f"Loop detected: the last {self.config.loop_detection_window} "
                        f"tool calls follow a repeating pattern. Try a different approach."
                    )
                    self.history.append(SteeringTurn(content=warning))
                    self._emit(EventKind.LOOP_DETECTION, message=warning)

        # Process follow-up
        if not self._followup_queue.empty():
            next_input = self._followup_queue.get()
            return self._process_input(next_input)

        self.state = SessionState.IDLE
        self._emit(EventKind.SESSION_END)
        return final_text

    def _drain_steering(self) -> None:
        while not self._steering_queue.empty():
            msg = self._steering_queue.get()
            self.history.append(SteeringTurn(content=msg))
            self._emit(EventKind.STEERING_INJECTED, content=msg)

    def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        results = []
        for tc in tool_calls:
            self._emit(EventKind.TOOL_CALL_START, tool_name=tc.name, call_id=tc.id)
            registered = self._tool_registry.get(tc.name)
            if registered is None:
                error_msg = f"Unknown tool: {tc.name}"
                self._emit(EventKind.TOOL_CALL_END, call_id=tc.id, error=error_msg)
                results.append(ToolResult(tool_call_id=tc.id, content=error_msg, is_error=True))
                continue

            try:
                raw_output = registered.executor(tc.arguments, self.execution_env)
                truncated = truncate_tool_output(
                    raw_output, tc.name, self.config.tool_output_limits or {}
                )
                self._emit(EventKind.TOOL_CALL_END, call_id=tc.id, output=raw_output)
                results.append(ToolResult(tool_call_id=tc.id, content=truncated, is_error=False))
            except Exception as e:
                error_msg = f"Tool error ({tc.name}): {e}"
                self._emit(EventKind.TOOL_CALL_END, call_id=tc.id, error=error_msg)
                results.append(ToolResult(tool_call_id=tc.id, content=error_msg, is_error=True))

        return results

    def _history_to_messages(self) -> list[Message]:
        messages = []
        for turn in self.history:
            if isinstance(turn, UserTurn):
                messages.append(Message.user(turn.content))
            elif isinstance(turn, AssistantTurn):
                parts = []
                if turn.content:
                    parts.append(ContentPart.text_part(turn.content))
                for tc in turn.tool_calls:
                    parts.append(ContentPart.tool_call_part(
                        ToolCallData(id=tc.id, name=tc.name, arguments=tc.arguments)
                    ))
                if parts:
                    messages.append(Message(role=Role.ASSISTANT, content=parts))
            elif isinstance(turn, ToolResultsTurn):
                for result in turn.results:
                    messages.append(Message.tool_result(
                        tool_call_id=result.tool_call_id,
                        content=str(result.content),
                        is_error=result.is_error,
                    ))
            elif isinstance(turn, SteeringTurn):
                messages.append(Message.user(turn.content))
            elif isinstance(turn, SystemTurn):
                messages.append(Message.user(f"[System] {turn.content}"))
        return messages

    def _build_system_prompt(self) -> str:
        env = self.execution_env
        working_dir = env.working_directory()
        platform = env.platform()

        # Check git
        is_git = os.path.exists(os.path.join(working_dir, ".git"))
        git_branch = ""
        if is_git:
            try:
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    capture_output=True, text=True, cwd=working_dir
                )
                git_branch = result.stdout.strip()
            except Exception:
                pass

        model = self.config.model
        provider = self.config.provider

        base = self._provider_base_instructions()

        env_block = f"""<environment>
Working directory: {working_dir}
Is git repository: {is_git}
Git branch: {git_branch or "unknown"}
Platform: {platform}
Today's date: {time.strftime("%Y-%m-%d")}
Model: {model}
Provider: {provider}
</environment>"""

        # Load project docs
        project_docs = self._load_project_docs(working_dir, provider)

        parts = [base, env_block]
        if project_docs:
            parts.append(f"<project_instructions>\n{project_docs}\n</project_instructions>")

        return "\n\n".join(parts)

    def _provider_base_instructions(self) -> str:
        provider = self.config.provider.lower()
        if provider == "anthropic":
            return _ANTHROPIC_BASE_INSTRUCTIONS
        if provider == "openai":
            return _OPENAI_BASE_INSTRUCTIONS
        return _GEMINI_BASE_INSTRUCTIONS

    def _load_project_docs(self, working_dir: str, provider: str) -> str:
        doc_files = ["AGENTS.md"]
        if provider == "anthropic":
            doc_files.extend(["CLAUDE.md"])
        elif provider == "openai":
            doc_files.extend([".codex/instructions.md"])
        elif provider == "gemini":
            doc_files.extend(["GEMINI.md"])

        total = []
        budget = 32 * 1024  # 32KB
        used = 0
        for fname in doc_files:
            path = os.path.join(working_dir, fname)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        content = f.read()
                    content_bytes = content.encode()
                    if used + len(content_bytes) > budget:
                        remaining = budget - used
                        content = content_bytes[:remaining].decode(errors="replace")
                        content += "\n[Project instructions truncated at 32KB]"
                        total.append(f"# {fname}\n{content}")
                        break
                    total.append(f"# {fname}\n{content}")
                    used += len(content_bytes)
                except Exception:
                    pass
        return "\n\n".join(total)

    def _get_client(self):
        if self._llm_client:
            return self._llm_client
        from ..llm.client import get_default_client
        return get_default_client()

    def _detect_loop(self) -> bool:
        window = self.config.loop_detection_window
        # Extract tool call signatures from recent history
        signatures = []
        for turn in reversed(self.history):
            if isinstance(turn, AssistantTurn) and turn.tool_calls:
                for tc in turn.tool_calls:
                    sig = hashlib.md5(f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True, default=str)}".encode()).hexdigest()[:8]
                    signatures.append(sig)
            if len(signatures) >= window:
                break
        signatures = signatures[:window]

        if len(signatures) < window:
            return False

        # Check repeating patterns of length 1, 2, 3
        for pattern_len in [1, 2, 3]:
            if window % pattern_len != 0:
                continue
            pattern = signatures[:pattern_len]
            all_match = all(
                signatures[i:i + pattern_len] == pattern
                for i in range(0, window, pattern_len)
            )
            if all_match:
                return True
        return False

    # Subagent tool executors
    def _exec_spawn_agent(self, args: dict, env: ExecutionEnvironment) -> str:
        task = args["task"]
        working_dir = args.get("working_dir")
        model = args.get("model", self.config.model)
        max_turns = args.get("max_turns", 0)

        sub_env = LocalExecutionEnvironment(working_dir or env.working_directory())
        sub_config = SessionConfig(
            model=model,
            provider=self.config.provider,
            max_turns=max_turns,
            default_command_timeout_ms=self.config.default_command_timeout_ms,
        )
        sub_session = Session(
            config=sub_config,
            execution_env=sub_env,
            llm_client=self._llm_client,
            depth=self._depth + 1,
        )
        agent_id = sub_session.id
        self._subagents[agent_id] = sub_session

        # Run in background thread
        result_holder = {"output": "", "done": False}
        def run():
            result_holder["output"] = sub_session.submit(task)
            result_holder["done"] = True

        t = threading.Thread(target=run, daemon=True)
        t.start()
        # Store thread for wait
        sub_session._thread = t
        sub_session._result = result_holder

        return f"Agent spawned with ID: {agent_id}\nTask: {task}"

    def _exec_send_input(self, args: dict, env: ExecutionEnvironment) -> str:
        agent_id = args["agent_id"]
        message = args["message"]
        sub = self._subagents.get(agent_id)
        if not sub:
            return f"Error: No agent with ID {agent_id}"
        sub.follow_up(message)
        return f"Message sent to agent {agent_id}"

    def _exec_wait_agent(self, args: dict, env: ExecutionEnvironment) -> str:
        agent_id = args["agent_id"]
        sub = self._subagents.get(agent_id)
        if not sub:
            return f"Error: No agent with ID {agent_id}"
        t = getattr(sub, "_thread", None)
        if t:
            t.join(timeout=300)
        result = getattr(sub, "_result", {})
        output = result.get("output", "(no output)")
        return f"Agent {agent_id} result:\n{output}"

    def _exec_close_agent(self, args: dict, env: ExecutionEnvironment) -> str:
        agent_id = args["agent_id"]
        sub = self._subagents.pop(agent_id, None)
        if sub:
            sub.abort()
            return f"Agent {agent_id} terminated"
        return f"Agent {agent_id} not found"


# Provider base instructions
_ANTHROPIC_BASE_INSTRUCTIONS = """You are a coding agent powered by Claude. Your role is to help with software engineering tasks.

## Core Principles
- Read files before editing them to understand their current content
- Prefer editing existing files over creating new ones
- Use edit_file with precise old_string/new_string pairs (old_string must be unique in the file)
- Keep changes focused and minimal - don't refactor code beyond what was asked
- Run commands to verify your changes work when possible

## Tool Usage
- Use read_file to understand existing code before making changes
- Use edit_file for targeted changes (the old_string/new_string format is your primary editing tool)
- Use write_file for creating new files or complete rewrites
- Use shell to run commands, tests, and verify changes
- Use grep to search for code patterns and glob to find files

## Code Quality
- Write clean, idiomatic code that follows the project's existing style
- Don't add unnecessary comments, docstrings, or type annotations
- Don't add features beyond what was requested"""

_OPENAI_BASE_INSTRUCTIONS = """You are a coding agent. Your role is to help with software engineering tasks.

## Core Principles
- Read files before modifying them
- Use apply_patch as your primary tool for code modifications (it supports multiple files in one operation)
- Keep changes minimal and focused

## Tool Usage
- Use read_file to understand existing code
- Use apply_patch for all file modifications (preferred over write_file for existing files)
- Use write_file only for creating new files without patch overhead
- Use shell to run commands and verify changes

## apply_patch Format
Use the v4a patch format:
```
*** Begin Patch
*** Update File: path/to/file.py
@@ context hint
 unchanged line
-line to remove
+line to add
*** End Patch
```"""

_GEMINI_BASE_INSTRUCTIONS = """You are a coding agent powered by Gemini. Your role is to help with software engineering tasks.

## Core Principles
- Read files before editing them
- Use edit_file for targeted changes, write_file for new files
- Keep changes minimal and focused
- Run commands to verify your changes

## Tool Usage
- Use read_file and read_many_files to understand existing code
- Use edit_file for search-and-replace style edits
- Use write_file for creating new files
- Use shell for running commands
- Use grep and glob for searching"""
