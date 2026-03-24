"""Tool output truncation (Section 5 of coding-agent-loop-spec.md)."""
from __future__ import annotations

# Default character limits per tool
DEFAULT_CHAR_LIMITS: dict[str, int] = {
    "read_file": 50_000,
    "shell": 30_000,
    "grep": 20_000,
    "glob": 20_000,
    "edit_file": 10_000,
    "apply_patch": 10_000,
    "write_file": 1_000,
    "spawn_agent": 20_000,
    "list_dir": 20_000,
}

# Default line limits per tool
DEFAULT_LINE_LIMITS: dict[str, int | None] = {
    "shell": 256,
    "grep": 200,
    "glob": 500,
    "read_file": None,
    "edit_file": None,
    "write_file": None,
    "apply_patch": None,
}

# Default truncation modes
DEFAULT_TRUNCATION_MODES: dict[str, str] = {
    "read_file": "head_tail",
    "shell": "head_tail",
    "grep": "tail",
    "glob": "tail",
    "edit_file": "tail",
    "apply_patch": "tail",
    "write_file": "tail",
    "spawn_agent": "head_tail",
}

DEFAULT_MAX_CHARS = 30_000


def truncate_output(output: str, max_chars: int, mode: str = "head_tail") -> str:
    if len(output) <= max_chars:
        return output

    if mode == "head_tail":
        half = max_chars // 2
        removed = len(output) - max_chars
        return (
            output[:half]
            + f"\n\n[WARNING: Tool output was truncated. "
            f"{removed} characters were removed from the middle. "
            f"The full output is available in the event stream. "
            f"If you need to see specific parts, re-run the tool with more targeted parameters.]\n\n"
            + output[-half:]
        )
    if mode == "tail":
        removed = len(output) - max_chars
        return (
            f"[WARNING: Tool output was truncated. First "
            f"{removed} characters were removed. "
            f"The full output is available in the event stream.]\n\n"
            + output[-max_chars:]
        )
    # Default: head
    return output[:max_chars]


def truncate_lines(output: str, max_lines: int) -> str:
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output
    head_count = max_lines // 2
    tail_count = max_lines - head_count
    omitted = len(lines) - head_count - tail_count
    return (
        "\n".join(lines[:head_count])
        + f"\n[... {omitted} lines omitted ...]\n"
        + "\n".join(lines[-tail_count:])
    )


def truncate_tool_output(
    output: str,
    tool_name: str,
    char_limits: dict[str, int] | None = None,
    line_limits: dict[str, int | None] | None = None,
) -> str:
    """Full truncation pipeline: character first, then lines."""
    cl = char_limits or DEFAULT_CHAR_LIMITS
    ll = line_limits or DEFAULT_LINE_LIMITS

    max_chars = cl.get(tool_name, DEFAULT_MAX_CHARS)
    mode = DEFAULT_TRUNCATION_MODES.get(tool_name, "head_tail")

    # Step 1: Character truncation (always)
    result = truncate_output(output, max_chars, mode)

    # Step 2: Line truncation (secondary)
    max_lines = ll.get(tool_name)
    if max_lines is not None:
        result = truncate_lines(result, max_lines)

    return result
