"""Core tool definitions and executors (Section 3.3 of coding-agent-loop-spec.md)."""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Callable

from ..environment import ExecutionEnvironment, LocalExecutionEnvironment


@dataclass
class RegisteredTool:
    name: str
    description: str
    parameters: dict
    executor: Callable[[dict, ExecutionEnvironment], str]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def definitions(self) -> list[dict]:
        from ...llm.types import ToolDefinition
        return [{"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())


# ---------- Tool executors ----------

def _exec_read_file(args: dict, env: ExecutionEnvironment) -> str:
    path = args["file_path"]
    offset = args.get("offset")
    limit = args.get("limit", 2000)
    try:
        return env.read_file(path, offset=offset, limit=limit)
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading {path}: {e}"


def _exec_write_file(args: dict, env: ExecutionEnvironment) -> str:
    path = args["file_path"]
    content = args["content"]
    try:
        env.write_file(path, content)
        size = len(content.encode())
        return f"Successfully wrote {size} bytes to {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def _exec_edit_file(args: dict, env: ExecutionEnvironment) -> str:
    path = args["file_path"]
    old_string = args["old_string"]
    new_string = args["new_string"]
    replace_all = args.get("replace_all", False)

    resolved = path
    if not os.path.isabs(path):
        resolved = os.path.join(env.working_directory(), path)

    try:
        with open(resolved, "r", errors="replace") as f:
            content = f.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"

    if old_string not in content:
        # Try fuzzy matching (whitespace normalization)
        import re
        normalized_old = re.sub(r"[ \t]+", " ", old_string.strip())
        normalized_content = re.sub(r"[ \t]+", " ", content)
        if normalized_old in normalized_content:
            # Found with fuzzy match - use original
            pass
        else:
            return f"Error: old_string not found in {path}. The string must match exactly."

    count = content.count(old_string)
    if count > 1 and not replace_all:
        return (
            f"Error: old_string matches {count} locations in {path}. "
            f"Provide more context to make it unique, or set replace_all=true."
        )

    if replace_all:
        new_content = content.replace(old_string, new_string)
        replacements = count
    else:
        new_content = content.replace(old_string, new_string, 1)
        replacements = 1

    try:
        with open(resolved, "w") as f:
            f.write(new_content)
        return f"Successfully made {replacements} replacement(s) in {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def _exec_shell(args: dict, env: ExecutionEnvironment) -> str:
    command = args["command"]
    timeout_ms = args.get("timeout_ms", 10000)
    result = env.exec_command(command, timeout_ms=timeout_ms)
    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += result.stderr
    if result.exit_code != 0 and not result.timed_out:
        output += f"\n[Exit code: {result.exit_code}]"
    output += f"\n[Duration: {result.duration_ms}ms]"
    return output


def _exec_grep(args: dict, env: ExecutionEnvironment) -> str:
    pattern = args["pattern"]
    path = args.get("path")
    options = {
        "glob_filter": args.get("glob_filter", ""),
        "case_insensitive": args.get("case_insensitive", False),
        "max_results": args.get("max_results", 100),
    }
    return env.grep(pattern, path, options)


def _exec_glob(args: dict, env: ExecutionEnvironment) -> str:
    pattern = args["pattern"]
    path = args.get("path")
    matches = env.glob(pattern, path)
    if not matches:
        return "No files found matching pattern"
    return "\n".join(matches)


def _exec_apply_patch(args: dict, env: ExecutionEnvironment) -> str:
    """Apply a v4a format patch."""
    patch_text = args["patch"]
    return _apply_v4a_patch(patch_text, env)


def _apply_v4a_patch(patch_text: str, env: ExecutionEnvironment) -> str:
    """Parse and apply a v4a patch."""
    lines = patch_text.split("\n")
    if not lines or lines[0].strip() != "*** Begin Patch":
        return "Error: Patch must start with '*** Begin Patch'"

    results = []
    i = 1
    while i < len(lines):
        line = lines[i]

        if line.strip() == "*** End Patch":
            break

        if line.startswith("*** Add File: "):
            path = line[len("*** Add File: "):].strip()
            i += 1
            file_lines = []
            while i < len(lines) and not lines[i].startswith("***"):
                if lines[i].startswith("+"):
                    file_lines.append(lines[i][1:])
                i += 1
            content = "\n".join(file_lines)
            env.write_file(path, content)
            results.append(f"Added: {path}")
            continue

        if line.startswith("*** Delete File: "):
            path = line[len("*** Delete File: "):].strip()
            resolved = path if os.path.isabs(path) else os.path.join(env.working_directory(), path)
            try:
                os.remove(resolved)
                results.append(f"Deleted: {path}")
            except Exception as e:
                results.append(f"Error deleting {path}: {e}")
            i += 1
            continue

        if line.startswith("*** Update File: "):
            path = line[len("*** Update File: "):].strip()
            i += 1
            new_path = None
            if i < len(lines) and lines[i].startswith("*** Move to: "):
                new_path = lines[i][len("*** Move to: "):].strip()
                i += 1

            # Read current file
            resolved = path if os.path.isabs(path) else os.path.join(env.working_directory(), path)
            try:
                with open(resolved) as f:
                    content_lines = f.readlines()
            except FileNotFoundError:
                results.append(f"Error: File not found: {path}")
                # Skip hunks
                while i < len(lines) and not lines[i].startswith("***"):
                    i += 1
                continue

            # Apply hunks
            content_lines, hunk_results = _apply_hunks(content_lines, lines, i)
            i = hunk_results["next_i"]

            new_content = "".join(content_lines)
            dest = new_path or path
            env.write_file(dest, new_content)
            if new_path:
                try:
                    os.remove(resolved)
                except Exception:
                    pass
                results.append(f"Updated and moved: {path} -> {new_path}")
            else:
                results.append(f"Updated: {path}")
            continue

        i += 1

    return "\n".join(results) if results else "Patch applied (no changes)"


def _apply_hunks(content_lines: list[str], patch_lines: list[str], start_i: int) -> tuple[list[str], dict]:
    """Apply @@ hunks to content_lines, return modified lines and next_i."""
    i = start_i
    lines = list(content_lines)

    while i < len(patch_lines):
        line = patch_lines[i]
        if line.startswith("***") or line.strip() == "*** End Patch":
            break
        if line.startswith("@@"):
            context_hint = line[2:].strip()
            i += 1
            # Collect hunk lines
            hunk = []
            while i < len(patch_lines) and not patch_lines[i].startswith("@@") and not patch_lines[i].startswith("***"):
                hunk.append(patch_lines[i])
                i += 1

            # Find context in file
            context_lines = [h[1:] for h in hunk if h.startswith(" ")]
            delete_lines = [h[1:] for h in hunk if h.startswith("-")]
            add_lines = [h[1:] for h in hunk if h.startswith("+")]

            # Find position
            pos = _find_hunk_position(lines, context_lines, delete_lines)
            if pos >= 0:
                # Apply: remove delete_lines, add add_lines at position
                new_lines = []
                j = 0
                applied = False
                search_seq = [h + "\n" for h in delete_lines] if delete_lines else [c + "\n" for c in context_lines[:1]]
                for idx, ln in enumerate(lines):
                    if not applied and idx == pos:
                        # Skip delete lines, add new lines
                        for al in add_lines:
                            new_lines.append(al + "\n")
                        # Keep non-deleted context
                        applied = True
                        # Skip the deleted lines
                        skip = len(delete_lines)
                        for dl in delete_lines:
                            new_lines_check = lines[idx:idx + len(delete_lines)]
                            break
                        # Simple approach: rebuild
                        break
                    new_lines.append(ln)

                # Simpler implementation
                if delete_lines:
                    delete_seq = [d + "\n" for d in delete_lines]
                    # Find and replace
                    for idx in range(len(lines)):
                        match = True
                        for di, dl in enumerate(delete_seq):
                            if idx + di >= len(lines) or lines[idx + di].rstrip("\n") != delete_lines[di]:
                                match = False
                                break
                        if match:
                            lines = lines[:idx] + [al + "\n" for al in add_lines] + lines[idx + len(delete_seq):]
                            break
        else:
            i += 1

    return lines, {"next_i": i}


def _find_hunk_position(lines: list[str], context: list[str], delete: list[str]) -> int:
    """Find position of context/delete lines in file."""
    search = delete or context
    if not search:
        return 0
    for i in range(len(lines)):
        match = True
        for j, s in enumerate(search):
            if i + j >= len(lines) or lines[i + j].rstrip("\n") != s:
                match = False
                break
        if match:
            return i
    return -1


# ---------- Tool definitions ----------

read_file_tool = RegisteredTool(
    name="read_file",
    description="Read a file from the filesystem. Returns line-numbered content.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute or relative path to the file"},
            "offset": {"type": "integer", "description": "1-based line number to start reading from"},
            "limit": {"type": "integer", "description": "Max lines to read (default: 2000)"},
        },
        "required": ["file_path"],
    },
    executor=_exec_read_file,
)

write_file_tool = RegisteredTool(
    name="write_file",
    description="Write content to a file. Creates the file and parent directories if needed.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute or relative path"},
            "content": {"type": "string", "description": "The full file content"},
        },
        "required": ["file_path", "content"],
    },
    executor=_exec_write_file,
)

edit_file_tool = RegisteredTool(
    name="edit_file",
    description=(
        "Replace an exact string occurrence in a file. "
        "old_string must match exactly (including whitespace and newlines). "
        "If old_string is not unique, provide more context or use replace_all=true."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file"},
            "old_string": {"type": "string", "description": "Exact text to find and replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences (default: false)"},
        },
        "required": ["file_path", "old_string", "new_string"],
    },
    executor=_exec_edit_file,
)

shell_tool = RegisteredTool(
    name="shell",
    description="Execute a shell command. Returns stdout, stderr, and exit code.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run"},
            "timeout_ms": {"type": "integer", "description": "Override default timeout in ms"},
            "description": {"type": "string", "description": "Human-readable description"},
        },
        "required": ["command"],
    },
    executor=_exec_shell,
)

grep_tool = RegisteredTool(
    name="grep",
    description="Search file contents using regex patterns.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern"},
            "path": {"type": "string", "description": "Directory or file to search"},
            "glob_filter": {"type": "string", "description": "File pattern filter (e.g., '*.py')"},
            "case_insensitive": {"type": "boolean", "description": "Case insensitive search"},
            "max_results": {"type": "integer", "description": "Max results (default: 100)"},
        },
        "required": ["pattern"],
    },
    executor=_exec_grep,
)

glob_tool = RegisteredTool(
    name="glob",
    description="Find files matching a glob pattern. Returns paths sorted by modification time.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.ts')"},
            "path": {"type": "string", "description": "Base directory (default: working dir)"},
        },
        "required": ["pattern"],
    },
    executor=_exec_glob,
)

apply_patch_tool = RegisteredTool(
    name="apply_patch",
    description=(
        "Apply code changes using the v4a patch format. "
        "Supports creating, deleting, and modifying files in a single operation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "patch": {"type": "string", "description": "The patch content in v4a format"},
        },
        "required": ["patch"],
    },
    executor=_exec_apply_patch,
)

# Tool sets per provider profile
CORE_TOOLS = [read_file_tool, write_file_tool, edit_file_tool, shell_tool, grep_tool, glob_tool]
ANTHROPIC_TOOLS = CORE_TOOLS  # edit_file is native format
OPENAI_TOOLS = [read_file_tool, apply_patch_tool, write_file_tool, shell_tool, grep_tool, glob_tool]
GEMINI_TOOLS = CORE_TOOLS
