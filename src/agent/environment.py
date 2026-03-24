"""Execution Environment abstraction (Section 4 of coding-agent-loop-spec.md)."""
from __future__ import annotations
import fnmatch
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: int


@dataclass
class DirEntry:
    name: str
    is_dir: bool
    size: int | None


class ExecutionEnvironment(Protocol):
    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str: ...
    def write_file(self, path: str, content: str) -> None: ...
    def file_exists(self, path: str) -> bool: ...
    def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]: ...
    def exec_command(self, command: str, timeout_ms: int = 10000,
                      working_dir: str | None = None,
                      env_vars: dict | None = None) -> ExecResult: ...
    def grep(self, pattern: str, path: str, options: dict | None = None) -> str: ...
    def glob(self, pattern: str, path: str | None = None) -> list[str]: ...
    def working_directory(self) -> str: ...
    def platform(self) -> str: ...
    def os_version(self) -> str: ...
    def initialize(self) -> None: ...
    def cleanup(self) -> None: ...


# Environment variables to exclude (security)
_SENSITIVE_PATTERNS = [
    r".*_API_KEY$", r".*_SECRET$", r".*_TOKEN$", r".*_PASSWORD$", r".*_CREDENTIAL$",
    r".*_PRIVATE_KEY$", r".*_ACCESS_KEY.*",
]
_ALWAYS_INCLUDE = {
    "PATH", "HOME", "USER", "SHELL", "LANG", "TERM", "TMPDIR",
    "GOPATH", "GOROOT", "CARGO_HOME", "NVM_DIR", "PYENV_ROOT",
    "NODE_PATH", "npm_config_prefix", "JAVA_HOME",
}


def _filter_env(env: dict[str, str]) -> dict[str, str]:
    result = {}
    for key, val in env.items():
        if key in _ALWAYS_INCLUDE:
            result[key] = val
            continue
        sensitive = any(re.match(p, key, re.IGNORECASE) for p in _SENSITIVE_PATTERNS)
        if not sensitive:
            result[key] = val
    return result


class LocalExecutionEnvironment:
    """Default execution environment - runs on local machine."""

    def __init__(self, working_dir: str | None = None):
        self._working_dir = working_dir or os.getcwd()

    def _resolve(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self._working_dir, path)

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        resolved = self._resolve(path)
        with open(resolved, "r", errors="replace") as f:
            lines = f.readlines()
        start = (offset - 1) if offset else 0
        start = max(0, start)
        if limit:
            lines = lines[start:start + limit]
        else:
            lines = lines[start:]
        # Add line numbers
        base = start + 1
        return "".join(f"{base + i:4d} | {line}" for i, line in enumerate(lines))

    def write_file(self, path: str, content: str) -> None:
        resolved = self._resolve(path)
        os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
        with open(resolved, "w") as f:
            f.write(content)

    def file_exists(self, path: str) -> bool:
        return os.path.exists(self._resolve(path))

    def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]:
        resolved = self._resolve(path)
        if not os.path.isdir(resolved):
            return []
        entries = []
        for name in sorted(os.listdir(resolved)):
            full = os.path.join(resolved, name)
            is_dir = os.path.isdir(full)
            size = None if is_dir else os.path.getsize(full)
            entries.append(DirEntry(name=name, is_dir=is_dir, size=size))
        return entries

    def exec_command(self, command: str, timeout_ms: int = 10000,
                      working_dir: str | None = None,
                      env_vars: dict | None = None) -> ExecResult:
        cwd = working_dir or self._working_dir
        env = _filter_env(dict(os.environ))
        if env_vars:
            env.update(env_vars)

        timeout_s = timeout_ms / 1000.0
        start = time.time()
        timed_out = False

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,  # new process group for clean killability
            )

            try:
                stdout, stderr = proc.communicate(timeout=timeout_s)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                # SIGTERM first
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
                time.sleep(2)
                # SIGKILL if still running
                try:
                    if proc.poll() is None:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
                stdout, stderr = proc.communicate()
                exit_code = -1

        except Exception as e:
            return ExecResult(
                stdout="", stderr=str(e), exit_code=1,
                timed_out=False,
                duration_ms=int((time.time() - start) * 1000),
            )

        duration_ms = int((time.time() - start) * 1000)
        if timed_out:
            timeout_msg = (
                f"\n[ERROR: Command timed out after {timeout_ms}ms. "
                f"Partial output is shown above.\n"
                f"You can retry with a longer timeout by setting the timeout_ms parameter.]"
            )
            stdout += timeout_msg

        return ExecResult(
            stdout=stdout, stderr=stderr, exit_code=exit_code,
            timed_out=timed_out, duration_ms=duration_ms,
        )

    def grep(self, pattern: str, path: str | None = None, options: dict | None = None) -> str:
        opts = options or {}
        search_path = self._resolve(path or ".")
        case_flag = "-i" if opts.get("case_insensitive") else ""
        glob_filter = opts.get("glob_filter", "")
        max_results = opts.get("max_results", 100)

        # Try ripgrep first
        try:
            cmd_parts = ["rg", "--line-number", "--no-heading"]
            if case_flag:
                cmd_parts.append("-i")
            if glob_filter:
                cmd_parts.extend(["-g", glob_filter])
            cmd_parts.extend(["-m", str(max_results)])
            cmd_parts.extend([pattern, search_path])
            result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=30)
            return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback to Python regex
        results = []
        for root, dirs, files in os.walk(search_path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                if glob_filter and not fnmatch.fnmatch(fname, glob_filter):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            flags = re.IGNORECASE if opts.get("case_insensitive") else 0
                            if re.search(pattern, line, flags):
                                results.append(f"{fpath}:{i}:{line.rstrip()}")
                                if len(results) >= max_results:
                                    return "\n".join(results)
                except Exception:
                    pass
        return "\n".join(results)

    def glob(self, pattern: str, path: str | None = None) -> list[str]:
        import glob as glob_mod
        base = self._resolve(path or ".")
        full_pattern = os.path.join(base, pattern)
        matches = glob_mod.glob(full_pattern, recursive=True)
        # Sort by modification time (newest first)
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return [os.path.relpath(m, self._working_dir) for m in matches]

    def working_directory(self) -> str:
        return self._working_dir

    def platform(self) -> str:
        return sys.platform.replace("darwin", "darwin").replace("win32", "windows").replace("linux", "linux")

    def os_version(self) -> str:
        try:
            result = subprocess.run(["uname", "-r"], capture_output=True, text=True)
            return result.stdout.strip()
        except Exception:
            return ""

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass
