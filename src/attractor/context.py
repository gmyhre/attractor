"""Context, Checkpoint, and ArtifactStore (Sections 5.1-5.5)."""
from __future__ import annotations
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any


class Context:
    """Thread-safe key-value store for pipeline run state."""

    def __init__(self):
        self._values: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._logs: list[str] = []

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._values.get(key, default)

    def get_string(self, key: str, default: str = "") -> str:
        val = self.get(key)
        if val is None:
            return default
        return str(val)

    def append_log(self, entry: str) -> None:
        with self._lock:
            self._logs.append(entry)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def clone(self) -> "Context":
        with self._lock:
            new = Context()
            new._values = dict(self._values)
            new._logs = list(self._logs)
        return new

    def apply_updates(self, updates: dict[str, Any]) -> None:
        with self._lock:
            for k, v in updates.items():
                self._values[k] = v

    @property
    def logs(self) -> list[str]:
        with self._lock:
            return list(self._logs)


@dataclass
class Checkpoint:
    timestamp: float = field(default_factory=time.time)
    current_node: str = ""
    completed_nodes: list[str] = field(default_factory=list)
    node_retries: dict[str, int] = field(default_factory=dict)
    context_values: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        data = {
            "timestamp": self.timestamp,
            "current_node": self.current_node,
            "completed_nodes": self.completed_nodes,
            "node_retries": self.node_retries,
            "context": self.context_values,
            "logs": self.logs,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        with open(path) as f:
            data = json.load(f)
        return cls(
            timestamp=data.get("timestamp", 0.0),
            current_node=data.get("current_node", ""),
            completed_nodes=data.get("completed_nodes", []),
            node_retries=data.get("node_retries", {}),
            context_values=data.get("context", {}),
            logs=data.get("logs", []),
        )


@dataclass
class ArtifactInfo:
    id: str
    name: str
    size_bytes: int
    stored_at: float
    is_file_backed: bool


FILE_BACKING_THRESHOLD = 100 * 1024  # 100KB


class ArtifactStore:
    def __init__(self, base_dir: str | None = None):
        self._artifacts: dict[str, tuple[ArtifactInfo, Any]] = {}
        self._lock = threading.RLock()
        self.base_dir = base_dir

    def store(self, artifact_id: str, name: str, data: Any) -> ArtifactInfo:
        import pickle
        size = len(pickle.dumps(data)) if not isinstance(data, (str, bytes)) else len(data)
        if isinstance(data, str):
            size = len(data.encode())

        is_file_backed = size > FILE_BACKING_THRESHOLD and self.base_dir is not None
        stored_data = data
        if is_file_backed and self.base_dir:
            path = os.path.join(self.base_dir, "artifacts", f"{artifact_id}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, default=str)
            stored_data = path

        info = ArtifactInfo(
            id=artifact_id, name=name, size_bytes=size,
            stored_at=time.time(), is_file_backed=is_file_backed,
        )
        with self._lock:
            self._artifacts[artifact_id] = (info, stored_data)
        return info

    def retrieve(self, artifact_id: str) -> Any:
        with self._lock:
            if artifact_id not in self._artifacts:
                raise KeyError(f"Artifact {artifact_id!r} not found")
            info, data = self._artifacts[artifact_id]
        if info.is_file_backed:
            with open(data) as f:
                return json.load(f)
        return data

    def has(self, artifact_id: str) -> bool:
        with self._lock:
            return artifact_id in self._artifacts

    def list(self) -> list[ArtifactInfo]:
        with self._lock:
            return [info for info, _ in self._artifacts.values()]

    def remove(self, artifact_id: str) -> None:
        with self._lock:
            self._artifacts.pop(artifact_id, None)

    def clear(self) -> None:
        with self._lock:
            self._artifacts.clear()
