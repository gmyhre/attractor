"""Attractor core data types."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    RETRY = "retry"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass
class Outcome:
    status: StageStatus
    preferred_label: str = ""
    suggested_next_ids: list[str] = field(default_factory=list)
    context_updates: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    failure_reason: str = ""

    @classmethod
    def success(cls, notes: str = "", **context_updates) -> "Outcome":
        return cls(status=StageStatus.SUCCESS, notes=notes, context_updates=context_updates)

    @classmethod
    def fail(cls, reason: str) -> "Outcome":
        return cls(status=StageStatus.FAIL, failure_reason=reason)

    @classmethod
    def retry(cls, reason: str) -> "Outcome":
        return cls(status=StageStatus.RETRY, failure_reason=reason)


@dataclass
class NodeAttrs:
    """Parsed attributes for a graph node."""
    label: str = ""
    shape: str = "box"
    type: str = ""
    prompt: str = ""
    max_retries: int = 0
    goal_gate: bool = False
    retry_target: str = ""
    fallback_retry_target: str = ""
    fidelity: str = ""
    thread_id: str = ""
    cls: str = ""  # 'class' attribute (renamed to avoid keyword)
    timeout: str = ""
    llm_model: str = ""
    llm_provider: str = ""
    reasoning_effort: str = "high"
    auto_status: bool = False
    allow_partial: bool = False
    # Extra/custom attributes
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            if val != "" and val is not None:
                return val
        return self.extra.get(key, default)


@dataclass
class Node:
    id: str
    attrs: NodeAttrs = field(default_factory=NodeAttrs)

    @property
    def label(self) -> str:
        return self.attrs.label or self.id

    @property
    def shape(self) -> str:
        return self.attrs.shape

    @property
    def type(self) -> str:
        return self.attrs.type


@dataclass
class EdgeAttrs:
    label: str = ""
    condition: str = ""
    weight: int = 0
    fidelity: str = ""
    thread_id: str = ""
    loop_restart: bool = False


@dataclass
class Edge:
    from_node: str
    to_node: str
    attrs: EdgeAttrs = field(default_factory=EdgeAttrs)


@dataclass
class GraphAttrs:
    goal: str = ""
    label: str = ""
    model_stylesheet: str = ""
    default_max_retry: int = 50
    retry_target: str = ""
    fallback_retry_target: str = ""
    default_fidelity: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    attrs: GraphAttrs = field(default_factory=GraphAttrs)
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.from_node == node_id]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.to_node == node_id]

    @property
    def goal(self) -> str:
        return self.attrs.goal


# Shape -> handler type mapping
SHAPE_TO_TYPE: dict[str, str] = {
    "Mdiamond": "start",
    "Msquare": "exit",
    "box": "codergen",
    "hexagon": "wait.human",
    "diamond": "conditional",
    "component": "parallel",
    "tripleoctagon": "parallel.fan_in",
    "parallelogram": "tool",
    "house": "stack.manager_loop",
}
