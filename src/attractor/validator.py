"""Validation and linting (Section 7)."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from .types import Graph, SHAPE_TO_TYPE
from .conditions import evaluate_condition
from .context import Context


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Diagnostic:
    rule: str
    severity: Severity
    message: str
    node_id: str = ""
    edge: tuple[str, str] | None = None
    fix: str = ""


class ValidationError(Exception):
    def __init__(self, diagnostics: list[Diagnostic]):
        messages = [f"[{d.rule}] {d.message}" for d in diagnostics]
        super().__init__("\n".join(messages))
        self.diagnostics = diagnostics


def validate(graph: Graph, extra_rules: list | None = None) -> list[Diagnostic]:
    rules = _BUILT_IN_RULES[:]
    if extra_rules:
        rules.extend(extra_rules)
    diagnostics = []
    for rule in rules:
        diagnostics.extend(rule(graph))
    return diagnostics


def validate_or_raise(graph: Graph, extra_rules: list | None = None) -> list[Diagnostic]:
    diagnostics = validate(graph, extra_rules)
    errors = [d for d in diagnostics if d.severity == Severity.ERROR]
    if errors:
        raise ValidationError(errors)
    return diagnostics


# ---------- Built-in lint rules ----------

def _rule_start_node(graph: Graph) -> list[Diagnostic]:
    starts = [n for n in graph.nodes.values()
               if n.attrs.shape == "Mdiamond" or n.id in ("start", "Start")]
    if len(starts) == 0:
        return [Diagnostic(
            rule="start_node", severity=Severity.ERROR,
            message="Pipeline must have exactly one start node (shape=Mdiamond)",
            fix="Add a node with shape=Mdiamond",
        )]
    if len(starts) > 1:
        return [Diagnostic(
            rule="start_node", severity=Severity.ERROR,
            message=f"Pipeline has {len(starts)} start nodes; exactly one required",
        )]
    return []


def _rule_terminal_node(graph: Graph) -> list[Diagnostic]:
    exits = [n for n in graph.nodes.values()
              if n.attrs.shape == "Msquare" or n.id in ("exit", "end")]
    if len(exits) == 0:
        return [Diagnostic(
            rule="terminal_node", severity=Severity.ERROR,
            message="Pipeline must have at least one exit node (shape=Msquare)",
            fix="Add a node with shape=Msquare",
        )]
    return []


def _rule_reachability(graph: Graph) -> list[Diagnostic]:
    start_nodes = [n.id for n in graph.nodes.values()
                   if n.attrs.shape == "Mdiamond" or n.id in ("start", "Start")]
    if not start_nodes:
        return []
    start = start_nodes[0]
    visited = set()
    queue = [start]
    while queue:
        nid = queue.pop()
        if nid in visited:
            continue
        visited.add(nid)
        for edge in graph.outgoing_edges(nid):
            queue.append(edge.to_node)

    unreachable = set(graph.nodes.keys()) - visited
    return [
        Diagnostic(
            rule="reachability", severity=Severity.ERROR,
            message=f"Node {nid!r} is not reachable from start",
            node_id=nid,
        )
        for nid in unreachable
    ]


def _rule_edge_targets_exist(graph: Graph) -> list[Diagnostic]:
    diags = []
    for edge in graph.edges:
        if edge.to_node not in graph.nodes:
            diags.append(Diagnostic(
                rule="edge_target_exists", severity=Severity.ERROR,
                message=f"Edge target {edge.to_node!r} does not exist",
                edge=(edge.from_node, edge.to_node),
            ))
        if edge.from_node not in graph.nodes:
            diags.append(Diagnostic(
                rule="edge_target_exists", severity=Severity.ERROR,
                message=f"Edge source {edge.from_node!r} does not exist",
                edge=(edge.from_node, edge.to_node),
            ))
    return diags


def _rule_start_no_incoming(graph: Graph) -> list[Diagnostic]:
    starts = [n.id for n in graph.nodes.values()
               if n.attrs.shape == "Mdiamond" or n.id in ("start", "Start")]
    diags = []
    for start in starts:
        incoming = graph.incoming_edges(start)
        if incoming:
            diags.append(Diagnostic(
                rule="start_no_incoming", severity=Severity.ERROR,
                message=f"Start node {start!r} must have no incoming edges",
                node_id=start,
            ))
    return diags


def _rule_exit_no_outgoing(graph: Graph) -> list[Diagnostic]:
    exits = [n.id for n in graph.nodes.values()
              if n.attrs.shape == "Msquare" or n.id in ("exit", "end")]
    diags = []
    for exit_id in exits:
        outgoing = graph.outgoing_edges(exit_id)
        if outgoing:
            diags.append(Diagnostic(
                rule="exit_no_outgoing", severity=Severity.ERROR,
                message=f"Exit node {exit_id!r} must have no outgoing edges",
                node_id=exit_id,
            ))
    return diags


def _rule_condition_syntax(graph: Graph) -> list[Diagnostic]:
    diags = []
    dummy_outcome_class = type("Outcome", (), {"status": type("Status", (), {"value": "success"})(), "preferred_label": ""})()
    dummy_ctx = Context()
    for edge in graph.edges:
        cond = edge.attrs.condition
        if not cond:
            continue
        try:
            evaluate_condition(cond, dummy_outcome_class, dummy_ctx)  # type: ignore
        except Exception as e:
            diags.append(Diagnostic(
                rule="condition_syntax", severity=Severity.ERROR,
                message=f"Invalid condition on edge {edge.from_node}->{edge.to_node}: {e}",
                edge=(edge.from_node, edge.to_node),
            ))
    return diags


def _rule_type_known(graph: Graph) -> list[Diagnostic]:
    known_types = set(SHAPE_TO_TYPE.values()) | {"codergen", "start", "exit",
                                                   "wait.human", "conditional",
                                                   "parallel", "parallel.fan_in",
                                                   "tool", "stack.manager_loop"}
    diags = []
    for node in graph.nodes.values():
        t = node.attrs.type
        if t and t not in known_types:
            diags.append(Diagnostic(
                rule="type_known", severity=Severity.WARNING,
                message=f"Node {node.id!r} has unrecognized type {t!r}",
                node_id=node.id,
            ))
    return diags


def _rule_fidelity_valid(graph: Graph) -> list[Diagnostic]:
    valid = {"full", "truncate", "compact", "summary:low", "summary:medium", "summary:high", ""}
    diags = []
    for node in graph.nodes.values():
        f = node.attrs.fidelity
        if f and f not in valid:
            diags.append(Diagnostic(
                rule="fidelity_valid", severity=Severity.WARNING,
                message=f"Node {node.id!r} has invalid fidelity {f!r}",
                node_id=node.id,
            ))
    return diags


def _rule_retry_target_exists(graph: Graph) -> list[Diagnostic]:
    diags = []
    for node in graph.nodes.values():
        for attr in ("retry_target", "fallback_retry_target"):
            target = getattr(node.attrs, attr, "")
            if target and target not in graph.nodes:
                diags.append(Diagnostic(
                    rule="retry_target_exists", severity=Severity.WARNING,
                    message=f"Node {node.id!r} {attr}={target!r} does not exist",
                    node_id=node.id,
                ))
    return diags


def _rule_goal_gate_has_retry(graph: Graph) -> list[Diagnostic]:
    diags = []
    for node in graph.nodes.values():
        if node.attrs.goal_gate:
            if not node.attrs.retry_target and not node.attrs.fallback_retry_target:
                if not graph.attrs.retry_target and not graph.attrs.fallback_retry_target:
                    diags.append(Diagnostic(
                        rule="goal_gate_has_retry", severity=Severity.WARNING,
                        message=f"Node {node.id!r} has goal_gate=true but no retry_target configured",
                        node_id=node.id,
                        fix="Set retry_target or fallback_retry_target on the node or graph",
                    ))
    return diags


def _rule_prompt_on_llm_nodes(graph: Graph) -> list[Diagnostic]:
    diags = []
    for node in graph.nodes.values():
        shape = node.attrs.shape
        node_type = node.attrs.type
        # Codergen nodes need a prompt
        if (shape == "box" or node_type == "codergen") and shape not in ("Mdiamond", "Msquare"):
            if not node.attrs.prompt and not node.attrs.label:
                diags.append(Diagnostic(
                    rule="prompt_on_llm_nodes", severity=Severity.WARNING,
                    message=f"LLM node {node.id!r} has no prompt or label",
                    node_id=node.id,
                    fix="Add a prompt or label attribute",
                ))
    return diags


_BUILT_IN_RULES = [
    _rule_start_node,
    _rule_terminal_node,
    _rule_reachability,
    _rule_edge_targets_exist,
    _rule_start_no_incoming,
    _rule_exit_no_outgoing,
    _rule_condition_syntax,
    _rule_type_known,
    _rule_fidelity_valid,
    _rule_retry_target_exists,
    _rule_goal_gate_has_retry,
    _rule_prompt_on_llm_nodes,
]
