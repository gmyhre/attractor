"""Pipeline execution engine (Section 3)."""
from __future__ import annotations
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .types import Graph, Node, Edge, Outcome, StageStatus, SHAPE_TO_TYPE
from .context import Context, Checkpoint
from .conditions import evaluate_condition
from .handlers import Handler, HandlerRegistry, make_default_registry, write_status
from .interviewer import Interviewer, normalize_label
from .stylesheet import apply_stylesheet


# ---------- Retry ----------

@dataclass
class BackoffConfig:
    initial_delay_ms: int = 200
    backoff_factor: float = 2.0
    max_delay_ms: int = 60000
    jitter: bool = True


@dataclass
class RetryPolicy:
    max_attempts: int = 1  # 1 = no retries
    backoff: BackoffConfig = field(default_factory=BackoffConfig)

    def delay_for_attempt(self, attempt: int) -> float:
        """Return delay in seconds for the given retry attempt (1-indexed)."""
        delay_ms = self.backoff.initial_delay_ms * (self.backoff.backoff_factor ** (attempt - 1))
        delay_ms = min(delay_ms, self.backoff.max_delay_ms)
        if self.backoff.jitter:
            delay_ms *= random.uniform(0.5, 1.5)
        return delay_ms / 1000.0


def build_retry_policy(node: Node, graph: Graph) -> RetryPolicy:
    max_retries = node.attrs.max_retries
    if max_retries == 0:
        max_retries = graph.attrs.default_max_retry
    # max_retries is *additional* attempts, so max_attempts = max_retries + 1
    # But use 0 retries if node explicitly set max_retries=0
    if node.attrs.max_retries == 0 and graph.attrs.default_max_retry == 50:
        # Default: no retries for most nodes
        return RetryPolicy(max_attempts=1)
    return RetryPolicy(max_attempts=max(1, node.attrs.max_retries + 1))


# ---------- Events ----------

class PipelineEvent:
    pass


@dataclass
class PipelineStarted(PipelineEvent):
    name: str
    id: str


@dataclass
class PipelineCompleted(PipelineEvent):
    duration: float
    artifact_count: int = 0


@dataclass
class PipelineFailed(PipelineEvent):
    error: str
    duration: float


@dataclass
class StageStarted(PipelineEvent):
    name: str
    index: int


@dataclass
class StageCompleted(PipelineEvent):
    name: str
    index: int
    duration: float


@dataclass
class StageFailed(PipelineEvent):
    name: str
    index: int
    error: str
    will_retry: bool


@dataclass
class CheckpointSaved(PipelineEvent):
    node_id: str


@dataclass
class HumanInteractionRequired(PipelineEvent):
    question: Any
    stage: str


# ---------- Engine ----------

@dataclass
class RunConfig:
    logs_root: str = "attractor-runs"
    run_id: str = ""
    resume: bool = False
    on_event: Callable[[PipelineEvent], None] | None = None
    dry_run: bool = False


def _find_start_node(graph: Graph) -> Node:
    for node in graph.nodes.values():
        if node.attrs.shape == "Mdiamond":
            return node
    for nid in ("start", "Start"):
        if nid in graph.nodes:
            return graph.nodes[nid]
    raise ValueError("No start node found (shape=Mdiamond or id='start'/'Start')")


def _is_terminal(node: Node) -> bool:
    return node.attrs.shape == "Msquare" or node.id in ("exit", "end")


def _check_goal_gates(graph: Graph, node_outcomes: dict[str, Outcome]) -> tuple[bool, Node | None]:
    for node_id, outcome in node_outcomes.items():
        node = graph.nodes.get(node_id)
        if node and node.attrs.goal_gate:
            if outcome.status not in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS):
                return False, node
    return True, None


def _get_retry_target(node: Node, graph: Graph) -> str | None:
    if node.attrs.retry_target:
        return node.attrs.retry_target
    if node.attrs.fallback_retry_target:
        return node.attrs.fallback_retry_target
    if graph.attrs.retry_target:
        return graph.attrs.retry_target
    if graph.attrs.fallback_retry_target:
        return graph.attrs.fallback_retry_target
    return None


def select_edge(node: Node, outcome: Outcome, context: Context, graph: Graph) -> Edge | None:
    """Select the next edge following the 5-step priority order (Section 3.3)."""
    edges = graph.outgoing_edges(node.id)
    if not edges:
        return None

    # Step 1: Condition matching
    condition_matched = []
    for edge in edges:
        if edge.attrs.condition:
            if evaluate_condition(edge.attrs.condition, outcome, context):
                condition_matched.append(edge)

    if condition_matched:
        return _best_by_weight_then_lexical(condition_matched)

    # Step 2: Preferred label match
    if outcome.preferred_label:
        norm_pref = normalize_label(outcome.preferred_label)
        for edge in edges:
            if normalize_label(edge.attrs.label) == norm_pref:
                return edge

    # Step 3: Suggested next IDs
    if outcome.suggested_next_ids:
        for suggested_id in outcome.suggested_next_ids:
            for edge in edges:
                if edge.to_node == suggested_id:
                    return edge

    # Step 4 & 5: Weight with lexical tiebreak (unconditional edges only)
    unconditional = [e for e in edges if not e.attrs.condition]
    if unconditional:
        return _best_by_weight_then_lexical(unconditional)

    # Fallback
    return _best_by_weight_then_lexical(edges)


def _best_by_weight_then_lexical(edges: list[Edge]) -> Edge:
    return sorted(edges, key=lambda e: (-e.attrs.weight, e.to_node))[0]


def execute_with_retry(
    node: Node, context: Context, graph: Graph,
    logs_root: str, registry: HandlerRegistry,
    retry_policy: RetryPolicy,
) -> Outcome:
    handler = registry.resolve(node)
    last_outcome = Outcome(status=StageStatus.FAIL, failure_reason="unknown")

    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            outcome = handler.execute(node, context, graph, logs_root)
        except Exception as e:
            outcome = Outcome(status=StageStatus.FAIL, failure_reason=str(e))

        last_outcome = outcome

        if outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS):
            return outcome

        if outcome.status == StageStatus.RETRY:
            if attempt < retry_policy.max_attempts:
                delay = retry_policy.delay_for_attempt(attempt)
                time.sleep(delay)
                continue
            else:
                if node.attrs.allow_partial:
                    return Outcome(
                        status=StageStatus.PARTIAL_SUCCESS,
                        notes="retries exhausted, partial accepted",
                    )
                return Outcome(status=StageStatus.FAIL, failure_reason="max retries exceeded")

        if outcome.status == StageStatus.FAIL:
            if attempt < retry_policy.max_attempts:
                delay = retry_policy.delay_for_attempt(attempt)
                time.sleep(delay)
                continue
            return outcome

    return last_outcome


def execute_subgraph(
    start_node_id: str, context: Context, graph: Graph, logs_root: str
) -> Outcome:
    """Execute a subgraph starting from a given node (for parallel branches)."""
    from .handlers import make_default_registry
    registry = make_default_registry()
    current_id = start_node_id
    last_outcome = Outcome(status=StageStatus.SUCCESS)

    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        node = graph.nodes.get(current_id)
        if not node:
            break
        if _is_terminal(node):
            break
        retry_policy = build_retry_policy(node, graph)
        outcome = execute_with_retry(node, context, graph, logs_root, registry, retry_policy)
        last_outcome = outcome
        context.apply_updates(outcome.context_updates)
        context.set("outcome", outcome.status.value)
        edge = select_edge(node, outcome, context, graph)
        if not edge:
            break
        current_id = edge.to_node

    return last_outcome


class PipelineRunner:
    """Main pipeline execution engine (Section 3.2)."""

    def __init__(
        self,
        registry: HandlerRegistry | None = None,
        on_event: Callable[[PipelineEvent], None] | None = None,
        transforms: list | None = None,
    ):
        self.registry = registry or make_default_registry()
        self._on_event = on_event
        self._transforms = transforms or []

    def _emit(self, event: PipelineEvent) -> None:
        if self._on_event:
            self._on_event(event)

    def _prepare(self, graph: Graph) -> Graph:
        """Apply transforms (stylesheet, variable expansion)."""
        # Built-in: stylesheet
        apply_stylesheet(graph)
        # Built-in: variable expansion (already done at execution time in CodergenHandler)
        # Custom transforms
        for transform in self._transforms:
            graph = transform.apply(graph)
        return graph

    def run(self, graph: Graph, config: RunConfig | None = None) -> Outcome:
        """Execute the pipeline and return the final outcome."""
        import uuid
        cfg = config or RunConfig()
        run_id = cfg.run_id or uuid.uuid4().hex[:8]
        logs_root = os.path.join(cfg.logs_root, run_id)
        os.makedirs(logs_root, exist_ok=True)

        start_time = time.time()

        # Prepare graph (transforms)
        graph = self._prepare(graph)

        # Initialize context
        context = Context()
        context.set("graph.goal", graph.attrs.goal)
        context.set("run.id", run_id)
        for k, v in graph.attrs.extra.items():
            context.set(f"graph.{k}", v)

        # Write manifest
        manifest = {
            "run_id": run_id,
            "goal": graph.attrs.goal,
            "label": graph.attrs.label,
            "start_time": start_time,
        }
        with open(os.path.join(logs_root, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        # Resume from checkpoint if requested
        checkpoint_path = os.path.join(logs_root, "checkpoint.json")
        completed_nodes: list[str] = []
        node_outcomes: dict[str, Outcome] = {}

        if cfg.resume and os.path.exists(checkpoint_path):
            checkpoint = Checkpoint.load(checkpoint_path)
            context.apply_updates(checkpoint.context_values)
            completed_nodes = checkpoint.completed_nodes[:]

        self._emit(PipelineStarted(name=graph.attrs.label or "pipeline", id=run_id))

        try:
            final_outcome = self._execute_loop(
                graph, context, logs_root, completed_nodes, node_outcomes,
                checkpoint_path, run_id,
            )
        except Exception as e:
            duration = time.time() - start_time
            self._emit(PipelineFailed(error=str(e), duration=duration))
            return Outcome(status=StageStatus.FAIL, failure_reason=str(e))

        duration = time.time() - start_time
        self._emit(PipelineCompleted(duration=duration))
        return final_outcome

    def _execute_loop(
        self,
        graph: Graph,
        context: Context,
        logs_root: str,
        completed_nodes: list[str],
        node_outcomes: dict[str, Outcome],
        checkpoint_path: str,
        run_id: str,
    ) -> Outcome:
        current_node = _find_start_node(graph)
        last_outcome = Outcome(status=StageStatus.SUCCESS)
        stage_index = 0

        while True:
            node = current_node
            context.set("current_node", node.id)

            # Step 1: Check terminal
            if _is_terminal(node):
                gate_ok, failed_gate = _check_goal_gates(graph, node_outcomes)
                if not gate_ok and failed_gate:
                    retry_target = _get_retry_target(failed_gate, graph)
                    if retry_target and retry_target in graph.nodes:
                        current_node = graph.nodes[retry_target]
                        continue
                    else:
                        return Outcome(
                            status=StageStatus.FAIL,
                            failure_reason=f"Goal gate unsatisfied: {failed_gate.id} and no retry target",
                        )
                # All goal gates satisfied, pipeline complete
                handler = self.registry.resolve(node)
                handler.execute(node, context, graph, logs_root)
                break

            # Step 2: Execute with retry
            self._emit(StageStarted(name=node.id, index=stage_index))
            stage_start = time.time()

            retry_policy = build_retry_policy(node, graph)
            outcome = execute_with_retry(node, context, graph, logs_root, self.registry, retry_policy)

            stage_duration = time.time() - stage_start

            # Step 3: Record
            completed_nodes.append(node.id)
            node_outcomes[node.id] = outcome

            if outcome.status == StageStatus.FAIL:
                self._emit(StageFailed(name=node.id, index=stage_index,
                                        error=outcome.failure_reason, will_retry=False))
            else:
                self._emit(StageCompleted(name=node.id, index=stage_index, duration=stage_duration))

            # Step 4: Apply context updates
            context.apply_updates(outcome.context_updates)
            context.set("outcome", outcome.status.value)
            if outcome.preferred_label:
                context.set("preferred_label", outcome.preferred_label)
            context.set("last_stage", node.id)

            # Step 5: Save checkpoint
            checkpoint = Checkpoint(
                current_node=node.id,
                completed_nodes=completed_nodes[:],
                context_values=context.snapshot(),
                logs=context.logs,
            )
            checkpoint.save(checkpoint_path)
            self._emit(CheckpointSaved(node_id=node.id))

            # Step 6: Select next edge
            next_edge = select_edge(node, outcome, context, graph)
            if next_edge is None:
                if outcome.status == StageStatus.FAIL:
                    # Try failure routing
                    fail_target = node.attrs.retry_target or node.attrs.fallback_retry_target
                    if fail_target and fail_target in graph.nodes:
                        current_node = graph.nodes[fail_target]
                        stage_index += 1
                        continue
                    raise RuntimeError(
                        f"Stage {node.id!r} failed with no outgoing edge. "
                        f"Reason: {outcome.failure_reason}"
                    )
                break  # No more edges, done

            # Step 7: Handle loop_restart
            if next_edge.attrs.loop_restart:
                # Re-launch with fresh logs
                new_config = RunConfig(logs_root=os.path.dirname(logs_root))
                return self.run(graph, new_config)

            # Step 8: Advance
            current_node = graph.nodes[next_edge.to_node]
            last_outcome = outcome
            stage_index += 1

        return last_outcome
