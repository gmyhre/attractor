"""Node handlers (Section 4)."""
from __future__ import annotations
import json
import os
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Protocol

from .types import Outcome, StageStatus, Node, Graph, SHAPE_TO_TYPE
from .context import Context
from .interviewer import (
    Interviewer, Question, QuestionType, Option, Answer, AnswerValue,
    parse_accelerator_key, normalize_label,
)


class Handler(Protocol):
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        ...


# ---------- Status file helpers ----------

def write_status(stage_dir: str, outcome: Outcome) -> None:
    os.makedirs(stage_dir, exist_ok=True)
    data = {
        "outcome": outcome.status.value,
        "preferred_next_label": outcome.preferred_label,
        "suggested_next_ids": outcome.suggested_next_ids,
        "context_updates": outcome.context_updates,
        "notes": outcome.notes,
        "failure_reason": outcome.failure_reason,
    }
    with open(os.path.join(stage_dir, "status.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)


def read_status(stage_dir: str) -> Outcome | None:
    path = os.path.join(stage_dir, "status.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    status_map = {
        "success": StageStatus.SUCCESS,
        "partial_success": StageStatus.PARTIAL_SUCCESS,
        "retry": StageStatus.RETRY,
        "fail": StageStatus.FAIL,
        "skipped": StageStatus.SKIPPED,
    }
    return Outcome(
        status=status_map.get(data.get("outcome", "success"), StageStatus.SUCCESS),
        preferred_label=data.get("preferred_next_label", ""),
        suggested_next_ids=data.get("suggested_next_ids", []),
        context_updates=data.get("context_updates", {}),
        notes=data.get("notes", ""),
        failure_reason=data.get("failure_reason", ""),
    )


# ---------- Start / Exit ----------

class StartHandler:
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS, notes="Pipeline started")


class ExitHandler:
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS, notes="Pipeline complete")


# ---------- Codergen ----------

class CodergenBackend(Protocol):
    def run(self, node: Node, prompt: str, context: Context) -> "str | Outcome":
        ...


class SimulationBackend:
    """No-op backend for testing without LLM."""
    def run(self, node: Node, prompt: str, context: Context) -> str:
        return f"[Simulated] Response for stage: {node.id}"


class CodergenHandler:
    def __init__(self, backend: CodergenBackend | None = None):
        self.backend = backend

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        # 1. Build prompt
        prompt = node.attrs.prompt or node.attrs.label or node.id
        prompt = prompt.replace("$goal", graph.attrs.goal)

        # 2. Write prompt to logs
        stage_dir = os.path.join(logs_root, node.id)
        os.makedirs(stage_dir, exist_ok=True)
        with open(os.path.join(stage_dir, "prompt.md"), "w") as f:
            f.write(prompt)

        # 3. Call backend
        if self.backend is not None:
            try:
                result = self.backend.run(node, prompt, context)
                if isinstance(result, Outcome):
                    write_status(stage_dir, result)
                    return result
                response_text = str(result)
            except Exception as e:
                err_outcome = Outcome(status=StageStatus.FAIL, failure_reason=str(e))
                write_status(stage_dir, err_outcome)
                return err_outcome
        else:
            response_text = f"[Simulated] Response for stage: {node.id}"

        # 4. Write response
        with open(os.path.join(stage_dir, "response.md"), "w") as f:
            f.write(response_text)

        # 5. Check for external status.json (status-file contract)
        existing = read_status(stage_dir)
        if existing:
            return existing

        # 6. Build outcome
        truncated = response_text[:200] if len(response_text) > 200 else response_text
        outcome = Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Stage completed: {node.id}",
            context_updates={
                "last_stage": node.id,
                "last_response": truncated,
            },
        )
        write_status(stage_dir, outcome)
        return outcome


# ---------- Wait for Human ----------

class WaitForHumanHandler:
    def __init__(self, interviewer: Interviewer | None = None):
        self.interviewer = interviewer or SimulationInterviewer()

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        edges = graph.outgoing_edges(node.id)
        if not edges:
            return Outcome(status=StageStatus.FAIL, failure_reason="No outgoing edges for human gate")

        # Build choices from edges
        choices = []
        for edge in edges:
            label = edge.attrs.label or edge.to_node
            key = parse_accelerator_key(label)
            choices.append({"key": key, "label": label, "to": edge.to_node})

        options = [Option(key=c["key"], label=c["label"]) for c in choices]
        question = Question(
            text=node.attrs.label or "Select an option:",
            type=QuestionType.MULTIPLE_CHOICE,
            options=options,
            stage=node.id,
        )

        answer = self.interviewer.ask(question)

        if isinstance(answer.value, AnswerValue) and answer.value == AnswerValue.TIMEOUT:
            default_choice = node.attrs.extra.get("human.default_choice")
            if default_choice:
                for c in choices:
                    if c["to"] == default_choice or c["key"] == default_choice:
                        return Outcome(
                            status=StageStatus.SUCCESS,
                            suggested_next_ids=[c["to"]],
                            context_updates={"human.gate.selected": c["key"], "human.gate.label": c["label"]},
                        )
            return Outcome(status=StageStatus.RETRY, failure_reason="human gate timeout, no default")

        if isinstance(answer.value, AnswerValue) and answer.value == AnswerValue.SKIPPED:
            return Outcome(status=StageStatus.FAIL, failure_reason="human skipped interaction")

        # Find matching choice
        selected = None
        if answer.selected_option:
            for c in choices:
                if c["key"] == answer.selected_option.key:
                    selected = c
                    break
        if not selected:
            response_str = str(answer.value)
            for c in choices:
                if c["key"].upper() == response_str.upper():
                    selected = c
                    break
        if not selected and choices:
            selected = choices[0]

        if not selected:
            return Outcome(status=StageStatus.FAIL, failure_reason="No matching choice found")

        return Outcome(
            status=StageStatus.SUCCESS,
            suggested_next_ids=[selected["to"]],
            context_updates={
                "human.gate.selected": selected["key"],
                "human.gate.label": selected["label"],
            },
        )


class SimulationInterviewer:
    """Used when no interviewer is configured."""
    def ask(self, question: Question) -> Answer:
        if question.options:
            opt = question.options[0]
            return Answer(value=opt.key, selected_option=opt)
        return Answer(value="auto")


# ---------- Conditional ----------

class ConditionalHandler:
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS, notes=f"Conditional node evaluated: {node.id}")


# ---------- Parallel ----------

class ParallelHandler:
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        branches = graph.outgoing_edges(node.id)
        join_policy = node.attrs.extra.get("join_policy", "wait_all")
        error_policy = node.attrs.extra.get("error_policy", "continue")
        max_parallel = int(node.attrs.extra.get("max_parallel", "4"))

        results = []
        # Import engine locally to avoid circular import
        from .engine import execute_subgraph
        with ThreadPoolExecutor(max_workers=min(max_parallel, len(branches) or 1)) as executor:
            futures = {
                executor.submit(execute_subgraph, branch.to_node, context.clone(), graph, logs_root): branch
                for branch in branches
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(Outcome(status=StageStatus.FAIL, failure_reason=str(e)))
                    if error_policy == "fail_fast":
                        break

        success_count = sum(1 for r in results if r.status == StageStatus.SUCCESS)
        fail_count = sum(1 for r in results if r.status == StageStatus.FAIL)

        serialized = [{"status": r.status.value, "notes": r.notes} for r in results]
        context.set("parallel.results", serialized)

        if join_policy == "wait_all":
            status = StageStatus.SUCCESS if fail_count == 0 else StageStatus.PARTIAL_SUCCESS
        elif join_policy == "first_success":
            status = StageStatus.SUCCESS if success_count > 0 else StageStatus.FAIL
        else:
            status = StageStatus.SUCCESS if fail_count == 0 else StageStatus.PARTIAL_SUCCESS

        return Outcome(
            status=status,
            notes=f"Parallel: {success_count} success, {fail_count} fail",
            context_updates={"parallel.results": serialized},
        )


# ---------- Fan-In ----------

class FanInHandler:
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        results = context.get("parallel.results", [])
        if not results:
            return Outcome(status=StageStatus.FAIL, failure_reason="No parallel results to evaluate")

        # Heuristic selection
        status_rank = {
            "success": 0, "partial_success": 1, "retry": 2, "fail": 3
        }
        best = min(results, key=lambda r: status_rank.get(r.get("status", "fail"), 3))

        context_updates = {
            "parallel.fan_in.best_id": best.get("id", ""),
            "parallel.fan_in.best_outcome": best.get("status", ""),
        }

        return Outcome(
            status=StageStatus.SUCCESS,
            context_updates=context_updates,
            notes=f"Selected best candidate: {best.get('status', '')}",
        )


# ---------- Tool ----------

class ToolHandler:
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        command = node.attrs.extra.get("tool_command", "")
        if not command:
            return Outcome(status=StageStatus.FAIL, failure_reason="No tool_command specified")

        timeout_str = node.attrs.timeout
        timeout_s = _parse_duration_s(timeout_str) if timeout_str else 30.0

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            output = result.stdout + result.stderr
            if result.returncode != 0:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Command exited {result.returncode}: {output[:500]}",
                )
            return Outcome(
                status=StageStatus.SUCCESS,
                context_updates={"tool.output": output},
                notes=f"Tool completed: {command[:100]}",
            )
        except subprocess.TimeoutExpired as e:
            return Outcome(status=StageStatus.FAIL, failure_reason=f"Tool timed out: {e}")
        except Exception as e:
            return Outcome(status=StageStatus.FAIL, failure_reason=str(e))


# ---------- Manager Loop ----------

class ManagerLoopHandler:
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        poll_interval_str = node.attrs.extra.get("manager.poll_interval", "45s")
        max_cycles = int(node.attrs.extra.get("manager.max_cycles", "1000"))
        stop_condition = node.attrs.extra.get("manager.stop_condition", "")
        actions = [a.strip() for a in node.attrs.extra.get("manager.actions", "observe,wait").split(",")]

        from .conditions import evaluate_condition
        for cycle in range(max_cycles):
            child_status = context.get_string("context.stack.child.status")
            if child_status in ("completed", "failed"):
                child_outcome = context.get_string("context.stack.child.outcome")
                if child_outcome == "success":
                    return Outcome(status=StageStatus.SUCCESS, notes="Child completed")
                return Outcome(status=StageStatus.FAIL, failure_reason="Child failed")

            if stop_condition:
                dummy_outcome = Outcome(status=StageStatus.SUCCESS)
                if evaluate_condition(stop_condition, dummy_outcome, context):
                    return Outcome(status=StageStatus.SUCCESS, notes="Stop condition satisfied")

            if "wait" in actions:
                interval = _parse_duration_s(poll_interval_str)
                time.sleep(interval)

        return Outcome(status=StageStatus.FAIL, failure_reason="Max cycles exceeded")


# ---------- Handler Registry ----------

class HandlerRegistry:
    def __init__(self, default_handler: Handler | None = None):
        self._handlers: dict[str, Handler] = {}
        self.default_handler = default_handler or CodergenHandler()

    def register(self, type_string: str, handler: Handler) -> None:
        self._handlers[type_string] = handler

    def resolve(self, node: Node) -> Handler:
        # 1. Explicit type attribute
        if node.attrs.type and node.attrs.type in self._handlers:
            return self._handlers[node.attrs.type]
        # 2. Shape-based resolution
        handler_type = SHAPE_TO_TYPE.get(node.attrs.shape, "codergen")
        if handler_type in self._handlers:
            return self._handlers[handler_type]
        # 3. Default
        return self.default_handler


def make_default_registry(
    backend: CodergenBackend | None = None,
    interviewer: Interviewer | None = None,
) -> HandlerRegistry:
    codergen = CodergenHandler(backend=backend)
    registry = HandlerRegistry(default_handler=codergen)
    registry.register("start", StartHandler())
    registry.register("exit", ExitHandler())
    registry.register("codergen", codergen)
    registry.register("wait.human", WaitForHumanHandler(interviewer=interviewer))
    registry.register("conditional", ConditionalHandler())
    registry.register("parallel", ParallelHandler())
    registry.register("parallel.fan_in", FanInHandler())
    registry.register("tool", ToolHandler())
    registry.register("stack.manager_loop", ManagerLoopHandler())
    return registry


def _parse_duration_s(duration_str: str) -> float:
    """Parse a duration string like '45s', '2m', '1h' to seconds."""
    import re
    m = re.match(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)?", str(duration_str))
    if not m:
        return 30.0
    val = float(m.group(1))
    unit = m.group(2) or "s"
    multipliers = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}
    return val * multipliers.get(unit, 1)
