"""Attractor CLI - run DOT-based AI pipelines."""
from __future__ import annotations
import json
import os
import sys
import time

import click

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


@click.group()
@click.version_option("0.1.0")
def main():
    """Attractor - DOT-based AI pipeline runner."""
    pass


@main.command()
@click.argument("dotfile", type=click.Path(exists=True))
@click.option("--logs-dir", default="attractor-runs", show_default=True,
              help="Directory to store run logs and checkpoints")
@click.option("--run-id", default="", help="Override the run ID (default: auto-generated)")
@click.option("--resume", is_flag=True, help="Resume from existing checkpoint")
@click.option("--dry-run", is_flag=True, help="Parse and validate without executing")
@click.option("--backend", type=click.Choice(["agent", "direct", "simulate"]),
              default="simulate", show_default=True,
              help="LLM backend: agent (full agent loop), direct (single LLM call), simulate (no-op)")
@click.option("--model", default="", help="Override LLM model")
@click.option("--provider", default="", help="Override LLM provider")
@click.option("--working-dir", default="", help="Working directory for agent tools")
@click.option("--auto-approve", is_flag=True, help="Auto-approve all human gates")
@click.option("--quiet", "-q", is_flag=True, help="Suppress event output")
def run(dotfile, logs_dir, run_id, resume, dry_run, backend, model, provider,
        working_dir, auto_approve, quiet):
    """Run a pipeline defined in DOTFILE."""
    from src.attractor.parser import parse_dot, ParseError
    from src.attractor.validator import validate_or_raise, ValidationError, Severity
    from src.attractor.engine import PipelineRunner, RunConfig
    from src.attractor.handlers import make_default_registry
    from src.attractor.interviewer import AutoApproveInterviewer, ConsoleInterviewer

    try:
        with open(dotfile) as f:
            source = f.read()
    except Exception as e:
        click.echo(f"Error reading {dotfile}: {e}", err=True)
        sys.exit(1)

    # Parse
    try:
        graph = parse_dot(source)
    except ParseError as e:
        click.echo(f"Parse error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Parsed: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    if graph.attrs.goal:
        click.echo(f"Goal: {graph.attrs.goal}")

    # Validate
    try:
        diagnostics = validate_or_raise(graph)
    except ValidationError as e:
        click.echo(f"Validation errors:\n{e}", err=True)
        sys.exit(1)

    warnings = [d for d in diagnostics if d.severity == Severity.WARNING]
    if warnings:
        for w in warnings:
            click.echo(f"  WARNING [{w.rule}] {w.message}", err=True)

    if dry_run:
        click.echo("Dry run complete - pipeline is valid")
        return

    # Build backend
    llm_backend = None
    if backend == "agent":
        from src.attractor.backends import AgentLoopBackend
        llm_backend = AgentLoopBackend(
            model=model or None,
            provider=provider or None,
            working_dir=working_dir or None,
        )
    elif backend == "direct":
        from src.attractor.backends import DirectLLMBackend
        llm_backend = DirectLLMBackend(
            model=model or None,
            provider=provider or None,
        )
    # simulate = None backend (default)

    # Build interviewer
    interviewer = AutoApproveInterviewer() if auto_approve else ConsoleInterviewer()

    # Build registry
    registry = make_default_registry(backend=llm_backend, interviewer=interviewer)

    # Event handler
    event_log = []
    def on_event(event):
        event_log.append(event)
        if not quiet:
            _print_event(event)

    runner = PipelineRunner(registry=registry, on_event=on_event)

    cfg = RunConfig(
        logs_root=logs_dir,
        run_id=run_id or "",
        resume=resume,
    )

    click.echo(f"Running pipeline...")
    start = time.time()
    outcome = runner.run(graph, cfg)
    duration = time.time() - start

    click.echo(f"\n{'='*50}")
    click.echo(f"Pipeline {'COMPLETED' if outcome.status.value in ('success', 'partial_success') else 'FAILED'}")
    click.echo(f"Status: {outcome.status.value}")
    click.echo(f"Duration: {duration:.1f}s")
    if outcome.failure_reason:
        click.echo(f"Failure: {outcome.failure_reason}", err=True)
    if outcome.notes:
        click.echo(f"Notes: {outcome.notes}")

    if outcome.status.value == "fail":
        sys.exit(1)


@main.command()
@click.argument("dotfile", type=click.Path(exists=True))
def validate(dotfile):
    """Validate a pipeline DOT file without running it."""
    from src.attractor.parser import parse_dot, ParseError
    from src.attractor.validator import validate as do_validate, Severity

    with open(dotfile) as f:
        source = f.read()

    try:
        graph = parse_dot(source)
    except ParseError as e:
        click.echo(f"Parse error: {e}", err=True)
        sys.exit(1)

    diagnostics = do_validate(graph)
    errors = [d for d in diagnostics if d.severity == Severity.ERROR]
    warnings = [d for d in diagnostics if d.severity == Severity.WARNING]

    for d in diagnostics:
        icon = "✗" if d.severity == Severity.ERROR else "⚠" if d.severity == Severity.WARNING else "ℹ"
        node_info = f" (node: {d.node_id})" if d.node_id else ""
        click.echo(f"{icon} [{d.rule}]{node_info} {d.message}")

    click.echo(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    if errors:
        sys.exit(1)


@main.command()
@click.option("--provider", type=click.Choice(["anthropic", "openai", "gemini"]), default="anthropic")
@click.option("--model", default="")
@click.argument("prompt")
def agent(provider, model, prompt):
    """Run a one-shot coding agent session."""
    from src.agent.session import Session, SessionConfig
    from src.agent.environment import LocalExecutionEnvironment

    config = SessionConfig(
        provider=provider,
        model=model or _default_model(provider),
    )

    def on_event(event):
        _print_agent_event(event)

    session = Session(config=config, on_event=on_event)
    result = session.submit(prompt)
    click.echo(f"\nResult:\n{result}")


@main.command()
@click.argument("model")
def model_info(model):
    """Show information about a model."""
    from src.llm.catalog import get_model_info, list_models

    info = get_model_info(model)
    if info:
        click.echo(f"Model: {info.display_name}")
        click.echo(f"Provider: {info.provider}")
        click.echo(f"Context window: {info.context_window:,} tokens")
        click.echo(f"Max output: {info.max_output or 'unknown'}")
        click.echo(f"Supports tools: {info.supports_tools}")
        click.echo(f"Supports vision: {info.supports_vision}")
        click.echo(f"Supports reasoning: {info.supports_reasoning}")
        if info.aliases:
            click.echo(f"Aliases: {', '.join(info.aliases)}")
    else:
        click.echo(f"Unknown model: {model}")
        click.echo("Known models:")
        for m in list_models():
            click.echo(f"  {m.id} ({m.provider})")


def _print_event(event):
    """Print pipeline events."""
    from src.attractor.engine import (
        PipelineStarted, PipelineCompleted, PipelineFailed,
        StageStarted, StageCompleted, StageFailed, CheckpointSaved,
    )
    if isinstance(event, StageStarted):
        click.echo(f"  → {event.name}")
    elif isinstance(event, StageCompleted):
        click.echo(f"  ✓ {event.name} ({event.duration:.1f}s)")
    elif isinstance(event, StageFailed):
        click.echo(f"  ✗ {event.name}: {event.error}", err=True)
    elif isinstance(event, CheckpointSaved):
        pass  # Suppress checkpoint noise


def _print_agent_event(event):
    """Print agent session events."""
    from src.agent.session import EventKind
    if event.kind == EventKind.TOOL_CALL_START:
        click.echo(f"  [tool] {event.data.get('tool_name', '?')}", nl=False)
    elif event.kind == EventKind.TOOL_CALL_END:
        click.echo(" ✓")
    elif event.kind == EventKind.ASSISTANT_TEXT_END:
        text = event.data.get("text", "")
        if text:
            click.echo(f"\n{text}")


def _default_model(provider: str) -> str:
    defaults = {
        "anthropic": "claude-sonnet-4-6",
        "openai": "gpt-5.2",
        "gemini": "gemini-3-flash-preview",
    }
    return defaults.get(provider, "claude-sonnet-4-6")


if __name__ == "__main__":
    main()
