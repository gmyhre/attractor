"""Condition expression language (Section 10 of attractor-spec.md).

Grammar:
    ConditionExpr ::= Clause ( '&&' Clause )*
    Clause        ::= Key Operator Literal
    Key           ::= 'outcome' | 'preferred_label' | 'context.' Path
    Operator      ::= '=' | '!='
    Literal       ::= String | Integer | Boolean
"""
from __future__ import annotations
from .types import Outcome


def evaluate_condition(condition: str, outcome: Outcome, context: "Context") -> bool:  # type: ignore
    """Evaluate a condition expression against outcome and context."""
    if not condition or not condition.strip():
        return True  # empty condition always passes

    clauses = condition.split("&&")
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if not _evaluate_clause(clause, outcome, context):
            return False
    return True


def _evaluate_clause(clause: str, outcome: Outcome, context) -> bool:
    if "!=" in clause:
        key, _, value = clause.partition("!=")
        return _resolve_key(key.strip(), outcome, context) != value.strip().strip('"').strip("'")
    if "=" in clause:
        key, _, value = clause.partition("=")
        return _resolve_key(key.strip(), outcome, context) == value.strip().strip('"').strip("'")
    # Bare key: truthy check
    val = _resolve_key(clause.strip(), outcome, context)
    return bool(val)


def _resolve_key(key: str, outcome: Outcome, context) -> str:
    if key == "outcome":
        return outcome.status.value if hasattr(outcome.status, "value") else str(outcome.status)
    if key == "preferred_label":
        return outcome.preferred_label or ""
    if key.startswith("context."):
        path = key[len("context."):]
        val = context.get(key)
        if val is not None:
            return str(val)
        val = context.get(path)
        if val is not None:
            return str(val)
        return ""
    # Direct context lookup
    val = context.get(key)
    if val is not None:
        return str(val)
    return ""
