"""DOT DSL parser for Attractor pipelines.

Parses the restricted DOT subset defined in Section 2 of attractor-spec.md.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Iterator

from .types import (
    Graph, GraphAttrs, Node, NodeAttrs, Edge, EdgeAttrs,
)


class ParseError(Exception):
    pass


# ---------- Tokenizer ----------

@dataclass
class Token:
    kind: str  # IDENT, STRING, LBRACE, RBRACE, LBRACKET, RBRACKET, EQUALS, COMMA, SEMI, ARROW, EOF
    value: str
    line: int = 0


_TOKEN_RE = re.compile(r"""
    (?P<COMMENT_LINE>//[^\n]*)
  | (?P<COMMENT_BLOCK>/\*.*?\*/)
  | (?P<STRING>"(?:[^"\\]|\\["ntr\\])*")
  | (?P<ARROW>->)
  | (?P<LBRACE>\{)
  | (?P<RBRACE>\})
  | (?P<LBRACKET>\[)
  | (?P<RBRACKET>\])
  | (?P<EQUALS>=)
  | (?P<COMMA>,)
  | (?P<SEMI>;)
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_./-]*)
  | (?P<NUMBER>-?[0-9]+(?:\.[0-9]+)?(?:ms|s|m|h|d)?)
  | (?P<WHITESPACE>[ \t\r\n]+)
""", re.VERBOSE | re.DOTALL)


def tokenize(source: str) -> list[Token]:
    tokens = []
    line = 1
    for m in _TOKEN_RE.finditer(source):
        kind = m.lastgroup
        val = m.group()
        if kind in ("COMMENT_LINE", "COMMENT_BLOCK", "WHITESPACE"):
            line += val.count("\n")
            continue
        if kind == "NUMBER":
            kind = "IDENT"  # treat numbers as identifiers for attribute values
        tokens.append(Token(kind=kind, value=val, line=line))
        line += val.count("\n")
    tokens.append(Token(kind="EOF", value="", line=line))
    return tokens


# ---------- Parser ----------

class Parser:
    def __init__(self, tokens: list[Token]):
        self._tokens = tokens
        self._pos = 0

    def peek(self) -> Token:
        return self._tokens[self._pos]

    def advance(self) -> Token:
        t = self._tokens[self._pos]
        if t.kind != "EOF":
            self._pos += 1
        return t

    def expect(self, kind: str, value: str | None = None) -> Token:
        t = self.advance()
        if t.kind != kind:
            raise ParseError(f"Line {t.line}: Expected {kind}, got {t.kind} ({t.value!r})")
        if value and t.value != value:
            raise ParseError(f"Line {t.line}: Expected {value!r}, got {t.value!r}")
        return t

    def match(self, kind: str, value: str | None = None) -> bool:
        t = self.peek()
        if t.kind == kind and (value is None or t.value == value):
            return True
        return False

    def consume_if(self, kind: str, value: str | None = None) -> Token | None:
        if self.match(kind, value):
            return self.advance()
        return None

    def parse_value(self) -> str:
        """Parse a string or identifier value."""
        t = self.peek()
        if t.kind == "STRING":
            self.advance()
            # Unescape the string value
            raw = t.value[1:-1]  # strip quotes
            return raw.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")
        if t.kind == "IDENT":
            self.advance()
            return t.value
        raise ParseError(f"Line {t.line}: Expected value, got {t.kind} ({t.value!r})")

    def parse_attrs(self) -> dict[str, str]:
        """Parse [ key=value, key=value, ... ]"""
        self.expect("LBRACKET")
        attrs: dict[str, str] = {}
        while not self.match("RBRACKET") and not self.match("EOF"):
            # key (may be qualified: foo.bar)
            key = self.parse_value()
            self.expect("EQUALS")
            val = self.parse_value()
            attrs[key] = val
            self.consume_if("COMMA")
        self.expect("RBRACKET")
        return attrs

    def parse_graph(self) -> Graph:
        """Parse: digraph Identifier { ... }"""
        # Accept optional 'strict'
        if self.match("IDENT", "strict"):
            self.advance()
        if not self.match("IDENT", "digraph"):
            raise ParseError(f"Line {self.peek().line}: Expected 'digraph'")
        self.advance()

        # Graph name (optional)
        if self.match("IDENT"):
            self.advance()

        self.expect("LBRACE")

        graph = Graph()
        # Current default attrs for nodes/edges
        node_defaults: dict[str, str] = {}
        edge_defaults: dict[str, str] = {}

        self._parse_stmts(graph, node_defaults, edge_defaults)

        self.expect("RBRACE")
        return graph

    def _parse_stmts(self, graph: Graph, node_defaults: dict, edge_defaults: dict,
                     subgraph_node_defaults: dict | None = None) -> None:
        """Parse statements inside a graph or subgraph body."""
        effective_node_defaults = dict(node_defaults)
        if subgraph_node_defaults:
            effective_node_defaults.update(subgraph_node_defaults)

        while not self.match("RBRACE") and not self.match("EOF"):
            t = self.peek()

            # graph [ ... ] or graph_attr = value
            if t.kind == "IDENT" and t.value == "graph":
                self.advance()
                if self.match("LBRACKET"):
                    attrs = self.parse_attrs()
                    _apply_graph_attrs(graph.attrs, attrs)
                elif self.match("EQUALS"):
                    # graph.attr = value (rare)
                    pass
                self.consume_if("SEMI")
                continue

            # node [ ... ] defaults
            if t.kind == "IDENT" and t.value == "node":
                self.advance()
                if self.match("LBRACKET"):
                    attrs = self.parse_attrs()
                    effective_node_defaults.update(attrs)
                    node_defaults.update(attrs)
                self.consume_if("SEMI")
                continue

            # edge [ ... ] defaults
            if t.kind == "IDENT" and t.value == "edge":
                self.advance()
                if self.match("LBRACKET"):
                    attrs = self.parse_attrs()
                    edge_defaults.update(attrs)
                self.consume_if("SEMI")
                continue

            # subgraph
            if t.kind == "IDENT" and t.value == "subgraph":
                self.advance()
                sub_label = ""
                if self.match("IDENT"):
                    self.advance()  # subgraph name
                self.expect("LBRACE")
                sub_node_defaults: dict[str, str] = {}
                # Check for label and node defaults inside subgraph
                self._parse_stmts(graph, effective_node_defaults, edge_defaults,
                                   sub_node_defaults)
                self.expect("RBRACE")
                self.consume_if("SEMI")
                continue

            # rankdir = LR (graph-level shorthand)
            if t.kind == "IDENT" and self._peek_next_is_equals():
                key = self.advance().value
                self.expect("EQUALS")
                val = self.parse_value()
                if key == "rankdir":
                    pass  # visualization hint, ignored for execution
                elif key == "label":
                    graph.attrs.label = val
                else:
                    _apply_single_graph_attr(graph.attrs, key, val)
                self.consume_if("SEMI")
                continue

            # Edge or node statement
            if t.kind == "IDENT":
                name = self.advance().value

                # Check for edge (chained or single)
                if self.match("ARROW"):
                    # Edge statement: name -> target1 -> target2 ... [attrs]
                    targets = [name]
                    while self.match("ARROW"):
                        self.advance()
                        targets.append(self.parse_value())

                    edge_attrs: dict[str, str] = {}
                    if self.match("LBRACKET"):
                        edge_attrs = self.parse_attrs()

                    # Expand chained edges
                    for i in range(len(targets) - 1):
                        src, dst = targets[i], targets[i + 1]
                        # Ensure nodes exist
                        if src not in graph.nodes:
                            graph.nodes[src] = Node(id=src, attrs=_make_node_attrs(src, effective_node_defaults))
                        if dst not in graph.nodes:
                            graph.nodes[dst] = Node(id=dst, attrs=_make_node_attrs(dst, effective_node_defaults))
                        final_edge_attrs = dict(edge_defaults)
                        final_edge_attrs.update(edge_attrs)
                        graph.edges.append(Edge(
                            from_node=src,
                            to_node=dst,
                            attrs=_make_edge_attrs(final_edge_attrs),
                        ))
                    self.consume_if("SEMI")
                    continue

                # Node statement: name [attrs]
                node_attrs: dict[str, str] = dict(effective_node_defaults)
                if self.match("LBRACKET"):
                    node_attrs.update(self.parse_attrs())
                if name not in graph.nodes:
                    graph.nodes[name] = Node(id=name, attrs=_make_node_attrs(name, node_attrs))
                else:
                    # Update existing node attrs
                    _update_node_attrs(graph.nodes[name].attrs, node_attrs)
                self.consume_if("SEMI")
                continue

            # Skip unknown tokens
            self.advance()

    def _peek_next_is_equals(self) -> bool:
        """Check if the token after current peek is '='."""
        pos = self._pos + 1
        if pos < len(self._tokens):
            return self._tokens[pos].kind == "EQUALS"
        return False


def _make_node_attrs(node_id: str, raw: dict[str, str]) -> NodeAttrs:
    attrs = NodeAttrs()
    attrs.label = raw.get("label", node_id)
    attrs.shape = raw.get("shape", "box")
    attrs.type = raw.get("type", "")
    attrs.prompt = raw.get("prompt", "")
    attrs.max_retries = _int(raw.get("max_retries", "0"))
    attrs.goal_gate = _bool(raw.get("goal_gate", "false"))
    attrs.retry_target = raw.get("retry_target", "")
    attrs.fallback_retry_target = raw.get("fallback_retry_target", "")
    attrs.fidelity = raw.get("fidelity", "")
    attrs.thread_id = raw.get("thread_id", "")
    attrs.cls = raw.get("class", "")
    attrs.timeout = raw.get("timeout", "")
    attrs.llm_model = raw.get("llm_model", "")
    attrs.llm_provider = raw.get("llm_provider", "")
    attrs.reasoning_effort = raw.get("reasoning_effort", "high")
    attrs.auto_status = _bool(raw.get("auto_status", "false"))
    attrs.allow_partial = _bool(raw.get("allow_partial", "false"))
    # Extra attrs
    known = {
        "label", "shape", "type", "prompt", "max_retries", "goal_gate",
        "retry_target", "fallback_retry_target", "fidelity", "thread_id",
        "class", "timeout", "llm_model", "llm_provider", "reasoning_effort",
        "auto_status", "allow_partial",
    }
    attrs.extra = {k: v for k, v in raw.items() if k not in known}
    return attrs


def _update_node_attrs(attrs: NodeAttrs, raw: dict[str, str]) -> None:
    """Update existing node attrs, overriding only explicitly set values."""
    if "label" in raw:
        attrs.label = raw["label"]
    if "shape" in raw:
        attrs.shape = raw["shape"]
    if "type" in raw:
        attrs.type = raw["type"]
    if "prompt" in raw:
        attrs.prompt = raw["prompt"]
    if "max_retries" in raw:
        attrs.max_retries = _int(raw["max_retries"])
    if "goal_gate" in raw:
        attrs.goal_gate = _bool(raw["goal_gate"])
    if "retry_target" in raw:
        attrs.retry_target = raw["retry_target"]
    if "fallback_retry_target" in raw:
        attrs.fallback_retry_target = raw["fallback_retry_target"]
    if "fidelity" in raw:
        attrs.fidelity = raw["fidelity"]
    if "thread_id" in raw:
        attrs.thread_id = raw["thread_id"]
    if "class" in raw:
        attrs.cls = raw["class"]
    if "timeout" in raw:
        attrs.timeout = raw["timeout"]
    if "llm_model" in raw:
        attrs.llm_model = raw["llm_model"]
    if "llm_provider" in raw:
        attrs.llm_provider = raw["llm_provider"]
    if "reasoning_effort" in raw:
        attrs.reasoning_effort = raw["reasoning_effort"]
    if "auto_status" in raw:
        attrs.auto_status = _bool(raw["auto_status"])
    if "allow_partial" in raw:
        attrs.allow_partial = _bool(raw["allow_partial"])
    # Extra attrs
    known = {
        "label", "shape", "type", "prompt", "max_retries", "goal_gate",
        "retry_target", "fallback_retry_target", "fidelity", "thread_id",
        "class", "timeout", "llm_model", "llm_provider", "reasoning_effort",
        "auto_status", "allow_partial",
    }
    for k, v in raw.items():
        if k not in known:
            attrs.extra[k] = v


def _make_edge_attrs(raw: dict[str, str]) -> EdgeAttrs:
    return EdgeAttrs(
        label=raw.get("label", ""),
        condition=raw.get("condition", ""),
        weight=_int(raw.get("weight", "0")),
        fidelity=raw.get("fidelity", ""),
        thread_id=raw.get("thread_id", ""),
        loop_restart=_bool(raw.get("loop_restart", "false")),
    )


def _apply_graph_attrs(attrs: GraphAttrs, raw: dict[str, str]) -> None:
    for k, v in raw.items():
        _apply_single_graph_attr(attrs, k, v)


def _apply_single_graph_attr(attrs: GraphAttrs, key: str, val: str) -> None:
    if key == "goal":
        attrs.goal = val
    elif key == "label":
        attrs.label = val
    elif key == "model_stylesheet":
        attrs.model_stylesheet = val
    elif key == "default_max_retry":
        attrs.default_max_retry = _int(val)
    elif key == "retry_target":
        attrs.retry_target = val
    elif key == "fallback_retry_target":
        attrs.fallback_retry_target = val
    elif key == "default_fidelity":
        attrs.default_fidelity = val
    else:
        attrs.extra[key] = val


def _int(val: str) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _bool(val: str) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def parse_dot(source: str) -> Graph:
    """Parse a DOT source string into a Graph."""
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_graph()
