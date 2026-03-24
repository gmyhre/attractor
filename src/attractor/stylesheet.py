"""Model stylesheet (Section 8) - CSS-like rules for LLM model/provider defaults."""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class StyleRule:
    selector: str  # "*", ".classname", "#node_id"
    properties: dict[str, str]
    specificity: int  # 0=universal, 1=class, 2=id


def parse_stylesheet(css: str) -> list[StyleRule]:
    """Parse model stylesheet string into rules."""
    rules = []
    # Strip comments
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    css = re.sub(r"//[^\n]*", "", css)

    # Match selector { props }
    for m in re.finditer(r"([*#.]?[a-zA-Z0-9_-]*)\s*\{([^}]*)\}", css):
        selector = m.group(1).strip()
        body = m.group(2).strip()

        if selector == "*":
            specificity = 0
        elif selector.startswith("."):
            specificity = 1
        elif selector.startswith("#"):
            specificity = 2
        else:
            specificity = 0  # treat as universal

        props: dict[str, str] = {}
        for decl in body.split(";"):
            decl = decl.strip()
            if not decl:
                continue
            if ":" in decl:
                key, _, val = decl.partition(":")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key in ("llm_model", "llm_provider", "reasoning_effort"):
                    props[key] = val

        if props:
            rules.append(StyleRule(selector=selector, properties=props, specificity=specificity))

    return rules


def apply_stylesheet(graph: "Graph") -> None:  # type: ignore
    """Apply model_stylesheet to all nodes in the graph (modifies in place)."""
    if not graph.attrs.model_stylesheet:
        return

    rules = parse_stylesheet(graph.attrs.model_stylesheet)
    if not rules:
        return

    for node in graph.nodes.values():
        # Collect all matching rules sorted by specificity (lowest first, last wins)
        matching = []
        for rule in rules:
            sel = rule.selector
            if sel == "*":
                matching.append(rule)
            elif sel.startswith("."):
                class_name = sel[1:]
                node_classes = [c.strip() for c in (node.attrs.cls or "").split(",")]
                if class_name in node_classes:
                    matching.append(rule)
            elif sel.startswith("#"):
                node_id_match = sel[1:]
                if node.id == node_id_match:
                    matching.append(rule)

        # Sort by specificity, then apply (later = higher priority)
        matching.sort(key=lambda r: r.specificity)

        for rule in matching:
            # Only apply if node doesn't have explicit attribute
            for prop, val in rule.properties.items():
                if prop == "llm_model" and not node.attrs.llm_model:
                    node.attrs.llm_model = val
                elif prop == "llm_provider" and not node.attrs.llm_provider:
                    node.attrs.llm_provider = val
                elif prop == "reasoning_effort":
                    # stylesheet can set reasoning_effort (node explicit still wins)
                    node.attrs.reasoning_effort = val
