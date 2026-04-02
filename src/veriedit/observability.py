from __future__ import annotations

from typing import Any


def record_node_event(state: dict[str, Any], *, node: str, phase: str, summary: dict[str, Any] | None = None) -> None:
    state.setdefault("observation_trace", []).append(
        {
            "kind": "node",
            "node": node,
            "phase": phase,
            "iteration": state.get("iteration", 0),
            "summary": summary or {},
        }
    )


def record_tool_event(
    state: dict[str, Any],
    *,
    tool: str,
    params: dict[str, Any],
    variant: str,
    status: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    state.setdefault("observation_trace", []).append(
        {
            "kind": "tool",
            "tool": tool,
            "params": params,
            "variant": variant,
            "status": status,
            "iteration": state.get("iteration", 0),
            "metrics": metrics or {},
        }
    )
