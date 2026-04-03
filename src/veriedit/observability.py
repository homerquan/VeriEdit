from __future__ import annotations

from typing import Any

from veriedit.schemas import AgentHandoff


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


def record_agent_handoff(
    state: dict[str, Any],
    *,
    from_agent: str,
    to_agent: str,
    summary: str,
    key_points: list[str] | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    handoff = AgentHandoff(
        from_agent=from_agent,
        to_agent=to_agent,
        iteration=state.get("iteration", 0),
        summary=summary,
        key_points=key_points or [],
        payload=payload or {},
    )
    state.setdefault("agent_handoffs", []).append(handoff.model_dump())
    state.setdefault("observation_trace", []).append(
        {
            "kind": "handoff",
            "from_agent": from_agent,
            "to_agent": to_agent,
            "iteration": state.get("iteration", 0),
            "summary": summary,
            "payload": payload or {},
        }
    )
