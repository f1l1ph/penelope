"""A tool that delegates a focused task to a specialist subagent.

The agent calls `spawn_subagent` with a subagent name/id and an instruction.
The tool loads the subagent's spec (instructions, model, tools) via an injected
`load_subagent`, builds a runnable `Agent` via an injected `build_subagent`, and
drives it to completion, returning the subagent's final answer. Blocking by
design: streaming the subagent's inner events up through this loop is deferred
to a later slice. Like every tool it returns a string and never raises.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable


class SpawnSubagentTool:
    __slots__ = ("_load_subagent", "_build_subagent")

    name = "spawn_subagent"
    description = (
        "Delegate a focused task to a specialist subagent (its own "
        "instructions, model, and tools). Returns the subagent's final answer."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The subagent's id or name.",
            },
            "task": {
                "type": "string",
                "description": "The instruction for the subagent to carry out.",
            },
        },
        "required": ["name", "task"],
    }

    def __init__(
        self,
        *,
        load_subagent: Callable[[str], dict | None | Awaitable[dict | None]],
        build_subagent: Callable[[dict], Any],
    ) -> None:
        # load_subagent(name) -> spec dict | None; may be sync OR a coroutine
        # (awaited below). build_subagent(spec) -> Agent (synchronous).
        self._load_subagent = load_subagent
        self._build_subagent = build_subagent

    async def run(self, arguments: dict) -> str:
        try:
            name = arguments.get("name")
            task = arguments.get("task")
            if not name:
                return "error: missing required argument 'name'"
            if not task:
                return "error: missing required argument 'task'"

            spec = self._load_subagent(name)
            if inspect.isawaitable(spec):
                spec = await spec
            if spec is None:
                return f"error: no subagent named {name}"

            agent = self._build_subagent(spec)

            text_parts: list[str] = []
            result: str | None = None
            # Blocking: streaming the subagent's inner events up is future work.
            async for ev in agent.run(task):
                if ev.type == "token":
                    text_parts.append(ev.data.get("text", ""))
                elif ev.type == "error":
                    return f"subagent failed: {ev.data.get('message', 'unknown error')}"
                elif ev.type == "done":
                    done_result = ev.data.get("result")
                    if done_result is not None:
                        result = str(done_result)

            if result:
                return result
            return "".join(text_parts)
        except Exception as exc:  # noqa: BLE001 - tools never raise out of run()
            return f"error: {exc}"
