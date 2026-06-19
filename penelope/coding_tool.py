"""A tool that delegates a coding task to a CodingExecutor.

The agent calls `delegate_coding` with a natural-language instruction; the tool
drives a CodingExecutor to completion in a workspace directory and returns a
concise summary of what was done.
"""

from __future__ import annotations

from typing import Callable

from .executors import CodingExecutor, CodingTask


class DelegateCodingTool:
    __slots__ = ("_executor", "_default_workspace", "_on_event")

    name = "delegate_coding"
    description = (
        "Delegate a self-contained coding task (creating or editing files, "
        "running code) to a coding agent that operates in a workspace "
        "directory. Returns a summary of what was done. Use for any task that "
        "requires writing or modifying files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The coding instruction to carry out.",
            },
            "workspace": {
                "type": "string",
                "description": (
                    "Absolute path to the workspace directory; defaults to the "
                    "tool's configured workspace."
                ),
            },
        },
        "required": ["task"],
    }

    def __init__(
        self,
        executor: CodingExecutor,
        *,
        default_workspace: str,
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self._executor = executor
        self._default_workspace = default_workspace
        self._on_event = on_event

    async def run(self, arguments: dict) -> str:
        task = CodingTask(
            prompt=arguments["task"],
            workspace=arguments.get("workspace") or self._default_workspace,
        )

        result: str | None = None
        error: str | None = None
        summaries: list[str] = []

        # Blocking by design: streaming inner events up through this loop is deferred to a later slice.
        async for ev in self._executor.run(task):
            if self._on_event is not None:
                try:
                    self._on_event({"type": ev.type, "data": dict(ev.data)})
                except Exception:
                    pass
            if ev.type == "tool_result":
                content = ev.data.get("content")
                if content is not None:
                    summaries.append(str(content))
            elif ev.type == "error":
                error = str(ev.data.get("message", "unknown error"))
            elif ev.type == "done":
                done_result = ev.data.get("result")
                if done_result is not None:
                    result = str(done_result)

        if error is not None:
            return f"coding task failed: {error}"
        if result:
            return result
        if summaries:
            return "\n".join(summaries[-3:])
        return "coding task completed with no output."
