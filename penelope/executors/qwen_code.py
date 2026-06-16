"""Qwen Code executor.

Wraps the `qwen` CLI (Qwen Code) running headless in one-shot mode and normalizes
its `stream-json` stdout into the backend-agnostic `ExecutorEvent` vocabulary. The
process inherits the ambient environment; Qwen Code resolves its own auth from the
user's `~/.qwen` config. Penelope supplies no keys and reads no `.env`.

The normalization is split into a pure module-level function so it can be tested
against canned envelopes without spawning a subprocess.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator

from penelope.executors.base import CodingExecutor, CodingTask, ExecutorEvent

# stderr tail kept on abnormal exit so the surfaced error has some context.
_STDERR_TAIL = 2000


def normalize_envelope(obj: dict) -> list[ExecutorEvent]:
    """Map one parsed stdout JSON object to zero or more ExecutorEvents.

    Never raises on missing/odd keys: everything is read with `.get` and content
    blocks default to an empty list. `permission` is never emitted - yolo mode
    auto-approves, so there is nothing to surface for approval.
    """
    kind = obj.get("type")

    if kind == "assistant":
        events: list[ExecutorEvent] = []
        content = obj.get("message", {}).get("content") or []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                events.append(
                    ExecutorEvent(
                        "tool_call",
                        {
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "input": block.get("input", {}),
                        },
                    )
                )
            elif btype == "text":
                events.append(ExecutorEvent("token", {"text": block.get("text", "")}))
            # thinking (and anything else) produces no event.
        return events

    if kind == "user":
        events = []
        content = obj.get("message", {}).get("content") or []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                events.append(
                    ExecutorEvent(
                        "tool_result",
                        {
                            "id": block.get("tool_use_id"),
                            "is_error": block.get("is_error", False),
                            "content": block.get("content"),
                        },
                    )
                )
        return events

    if kind == "result":
        is_error = bool(obj.get("is_error"))
        subtype = obj.get("subtype")
        result = obj.get("result")
        if is_error or subtype != "success":
            return [
                ExecutorEvent("error", {"message": result or subtype}),
                ExecutorEvent("done", {}),
            ]
        return [
            ExecutorEvent(
                "done",
                {
                    "result": result,
                    "num_turns": obj.get("num_turns"),
                    "usage": obj.get("usage"),
                },
            )
        ]

    # system / unknown / missing type -> nothing.
    return []


class QwenCodeExecutor(CodingExecutor):
    """Runs coding tasks via the headless `qwen` CLI."""

    name = "qwen-code"

    def __init__(
        self,
        *,
        default_wall_time: int = 300,
        default_max_tool_calls: int = 50,
        qwen_path: str = "qwen",
    ) -> None:
        self._default_wall_time = default_wall_time
        self._default_max_tool_calls = default_max_tool_calls
        self._qwen_path = qwen_path
        self._processes: dict[str, asyncio.subprocess.Process] = {}

    async def run(self, task: CodingTask) -> AsyncIterator[ExecutorEvent]:
        wall = task.context.get("max_wall_time", self._default_wall_time)
        calls = task.context.get("max_tool_calls", self._default_max_tool_calls)

        argv = [
            self._qwen_path,
            task.prompt,
            "--output-format",
            "stream-json",
            "--approval-mode",
            "yolo",
            "--max-wall-time",
            str(wall),
            "--max-tool-calls",
            str(calls),
        ]
        if task.model:
            argv += ["-m", task.model]

        emitted_done = False
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=task.workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "QWEN_CODE_SUPPRESS_YOLO_WARNING": "1"},
            )
            if task.session_id is not None:
                self._processes[task.session_id] = proc

            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                for ev in normalize_envelope(obj):
                    if ev.type == "done":
                        emitted_done = True
                    yield ev

            code = await proc.wait()

            if not emitted_done:
                stderr_tail = ""
                if proc.stderr is not None:
                    data = await proc.stderr.read()
                    stderr_tail = data.decode("utf-8", "replace")[-_STDERR_TAIL:].strip()
                message = f"qwen exited {code}"
                if stderr_tail:
                    message = f"{message}: {stderr_tail}"
                yield ExecutorEvent("error", {"message": message})
                yield ExecutorEvent("done", {})
                emitted_done = True
        except Exception as exc:  # noqa: BLE001 - any failure becomes a uniform error event.
            if not emitted_done:
                yield ExecutorEvent("error", {"message": str(exc)})
                yield ExecutorEvent("done", {})
        finally:
            if task.session_id is not None:
                self._processes.pop(task.session_id, None)

    async def health(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                self._qwen_path,
                "--version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            code = await proc.wait()
            return code == 0
        except Exception:  # noqa: BLE001
            return False

    async def cancel(self, session_id: str) -> None:
        proc = self._processes.get(session_id)
        if proc is None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
