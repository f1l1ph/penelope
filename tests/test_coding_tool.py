"""Tests for DelegateCodingTool against a scripted fake executor.

No subprocess, no network: the FakeExecutor yields ExecutorEvents directly.
"""

from __future__ import annotations

from typing import AsyncIterator

from penelope.coding_tool import DelegateCodingTool
from penelope.executors import CodingExecutor, CodingTask, ExecutorEvent


class FakeExecutor(CodingExecutor):
    def __init__(self, events: list[ExecutorEvent]) -> None:
        self._events = events
        self.received_task: CodingTask | None = None

    async def run(self, task: CodingTask) -> AsyncIterator[ExecutorEvent]:
        self.received_task = task
        for ev in self._events:
            yield ev

    async def health(self) -> bool:
        return True

    async def cancel(self, session_id: str) -> None:
        return None


async def test_run_returns_final_result():
    fake = FakeExecutor(
        [
            ExecutorEvent(
                "tool_result",
                {"id": "1", "is_error": False, "content": "wrote file"},
            ),
            ExecutorEvent("done", {"result": "Created hello.txt"}),
        ]
    )
    tool = DelegateCodingTool(fake, default_workspace="/tmp")
    out = await tool.run({"task": "make hello.txt"})
    assert "Created hello.txt" in out


async def test_run_on_error_does_not_raise():
    fake = FakeExecutor(
        [
            ExecutorEvent("error", {"message": "boom"}),
            ExecutorEvent("done", {}),
        ]
    )
    tool = DelegateCodingTool(fake, default_workspace="/tmp")
    out = await tool.run({"task": "do a thing"})
    assert "failed" in out
    assert "boom" in out


async def test_on_event_forwards_events_in_order():
    fake = FakeExecutor(
        [
            ExecutorEvent(
                "tool_result",
                {"id": "1", "is_error": False, "content": "wrote file"},
            ),
            ExecutorEvent("done", {"result": "Created hello.txt"}),
        ]
    )
    forwarded: list[dict] = []
    tool = DelegateCodingTool(
        fake, default_workspace="/tmp", on_event=forwarded.append
    )
    out = await tool.run({"task": "make hello.txt"})

    # Return value is unchanged when on_event is set.
    assert "Created hello.txt" in out

    assert forwarded == [
        {
            "type": "tool_result",
            "data": {"id": "1", "is_error": False, "content": "wrote file"},
        },
        {"type": "done", "data": {"result": "Created hello.txt"}},
    ]


async def test_on_event_that_raises_does_not_break_run():
    fake = FakeExecutor([ExecutorEvent("done", {"result": "ok"})])

    def boom(event):
        raise RuntimeError("callback exploded")

    tool = DelegateCodingTool(fake, default_workspace="/tmp", on_event=boom)
    out = await tool.run({"task": "t"})
    assert "ok" in out


async def test_workspace_passthrough():
    fake = FakeExecutor([ExecutorEvent("done", {"result": "ok"})])
    tool = DelegateCodingTool(fake, default_workspace="/tmp")

    await tool.run({"task": "t", "workspace": "/abs/path"})
    assert fake.received_task is not None
    assert fake.received_task.workspace == "/abs/path"

    await tool.run({"task": "t"})
    assert fake.received_task is not None
    assert fake.received_task.workspace == "/tmp"
