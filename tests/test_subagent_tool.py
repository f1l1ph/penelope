"""Tests for SpawnSubagentTool.

No network. The suite uses pytest-asyncio in auto mode, so test functions are
plain `async def`. A FakeAgent yields scripted events; injected callables are
recorded so we can assert they were called with the right arguments.
"""

from __future__ import annotations

from penelope.events import Event
from penelope.subagent_tool import SpawnSubagentTool


class FakeAgent:
    def __init__(self, events: list[Event]) -> None:
        self._events = events

    async def run(self, task: str, history=None):
        for ev in self._events:
            yield ev


async def test_returns_done_result_when_present():
    spec = {"name": "writer", "model": "m", "system_prompt": "p"}
    calls: dict = {"load": [], "build": []}

    def load_subagent(name):
        calls["load"].append(name)
        return spec

    def build_subagent(s):
        calls["build"].append(s)
        return FakeAgent(
            [
                Event("token", {"text": "partial "}),
                Event("done", {"result": "final answer"}),
            ]
        )

    tool = SpawnSubagentTool(load_subagent=load_subagent, build_subagent=build_subagent)
    out = await tool.run({"name": "writer", "task": "do a thing"})

    assert out == "final answer"
    assert calls["load"] == ["writer"]
    assert calls["build"] == [spec]


async def test_returns_tokens_when_done_has_no_result():
    def load_subagent(name):
        return {"name": name, "model": "m", "system_prompt": "p"}

    def build_subagent(spec):
        return FakeAgent(
            [
                Event("token", {"text": "hello "}),
                Event("token", {"text": "world"}),
                Event("done", {}),
            ]
        )

    tool = SpawnSubagentTool(load_subagent=load_subagent, build_subagent=build_subagent)
    out = await tool.run({"name": "writer", "task": "do a thing"})
    assert out == "hello world"


async def test_unknown_name_returns_error():
    def load_subagent(name):
        return None

    def build_subagent(spec):  # pragma: no cover - must not be called
        raise AssertionError("build_subagent should not be called for unknown name")

    tool = SpawnSubagentTool(load_subagent=load_subagent, build_subagent=build_subagent)
    out = await tool.run({"name": "ghost", "task": "x"})
    assert out == "error: no subagent named ghost"


async def test_async_load_subagent_is_awaited():
    spec = {"name": "writer", "model": "m", "system_prompt": "p"}

    async def load_subagent(name):
        return spec

    def build_subagent(s):
        assert s == spec
        return FakeAgent([Event("done", {"result": "ok"})])

    tool = SpawnSubagentTool(load_subagent=load_subagent, build_subagent=build_subagent)
    out = await tool.run({"name": "writer", "task": "x"})
    assert out == "ok"


async def test_on_event_forwards_events_in_order():
    def load_subagent(name):
        return {"name": name, "model": "m", "system_prompt": "p"}

    def build_subagent(spec):
        return FakeAgent(
            [
                Event("token", {"text": "hello "}),
                Event("token", {"text": "world"}),
                Event("done", {"result": "final answer"}),
            ]
        )

    forwarded: list[dict] = []
    tool = SpawnSubagentTool(
        load_subagent=load_subagent,
        build_subagent=build_subagent,
        on_event=forwarded.append,
    )
    out = await tool.run({"name": "writer", "task": "do a thing"})

    assert out == "final answer"
    assert forwarded == [
        {"type": "token", "data": {"text": "hello "}},
        {"type": "token", "data": {"text": "world"}},
        {"type": "done", "data": {"result": "final answer"}},
    ]


async def test_on_event_that_raises_does_not_break_run():
    def load_subagent(name):
        return {"name": name, "model": "m", "system_prompt": "p"}

    def build_subagent(spec):
        return FakeAgent(
            [
                Event("token", {"text": "hello "}),
                Event("token", {"text": "world"}),
                Event("done", {}),
            ]
        )

    def boom(event):
        raise RuntimeError("callback exploded")

    tool = SpawnSubagentTool(
        load_subagent=load_subagent,
        build_subagent=build_subagent,
        on_event=boom,
    )
    out = await tool.run({"name": "writer", "task": "do a thing"})
    assert out == "hello world"


async def test_error_event_reports_failure():
    def load_subagent(name):
        return {"name": name, "model": "m", "system_prompt": "p"}

    def build_subagent(spec):
        return FakeAgent(
            [
                Event("token", {"text": "starting"}),
                Event("error", {"message": "boom"}),
                Event("done", {}),
            ]
        )

    tool = SpawnSubagentTool(load_subagent=load_subagent, build_subagent=build_subagent)
    out = await tool.run({"name": "writer", "task": "x"})
    assert out == "subagent failed: boom"
