"""Agent-loop tests. No network: a scripted FakeProvider stands in for any model."""

from __future__ import annotations

from collections.abc import AsyncIterator

from penelope.agent import Agent
from penelope.provider import Provider, ProviderChunk, ToolCall
from penelope.tools import AddTool, ToolRegistry


class FakeProvider(Provider):
    """Yields pre-baked chunks turn by turn, ignoring the model.

    `turns` is a list of turns; each turn is a list of ProviderChunks to yield.
    Every call to `stream` consumes the next turn and records the messages it saw.
    """

    def __init__(self, turns: list[list[ProviderChunk]]) -> None:
        self._turns = turns
        self._call = 0
        self.seen_messages: list[list[dict]] = []

    async def stream(
        self, messages: list[dict], tools: list[dict]
    ) -> AsyncIterator[ProviderChunk]:
        self.seen_messages.append([dict(m) for m in messages])
        turn = self._turns[self._call]
        self._call += 1
        for chunk in turn:
            yield chunk


class RaisingProvider(Provider):
    async def stream(
        self, messages: list[dict], tools: list[dict]
    ) -> AsyncIterator[ProviderChunk]:
        raise RuntimeError("boom")
        yield  # pragma: no cover - makes this an async generator


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(AddTool())
    return reg


async def test_tool_round_trip() -> None:
    provider = FakeProvider(
        turns=[
            [
                ProviderChunk(text_delta="let me "),
                ProviderChunk(text_delta="add"),
                ProviderChunk(
                    tool_calls=[ToolCall("c1", "add", {"a": 21, "b": 21})],
                    finish_reason="tool_calls",
                ),
            ],
            [
                ProviderChunk(text_delta="the answer "),
                ProviderChunk(text_delta="is 42"),
                ProviderChunk(tool_calls=None, finish_reason="stop"),
            ],
        ]
    )
    agent = Agent(provider, _registry())

    events = [ev async for ev in agent.run("what is 21 + 21?")]
    types = [ev.type for ev in events]

    assert types[-1] == "done"
    assert types.count("done") == 1
    assert "token" in types
    assert types.index("tool_call") < types.index("tool_result")

    tool_call = next(ev for ev in events if ev.type == "tool_call")
    assert tool_call.data["name"] == "add"

    tool_result = next(ev for ev in events if ev.type == "tool_result")
    assert tool_result.data["result"] == "42"

    # A second token batch must come AFTER the tool result.
    assert any(
        ev.type == "token"
        for ev in events[events.index(tool_result) + 1 :]
    )

    # Turn 2's messages must have carried the tool-result message back.
    turn2_messages = provider.seen_messages[1]
    assert any(
        m.get("role") == "tool"
        and m.get("tool_call_id") == "c1"
        and m.get("content") == "42"
        for m in turn2_messages
    )


async def test_no_tools() -> None:
    provider = FakeProvider(
        turns=[
            [
                ProviderChunk(text_delta="hello "),
                ProviderChunk(text_delta="world"),
                ProviderChunk(tool_calls=None, finish_reason="stop"),
            ]
        ]
    )
    agent = Agent(provider, _registry())

    events = [ev async for ev in agent.run("hi")]
    types = [ev.type for ev in events]

    assert "tool_call" not in types
    assert "token" in types
    assert types[-1] == "done"
    assert types.count("done") == 1


async def test_error_path() -> None:
    agent = Agent(RaisingProvider(), _registry())

    events = [ev async for ev in agent.run("hi")]
    types = [ev.type for ev in events]

    assert types == ["error", "done"]
    assert events[0].data["message"] == "boom"


async def test_registry() -> None:
    reg = _registry()
    schemas = reg.schemas()

    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert "parameters" in schema["function"]

    result = await reg.run("add", {"a": 2, "b": 3})
    assert result == "5"
