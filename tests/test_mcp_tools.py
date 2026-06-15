"""MCP adapter tests. No network, no subprocess: a FakeSession stands in for any server."""

from __future__ import annotations

from typing import Any

from penelope.mcp_tools import MCPToolAdapter
from penelope.tools import Tool, ToolRegistry


class FakeTool:
    def __init__(self, name: str, description: str, input_schema: dict[str, Any] | None) -> None:
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class FakeNonTextBlock:
    def __init__(self) -> None:
        self.type = "image"
        self.data = "ignored"


class FakeResult:
    def __init__(self, content: list[Any], structured_content: Any = None) -> None:
        self.content = content
        self.structuredContent = structured_content


class FakeListed:
    def __init__(self, tools: list[FakeTool]) -> None:
        self.tools = tools


class FakeSession:
    def __init__(self, tools: list[FakeTool], result: FakeResult) -> None:
        self._tools = tools
        self._result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> FakeListed:
        return FakeListed(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeResult:
        self.calls.append((name, arguments))
        return self._result


async def test_adapter_satisfies_protocol_and_schema() -> None:
    tool = FakeTool(
        "weather",
        "Get the weather.",
        {"type": "object", "properties": {"city": {"type": "string"}}},
    )
    session = FakeSession([tool], FakeResult([]))
    adapter = MCPToolAdapter(tool, session)

    assert isinstance(adapter, Tool)

    registry = ToolRegistry()
    registry.register(adapter)
    schemas = registry.schemas()

    assert len(schemas) == 1
    fn = schemas[0]["function"]
    assert schemas[0]["type"] == "function"
    assert fn["name"] == "weather"
    assert fn["parameters"] == tool.inputSchema

    empty = MCPToolAdapter(FakeTool("noargs", "", None), session)
    assert empty.parameters == {"type": "object", "properties": {}}
    assert empty.description == ""


async def test_run_routes_to_call_tool_and_joins_text() -> None:
    tool = FakeTool("echo", "echo", {})
    result = FakeResult([FakeTextBlock("hello "), FakeNonTextBlock(), FakeTextBlock("world")])
    session = FakeSession([tool], result)
    adapter = MCPToolAdapter(tool, session)

    out = await adapter.run({"x": 1})

    assert session.calls == [("echo", {"x": 1})]
    assert out == "hello world"


async def test_run_falls_back_to_structured_content() -> None:
    tool = FakeTool("structured", "", {})
    result = FakeResult([FakeNonTextBlock()], structured_content={"value": 42})
    session = FakeSession([tool], result)
    adapter = MCPToolAdapter(tool, session)

    out = await adapter.run({})

    assert out == str({"value": 42})


async def test_run_returns_empty_string_when_no_text_no_structured() -> None:
    tool = FakeTool("blank", "", {})
    session = FakeSession([tool], FakeResult([FakeNonTextBlock()]))
    adapter = MCPToolAdapter(tool, session)

    assert await adapter.run({}) == ""
