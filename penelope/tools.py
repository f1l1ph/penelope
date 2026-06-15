"""Tools and the registry the loop invokes them through.

A `Tool` advertises an OpenAI-style JSON-Schema for its arguments and runs async.
The `ToolRegistry` maps names to tools and emits the schema list a provider needs.
`AddTool` is a minimal demo so the loop is exercisable end to end.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    async def run(self, arguments: dict[str, Any]) -> str: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def run(self, name: str, args: dict[str, Any]) -> str:
        return await self._tools[name].run(args)


class AddTool:
    name = "add"
    description = "Add two numbers."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        return str(arguments["a"] + arguments["b"])
