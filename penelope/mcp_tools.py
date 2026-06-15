"""Expose tools served by MCP servers as `Tool`-protocol objects.

Each remote MCP tool is wrapped in an `MCPToolAdapter`, which carries the
`name`/`description`/`parameters` attributes plus an async `run` method the
tool registry and agent loop consume unchanged. Session lifecycle is owned by
`MCPToolSource`, an async context manager that spins up every configured server,
yields the adapters, and tears the sessions down on exit.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass(slots=True)
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


class MCPToolAdapter:
    """Adapt a single MCP tool to the `Tool` protocol over a live session."""

    def __init__(self, tool: Any, session: ClientSession) -> None:
        self.name: str = tool.name
        self.description: str = getattr(tool, "description", "") or ""
        schema = getattr(tool, "inputSchema", None)
        self.parameters: dict[str, Any] = schema or dict(_EMPTY_SCHEMA)
        self._session = session

    async def run(self, arguments: dict[str, Any]) -> str:
        result = await self._session.call_tool(self.name, arguments)
        texts = [
            block.text
            for block in getattr(result, "content", []) or []
            if getattr(block, "type", None) == "text"
        ]
        if texts:
            return "".join(texts)
        return str(getattr(result, "structuredContent", "") or "")


class MCPToolSource:
    """Async context manager owning the sessions for a set of MCP servers."""

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self._configs = configs
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> list[MCPToolAdapter]:
        await self._stack.__aenter__()
        adapters: list[MCPToolAdapter] = []
        try:
            for config in self._configs:
                params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env=config.env,
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed = await session.list_tools()
                # Tool-name collisions across servers are out of scope: last registered wins.
                for tool in listed.tools:
                    adapters.append(MCPToolAdapter(tool, session))
        except Exception as exc:
            await self._stack.aclose()
            raise RuntimeError(f"failed to start MCP server {config.name!r}: {exc}") from exc
        return adapters

    async def __aexit__(self, *exc_info: Any) -> bool | None:
        return await self._stack.__aexit__(*exc_info)
