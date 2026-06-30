"""Expose tools served by MCP servers as `Tool`-protocol objects.

Each remote MCP tool is wrapped in an `MCPToolAdapter`, which carries the
`name`/`description`/`parameters` attributes plus an async `run` method the
tool registry and agent loop consume unchanged. Session lifecycle is owned by
`MCPToolSource`, an async context manager that spins up every configured server,
yields the adapters, and tears the sessions down on exit.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

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
        self._stacks: list[AsyncExitStack] = []

    async def __aenter__(self) -> list[MCPToolAdapter]:
        adapters: list[MCPToolAdapter] = []
        # Each server gets its OWN AsyncExitStack, entered and (on failure) closed
        # within this same task. The MCP SDK runs each server inside an anyio cancel
        # scope; sharing one stack across servers and unwinding them together is the
        # documented cause of "exit cancel scope in a different task" crashes once
        # two or more servers are configured. Isolated stacks also let one server
        # fail without dropping the others.
        for config in self._configs:
            stack = AsyncExitStack()
            await stack.__aenter__()
            try:
                params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env=config.env,
                )
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed = await session.list_tools()
                # Tool-name collisions across servers are out of scope: last registered wins.
                server_adapters = [MCPToolAdapter(tool, session) for tool in listed.tools]
            except Exception as exc:
                logger.warning(
                    "MCP server %r unavailable, skipping: %s", config.name, exc
                )
                await stack.aclose()
                continue
            self._stacks.append(stack)
            adapters.extend(server_adapters)
        return adapters

    async def __aexit__(self, *exc_info: Any) -> bool | None:
        for stack in reversed(self._stacks):
            try:
                await stack.aclose()
            except Exception:  # noqa: BLE001 - one noisy teardown must not mask others
                pass
        return None
