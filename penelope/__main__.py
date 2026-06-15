"""CLI entrypoint: `python -m penelope "what is 21 + 21?"`.

Reads the prompt from argv, builds an OpenAI-compatible provider from the
environment, registers the demo `add` tool plus any tools served by MCP servers
configured via PENELOPE_MCP, and streams the agent's events to stdout.
Configuration is environment-only; the API key is never printed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from .agent import Agent
from .mcp_tools import MCPServerConfig, MCPToolSource
from .provider import OpenAIProvider
from .tools import AddTool, ToolRegistry

SYSTEM_PROMPT = (
    "You are a helpful assistant. When a calculation is needed, use the provided "
    "tools rather than computing the answer yourself."
)


def _parse_mcp_config(raw: str) -> list[MCPServerConfig]:
    data = json.loads(raw)
    configs: list[MCPServerConfig] = []
    for entry in data:
        configs.append(
            MCPServerConfig(
                name=entry["name"],
                command=entry["command"],
                args=list(entry.get("args", [])),
                env=entry.get("env"),
            )
        )
    return configs


async def _drive(agent: Agent, prompt: str) -> None:
    async for ev in agent.run(prompt):
        if ev.type == "token":
            print(ev.data.get("text", ""), end="", flush=True)
        elif ev.type == "tool_call":
            print(
                f"\n[tool_call] {ev.data['name']}({ev.data['arguments']})",
                flush=True,
            )
        elif ev.type == "tool_result":
            print(
                f"[tool_result] {ev.data['name']} -> {ev.data['result']}",
                flush=True,
            )
        elif ev.type == "error":
            print(f"\n[error] {ev.data['message']}", file=sys.stderr, flush=True)
        elif ev.type == "done":
            print(flush=True)


async def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        print('usage: python -m penelope "your prompt here"', file=sys.stderr)
        return 2

    try:
        api_key = os.environ["VENICE_API_KEY"]
    except KeyError:
        print(
            "error: VENICE_API_KEY is not set in the environment.",
            file=sys.stderr,
        )
        return 1

    raw_mcp = os.environ.get("PENELOPE_MCP", "").strip()
    configs: list[MCPServerConfig] = []
    if raw_mcp:
        try:
            configs = _parse_mcp_config(raw_mcp)
        except (ValueError, KeyError, TypeError) as exc:
            print(f"error: PENELOPE_MCP is not valid JSON: {exc}", file=sys.stderr)
            return 1

    model = os.environ.get("PENELOPE_MODEL", "qwen-3-6-plus:disable_thinking=true")
    base_url = os.environ.get("PENELOPE_BASE_URL", "https://api.venice.ai/api/v1")

    provider = OpenAIProvider(model, api_key=api_key, base_url=base_url)
    registry = ToolRegistry()
    registry.register(AddTool())

    if not configs:
        agent = Agent(provider, registry, system_prompt=SYSTEM_PROMPT)
        await _drive(agent, prompt)
        return 0

    async with MCPToolSource(configs) as mcp_tools:
        for t in mcp_tools:
            registry.register(t)
        agent = Agent(provider, registry, system_prompt=SYSTEM_PROMPT)
        await _drive(agent, prompt)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
