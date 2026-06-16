"""CLI entrypoint: `python -m penelope "what is 21 + 21?"`.

Reads the prompt from argv, builds an OpenAI-compatible provider from the
environment, registers the demo `add` tool plus any tools served by MCP servers
configured via PENELOPE_MCP, and streams the agent's events to stdout.

Agents can also be loaded from a database: `--list-agents` prints the catalog and
`--agent <id> "prompt"` runs a stored agent (model + system prompt from the row).
The database is only touched when one of those flags is used; without them the CLI
behaves exactly as it does with no database at all. Configuration is
environment-only; the API key and connection string are never printed.
"""

from __future__ import annotations

import argparse
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


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(AddTool())
    return registry


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


async def _run_with_tools(make_agent, prompt: str, configs: list[MCPServerConfig]) -> None:
    """Build the registry (plus optional MCP tools), construct the agent via
    `make_agent(registry)`, and drive it. MCP servers stay scoped to the run."""
    registry = _build_registry()
    if not configs:
        await _drive(make_agent(registry), prompt)
        return
    async with MCPToolSource(configs) as mcp_tools:
        for t in mcp_tools:
            registry.register(t)
        await _drive(make_agent(registry), prompt)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="penelope")
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--agent", default=None, help="run a stored agent by id")
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="list stored agents and exit",
    )
    parser.add_argument(
        "--code",
        action="store_true",
        help="run the prompt as a coding task via the qwen-code executor",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="workspace directory for --code (default: current directory)",
    )
    return parser.parse_args(argv)


async def _run_code(prompt: str, workspace: str) -> int:
    """Run a coding task through the qwen-code executor and stream events.

    This path requires no model provider and no API key: the executor inherits
    the ambient environment and the `qwen` CLI resolves its own configuration.
    """
    from .executors import CodingTask
    from .executors.qwen_code import QwenCodeExecutor

    executor = QwenCodeExecutor()
    task = CodingTask(prompt=prompt, workspace=os.path.abspath(workspace))

    saw_error = False
    async for ev in executor.run(task):
        if ev.type == "token":
            print(ev.data.get("text", ""), end="", flush=True)
        elif ev.type == "tool_call":
            print(f"\n[tool_call] {ev.data.get('name')}({ev.data.get('input')})", flush=True)
        elif ev.type == "tool_result":
            content = ev.data.get("content")
            text = str(content)
            if len(text) > 200:
                text = text[:200] + "..."
            print(
                f"[tool_result] {ev.data.get('id')} ({ev.data.get('is_error')}) -> {text}",
                flush=True,
            )
        elif ev.type == "error":
            print(f"\n[error] {ev.data.get('message')}", file=sys.stderr, flush=True)
            saw_error = True
        elif ev.type == "done":
            print(flush=True)
            if ev.data.get("result"):
                print(ev.data["result"], flush=True)
    return 1 if saw_error else 0


async def main() -> int:
    args = _parse_args(sys.argv[1:])

    if args.code:
        prompt = (args.prompt or "").strip()
        if not prompt:
            print('usage: python -m penelope --code "your coding task"', file=sys.stderr)
            return 2
        return await _run_code(prompt, args.workspace)

    use_db = args.list_agents or args.agent is not None

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

    base_url = os.environ.get("PENELOPE_BASE_URL", "https://api.venice.ai/api/v1")

    if not use_db:
        prompt = (args.prompt or "").strip()
        if not prompt:
            print('usage: python -m penelope "your prompt here"', file=sys.stderr)
            return 2
        model = os.environ.get("PENELOPE_MODEL", "qwen-3-6-plus:disable_thinking=true")
        provider = OpenAIProvider(model, api_key=api_key, base_url=base_url)

        def make_agent(registry: ToolRegistry) -> Agent:
            return Agent(provider, registry, system_prompt=SYSTEM_PROMPT)

        await _run_with_tools(make_agent, prompt, configs)
        return 0

    dsn = os.environ.get("PENELOPE_DATABASE_URL", "").strip()
    if not dsn:
        print(
            "error: PENELOPE_DATABASE_URL is not set in the environment.",
            file=sys.stderr,
        )
        return 1

    from .registry import Catalog, build_agent
    from .store.catalog import apply_schema, connect, seed

    pool = await connect(dsn)
    try:
        await apply_schema(pool)
        await seed(pool)
        catalog = await Catalog.load(pool)
    finally:
        await pool.close()

    if args.list_agents:
        for record in catalog.list_agents():
            print(f"{record.id}\t{record.name}\t{record.model}")
        return 0

    prompt = (args.prompt or "").strip()
    if not prompt:
        print('usage: python -m penelope --agent <id> "your prompt here"', file=sys.stderr)
        return 2

    try:
        record = catalog.get_agent(args.agent)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    def make_agent(registry: ToolRegistry) -> Agent:
        return build_agent(
            catalog, record.id, registry, api_key=api_key, base_url=base_url
        )

    await _run_with_tools(make_agent, prompt, configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
