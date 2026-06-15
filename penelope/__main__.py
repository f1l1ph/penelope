"""CLI entrypoint: `python -m penelope "what is 21 + 21?"`.

Reads the prompt from argv, builds an OpenAI-compatible provider from the
environment, registers the demo `add` tool, and streams the agent's events to
stdout. Configuration is environment-only; the API key is never printed.
"""

from __future__ import annotations

import asyncio
import os
import sys

from .agent import Agent
from .provider import OpenAIProvider
from .tools import AddTool, ToolRegistry

SYSTEM_PROMPT = (
    "You are a helpful assistant. When a calculation is needed, use the provided "
    "tools rather than computing the answer yourself."
)


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

    model = os.environ.get("PENELOPE_MODEL", "qwen-3-6-plus:disable_thinking=true")
    base_url = os.environ.get("PENELOPE_BASE_URL", "https://api.venice.ai/api/v1")

    provider = OpenAIProvider(model, api_key=api_key, base_url=base_url)
    registry = ToolRegistry()
    registry.register(AddTool())
    agent = Agent(provider, registry, system_prompt=SYSTEM_PROMPT)

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

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
