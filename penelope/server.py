"""FastAPI surface for the agent runtime: `/health` and a streaming `/chat`.

`create_app` accepts an injectable `agent_factory` so tests can supply a fake
agent and avoid any network or provider dependency. The default factory builds a
provider and registry from the environment, matching the CLI's default path.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .agent import Agent


class ChatRequest(BaseModel):
    message: str
    agent_id: str | None = None


def _default_agent_factory(message_meta: dict[str, Any]) -> Agent:
    from .__main__ import SYSTEM_PROMPT, _build_registry
    from .provider import OpenAIProvider

    api_key = os.environ["VENICE_API_KEY"]
    model = os.environ.get("PENELOPE_MODEL", "qwen-3-6-plus:disable_thinking=true")
    base_url = os.environ.get("PENELOPE_BASE_URL", "https://api.venice.ai/api/v1")
    provider = OpenAIProvider(model, api_key=api_key, base_url=base_url)
    return Agent(provider, _build_registry(), system_prompt=SYSTEM_PROMPT)


def create_app(*, agent_factory: Callable[[dict[str, Any]], Agent] | None = None) -> FastAPI:
    factory = agent_factory or _default_agent_factory
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/chat")
    async def chat(body: ChatRequest) -> StreamingResponse:
        # agent_id catalog resolution is deferred; always use the default/injected agent.
        # MCP tools in the server path are deferred (session lifecycle vs request scope);
        # server agents get AddTool + delegate_coding only. The CLI keeps full MCP.
        # Auth is deferred: the server binds to 127.0.0.1 only.
        agent = factory({"agent_id": body.agent_id})

        async def generator() -> AsyncIterator[str]:
            async for ev in agent.run(body.message):
                yield f"data: {json.dumps({'type': ev.type, 'data': ev.data})}\n\n"
                if ev.type == "done":
                    break

        return StreamingResponse(generator(), media_type="text/event-stream")

    return app
