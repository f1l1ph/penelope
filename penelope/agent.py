"""The streaming tool-calling loop.

`Agent.run` drives a provider in turns: stream text as `token` events, run any
tool calls the provider returns, feed the results back, and repeat until the model
stops calling tools or the turn cap is hit. Every path ends in exactly one `done`,
and the loop never raises - any failure surfaces as a single `error` event.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from .events import Event
from .provider import Provider, ToolCall
from .tools import ToolRegistry


class Agent:
    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry,
        system_prompt: str | None = None,
        max_turns: int = 8,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_turns = max_turns

    async def run(
        self, user_message: str, history: list[dict] | None = None
    ) -> AsyncIterator[Event]:
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        schemas = self.tools.schemas()

        try:
            for _turn in range(self.max_turns):
                text_parts: list[str] = []
                tool_calls: list[ToolCall] | None = None

                async for chunk in self.provider.stream(messages, schemas):
                    if chunk.text_delta:
                        text_parts.append(chunk.text_delta)
                        yield Event("token", {"text": chunk.text_delta})
                    if chunk.tool_calls is not None:
                        tool_calls = chunk.tool_calls

                assistant_text = "".join(text_parts)
                assistant_msg: dict = {"role": "assistant", "content": assistant_text}
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in tool_calls
                    ]
                messages.append(assistant_msg)

                if not tool_calls:
                    yield Event("done", {})
                    return

                for tc in tool_calls:
                    yield Event(
                        "tool_call",
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments},
                    )
                    result = await self.tools.run(tc.name, tc.arguments)
                    yield Event(
                        "tool_result",
                        {"id": tc.id, "name": tc.name, "result": result},
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )

            yield Event("error", {"message": "max turns exceeded"})
            yield Event("done", {})
        except Exception as e:  # noqa: BLE001 - single uniform failure path
            yield Event("error", {"message": str(e)})
            yield Event("done", {})
