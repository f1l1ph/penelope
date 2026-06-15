"""The provider contract and an OpenAI-compatible implementation.

A `Provider` turns a message list plus tool schemas into a stream of
`ProviderChunk`s. Text arrives incrementally as `text_delta`s; tool calls are
assembled internally from streamed fragments and returned complete, once, on the
final chunk. The loop never sees partial tool-call fragments.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import AsyncOpenAI


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ProviderChunk:
    text_delta: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None


class Provider(ABC):
    @abstractmethod
    def stream(
        self, messages: list[dict], tools: list[dict]
    ) -> AsyncIterator[ProviderChunk]:
        """Stream model output for `messages`. `tools` is the OpenAI tool-schema list."""
        ...


class OpenAIProvider(Provider):
    """Thin wrapper over the async OpenAI SDK, usable against any OpenAI-compatible
    endpoint (base_url is configurable)."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._client = client

    async def stream(
        self, messages: list[dict], tools: list[dict]
    ) -> AsyncIterator[ProviderChunk]:
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or None,
            stream=True,
        )

        # Tool calls stream as partial fragments keyed by index: the first fragment
        # for an index carries id+name, later fragments append to the arguments string.
        fragments: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None

        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta is not None and delta.content:
                yield ProviderChunk(text_delta=delta.content)

            if delta is not None and delta.tool_calls:
                for tc in delta.tool_calls:
                    frag = fragments.setdefault(
                        tc.index, {"id": None, "name": None, "arguments": ""}
                    )
                    if tc.id:
                        frag["id"] = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            frag["name"] = tc.function.name
                        if tc.function.arguments:
                            frag["arguments"] += tc.function.arguments

            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

        assembled: list[ToolCall] | None = None
        if fragments:
            assembled = []
            for _, frag in sorted(fragments.items()):
                raw = frag["arguments"]
                arguments = json.loads(raw) if raw else {}
                assembled.append(
                    ToolCall(
                        id=frag["id"] or "",
                        name=frag["name"] or "",
                        arguments=arguments,
                    )
                )

        yield ProviderChunk(tool_calls=assembled, finish_reason=finish_reason)
