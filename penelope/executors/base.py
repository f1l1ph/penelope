"""The coding-executor contract.

Every coding backend implements `CodingExecutor`. The runtime never depends on a
specific backend's API shape: a backend's native output is normalized into
`ExecutorEvent`s here, so the loop and the UI see one event vocabulary regardless
of which CLI ran the task.

This is the single interface the whole pluggable-coder design rests on. It is the
one part of the runtime defined up front, deliberately, so that adding OpenCode,
Gemini CLI, or Claude Code later is a new subclass and nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class CodingTask:
    """A unit of coding work handed to an executor.

    Attributes:
        prompt: The natural-language instruction for the coding agent.
        workspace: Absolute path the executor is allowed to operate in. The
            executor must not act outside this root.
        session_id: Stable id so an executor that supports resume can continue an
            existing conversation instead of starting cold.
        model: Optional model override; when None the executor uses its default.
        context: Free-form extra context (files, constraints) the loop wants to
            pass through without the executor needing to rediscover it.
    """

    prompt: str
    workspace: str
    session_id: str | None = None
    model: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


EventType = Literal[
    "token",            # streamed assistant text
    "tool_call",        # the coder requested a tool/action
    "tool_result",      # an action completed
    "permission",       # an action needs human approval before it runs
    "error",            # the executor failed; carries a message
    "done",             # terminal event for this task
]


@dataclass(slots=True)
class ExecutorEvent:
    """A backend-agnostic event. Executors translate native output into these so
    the loop and UI never see backend-specific shapes."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)


class CodingExecutor(ABC):
    """Interface every coding backend implements.

    Implementations live alongside this file (e.g. `qwen_code.py`). They own the
    process/transport details of one CLI and nothing else - no loop logic, no
    persistence, no permission policy. Those belong to the runtime.
    """

    #: Short, stable identifier used in agent definitions and the UI (e.g. "qwen-code").
    name: str

    @abstractmethod
    async def run(self, task: CodingTask) -> AsyncIterator[ExecutorEvent]:
        """Run a coding task and yield normalized events until a terminal event.

        Must yield exactly one terminal event (`done` or `error`) and must not
        raise for ordinary backend failures - surface them as an `error` event so
        the loop has a single, uniform failure path.
        """
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> bool:
        """Return True if the backend is installed, reachable, and ready."""
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, session_id: str) -> None:
        """Best-effort cancel of in-flight work for a session."""
        raise NotImplementedError
