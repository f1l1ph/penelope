"""The normalized event vocabulary the agent loop emits.

`Agent.run` yields a stream of `Event`s. The five `EventType`s are the only shapes
a consumer (CLI, web UI, test) ever sees, regardless of provider or tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EventType = Literal["token", "tool_call", "tool_result", "error", "done"]


@dataclass(slots=True)
class Event:
    """A single streamed event.

    Payload conventions by type:
        token        {"text": str}
        tool_call    {"id": str, "name": str, "arguments": dict}
        tool_result  {"id": str, "name": str, "result": str}
        error        {"message": str}
        done         {}
    """

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
