"""Coding executors.

Each executor wraps an external coding CLI (Qwen Code, OpenCode, Gemini CLI,
Claude Code, ...) behind the single CodingExecutor contract defined in `base`.
The loop dispatches a coding task to whichever executor an agent is configured to
use; adding a new backend is a new plugin here, not a change to the loop.
"""

from penelope.executors.base import CodingExecutor, CodingTask, ExecutorEvent

__all__ = ["CodingExecutor", "CodingTask", "ExecutorEvent"]
