"""Tests for RunShellTool.

These run real subprocesses (no network). Commands are trivial and kept
reasonably portable; the suite uses pytest-asyncio in auto mode, so test
functions are plain `async def` with no decorator.
"""

from __future__ import annotations

from penelope.shell_tool import RunShellTool


async def test_echo_returns_output_and_exit_code():
    tool = RunShellTool(cwd=".")
    result = await tool.run({"command": "echo hello"})
    assert "hello" in result
    assert "[exit code: 0]" in result


async def test_timeout_kills_and_reports():
    tool = RunShellTool(cwd=".", timeout=1)
    result = await tool.run({"command": "sleep 5"})
    assert result == "error: timed out after 1s"


async def test_disabled_returns_message_without_running():
    tool = RunShellTool(cwd=".", enabled=False)
    result = await tool.run({"command": "echo should-not-run"})
    assert result == "error: shell is disabled"


async def test_nonzero_exit_code_reported():
    tool = RunShellTool(cwd=".")
    result = await tool.run({"command": "exit 3"})
    assert "[exit code: 3]" in result
