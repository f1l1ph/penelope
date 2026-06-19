"""A tool that runs a shell command in a fixed working directory.

`RunShellTool` executes a command via the system shell in its configured
`cwd`, enforces a wall-clock timeout (killing the process on expiry), and
returns the combined stdout+stderr plus the exit code. It can be disabled at
construction so a caller can gate it behind an external flag. Like every tool
it returns a string and never raises out of `run`.
"""

from __future__ import annotations

import asyncio

_OUTPUT_CAP = 20_000


class RunShellTool:
    __slots__ = ("_cwd", "_timeout", "_enabled")

    name = "run_shell"
    description = (
        "Run a shell command in the tool's working directory and return the "
        "combined stdout and stderr followed by the process exit code. The "
        "command runs through the system shell."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
        },
        "required": ["command"],
    }

    def __init__(self, *, cwd: str, timeout: int = 60, enabled: bool = True) -> None:
        self._cwd = cwd
        self._timeout = timeout
        self._enabled = enabled

    async def run(self, arguments: dict) -> str:
        if not self._enabled:
            return "error: shell is disabled"

        command = arguments.get("command")
        if not command:
            return "error: missing required argument 'command'"

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=self._cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"error: timed out after {self._timeout}s"

            output = (stdout or b"").decode("utf-8", errors="replace")
            if len(output) > _OUTPUT_CAP:
                output = output[:_OUTPUT_CAP] + "\n[output truncated]"
            return f"{output}\n[exit code: {proc.returncode}]"
        except Exception as exc:  # noqa: BLE001 - tools never raise out of run()
            return f"error: {exc}"
