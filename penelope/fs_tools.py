"""Filesystem tools confined to a set of allowed roots.

Every tool here takes `roots` (a list of absolute directories) and refuses to
touch anything outside them. `_resolve_in_roots` resolves a candidate path and
each root to their realpaths before comparing, so both `..` traversal and
symlink escapes are blocked. Tools return strings and never raise out of
`run` - any failure (including a confinement violation) becomes an
`"error: ..."` string. `WriteFileTool` is root-scoped, which is the only write
guard until a permission policy lands.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path

_READ_CAP = 100_000
_LIST_CAP = 500
_GLOB_CAP = 300
_GREP_LINE_CAP = 200
_GREP_CHAR_CAP = 20_000


def _resolve_in_roots(path: str, roots: list[str]) -> Path:
    """Resolve `path` to a realpath and confirm it is within an allowed root.

    Raises ValueError if `path` resolves outside every root. Both the candidate
    and the roots are fully resolved first, so `..` and symlink escapes fail.
    """
    resolved = Path(path).expanduser().resolve()
    for root in roots:
        root_resolved = Path(root).expanduser().resolve()
        if resolved == root_resolved or resolved.is_relative_to(root_resolved):
            return resolved
    raise ValueError(f"path is outside the allowed roots: {path}")


class ReadFileTool:
    __slots__ = ("_roots",)

    name = "read_file"
    description = "Read a UTF-8 text file within the allowed roots and return its contents."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
        },
        "required": ["path"],
    }

    def __init__(self, roots: list[str]) -> None:
        self._roots = roots

    async def run(self, arguments: dict) -> str:
        try:
            resolved = _resolve_in_roots(arguments["path"], self._roots)
            text = resolved.read_text(encoding="utf-8", errors="replace")
            if len(text) > _READ_CAP:
                return text[:_READ_CAP] + "\n[truncated]"
            return text
        except Exception as exc:  # noqa: BLE001 - tools never raise out of run()
            return f"error: {exc}"


class ListDirTool:
    __slots__ = ("_roots",)

    name = "list_dir"
    description = "List the entries of a directory within the allowed roots."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the directory to list."},
        },
        "required": ["path"],
    }

    def __init__(self, roots: list[str]) -> None:
        self._roots = roots

    async def run(self, arguments: dict) -> str:
        try:
            resolved = _resolve_in_roots(arguments["path"], self._roots)
            if not resolved.is_dir():
                return f"error: not a directory: {arguments['path']}"
            entries = sorted(
                p.name + ("/" if p.is_dir() else "") for p in resolved.iterdir()
            )
            truncated = len(entries) > _LIST_CAP
            entries = entries[:_LIST_CAP]
            out = "\n".join(entries)
            if truncated:
                out += "\n[truncated]"
            return out or "(empty)"
        except Exception as exc:  # noqa: BLE001 - tools never raise out of run()
            return f"error: {exc}"


class GlobTool:
    __slots__ = ("_roots",)

    name = "glob"
    description = (
        "Find files matching a glob pattern within an allowed root. Returns "
        "paths relative to the searched root."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'."},
            "path": {
                "type": "string",
                "description": "Root to search within; defaults to the first allowed root.",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, roots: list[str]) -> None:
        self._roots = roots

    async def run(self, arguments: dict) -> str:
        try:
            base = arguments.get("path") or (self._roots[0] if self._roots else ".")
            base_resolved = _resolve_in_roots(base, self._roots)
            matches = []
            for p in base_resolved.glob(arguments["pattern"]):
                matches.append(str(p.relative_to(base_resolved)))
                if len(matches) >= _GLOB_CAP:
                    break
            matches.sort()
            return "\n".join(matches) or "(no matches)"
        except Exception as exc:  # noqa: BLE001 - tools never raise out of run()
            return f"error: {exc}"


class GrepTool:
    __slots__ = ("_roots",)

    name = "grep"
    description = (
        "Search file contents for a regular expression within an allowed root. "
        "Returns 'path:lineno: line' matches."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for."},
            "path": {
                "type": "string",
                "description": "Root to search within; defaults to the first allowed root.",
            },
            "glob": {
                "type": "string",
                "description": "Optional glob to limit which files are searched.",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, roots: list[str]) -> None:
        self._roots = roots

    async def run(self, arguments: dict) -> str:
        try:
            base = arguments.get("path") or (self._roots[0] if self._roots else ".")
            base_resolved = _resolve_in_roots(base, self._roots)
            pattern = arguments["pattern"]
            glob = arguments.get("glob")

            rg = shutil.which("rg")
            if rg is not None:
                return await self._grep_rg(rg, pattern, base_resolved, glob)
            return self._grep_python(pattern, base_resolved, glob)
        except Exception as exc:  # noqa: BLE001 - tools never raise out of run()
            return f"error: {exc}"

    async def _grep_rg(
        self, rg: str, pattern: str, base: Path, glob: str | None
    ) -> str:
        argv = [rg, "--no-heading", "--line-number", "--color", "never"]
        if glob:
            argv += ["--glob", glob]
        argv += [pattern, str(base)]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        text = (stdout or b"").decode("utf-8", errors="replace")
        lines = text.splitlines()
        return self._cap(lines)

    def _grep_python(self, pattern: str, base: Path, glob: str | None) -> str:
        regex = re.compile(pattern)
        results: list[str] = []
        paths = base.rglob(glob) if glob else base.rglob("*")
        for p in paths:
            if not p.is_file():
                continue
            try:
                with p.open("r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if regex.search(line):
                            results.append(f"{p}:{lineno}: {line.rstrip()}")
                            if len(results) >= _GREP_LINE_CAP:
                                return self._cap(results)
            except OSError:
                continue
        return self._cap(results)

    @staticmethod
    def _cap(lines: list[str]) -> str:
        lines = lines[:_GREP_LINE_CAP]
        out = "\n".join(lines)
        if len(out) > _GREP_CHAR_CAP:
            out = out[:_GREP_CHAR_CAP] + "\n[truncated]"
        return out or "(no matches)"


class WriteFileTool:
    __slots__ = ("_roots",)

    name = "write_file"
    description = (
        "Write UTF-8 text to a file within the allowed roots, creating parent "
        "directories as needed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write."},
            "content": {"type": "string", "description": "Text content to write."},
        },
        "required": ["path", "content"],
    }

    def __init__(self, roots: list[str]) -> None:
        self._roots = roots

    async def run(self, arguments: dict) -> str:
        try:
            resolved = _resolve_in_roots(arguments["path"], self._roots)
            content = arguments["content"]
            resolved.parent.mkdir(parents=True, exist_ok=True)
            data = content.encode("utf-8")
            resolved.write_bytes(data)
            return f"wrote {len(data)} bytes to {resolved}"
        except Exception as exc:  # noqa: BLE001 - tools never raise out of run()
            return f"error: {exc}"
