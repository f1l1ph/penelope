"""Tests for the root-confined filesystem tools.

No network; everything runs against pytest tmp_path dirs. The suite uses
pytest-asyncio in auto mode, so test functions are plain `async def`.
"""

from __future__ import annotations

from penelope.fs_tools import (
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)


async def test_read_happy_path(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world", encoding="utf-8")
    tool = ReadFileTool([str(tmp_path)])
    assert await tool.run({"path": str(f)}) == "hello world"


async def test_list_dir_marks_directories(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    tool = ListDirTool([str(tmp_path)])
    out = await tool.run({"path": str(tmp_path)})
    assert "sub/" in out
    assert "a.txt" in out


async def test_glob_returns_relative_matches(tmp_path):
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    tool = GlobTool([str(tmp_path)])
    out = await tool.run({"pattern": "*.py", "path": str(tmp_path)})
    assert out == "a.py"


async def test_grep_finds_known_line(tmp_path):
    (tmp_path / "code.txt").write_text("alpha\nNEEDLE here\nbeta\n", encoding="utf-8")
    tool = GrepTool([str(tmp_path)])
    out = await tool.run({"pattern": "NEEDLE", "path": str(tmp_path)})
    assert "NEEDLE here" in out
    assert "code.txt" in out


async def test_write_lands_inside_root(tmp_path):
    target = tmp_path / "nested" / "out.txt"
    tool = WriteFileTool([str(tmp_path)])
    out = await tool.run({"path": str(target), "content": "data"})
    assert "wrote" in out
    assert target.read_text(encoding="utf-8") == "data"


async def test_read_outside_roots_blocked(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret", encoding="utf-8")
    tool = ReadFileTool([str(root)])
    out = await tool.run({"path": str(outside)})
    assert out.startswith("error:")
    assert "top secret" not in out


async def test_dotdot_traversal_blocked(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret", encoding="utf-8")
    tool = ReadFileTool([str(root)])
    out = await tool.run({"path": str(root / ".." / "secret.txt")})
    assert out.startswith("error:")
    assert "top secret" not in out


async def test_write_outside_roots_blocked(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "escape.txt"
    tool = WriteFileTool([str(root)])
    out = await tool.run({"path": str(target), "content": "nope"})
    assert out.startswith("error:")
    assert not target.exists()
