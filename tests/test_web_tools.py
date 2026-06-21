"""Network-free tests for WebSearchTool and WebFetchTool.

All network I/O is bypassed via injected async callables so the suite runs
offline and deterministically.
"""

from __future__ import annotations

import pytest
from penelope.web_tools import WebSearchTool, WebFetchTool


async def _make_searcher(results):
    async def searcher(query, n):
        return results[:n]
    return searcher


@pytest.mark.asyncio
async def test_search_returns_numbered_list():
    fake_results = [
        {"title": "Result A", "url": "https://a.example", "snippet": "Snippet A"},
        {"title": "Result B", "url": "https://b.example", "snippet": "Snippet B"},
    ]

    async def searcher(query, n):
        return fake_results[:n]

    tool = WebSearchTool(searcher=searcher)
    output = await tool.run({"query": "test"})

    assert "1." in output
    assert "Result A" in output
    assert "https://a.example" in output
    assert "Result B" in output


@pytest.mark.asyncio
async def test_search_respects_max_results():
    fake_results = [
        {"title": f"Result {i}", "url": f"https://{i}.example", "snippet": f"Snippet {i}"}
        for i in range(5)
    ]

    async def searcher(query, n):
        return fake_results[:n]

    tool = WebSearchTool(searcher=searcher)
    output = await tool.run({"query": "test", "max_results": 2})

    assert "3." not in output


@pytest.mark.asyncio
async def test_fetch_extracts_text_from_html():
    async def fetcher(url):
        return "<html><body><p>Hello readable world.</p></body></html>"

    tool = WebFetchTool(fetcher=fetcher)
    output = await tool.run({"url": "https://example.com"})

    assert "Hello readable world" in output
    assert not output.startswith("<")


@pytest.mark.asyncio
async def test_fetch_truncates_at_max_chars():
    async def fetcher(url):
        return "<html><body><p>" + "X" * 1000 + "</p></body></html>"

    tool = WebFetchTool(fetcher=fetcher, max_chars=50)
    output = await tool.run({"url": "https://example.com"})

    assert len(output) <= 65  # max_chars=50 + len("\n[truncated]")=12 = 62
    assert "[truncated]" in output


@pytest.mark.asyncio
async def test_search_error_returns_error_string():
    async def searcher(query, n):
        raise RuntimeError("boom")

    tool = WebSearchTool(searcher=searcher)
    output = await tool.run({"query": "test"})

    assert output.startswith("error:")


@pytest.mark.asyncio
async def test_fetch_error_returns_error_string():
    async def fetcher(url):
        raise RuntimeError("bang")

    tool = WebFetchTool(fetcher=fetcher)
    output = await tool.run({"url": "https://example.com"})

    assert output.startswith("error:")
