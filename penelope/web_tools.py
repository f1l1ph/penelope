"""Web search and fetch tools for Penelope.

WebSearchTool: search the public web via SearXNG (when searx_url is set),
DuckDuckGo (ddgs), or an injected async callable (for tests). Returns a
compact numbered list.

WebFetchTool: fetch a URL with httpx and extract readable text via
trafilatura. Falls back to tag-stripped text if trafilatura returns nothing.

A Playwright fallback for JS-heavy pages is future work (this slice is
httpx + trafilatura only).
"""

from __future__ import annotations

import re


class WebSearchTool:
    __slots__ = ("_searx_url", "_searcher")

    name = "web_search"
    description = (
        "Search the public web and return the top results (title, url, snippet). "
        "Use to find current information, sources, and links."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, *, searx_url: str | None = None, searcher=None) -> None:
        self._searx_url = searx_url
        self._searcher = searcher

    async def run(self, arguments: dict) -> str:
        try:
            query = arguments["query"]
            max_results = arguments.get("max_results") or 5

            if self._searcher is not None:
                results = await self._searcher(query, max_results)
            elif self._searx_url:
                import httpx

                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{self._searx_url}/search",
                        params={"q": query, "format": "json"},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                raw = data.get("results", [])[:max_results]
                results = [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                    }
                    for item in raw
                ]
            else:
                try:
                    from ddgs import DDGS
                except ImportError:
                    return "error: web search failed: ddgs not installed and no searx_url configured"

                raw = DDGS().text(query, max_results=max_results)
                results = [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("href", item.get("url", "")),
                        "snippet": item.get("body", item.get("snippet", "")),
                    }
                    for item in (raw or [])
                ]

            lines: list[str] = []
            for i, r in enumerate(results[:max_results], 1):
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("snippet", "")
                lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
            return "\n\n".join(lines)
        except Exception as exc:  # noqa: BLE001 - tools never raise out of run()
            return f"error: web search failed: {exc}"


class WebFetchTool:
    __slots__ = ("_fetcher", "_max_chars")

    name = "web_fetch"
    description = "Fetch a web page and return its readable text content (not raw HTML)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch.",
            },
        },
        "required": ["url"],
    }

    def __init__(self, *, fetcher=None, max_chars: int = 20000) -> None:
        self._fetcher = fetcher
        self._max_chars = max_chars

    async def run(self, arguments: dict) -> str:
        try:
            url = arguments["url"]

            if self._fetcher is not None:
                html = await self._fetcher(url)
            else:
                import httpx

                async with httpx.AsyncClient(
                    headers={"User-Agent": "Mozilla/5.0 (compatible; PenelopeBot/1.0)"},
                    timeout=15,
                    follow_redirects=True,
                ) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    html = resp.text

            try:
                import trafilatura

                text = trafilatura.extract(html)
            except Exception:  # noqa: BLE001
                text = None

            if not text:
                text = re.sub(r"<[^>]+>", " ", html)
                text = re.sub(r"\s+", " ", text).strip()

            if len(text) > self._max_chars:
                text = text[: self._max_chars] + "\n[truncated]"

            return text
        except Exception as exc:  # noqa: BLE001 - tools never raise out of run()
            return f"error: web fetch failed: {exc}"
