from __future__ import annotations

import json

from fastapi.testclient import TestClient

from penelope.events import Event
from penelope.server import create_app

_SCRIPT = [
    Event("token", {"text": "hi"}),
    Event("tool_call", {"id": "1", "name": "add", "arguments": {"a": 12, "b": 30}}),
    Event("tool_result", {"id": "1", "name": "add", "result": "42"}),
    Event("done", {}),
]


class FakeAgent:
    async def run(self, message):
        for ev in _SCRIPT:
            yield ev


def _client() -> TestClient:
    return TestClient(create_app(agent_factory=lambda meta: FakeAgent()))


def _frames(body: str) -> list[dict]:
    frames = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        assert chunk.startswith("data: ")
        frames.append(json.loads(chunk[len("data: "):]))
    return frames


def test_health():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_chat_streams_scripted_events():
    resp = _client().post("/chat", json={"message": "x"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    frames = _frames(resp.text)
    assert [f["type"] for f in frames] == ["token", "tool_call", "tool_result", "done"]
    assert frames[0]["data"] == {"text": "hi"}
    assert frames[-1]["type"] == "done"


def test_chat_frame_shape():
    resp = _client().post("/chat", json={"message": "x"})
    frames = _frames(resp.text)
    assert len(frames) == 4
    for f in frames:
        assert isinstance(f, dict)
        assert "type" in f and "data" in f
