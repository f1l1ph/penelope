"""Pure-normalizer tests for the qwen-code executor.

These feed real captured stdout envelopes through `normalize_envelope` and assert
the resulting ExecutorEvents. No subprocess, no network.
"""

from __future__ import annotations

from penelope.executors.qwen_code import normalize_envelope


def test_assistant_tool_use() -> None:
    envelope = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "thinking", "thinking": "let me write the file"},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "write_file",
                    "input": {"file_path": "/x/hello.txt", "content": "penelope"},
                },
            ]
        },
    }

    events = normalize_envelope(envelope)

    assert len(events) == 1
    ev = events[0]
    assert ev.type == "tool_call"
    assert ev.data == {
        "id": "call_1",
        "name": "write_file",
        "input": {"file_path": "/x/hello.txt", "content": "penelope"},
    }


def test_user_tool_result() -> None:
    envelope = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "is_error": False,
                    "content": "Successfully wrote file.",
                }
            ]
        },
    }

    events = normalize_envelope(envelope)

    assert len(events) == 1
    ev = events[0]
    assert ev.type == "tool_result"
    assert ev.data["id"] == "call_1"
    assert ev.data["is_error"] is False
    assert ev.data["content"] == "Successfully wrote file."


def test_assistant_text() -> None:
    envelope = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "Done."}]},
    }

    events = normalize_envelope(envelope)

    assert len(events) == 1
    assert events[0].type == "token"
    assert events[0].data == {"text": "Done."}


def test_result_success() -> None:
    envelope = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "Done.",
    }

    events = normalize_envelope(envelope)

    assert len(events) == 1
    assert events[0].type == "done"
    assert events[0].data["result"] == "Done."
    assert all(ev.type != "error" for ev in events)


def test_result_error() -> None:
    envelope = {
        "type": "result",
        "subtype": "error",
        "is_error": True,
        "result": "boom",
    }

    events = normalize_envelope(envelope)

    assert [ev.type for ev in events] == ["error", "done"]
    assert events[0].data["message"] == "boom"


def test_system_and_unknown_yield_nothing() -> None:
    system_envelope = {
        "type": "system",
        "subtype": "init",
        "session_id": "s1",
        "model": "some-model",
        "cwd": "/x",
        "tools": [],
    }
    unknown_envelope = {"type": "mystery", "payload": 1}

    assert normalize_envelope(system_envelope) == []
    assert normalize_envelope(unknown_envelope) == []
    assert normalize_envelope({}) == []
