"""Catalog cache and agent-builder tests. No database: the Catalog is built directly
from in-memory records, so nothing here touches Postgres or the network."""

from __future__ import annotations

import pytest

from penelope.registry import Catalog, build_agent
from penelope.store.catalog import AgentRecord, SystemPromptRecord
from penelope.tools import AddTool, ToolRegistry


def _catalog() -> Catalog:
    agents = {
        "default": AgentRecord("default", "Default", "model-x", "default"),
        "no_prompt": AgentRecord("no_prompt", "No Prompt", "model-y", None),
        "dangling": AgentRecord("dangling", "Dangling", "model-z", "missing"),
    }
    prompts = {
        "default": SystemPromptRecord("default", "Default", "be helpful"),
    }
    return Catalog(agents, prompts)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(AddTool())
    return registry


def test_get_agent_and_unknown_id() -> None:
    catalog = _catalog()

    assert catalog.get_agent("default").model == "model-x"
    with pytest.raises(KeyError):
        catalog.get_agent("nope")


def test_system_prompt_body_resolution() -> None:
    catalog = _catalog()

    assert catalog.system_prompt_body(catalog.get_agent("default")) == "be helpful"
    assert catalog.system_prompt_body(catalog.get_agent("no_prompt")) is None
    assert catalog.system_prompt_body(catalog.get_agent("dangling")) is None


def test_build_agent_wires_model_and_prompt() -> None:
    catalog = _catalog()
    agent = build_agent(
        catalog, "default", _registry(), api_key="x", base_url="y"
    )

    assert agent.provider.model == "model-x"
    assert agent.system_prompt == "be helpful"
