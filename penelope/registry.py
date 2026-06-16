"""In-memory catalog of agents and system prompts, plus the agent builder.

`Catalog` is loaded once from the database and then read on the hot path: the loop
never touches the database directly. `build_agent` turns a catalog row into a live
`Agent` wired to the row's model and resolved system prompt.
"""

from __future__ import annotations

from .agent import Agent
from .provider import OpenAIProvider
from .store.catalog import (
    AgentRecord,
    SystemPromptRecord,
    fetch_agents,
    fetch_system_prompts,
)
from .tools import ToolRegistry


class Catalog:
    def __init__(
        self,
        agents: dict[str, AgentRecord],
        system_prompts: dict[str, SystemPromptRecord],
    ) -> None:
        self._agents = agents
        self._system_prompts = system_prompts

    @classmethod
    async def load(cls, pool) -> "Catalog":
        agents = await fetch_agents(pool)
        system_prompts = await fetch_system_prompts(pool)
        return cls(
            {a.id: a for a in agents},
            {p.id: p for p in system_prompts},
        )

    def get_agent(self, agent_id: str) -> AgentRecord:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise KeyError(f"no agent with id {agent_id!r}") from None

    def list_agents(self) -> list[AgentRecord]:
        return list(self._agents.values())

    def system_prompt_body(self, agent: AgentRecord) -> str | None:
        if agent.system_prompt_id is None:
            return None
        prompt = self._system_prompts.get(agent.system_prompt_id)
        return prompt.body if prompt is not None else None


def build_agent(
    catalog: Catalog,
    agent_id: str,
    tools: ToolRegistry,
    *,
    api_key: str,
    base_url: str,
) -> Agent:
    record = catalog.get_agent(agent_id)
    provider = OpenAIProvider(record.model, api_key=api_key, base_url=base_url)
    body = catalog.system_prompt_body(record)
    return Agent(provider, tools, system_prompt=body)
