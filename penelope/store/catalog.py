"""asyncpg-backed catalog store.

The database is the source of truth for agents and system prompts. This module
owns connecting, applying the (idempotent) schema, seeding a runnable default, and
fetching rows. It never logs the connection string.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import asyncpg

_DEFAULT_MODEL = "qwen-3-6-plus:disable_thinking=true"
_DEFAULT_PROMPT_BODY = (
    "You are a helpful assistant. Use the provided tools when they help answer the request."
)


@dataclass(slots=True)
class AgentRecord:
    id: str
    name: str
    model: str
    system_prompt_id: str | None


@dataclass(slots=True)
class SystemPromptRecord:
    id: str
    name: str
    body: str


def _load_schema() -> str:
    return resources.files(__package__).joinpath("schema.sql").read_text(encoding="utf-8")


async def connect(dsn: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn)


async def apply_schema(pool: asyncpg.Pool) -> None:
    sql = _load_schema()
    async with pool.acquire() as con:
        await con.execute(sql)


async def seed(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as con:
        await con.execute(
            """
            INSERT INTO penelope.system_prompts (id, name, body)
            VALUES ($1, $2, $3)
            ON CONFLICT (id) DO NOTHING
            """,
            "default",
            "Default",
            _DEFAULT_PROMPT_BODY,
        )
        await con.execute(
            """
            INSERT INTO penelope.agents (id, name, model, system_prompt_id)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (id) DO NOTHING
            """,
            "default",
            "Default",
            _DEFAULT_MODEL,
            "default",
        )


async def fetch_agents(pool: asyncpg.Pool) -> list[AgentRecord]:
    rows = await pool.fetch(
        "SELECT id, name, model, system_prompt_id FROM penelope.agents"
    )
    return [
        AgentRecord(
            id=row["id"],
            name=row["name"],
            model=row["model"],
            system_prompt_id=row["system_prompt_id"],
        )
        for row in rows
    ]


async def fetch_system_prompts(pool: asyncpg.Pool) -> list[SystemPromptRecord]:
    rows = await pool.fetch("SELECT id, name, body FROM penelope.system_prompts")
    return [
        SystemPromptRecord(id=row["id"], name=row["name"], body=row["body"])
        for row in rows
    ]
