-- Penelope catalog schema. Idempotent: safe to run on every startup.
-- Tool allowlists, subagents, skills, and versioning are intentionally deferred.

CREATE SCHEMA IF NOT EXISTS penelope;

CREATE TABLE IF NOT EXISTS penelope.system_prompts (
    id          text PRIMARY KEY,
    name        text NOT NULL,
    body        text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS penelope.agents (
    id                text PRIMARY KEY,
    name              text NOT NULL,
    model             text NOT NULL,
    system_prompt_id  text REFERENCES penelope.system_prompts(id),
    created_at        timestamptz NOT NULL DEFAULT now()
);
