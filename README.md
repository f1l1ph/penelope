# Penelope

A self-hosted agent runtime. Penelope runs a streaming tool-calling loop whose **agents, subagents, skills, and system prompts live in a database** and are edited from a web UI - not scattered across config files in a folder. Coding work is delegated to pluggable coding CLIs (Qwen Code, OpenCode, Gemini CLI, Claude Code, ...) rather than reimplemented.

> **Status: pre-alpha scaffold.** The interfaces and architecture are defined; the loop is not yet built. Not usable yet.

## Why

Most agent setups keep their agent definitions, prompts, and skills as files next to each other on disk. That works for one agent and falls apart for many - no clean editing surface, no history, no way for a UI (or a helper AI) to manage them. Penelope treats every agent, subagent, skill, and system prompt as a **database row**: load from Postgres, cache locally, edit in the UI, chat with the result.

Penelope also does not try to be a coding agent. Good coding agents already exist. Penelope is the **orchestrator** that calls them when a task needs code, and owns the parts they don't: a durable session store, a permission gate, a unified event stream, and a database-backed agent catalog.

## Principles

- **Database is the source of truth, local disk is a cache.** Definitions live in Postgres; Penelope syncs them to a local cache on startup and on change.
- **One executor interface.** Every coding backend implements the same `CodingExecutor` contract, so adding a new one is a plugin, not a rewrite.
- **Tools via MCP.** Tool calling speaks the Model Context Protocol, so the existing MCP ecosystem plugs in directly.
- **One event schema.** Every backend's output is normalized to Penelope's own streaming event types, so the UI never sees backend-specific shapes.
- **Permission-gated by default.** Tool and shell actions surface for approval rather than running silently.

## Architecture

```
            ┌──────────────────────────────────────────┐
            │                Web UI                     │
            │   edit agents / skills / prompts · chat   │
            └───────────────────┬──────────────────────┘
                                │ HTTP + SSE
            ┌───────────────────▼──────────────────────┐
            │                Penelope                   │
            │  ┌─────────────┐   ┌──────────────────┐   │
            │  │ Registry    │   │ Agent loop       │   │
            │  │ Postgres →  │──▶│ model · tools ·  │   │
            │  │ local cache │   │ permissions ·    │   │
            │  └─────────────┘   │ streaming        │   │
            │                    └────────┬─────────┘   │
            │   ┌────────────┐   ┌─────────▼─────────┐  │
            │   │ MCP tools  │   │ Coding executors  │  │
            │   └────────────┘   │ (pluggable)       │  │
            │                    │  qwen · opencode  │  │
            │                    │  gemini · claude  │  │
            │                    └───────────────────┘  │
            └──────────────────────────────────────────┘
```

## Concepts

| Concept | What it is | Stored in |
| --- | --- | --- |
| **Agent** | A named persona: model, system prompt, allowed tools, default executor | Postgres |
| **Subagent** | A scoped helper an agent can delegate to | Postgres |
| **Skill** | A reusable capability/instruction bundle an agent can load | Postgres |
| **System prompt** | A versioned prompt body, attachable to agents | Postgres |
| **Tool** | An MCP-exposed capability | MCP registry |
| **Coding executor** | A plugin that runs a coding task on an external CLI | Code (this repo) |

## Project layout

```
penelope/
  penelope/
    loop.py            # the agent loop (planned)
    registry.py        # load definitions from Postgres → local cache (planned)
    permissions.py     # approval gating (planned)
    events.py          # normalized streaming event schema (planned)
    tools/             # MCP client wiring (planned)
    executors/
      base.py          # the CodingExecutor interface (defined)
      qwen_code.py     # first executor plugin (planned)
  docs/
    ARCHITECTURE.md    # design detail
```

## Status & roadmap

This repo is a scaffold. The build order, roughly:

1. Provider abstraction + a minimal tool-calling loop.
2. The `CodingExecutor` interface + the first plugin (Qwen Code).
3. Postgres-backed registry with local cache.
4. Permission gating + the normalized event stream.
5. Additional executors (OpenCode, Gemini CLI, Claude Code).
6. Sandboxed execution.

## License

MIT - see [LICENSE](./LICENSE).
