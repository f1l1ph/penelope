# Architecture

Penelope is an orchestrator, not a coding agent. It owns four things that coding CLIs don't give you - a database-backed catalog of agents, a durable session store, a permission gate, and a single event vocabulary - and delegates the actual coding to whichever CLI an agent is configured to use.

## The loop

The core is an ordinary tool-calling loop:

```
load agent definition (from cache) → assemble context → call model
   → model requests a tool         → gate → run tool      → feed result back
   → model requests coding work     → gate → dispatch to a CodingExecutor
   → model emits final text         → done
```

Streaming, cancellation, and a terminal event are guaranteed for every turn. The loop never blocks on a backend-specific signal; it consumes the normalized `ExecutorEvent` / event stream described below.

## Registry: database is truth, disk is cache

Agents, subagents, skills, and system prompts are rows in Postgres. On startup and on change, the registry syncs them into a local cache directory; the loop reads the cache, never the database directly, so a single request never waits on a query and a transient DB blip doesn't stall a live chat.

- **Edit** happens in the UI (or via a helper AI), which writes to Postgres.
- **Sync** pulls the changed rows into the local cache.
- **Load** is a cache read at dispatch time.

This keeps the editing surface rich (a real database, with history and a UI) while keeping the hot path fast and file-simple.

## Coding executors

The one interface the design rests on is [`CodingExecutor`](../penelope/executors/base.py). Each backend - Qwen Code first, then OpenCode, Gemini CLI, Claude Code - implements `run`, `health`, and `cancel`, and translates its native output into `ExecutorEvent`s. The loop picks an executor from the agent's configuration and consumes its events identically regardless of which CLI is underneath.

Adding a backend is a new subclass and a registry entry. It is never a change to the loop, the UI, or the event schema.

## Events

Every backend's output and every loop step is normalized to one set of event types (`token`, `tool_call`, `tool_result`, `permission`, `error`, `done`). The UI subscribes to this one stream. Backends that close their transport differently, signal completion differently, or nest tool calls differently are reconciled at the executor boundary - downstream code sees uniform events, including a single uniform failure path (`error`).

## Permissions

Tool and shell actions surface as a `permission` event and wait for an explicit decision before running. The default is to ask; auto-approval is opt-in per agent or per tool. The permission policy lives in the runtime, not in any backend, so it holds no matter which executor runs the action.

## Tools (MCP)

Tool calling speaks the Model Context Protocol. The runtime maintains an MCP client; configured MCP servers expose their tools to the loop. This avoids hand-writing tool plumbing and lets the existing MCP ecosystem plug in.

## What lives where

| Concern                                   | Owner                     |
| ----------------------------------------- | ------------------------- |
| Agent loop, context assembly              | runtime                   |
| Agent / subagent / skill / prompt catalog | Postgres + local cache    |
| Sessions, history                         | Postgres                  |
| Permission policy                         | runtime                   |
| Event schema / streaming                  | runtime                   |
| Running coding work                       | a `CodingExecutor` plugin |
| Tool execution                            | MCP servers               |
