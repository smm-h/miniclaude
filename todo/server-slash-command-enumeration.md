# Server slash-command enumeration and completion

## Context

The claude CLI's `initialize` control-response returns a `commands` array (~29 entries
observed on CLI 2.1.197) enumerating server-side slash commands (skills, plugins,
built-ins usable in print mode). miniclaude v0.1 forwards unknown `/...` input to Claude
verbatim, which works, but the user gets no discovery or completion.

## Problem

- No completion/listing for server-side commands; users must know them by heart.
- `/help` only lists miniclaude's own client-side commands.

## Solutions

1. **Consume the initialize response** (claudestream already performs the handshake when
   `intercept_permissions=True`): expose the `commands` list on the session (small
   claudestream addition — e.g. a `server_commands` property populated from the
   initialize control-response), then feed it into the prompt_toolkit completer so `/`
   triggers completion, and append a dim "server commands" section to `/help`.
   Pros: real data, zero extra round-trips. Cons: needs a claudestream release.
2. Client-side only: no claudestream change; issue a one-off control request at startup if
   a suitable subtype exists. Pros: none over (1). Cons: duplicated machinery.

Option 1 is the correct one.

## Affected files

- claudestream: `_async_session.py` (capture initialize response), property + tests.
- miniclaude: `_repl.py` (completer wiring, /help), `_cli.py` (none), tests.

## Effort

Small: ~half a day including the claudestream property and tests.
