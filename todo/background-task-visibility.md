# Background task visibility and control

## Context

The CLI runs background tasks (background bash, background subagents) and exposes wire
surfaces for them: task/status system messages and control subtypes (`stop_task`,
`background_tasks` per the Agent SDK typings). miniclaude v0.1 ignores these events;
claudestream does not yet type them (they fall through as UnknownEvent/HookEvent).

## Problem

- A turn can finish while background tasks still run; the user has no indication and no
  way to stop them from miniclaude.
- In `-p`-style headless mode the CLI reportedly kills background bash shortly after the
  final result, so semantics for a long-lived interactive stream session need verifying
  first (probe before building).

## Solutions

1. **Probe, then minimal surfacing**: extend the claudestream probe script to observe what
   task-related frames actually flow in a live stream-json session; add typed claudestream
   events for whatever exists; miniclaude renders dim one-liners (`task started/finished`)
   and a `/tasks` command listing + `stop_task` control.
2. Full task tree UI — over-scope for an inline client; rejected.

## Affected files

- claudestream: `scripts/probe_user_dialogs.py` (new scenario), `events.py`,
  `_protocol.py`, session control method, tests.
- miniclaude: `_repl.py` dispatch + `/tasks`, `_toolline.py`, tests.

## Effort

Medium: probe first (half a day), then ~a day of implementation.
