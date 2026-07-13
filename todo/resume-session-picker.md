# Session picker for resume

## Context

`miniclaude repl --resume <id>` requires an exact session id; bare resume (pick from a
list, like the official TUI) does not exist. The claudewheel `--client miniclaude`
adapter deliberately hard-errors on a bare `--resume` selection for this reason.
Sessions are stored as JSONL under the profile's shared projects dir (claudewheel
centralized layout), so listing them locally is feasible; the Agent SDK also ships
session-listing helpers, suggesting stable on-disk conventions.

## Problem

Resuming requires copy-pasting a session id from elsewhere; the most common resume flow
("continue what I was doing in this directory yesterday") is high-friction.

## Solutions

1. **Local session listing**: read the profile's session store for the current cwd's
   project dir, show an inline numbered picker (ChoiceInput) with timestamp + first user
   message snippet, launch with the chosen id. Pros: no engine changes, fast. Cons:
   depends on the on-disk layout remaining stable — isolate parsing in one module and
   fail hard with a clear message on layout drift (no silent guessing).
2. Engine-mediated listing via a control request, if a subtype exists for it. Pros:
   layout-proof. Cons: requires a live subprocess before picking; probe needed to confirm
   such a subtype exists.

Do (1) with hard-fail parsing; revisit (2) if layout drift bites.

## Affected files

- miniclaude: new `_sessions.py` (store listing, pure + tested), `_cli.py`
  (`--resume` with empty value or a `resume` subcommand triggering the picker),
  `_repl.py` (none), tests. Update the claudewheel adapter's bare-resume hard error once
  supported (separate project, file a todo there when ready).

## Effort

Medium: ~a day including store-format tests.
