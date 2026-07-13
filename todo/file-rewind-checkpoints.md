# File rewind / checkpoints

## Context

The CLI supports file checkpointing with a `rewind_files` control subtype (rewind the
working tree to the state at a given user-message id, with a dry-run option), per the
Agent SDK typings. Neither claudestream nor miniclaude exposes it.

## Problem

After a bad turn, undoing the model's file edits requires git archaeology by hand; the
official TUI has checkpoint UX, miniclaude has none.

## Solutions

1. **claudestream control method + miniclaude command**: `rewind_files(message_uuid,
   dry_run)` on the session (one more subtype over the existing correlation registry —
   trivial); miniclaude tracks user-message uuids per turn and offers `/rewind` (list
   recent turns, pick one, dry-run preview, confirm). Pros: parity with a genuinely
   valuable safety feature. Cons: needs careful UX around what checkpointing covers
   (probe the actual behavior first — what is checkpointed, when is it enabled, does it
   need a spawn flag like file checkpointing enablement).
2. Rely on git discipline only (status quo). Pros: nothing to build. Cons: unstaged or
   non-repo work is unprotected.

Probe first, then (1).

## Affected files

- claudestream: probe scenario, session method + tests.
- miniclaude: `_repl.py` (`/rewind`, uuid tracking, confirm modal via _dialogs), tests.

## Effort

Medium: probe half a day; implementation ~a day.
