# Richer subagent activity display

## Context

Events carry `parent_tool_use_id` when they originate inside a subagent (Agent/Task tool).
miniclaude v0.1 renders subagent tool one-liners indented two spaces and ignores subagent
text/thinking deltas entirely. For heavy multi-agent workflows the transcript shows only
the top-level Agent tool line plus indented tool lines, with no sense of progress or
which subagent is which.

## Problem

- Multiple concurrent subagents interleave indistinguishably (all just indented).
- No progress signal for long-running subagents; no subagent result summary beyond the
  final ToolResult one-liner.

## Solutions

1. **Label by agent**: maintain a map from the Agent tool_use_id to its description (from
   the ToolUse input); prefix indented lines with a short colored tag derived from it
   (e.g. first word + counter). Pros: cheap, keeps one-liner density. Cons: no live text.
2. Optional verbose mode (`/verbose` toggle or flag) that also streams subagent text dimmed
   under its tag. Pros: full visibility when wanted. Cons: scrollback volume — must stay
   opt-in to honor the quiet-by-default stance.

Do (1); add (2) behind an explicit toggle.

## Affected files

- miniclaude: `_toolline.py` (tagging), `_repl.py` (map + dispatch + toggle), tests.

## Effort

Small: ~half a day.
