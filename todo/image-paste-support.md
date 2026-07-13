# Image paste / attachment support

## Context

The wire protocol accepts user messages with `image` content blocks
(`{"type":"image","source":{"type":"base64","media_type":...,"data":...}}`).
claudestream's `send()` already accepts `str | list` prompts, so structured content can
likely pass through today (verify). The official TUI supports pasting images; miniclaude
v0.1 is text-only.

## Problem

No way to attach an image to a prompt from the terminal.

## Solutions

1. **Explicit file reference**: an `/attach <path>` client command (or `@image:<path>`
   inline token) that base64-encodes the file and sends a content-block list for the next
   prompt. Pros: simple, terminal-portable, explicit. Cons: no clipboard magic.
2. Clipboard paste detection: on bracketed paste of binary/OSC52 or via `wl-paste`
   integration, detect image MIME and attach. Pros: TUI-parity UX. Cons: compositor- and
   terminal-specific, fragile — exactly the kind of magic miniclaude avoids.

Option 1 first; option 2 only if daily use demands it.

## Affected files

- miniclaude: `_repl.py` (command + pending-attachment state), `_cli.py` (none),
  `_toolline.py` (none), tests.
- claudestream: verify `UserMessage` serialization of content-block lists; add tests if
  gaps found.

## Effort

Small-medium: a day including MIME sniffing and tests.
