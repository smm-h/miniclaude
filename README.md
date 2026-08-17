# miniclaude

A lean, snappy fullscreen terminal client for Claude Code sessions — a minimal alternative frontend to the official TUI. It drives the official `claude` CLI through the [claudestream](https://pypi.org/project/claudestream/) library, so tools, permissions, sessions, and authentication are all the real thing; miniclaude only replaces the presentation layer.

## Design stance

- **Fullscreen, block-backed rendering.** Output is modelled as blocks and materialized into a scrollable output region on the alternate screen. Scroll with the mouse wheel; the view holds where you leave it (scroll-lock) and re-flows tables to the live terminal width on resize.
- **Line-grain streaming markdown.** Assistant prose is styled and emitted line by line as it arrives (headers, bullets, inline code, links, code fences). Tables are the one buffered exception, held until complete and printed with aligned columns.
- **Dense tool activity.** Each tool call is one line (`▸ ToolName arg`); each result is one dim line (`✓`/`✗` plus a `(+N lines)` count). Subagent activity is indented.
- **Interactive tools that work in the terminal.** Permission prompts show a real decision surface (the Bash command, a colored diff for edits, a preview for writes) above numbered Allow/Deny choices. AskUserQuestion is answered through plain numbered prompts (single- and multi-select).
- **Type-ahead with interrupt.** Type while a turn runs and the line is queued; press Esc to interrupt the in-flight turn.

## Requirements

- Python >= 3.11
- The `claude` CLI installed and logged in
- A [claudewheel](https://pypi.org/project/claudewheel/) profile

## Install

```
uv tool install miniclaude
pip install miniclaude
npm i -g miniclaude
```

The npm package is a Node shim that runs the installed Python `miniclaude`; install the Python package as well.

## Usage

```
miniclaude repl --profile <name> --model <model> --permission-mode <mode>
```

Example:

```
miniclaude repl --profile default --model sonnet --permission-mode default
```

| Flag | Presence | Description |
| --- | --- | --- |
| `--profile <str>` | required | claudewheel profile to use |
| `--model <str>` | required | model to use, e.g. `sonnet`, `haiku` |
| `--permission-mode <str>` | required | one of `default`, `acceptEdits`, `plan`, `bypassPermissions`, `dontAsk`, `auto` |
| `--cwd <str>` | optional | working directory; omitted, the REPL runs in the current directory |

Which session the REPL runs is one selection over three named alternatives —
pass exactly one, or none and get a new session:

| Flag | Effect |
| --- | --- |
| `--resume <session-id>` | resume that previous session |
| `--continue-session` | continue the most recent session |
| `--new-session` | start a fresh session (what you get by passing none of the three) |

Passing two of them is refused by name. Run `miniclaude version` to print the
version.

## In-session

Slash commands:

| Command | Effect |
| --- | --- |
| `/model <name>` | switch model (no argument: show current) |
| `/mode <mode>` | change permission mode (no argument: show current) |
| `/context` | show context-window usage |
| `/cost` | show session cost and token totals |
| `/help` | list the slash commands |
| `/quit`, `/exit` | leave the REPL |

Unknown `/` commands are sent to the session verbatim, so server-side commands still work.

Keybindings:

| Key | Action |
| --- | --- |
| Enter | submit the prompt |
| Alt+Enter | insert a newline |
| Esc | interrupt the running turn |
| Ctrl+C | clear the input when idle; interrupt the running turn |
| Ctrl+D | exit (on an empty input) |

## claudewheel integration

Launch it from claudewheel with `claudewheel --client miniclaude`.

## License

MIT
