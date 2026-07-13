# miniclaude

A lean, snappy inline terminal client for Claude Code sessions — a minimal alternative frontend to the official TUI. It drives the official `claude` CLI through the [claudestream](https://pypi.org/project/claudestream/) library, so tools, permissions, sessions, and authentication are all the real thing; miniclaude only replaces the presentation layer.

## Design stance

- **Inline, scrollback-native rendering.** Output goes straight into your terminal's own scrollback, so its native search and copy keep working. No alternate screen, minimal redraw.
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

| Flag | Required | Description |
| --- | --- | --- |
| `--profile <str>` | yes | claudewheel profile to use |
| `--model <str>` | yes | model to use, e.g. `sonnet`, `haiku` |
| `--permission-mode <str>` | yes | one of `default`, `acceptEdits`, `plan`, `bypassPermissions`, `dontAsk`, `auto` |
| `--cwd <str>` | no | working directory (default: current) |
| `--resume <str>` | no | resume a previous session by ID |
| `--continue-session` / `--no-continue-session` | no | continue the most recent session (default: off) |

`--resume` and `--continue-session` are mutually exclusive. Run `miniclaude version` to print the version.

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
