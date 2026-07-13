"""One-liner tool-activity formatters (pure, no I/O).

Every function returns a single ANSI-styled line (no trailing newline). Rendering is
separated from I/O so it is unit-testable without a terminal or a real session.
"""

from __future__ import annotations

from typing import Any

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GLYPH = "\033[36m"  # cyan for the activity marker


def _truncate(text: str, limit: int) -> str:
    """Collapse newlines to spaces and truncate to ``limit`` chars with an ellipsis."""
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _kv_summary(data: dict[str, Any]) -> str:
    """Render ``key=value`` pairs for an unrecognized tool, kept to one truncated line."""
    if not data:
        return ""
    pairs = [f"{k}={_truncate(str(v), 40)}" for k, v in data.items()]
    return _truncate(" ".join(pairs), 80)


def _describe(name: str, data: dict[str, Any]) -> tuple[str, str]:
    """Return ``(display_name, arg_summary)`` for a tool invocation."""
    data = data or {}

    if name in ("Read", "Write", "Edit"):
        return name, _truncate(str(data.get("file_path", "")), 80)

    if name == "Bash":
        command = str(data.get("command", ""))
        first_line = command.split("\n", 1)[0]
        return name, _truncate(first_line, 80)

    if name in ("Glob", "Grep"):
        pattern = str(data.get("pattern", ""))
        path = data.get("path")
        summary = f"{pattern} in {path}" if path else pattern
        return name, _truncate(summary, 80)

    if name == "WebFetch":
        return name, _truncate(str(data.get("url", "")), 80)

    if name == "WebSearch":
        return name, _truncate(str(data.get("query", "")), 80)

    if name in ("Agent", "Task"):
        return name, _truncate(str(data.get("description", "")), 80)

    if name == "TodoWrite":
        todos = data.get("todos", [])
        count = len(todos) if isinstance(todos, list) else 0
        return name, f"{count} item{'' if count == 1 else 's'}"

    if name.startswith("mcp__"):
        parts = name.split("__")
        server = parts[1] if len(parts) > 1 else ""
        tool = "__".join(parts[2:]) if len(parts) > 2 else ""
        display = f"{server}:{tool}"
        value = ""
        if data:
            first_key = next(iter(data))
            value = _truncate(str(data[first_key]), 80)
        return display, value

    return name, _kv_summary(data)


def format_tool_use(name: str, input: dict, parent: str | None = None) -> str:
    """Format a tool invocation as one line: ``▸ ToolName arg-summary``.

    Subagent activity (``parent`` is not None) is indented two spaces.
    """
    display, summary = _describe(name, input)
    indent = "  " if parent is not None else ""
    line = f"{indent}{GLYPH}▸{RESET} {BOLD}{display}{RESET}"
    if summary:
        line += f" {summary}"
    return line


def _content_to_text(content: Any) -> str:
    """Normalize tool-result content (str, list of blocks, or arbitrary) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict):
                pieces.append(str(item.get("text", "")))
            else:
                pieces.append(str(item))
        return "\n".join(pieces)
    return str(content)


def format_tool_result(
    tool_name: str | None,
    content: Any,
    is_error: bool,
    parent: str | None = None,
) -> str:
    """Format a tool result as one dim line: glyph + first line + ``(+N lines)``.

    Success uses a dim ``✓``; errors use a red ``✗`` and render the first line in red.
    """
    text = _content_to_text(content)
    lines = text.split("\n")
    while len(lines) > 1 and lines[-1] == "":
        lines.pop()
    first = _truncate(lines[0], 100)
    extra = len(lines) - 1
    suffix = f" (+{extra} lines)" if extra > 0 else ""
    indent = "  " if parent is not None else ""

    if is_error:
        line = f"{indent}{RED}✗ {first}{RESET}"
        if suffix:
            line += f"{DIM}{suffix}{RESET}"
        return line

    return f"{indent}{DIM}✓ {first}{suffix}{RESET}"
