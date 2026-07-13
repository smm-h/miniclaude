"""Interactive prompts: permission modals, AskUserQuestion answering, dialog safety net.

Every user-facing prompt is split into a PURE builder (unit-testable without a
terminal or a live session) and a thin async UI runner that drives prompt_toolkit
inline widgets (`ChoiceInput` for numbered menus, a short `PromptSession` for free
text). Nothing here ever enters the alternate screen: `ChoiceInput` and
`PromptSession` are both `full_screen=False`.

WIRE REALITY (probe-confirmed on CLI 2.1.197):
- Permission prompts AND AskUserQuestion both arrive as PermissionRequest events.
- AskUserQuestion is ANSWERED via the permission response: respond_allow(request_id,
  updated_input={**original_input, "answers": {<question text>: <answer string>}}).
  Single-select answer = the chosen option's label; "Other" = the user's free text;
  multiSelect = the selected labels comma-joined with ", ".
- request_user_dialog is never emitted by the current CLI. Any UserDialogRequest that
  does arrive gets a single dim notice line + respond_dialog_cancelled (safety net).

Request objects (`req`) are duck-typed: the builders read `.tool_name`, `.tool_input`,
`.title`, `.permission_suggestions`, `.request_id`, `.dialog_kind`. This keeps the
module decoupled from the exact claudestream event classes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

# --- ANSI helpers (self-contained; raw SGR strings for patch_stdout(raw=True)) ---

RESET = "\x1b[0m"


def _sgr(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}{RESET}"


def _bold(text: str) -> str:
    return _sgr("1", text)


def _dim(text: str) -> str:
    return _sgr("2", text)


def _red(text: str) -> str:
    return _sgr("31", text)


def _green(text: str) -> str:
    return _sgr("32", text)


# Truncation limits for the permission decision surface.
_EDIT_MAX_LINES = 40
_WRITE_MAX_LINES = 20
_OTHER_MAX_CHARS = 200


# --- Permission choices (pure) ---


@dataclass(frozen=True)
class Choice:
    """One selectable action in the permission modal.

    ``action`` is a stable discriminator consumed by the runner:
    ``allow_once`` / ``allow_always`` / ``deny`` / ``deny_message``.
    ``suggestion`` carries the originating permission-rule dict for ``allow_always``.
    """

    action: str
    label: str
    suggestion: dict | None = None


def suggestion_label(suggestion: dict) -> str:
    """Render a permission suggestion as e.g. ``Bash(git status)`` or ``Bash``.

    Handles the plural ``addRules`` shape (``{"rules": [{"toolName", "ruleContent"}]}``)
    and the singular ``{"rule": {...}}`` shape. Falls back to compact JSON when the
    suggestion carries no recognizable rule content.
    """
    rules = suggestion.get("rules")
    if rules is None:
        single = suggestion.get("rule")
        if isinstance(single, dict):
            rules = [single]
    parts: list[str] = []
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        tool = rule.get("toolName") or rule.get("tool_name") or ""
        content = rule.get("ruleContent") or rule.get("rule_content") or ""
        if tool and content:
            parts.append(f"{tool}({content})")
        elif tool:
            parts.append(tool)
        elif content:
            parts.append(content)
    if parts:
        return ", ".join(parts)
    return _truncate(json.dumps(suggestion, separators=(",", ":"), ensure_ascii=False), 60)


def build_permission_choices(req: Any) -> list[Choice]:
    """Build the ordered permission menu for a PermissionRequest.

    Order: (1) Allow once; (2..) one "Allow always: <rule>" per permission_suggestions
    item; (N-1) Deny; (N) Deny with a message...
    """
    choices: list[Choice] = [Choice("allow_once", "Allow once")]
    for suggestion in getattr(req, "permission_suggestions", None) or []:
        choices.append(
            Choice(
                "allow_always",
                f"Allow always: {suggestion_label(suggestion)}",
                suggestion=suggestion,
            )
        )
    choices.append(Choice("deny", "Deny"))
    choices.append(Choice("deny_message", "Deny with a message..."))
    return choices


# --- Decision surface (pure) ---


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _diff_side(text: str, prefix: str, color: Callable[[str], str]) -> list[str]:
    lines = text.splitlines()
    out = [color(prefix + line) for line in lines[:_EDIT_MAX_LINES]]
    extra = len(lines) - _EDIT_MAX_LINES
    if extra > 0:
        out.append(_dim(f"  ... ({extra} more line{'s' if extra != 1 else ''})"))
    return out


def build_decision_surface(req: Any) -> str:
    """Render the human-facing detail shown ABOVE the permission choices.

    This is the one place where detail beats density: the user must be able to decide.
    Returns a block terminated by a trailing newline.
    """
    tool = getattr(req, "tool_name", "") or ""
    inp = getattr(req, "tool_input", None) or {}
    title = getattr(req, "title", "") or ""
    header = title or f"Claude wants to use {tool}"
    lines: list[str] = [_bold(header)]

    if tool == "Bash":
        command = str(inp.get("command", ""))
        if command:
            lines.append(command)
    elif tool == "Edit":
        path = str(inp.get("file_path") or inp.get("path") or "")
        if path:
            lines.append(_dim(path))
        lines.extend(_diff_side(str(inp.get("old_string", "")), "- ", _red))
        lines.extend(_diff_side(str(inp.get("new_string", "")), "+ ", _green))
    elif tool == "Write":
        path = str(inp.get("file_path") or inp.get("path") or "")
        content = str(inp.get("content", ""))
        lines.append(f"{_dim(path)}  {len(content)} bytes")
        body = content.splitlines()
        for line in body[:_WRITE_MAX_LINES]:
            lines.append("  " + line)
        extra = len(body) - _WRITE_MAX_LINES
        if extra > 0:
            lines.append(_dim(f"  ... ({extra} more line{'s' if extra != 1 else ''})"))
    else:
        compact = json.dumps(inp, separators=(",", ":"), ensure_ascii=False)
        lines.append(_truncate(compact, _OTHER_MAX_CHARS))

    return "\n".join(lines) + "\n"


# --- AskUserQuestion answering (pure) ---


def build_question_answers(input: dict, selections: Sequence[Any]) -> dict:
    """Build the answers-injected updated_input for an AskUserQuestion response.

    Returns ``{**input, "answers": {<question text>: <answer string>}}``. Each
    ``selections`` entry is resolved by position against ``input["questions"]``:
    a plain string is used verbatim (a chosen label or "Other" free text); a list or
    tuple (multiSelect) is comma-joined with ", ".
    """
    questions = input.get("questions", [])
    answers: dict[str, str] = {}
    for question, selection in zip(questions, selections):
        qtext = question.get("question", "")
        if isinstance(selection, (list, tuple)):
            answers[qtext] = ", ".join(str(item) for item in selection)
        else:
            answers[qtext] = str(selection)
    return {**input, "answers": answers}


def parse_multiselect(text: str, count: int) -> list[int] | None:
    """Parse a comma-separated list of 1-based option numbers into 0-based indices.

    Returns ``None`` on any garbage (empty, non-numeric, or out-of-range token) so the
    caller can re-ask. Duplicates are collapsed, order of first appearance preserved.
    """
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    if not tokens:
        return None
    idxs: list[int] = []
    for token in tokens:
        if not token.isdigit():
            return None
        n = int(token)
        if n < 1 or n > count:
            return None
        if (n - 1) not in idxs:
            idxs.append(n - 1)
    return idxs


def _option_label(option: dict) -> str:
    label = str(option.get("label", ""))
    description = option.get("description")
    if description and description != label:
        return f"{label}  {_dim(str(description))}"
    return label


def _numbered_options(options: Sequence[dict]) -> str:
    lines = []
    for i, option in enumerate(options, 1):
        lines.append(f"  {i}. {_option_label(option)}")
    return "\n".join(lines) + "\n"


# --- UI abstraction (injectable so runners are unit-testable) ---

Printer = Callable[[str], None]


class Interaction(Protocol):
    """The inline UI surface the runners drive; production impl uses prompt_toolkit."""

    async def ask_choice(
        self, message: str, options: Sequence[tuple[Any, Any]], default: Any = None
    ) -> Any:
        """Show a numbered single-select menu; return the chosen option's value."""
        ...

    async def ask_text(self, message: str) -> str:
        """Prompt for a single line of free text; return it."""
        ...


# --- Async runners (thin) ---

_OTHER_SENTINEL = ("other", None)


async def run_permission_flow(
    req: Any, session: Any, interaction: Interaction, printer: Printer
) -> None:
    """Render the decision surface, ask the user, and respond via the session.

    Esc/Ctrl+C anywhere in the modal denies (fail closed).
    """
    printer(build_decision_surface(req))
    choices = build_permission_choices(req)
    options = [(choice, choice.label) for choice in choices]
    try:
        chosen: Choice = await interaction.ask_choice(
            "Choose an action:", options, default=choices[0]
        )
    except (KeyboardInterrupt, EOFError):
        await session.respond_deny(req.request_id, "Denied by user")
        return

    if chosen.action == "allow_once":
        await session.respond_allow(req.request_id, req.tool_input)
    elif chosen.action == "allow_always":
        await session.respond_allow(
            req.request_id, req.tool_input, updated_permissions=[chosen.suggestion]
        )
    elif chosen.action == "deny_message":
        try:
            message = await interaction.ask_text("Denial message: ")
        except (KeyboardInterrupt, EOFError):
            message = ""
        await session.respond_deny(req.request_id, message or "Denied by user")
    else:  # deny
        await session.respond_deny(req.request_id, "Denied by user")


async def _ask_one_question(
    question: dict, interaction: Interaction, printer: Printer
) -> Any:
    """Ask a single AskUserQuestion question; return the resolved answer.

    Single-select returns the chosen label (or free text for "Other"); multiSelect
    returns a list of chosen labels.
    """
    printer(_bold(str(question.get("question", ""))) + "\n")
    options = question.get("options", []) or []
    labels = [str(option.get("label", "")) for option in options]

    if question.get("multiSelect"):
        while True:
            printer(_numbered_options(options))
            raw = await interaction.ask_text("Select (comma-separated numbers): ")
            idxs = parse_multiselect(raw, len(options))
            if idxs is not None:
                return [labels[i] for i in idxs]
            printer(_dim("Invalid selection; enter option numbers like 1,3.") + "\n")

    menu: list[tuple[Any, Any]] = [
        (("opt", i), _option_label(option)) for i, option in enumerate(options)
    ]
    menu.append((_OTHER_SENTINEL, "Other (type your own)"))
    picked = await interaction.ask_choice("", menu, default=menu[0][0])
    kind, idx = picked
    if kind == "other":
        return await interaction.ask_text("Your answer: ")
    return labels[idx]


async def run_question_flow(
    req: Any, session: Any, interaction: Interaction, printer: Printer
) -> None:
    """Drive the AskUserQuestion UI and answer via respond_allow.

    Esc/Ctrl+C dismisses the question (respond_deny, fail closed) so the turn continues.
    """
    inp = getattr(req, "tool_input", None) or {}
    questions = inp.get("questions", []) or []
    selections: list[Any] = []
    try:
        for question in questions:
            selections.append(await _ask_one_question(question, interaction, printer))
    except (KeyboardInterrupt, EOFError):
        await session.respond_deny(req.request_id, "User dismissed the question.")
        return
    await session.respond_allow(req.request_id, build_question_answers(inp, selections))


async def run_dialog_notice(req: Any, session: Any, printer: Printer) -> None:
    """Safety net for UserDialogRequest: one dim notice + respond_dialog_cancelled.

    The current CLI never emits these; a future one might. We never build UI here.
    """
    kind = getattr(req, "dialog_kind", "") or "unknown"
    printer(_dim(f"dialog '{kind}' not supported here; cancelling") + "\n")
    await session.respond_dialog_cancelled(req.request_id)


# --- Production interaction (thin prompt_toolkit wiring; inline, never full_screen) ---


class PromptToolkitInteraction:
    """Inline prompt_toolkit implementation of :class:`Interaction`.

    ``ask_choice`` uses ``ChoiceInput`` (a numbered ``RadioList``, ``full_screen=False``);
    ``ask_text`` uses a short single-line ``PromptSession``. Option labels may contain
    raw ANSI, so they are wrapped in ``ANSI(...)`` for prompt_toolkit's renderer.
    """

    async def ask_choice(
        self, message: str, options: Sequence[tuple[Any, Any]], default: Any = None
    ) -> Any:
        from prompt_toolkit.formatted_text import ANSI
        from prompt_toolkit.shortcuts.choice_input import ChoiceInput

        wrapped = [
            (value, ANSI(label) if isinstance(label, str) else label)
            for value, label in options
        ]
        widget: ChoiceInput = ChoiceInput(
            message=ANSI(message) if message else "",
            options=wrapped,
            default=default,
            enable_interrupt=True,
        )
        return await widget.prompt_async()

    async def ask_text(self, message: str) -> str:
        from prompt_toolkit import PromptSession

        session: PromptSession = PromptSession()
        return await session.prompt_async(message)
