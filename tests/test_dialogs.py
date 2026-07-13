"""Unit tests for miniclaude._dialogs.

Covers the pure builders (permission choices, decision surface, AskUserQuestion
answer building, multiSelect parsing) and the thin async runners driven with
lightweight fakes (no terminal, no live session). Request objects are duck-typed
SimpleNamespace instances so the tests never depend on claudestream's event classes.
"""

from __future__ import annotations

import asyncio
import functools
import re
from types import SimpleNamespace

import pytest

from miniclaude import _dialogs
from miniclaude._dialogs import (
    Choice,
    build_decision_surface,
    build_permission_choices,
    build_question_answers,
    parse_multiselect,
    run_dialog_notice,
    run_permission_flow,
    run_question_flow,
    suggestion_label,
)


def sync(fn):
    """Run an async test body via asyncio.run (avoids a pytest-asyncio dependency)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip SGR escape sequences so assertions read against visible text."""
    return _ANSI_RE.sub("", text)


def _perm_req(**kwargs):
    base = dict(
        request_id="perm_1",
        tool_name="Bash",
        tool_input={"command": "ls"},
        title="",
        permission_suggestions=[],
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


# --- Fakes -------------------------------------------------------------------


class FakeSession:
    """Records every response call the runners make."""

    def __init__(self):
        self.calls = []

    async def respond_allow(self, request_id, updated_input, *, updated_permissions=None):
        self.calls.append(("allow", request_id, updated_input, updated_permissions))

    async def respond_deny(self, request_id, message="Denied by user"):
        self.calls.append(("deny", request_id, message))

    async def respond_dialog_cancelled(self, request_id):
        self.calls.append(("cancelled", request_id))


class FakeInteraction:
    """Scripted UI. ``choices`` are indices into the options list (or exceptions);
    ``texts`` are free-text replies (or exceptions)."""

    def __init__(self, choices=None, texts=None):
        self._choices = list(choices or [])
        self._texts = list(texts or [])
        self.choice_messages = []
        self.text_messages = []

    async def ask_choice(self, message, options, default=None):
        self.choice_messages.append(message)
        sel = self._choices.pop(0)
        if isinstance(sel, BaseException):
            raise sel
        return options[sel][0]

    async def ask_text(self, message):
        self.text_messages.append(message)
        val = self._texts.pop(0)
        if isinstance(val, BaseException):
            raise val
        return val


class ListPrinter:
    def __init__(self):
        self.chunks = []

    def __call__(self, text):
        self.chunks.append(text)

    @property
    def text(self):
        return "".join(self.chunks)


# --- suggestion_label --------------------------------------------------------


def test_suggestion_label_plural_rules_with_content():
    sug = {
        "type": "addRules",
        "rules": [{"toolName": "Bash", "ruleContent": "git status"}],
        "behavior": "allow",
    }
    assert suggestion_label(sug) == "Bash(git status)"


def test_suggestion_label_tool_only():
    assert suggestion_label({"rules": [{"toolName": "Read"}]}) == "Read"


def test_suggestion_label_singular_rule_shape():
    sug = {"type": "addRule", "rule": {"toolName": "Bash", "ruleContent": "ls -la"}}
    assert suggestion_label(sug) == "Bash(ls -la)"


def test_suggestion_label_multiple_rules_joined():
    sug = {"rules": [{"toolName": "Bash", "ruleContent": "a"}, {"toolName": "Read"}]}
    assert suggestion_label(sug) == "Bash(a), Read"


def test_suggestion_label_snake_case_fields():
    sug = {"rules": [{"tool_name": "Bash", "rule_content": "pwd"}]}
    assert suggestion_label(sug) == "Bash(pwd)"


def test_suggestion_label_unrecognized_falls_back_to_json():
    out = suggestion_label({"weird": 1})
    assert "weird" in out


# --- build_permission_choices ------------------------------------------------


def test_build_permission_choices_without_suggestions():
    choices = build_permission_choices(_perm_req())
    assert [c.action for c in choices] == ["allow_once", "deny", "deny_message"]
    assert choices[0].label == "Allow once"
    assert choices[1].label == "Deny"
    assert choices[2].label == "Deny with a message..."


def test_build_permission_choices_with_suggestions_order_and_labels():
    sug = {"rules": [{"toolName": "Bash", "ruleContent": "git status"}]}
    req = _perm_req(permission_suggestions=[sug])
    choices = build_permission_choices(req)
    assert [c.action for c in choices] == [
        "allow_once",
        "allow_always",
        "deny",
        "deny_message",
    ]
    assert choices[1].label == "Allow always: Bash(git status)"
    assert choices[1].suggestion is sug


def test_build_permission_choices_multiple_suggestions():
    reqs = [
        {"rules": [{"toolName": "Bash", "ruleContent": "a"}]},
        {"rules": [{"toolName": "Bash", "ruleContent": "b"}]},
    ]
    choices = build_permission_choices(_perm_req(permission_suggestions=reqs))
    always = [c for c in choices if c.action == "allow_always"]
    assert len(always) == 2
    assert always[0].label == "Allow always: Bash(a)"
    assert always[1].label == "Allow always: Bash(b)"


# --- build_decision_surface --------------------------------------------------


def test_decision_surface_bash_shows_full_command():
    req = _perm_req(tool_name="Bash", tool_input={"command": "git status\ngit log"})
    out = _plain(build_decision_surface(req))
    assert "Claude wants to use Bash" in out
    assert "git status\ngit log" in out
    assert out.endswith("\n")


def test_decision_surface_uses_title_when_present():
    req = _perm_req(title="Run a shell command", tool_name="Bash",
                    tool_input={"command": "ls"})
    out = _plain(build_decision_surface(req))
    assert out.splitlines()[0] == "Run a shell command"


def test_decision_surface_edit_shows_diff_lines():
    req = _perm_req(
        tool_name="Edit",
        tool_input={
            "file_path": "/x/y.py",
            "old_string": "old line",
            "new_string": "new line",
        },
    )
    raw = build_decision_surface(req)
    out = _plain(raw)
    assert "/x/y.py" in out
    assert "- old line" in out
    assert "+ new line" in out
    # old side colored red, new side colored green
    assert "\x1b[31m- old line" in raw
    assert "\x1b[32m+ new line" in raw


def test_decision_surface_edit_truncates_long_sides():
    old = "\n".join(f"o{i}" for i in range(100))
    req = _perm_req(
        tool_name="Edit",
        tool_input={"file_path": "/x", "old_string": old, "new_string": "n"},
    )
    out = _plain(build_decision_surface(req))
    assert "more lines)" in out
    assert "o40" not in out  # beyond the 40-line cap
    assert "o39" in out


def test_decision_surface_write_shows_path_size_and_head():
    content = "\n".join(f"line{i}" for i in range(50))
    req = _perm_req(
        tool_name="Write",
        tool_input={"file_path": "/a/b.txt", "content": content},
    )
    out = _plain(build_decision_surface(req))
    assert "/a/b.txt" in out
    assert f"{len(content)} bytes" in out
    assert "line0" in out
    assert "line19" in out
    assert "line20" not in out  # only first 20 lines
    assert "more lines)" in out


def test_decision_surface_other_tool_compact_json():
    req = _perm_req(tool_name="WebFetch", tool_input={"url": "http://x", "n": 3})
    out = _plain(build_decision_surface(req))
    assert "Claude wants to use WebFetch" in out
    assert '"url":"http://x"' in out


# --- build_question_answers --------------------------------------------------


def test_question_answers_single_choice():
    inp = {"questions": [{"question": "Which color do you prefer?",
                          "options": [{"label": "Red"}, {"label": "Blue"}],
                          "multiSelect": False}]}
    out = build_question_answers(inp, ["Red"])
    assert out["answers"] == {"Which color do you prefer?": "Red"}
    # original input echoed
    assert out["questions"] == inp["questions"]


def test_question_answers_other_free_text():
    inp = {"questions": [{"question": "Favorite?", "options": [{"label": "A"}]}]}
    out = build_question_answers(inp, ["my own custom answer"])
    assert out["answers"] == {"Favorite?": "my own custom answer"}


def test_question_answers_multiselect_comma_joined():
    inp = {"questions": [{"question": "Pick some", "multiSelect": True,
                          "options": [{"label": "Red"}, {"label": "Blue"}]}]}
    out = build_question_answers(inp, [["Red", "Blue"]])
    assert out["answers"] == {"Pick some": "Red, Blue"}


def test_question_answers_multiple_questions():
    inp = {"questions": [
        {"question": "Q1"},
        {"question": "Q2"},
    ]}
    out = build_question_answers(inp, ["A1", ["x", "y"]])
    assert out["answers"] == {"Q1": "A1", "Q2": "x, y"}


# --- parse_multiselect -------------------------------------------------------


def test_parse_multiselect_valid():
    assert parse_multiselect("1, 3", 3) == [0, 2]


def test_parse_multiselect_dedupes_preserving_order():
    assert parse_multiselect("2,2,1", 3) == [1, 0]


def test_parse_multiselect_rejects_garbage():
    assert parse_multiselect("abc", 3) is None
    assert parse_multiselect("", 3) is None
    assert parse_multiselect("1,foo", 3) is None


def test_parse_multiselect_rejects_out_of_range():
    assert parse_multiselect("4", 3) is None
    assert parse_multiselect("0", 3) is None


# --- run_permission_flow -----------------------------------------------------


@sync
async def test_permission_flow_allow_once():
    session = FakeSession()
    interaction = FakeInteraction(choices=[0])  # "Allow once"
    printer = ListPrinter()
    req = _perm_req(tool_input={"command": "ls"})
    await run_permission_flow(req, session, interaction, printer)
    assert session.calls == [("allow", "perm_1", {"command": "ls"}, None)]
    # decision surface was printed above the choices
    assert "Claude wants to use Bash" in _plain(printer.text)


@sync
async def test_permission_flow_allow_always_passes_suggestion():
    session = FakeSession()
    sug = {"rules": [{"toolName": "Bash", "ruleContent": "ls"}]}
    req = _perm_req(permission_suggestions=[sug])
    interaction = FakeInteraction(choices=[1])  # "Allow always: ..."
    await run_permission_flow(req, session, interaction, ListPrinter())
    assert session.calls == [("allow", "perm_1", {"command": "ls"}, [sug])]


@sync
async def test_permission_flow_deny():
    session = FakeSession()
    req = _perm_req()  # no suggestions: index 1 == Deny
    interaction = FakeInteraction(choices=[1])
    await run_permission_flow(req, session, interaction, ListPrinter())
    assert session.calls == [("deny", "perm_1", "Denied by user")]


@sync
async def test_permission_flow_deny_with_message():
    session = FakeSession()
    req = _perm_req()  # index 2 == Deny with a message...
    interaction = FakeInteraction(choices=[2], texts=["please no"])
    await run_permission_flow(req, session, interaction, ListPrinter())
    assert session.calls == [("deny", "perm_1", "please no")]


@sync
async def test_permission_flow_deny_with_empty_message_defaults():
    session = FakeSession()
    req = _perm_req()
    interaction = FakeInteraction(choices=[2], texts=[""])
    await run_permission_flow(req, session, interaction, ListPrinter())
    assert session.calls == [("deny", "perm_1", "Denied by user")]


@sync
async def test_permission_flow_escape_fails_closed():
    session = FakeSession()
    req = _perm_req()
    interaction = FakeInteraction(choices=[KeyboardInterrupt()])
    await run_permission_flow(req, session, interaction, ListPrinter())
    assert session.calls == [("deny", "perm_1", "Denied by user")]


@sync
async def test_permission_flow_deny_message_escape_fails_closed():
    session = FakeSession()
    req = _perm_req()
    interaction = FakeInteraction(choices=[2], texts=[EOFError()])
    await run_permission_flow(req, session, interaction, ListPrinter())
    assert session.calls == [("deny", "perm_1", "Denied by user")]


# --- run_question_flow -------------------------------------------------------


def _question_req(questions):
    return SimpleNamespace(request_id="q_1", tool_name="AskUserQuestion",
                           tool_input={"questions": questions})


@sync
async def test_question_flow_single_select():
    req = _question_req([
        {"question": "Which color do you prefer?",
         "options": [{"label": "Red", "description": "Red"},
                     {"label": "Blue", "description": "Blue"}],
         "multiSelect": False},
    ])
    session = FakeSession()
    interaction = FakeInteraction(choices=[0])  # picks first real option
    await run_question_flow(req, session, interaction, ListPrinter())
    kind, request_id, updated, perms = session.calls[0]
    assert kind == "allow"
    assert request_id == "q_1"
    assert updated["answers"] == {"Which color do you prefer?": "Red"}
    assert updated["questions"] == req.tool_input["questions"]


@sync
async def test_question_flow_other_free_text():
    req = _question_req([
        {"question": "Favorite?",
         "options": [{"label": "A"}, {"label": "B"}],
         "multiSelect": False},
    ])
    session = FakeSession()
    # last menu entry is the "Other" sentinel -> index 2 (two options + Other)
    interaction = FakeInteraction(choices=[2], texts=["something else"])
    await run_question_flow(req, session, interaction, ListPrinter())
    _, _, updated, _ = session.calls[0]
    assert updated["answers"] == {"Favorite?": "something else"}


@sync
async def test_question_flow_multiselect():
    req = _question_req([
        {"question": "Pick some",
         "options": [{"label": "Red"}, {"label": "Blue"}, {"label": "Green"}],
         "multiSelect": True},
    ])
    session = FakeSession()
    interaction = FakeInteraction(texts=["1, 3"])
    await run_question_flow(req, session, interaction, ListPrinter())
    _, _, updated, _ = session.calls[0]
    assert updated["answers"] == {"Pick some": "Red, Green"}


@sync
async def test_question_flow_multiselect_reasks_on_garbage():
    req = _question_req([
        {"question": "Pick", "options": [{"label": "A"}, {"label": "B"}],
         "multiSelect": True},
    ])
    session = FakeSession()
    interaction = FakeInteraction(texts=["nonsense", "2"])
    printer = ListPrinter()
    await run_question_flow(req, session, interaction, printer)
    _, _, updated, _ = session.calls[0]
    assert updated["answers"] == {"Pick": "B"}
    assert "Invalid selection" in _plain(printer.text)


@sync
async def test_question_flow_dismiss_fails_closed():
    req = _question_req([
        {"question": "Q", "options": [{"label": "A"}], "multiSelect": False},
    ])
    session = FakeSession()
    interaction = FakeInteraction(choices=[KeyboardInterrupt()])
    await run_question_flow(req, session, interaction, ListPrinter())
    assert session.calls == [("deny", "q_1", "User dismissed the question.")]


@sync
async def test_question_flow_multiple_questions_injected_together():
    req = _question_req([
        {"question": "Q1", "options": [{"label": "A"}, {"label": "B"}],
         "multiSelect": False},
        {"question": "Q2", "options": [{"label": "X"}, {"label": "Y"}],
         "multiSelect": False},
    ])
    session = FakeSession()
    interaction = FakeInteraction(choices=[0, 1])
    await run_question_flow(req, session, interaction, ListPrinter())
    _, _, updated, _ = session.calls[0]
    assert updated["answers"] == {"Q1": "A", "Q2": "Y"}


# --- run_dialog_notice (unknown kind -> cancelled) ---------------------------


@sync
async def test_dialog_notice_cancels_unknown_kind():
    req = SimpleNamespace(request_id="d_1", dialog_kind="mystery_kind")
    session = FakeSession()
    printer = ListPrinter()
    await run_dialog_notice(req, session, printer)
    assert session.calls == [("cancelled", "d_1")]
    assert "mystery_kind" in _plain(printer.text)


@sync
async def test_dialog_notice_handles_missing_kind():
    req = SimpleNamespace(request_id="d_2", dialog_kind="")
    session = FakeSession()
    await run_dialog_notice(req, session, ListPrinter())
    assert session.calls == [("cancelled", "d_2")]


def test_choice_dataclass_is_frozen():
    c = Choice("allow_once", "Allow once")
    with pytest.raises(Exception):
        c.action = "x"  # type: ignore[misc]


def test_module_exposes_production_interaction():
    # Production wiring exists but is not exercised here (needs a terminal).
    assert hasattr(_dialogs, "PromptToolkitInteraction")
