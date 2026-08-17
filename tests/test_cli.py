"""CLI-level tests for the strictcli app registration.

strictcli validates every command handler's signature against its declared
flags at decoration time (i.e. at ``import miniclaude._cli``). Since strictcli
0.29 the first handler parameter is unconditionally the injected context slot
and is excluded from flag matching, so every handler must lead with a ``ctx``
parameter. Importing the module and inspecting the registered commands is
enough to catch a handler-contract skew: under the bug, the import itself
raises ``ValueError: handler missing parameter ...``.

The rest of this file pins the declaration regime strictcli 0.41 made
mandatory: every flag states its presence, no flag on a mutating command
carries a value default, ``choices=`` entries are records, and the
exactly-one-session selection is a member-spelled choice flag rather than a
hand-written check inside the handler.
"""

from __future__ import annotations

from typing import Any

import pytest

from miniclaude._cli import NewSession, app

_BASE = ["repl", "--profile", "p", "--model", "m", "--permission-mode", "default"]


def test_cli_registers_all_commands() -> None:
    import miniclaude._cli as cli

    registered = set(cli.app._collect_all_command_paths())
    assert {"repl", "mock", "version"} <= registered


# -- Presence -------------------------------------------------------------------


def _flag(command: str, name: str) -> Any:
    return next(f for f in app._commands[command].flags if f.name == name)


def _selector(command: str, name: str) -> Any:
    return next(s for s in app._commands[command].selectors if s.name == name)


@pytest.mark.parametrize(
    ("command", "flag", "presence"),
    [
        ("repl", "profile", "required"),
        ("repl", "model", "required"),
        ("repl", "permission-mode", "required"),
        ("repl", "cwd", "optional"),
        ("mock", "seed", "optional"),
    ],
)
def test_every_flag_declares_its_presence(command: str, flag: str, presence: str) -> None:
    """``--profile``, ``--model`` and ``--permission-mode`` were required only

    because they carried no default, and said so by writing "(required)" into
    their own help text. ``--cwd`` and ``--seed`` carried ``default=""`` and
    read absence back out of the empty string. Both idioms are now a
    declaration.
    """
    assert _flag(command, flag).presence == presence


def test_no_flag_on_a_mutating_command_declares_a_value_default() -> None:
    """Both commands are mutating, and a mutating command may not default a value.

    A value the framework picked on a mutating command is a value the framework
    writes, so the fallbacks that remain (a random mock seed, the current
    directory) live in the handler and are stated in the flag's help.
    """
    for name in ("repl", "mock"):
        command = app._commands[name]
        assert command.effect == "mutating"
        for flag in command.flags:
            assert flag.presence != "default", f"{name} --{flag.name}"


def test_permission_mode_choices_are_records_with_help() -> None:
    """A bare ``choices=["default", ...]`` list is a registration error now.

    Every entry carries its own help, so ``--help`` renders the modes as an
    indented block explaining what each one does instead of a bare enumeration.
    """
    entries = _flag("repl", "permission-mode").choice_records
    assert [c.value for c in entries] == [
        "default",
        "acceptEdits",
        "plan",
        "bypassPermissions",
        "dontAsk",
        "auto",
    ]
    assert all(c.help for c in entries)


# -- The session selection ------------------------------------------------------


def test_session_is_a_member_spelled_choice_over_three_named_alternatives() -> None:
    """Starting fresh is a choice with a name, not the absence of two others."""
    selector = _selector("repl", "session")
    assert selector.elect_by == "member-flags"
    assert [c.name for c in selector.choices] == [
        "resume",
        "continue-session",
        "new-session",
    ]


def test_omitting_every_member_elects_new_session() -> None:
    """The argv of the previous releases is unchanged: neither flag means new."""
    assert isinstance(_selector("repl", "session").default, NewSession)


@pytest.mark.parametrize(
    "argv",
    [
        ["--resume", "abc", "--continue-session"],
        ["--resume", "abc", "--new-session"],
        ["--continue-session", "--new-session"],
    ],
)
def test_two_session_members_are_refused_by_the_framework(argv: list[str]) -> None:
    """The handler used to check ``--resume``/``--continue-session`` itself, and

    only that one pair -- a third alternative would have needed a third branch.
    The refusal is the declaration's now, and it covers every pair.
    """
    result = app.test([*_BASE, *argv])
    assert result.exit_code == 1
    assert "mutually exclusive" in result.stderr


def test_declining_continue_session_still_starts_a_new_one(monkeypatch: Any) -> None:
    """``--no-continue-session`` declines a member; it elects nothing.

    Nothing elected falls to the declared default, which is what the flag's
    ``default=False`` meant before, so the spelling keeps working.
    """
    captured = _run_repl(monkeypatch, ["--no-continue-session"])
    assert captured["resume_session_id"] is None
    assert captured["session_resolution"] is None


# -- What the elected member reaches the session with ---------------------------


def _run_repl(monkeypatch: Any, argv: list[str]) -> dict[str, Any]:
    """Dispatch ``repl`` with the session plumbing stubbed, returning the config.

    The handler builds a ``SessionConfig`` and hands the REPL a factory that
    would spawn a real Claude Code process. Both are replaced here, so the test
    reads what the elected member resolved to without starting anything.
    """
    import claudestream

    import miniclaude._repl

    captured: dict[str, Any] = {}

    def _record(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    class _Repl:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run(self) -> None:
            return None

    monkeypatch.setattr(claudestream, "SessionConfig", _record)
    monkeypatch.setattr(miniclaude._repl, "Repl", _Repl)

    result = app.test([*_BASE, *argv])
    assert result.exit_code == 0, result.stderr
    return captured


def test_resume_carries_its_session_id_as_the_member_payload(monkeypatch: Any) -> None:
    captured = _run_repl(monkeypatch, ["--resume", "sess-42"])
    assert captured["resume_session_id"] == "sess-42"
    assert captured["session_resolution"] is None


def test_continue_session_asks_for_the_most_recent_session(monkeypatch: Any) -> None:
    captured = _run_repl(monkeypatch, ["--continue-session"])
    assert captured["resume_session_id"] is None
    assert captured["session_resolution"].continue_last is True


def test_new_session_is_a_spelling_of_its_own(monkeypatch: Any) -> None:
    captured = _run_repl(monkeypatch, ["--new-session"])
    assert captured["resume_session_id"] is None
    assert captured["session_resolution"] is None


def test_an_omitted_cwd_reaches_the_session_as_absence(monkeypatch: Any) -> None:
    """It reached it as ``"" or None`` before -- the sentinel round trip."""
    captured = _run_repl(monkeypatch, [])
    assert captured["cwd"] is None


def test_a_supplied_cwd_reaches_the_session_verbatim(monkeypatch: Any) -> None:
    captured = _run_repl(monkeypatch, ["--cwd", "/tmp/somewhere"])
    assert captured["cwd"] == "/tmp/somewhere"


# -- The mock seed --------------------------------------------------------------


def test_an_omitted_seed_is_absence_rather_than_an_empty_string() -> None:
    """``_resolve_seed`` picked at random when the seed was falsy, so a caller

    passing ``--seed ""`` silently got a random seed instead of a refusal. The
    empty string is now a value like any other, and it does not parse.
    """
    from miniclaude._cli import _resolve_seed

    assert 0 <= _resolve_seed(None) < 2**31
    assert _resolve_seed("7") == 7
    with pytest.raises(ValueError, match="invalid literal"):
        _resolve_seed("")
