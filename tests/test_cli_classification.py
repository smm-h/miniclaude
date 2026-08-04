"""The CLI's effect classification, pinned so a change has to be deliberate.

strictcli requires every command to declare ``effect="read_only"`` or
``effect="mutating"``; there is no default and a missing declaration is a
registration-time hard error. The classification answers exactly one question:
*should a dry run record this operation rather than perform it?*

Separately, a command may declare itself ``consequential``, which is what the
framework's confirm protocol keys on. It is NOT inferred from ``mutating`` --
that inference was measured at a ~1:10 signal-to-noise ratio across the fleet
and removed, because a guardrail that fires on two thirds of a CLI's commands
trains the reflex that hollows it out.

This file pins both tables in both directions. A new command shows up as an
unexpected entry; a reclassified one shows up as a mismatch. Either way the
edit has to come here, which is the point.
"""

from __future__ import annotations

from typing import Any

from miniclaude._cli import app

# `version` reads the string resolved at import and prints it.
#
# `repl` spawns a real Claude Code subprocess: it reads and writes files under
# --cwd, runs shell commands, calls the network, spends money and persists a
# session transcript. What the model chooses to do is not knowable in advance,
# which is exactly why a preview must record the spawn rather than perform it.
#
# `mock` is the interesting one. Its session is fake -- no Claude binary, no
# network, no spend -- so the temptation is to call it read-only. It is not:
# the REPL it drives is the production REPL, and that REPL creates
# ~/.miniclaude/ and appends every prompt typed into it to
# ~/.miniclaude/history. A persistent write outside the working directory is a
# mutation regardless of how harmless the thing that caused it was.
EFFECTS = {
    "version": "read_only",
    "repl": "mutating",
    "mock": "mutating",
}

# Empty, deliberately.
#
# `consequential` means "this act is worth interrupting someone for". Starting
# a Claude session is not an incidental side effect of `repl` -- it is the
# entire command, and it is precisely what the user typing `miniclaude repl`
# asked for. Prompting first would put a `Proceed? [y/N]` in front of a
# fullscreen TUI on every single launch, which is the exact reflex-training
# noise the declaration was introduced to remove.
#
# `--permission-mode bypassPermissions` is the sharpest thing here, and it is
# deliberately NOT addressed by declaring the command consequential:
# `consequential` is per-command, so declaring `repl` would prompt in front of
# every safe launch too. A flag-granular seam belongs inside the handler if one
# is ever wanted; it is not this table's job.
CONSEQUENTIAL: set[str] = set()

# strictcli owns these four names at every level -- command flags, flag-set
# flags, mutex-group flags and app globals alike. `yes` names no framework flag
# any more (the skip flag is --approve-consequential) but stays banned so a
# consumer cannot restate it in the spelling the rename removed.
RESERVED_FLAG_NAMES = {
    "dry-run",
    "approve-consequential",
    "quiet",
    "verbose",
    "yes",
}


def _walk() -> dict[str, Any]:
    """Map dotted command path -> Command for every registered command."""
    found: dict[str, Any] = {}

    def visit(container: Any, prefix: str) -> None:
        registry = getattr(container, "_commands", None) or container.commands
        for name, cmd in registry.items():
            found[prefix + name] = cmd
        for name, group in container._groups.items():
            visit(group, prefix + name + ".")

    visit(app, "")
    return found


def test_every_command_is_classified_exactly_as_reviewed() -> None:
    declared = {path: cmd.effect for path, cmd in _walk().items()}
    assert declared == EFFECTS


def test_consequential_declarations_match_the_reviewed_set() -> None:
    """Both directions matter.

    A missing declaration removes a prompt somebody decided was owed; a stray
    one puts a blind ``Proceed? [y/N]`` in front of a fullscreen TUI launch.
    """
    declared = {path for path, cmd in _walk().items() if cmd.consequential}
    assert declared == CONSEQUENTIAL


def test_no_command_redeclares_a_framework_reserved_flag_name() -> None:
    """A collision is a registration-time error, so reaching here means it built.

    Pinning the absence keeps a future flag from reintroducing one under a
    spelling that would break the CLI at import time.
    """
    assert not ({f.name for f in app._global_flags} & RESERVED_FLAG_NAMES)
    for path, cmd in _walk().items():
        names = {f.name for f in cmd.flags}
        collisions = names & RESERVED_FLAG_NAMES
        assert not collisions, f"'{path}' declares reserved flag(s) {sorted(collisions)}"


def test_the_reserved_quartet_reaches_the_context_not_the_handler() -> None:
    """--quiet is framework-owned and never lands in a handler's kwargs.

    Passing a quartet member must not become an unknown-flag error and must not
    be forwarded into the handler, which is what guard v2 would reject.
    `version` is the one command safe to actually dispatch here.
    """
    result = app.test(["version", "--quiet"])
    assert result.exit_code == 0
