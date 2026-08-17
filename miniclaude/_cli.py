"""Command-line interface entry point for miniclaude, a lean fullscreen terminal client for Claude Code."""

from __future__ import annotations

import asyncio
import shutil
import sys
from typing import assert_never

import strictcli
from strictcli import Context

from miniclaude import __version__


def _get_version() -> str:
    """Read version from pyproject.toml (editable installs) or fall back to package metadata."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject.exists():
        with open(pyproject, "rb") as f:
            return tomllib.load(f)["project"]["version"]
    from importlib.metadata import version

    return version("miniclaude")


app = strictcli.App(
    name="miniclaude",
    version=_get_version(),
    help="A lean, snappy fullscreen terminal client for Claude Code",
)


# read_only: reads the version string that was resolved at import and prints it.
@app.command("version", effect="read_only", help="Print the miniclaude version")
def cmd_version(ctx: Context) -> None:
    print(__version__)


# Which session the REPL runs is an exactly-one selection over three named
# alternatives, so it is a member-spelled choice flag: each member IS its own
# flag, and the argv the previous releases accepted is unchanged. What changes
# is that starting fresh has a name of its own -- `--new-session` -- instead of
# being the state left over when neither of the other two was typed, and that
# the mutual exclusion is the framework's refusal rather than a hand-written
# check inside the handler.
@strictcli.choice("resume", help="Resume a previous session by ID")
class ResumeSession:
    value: str = strictcli.member_value(help="ID of the session to resume")


@strictcli.choice("continue-session", help="Continue the most recent session")
class ContinueSession:
    pass


@strictcli.choice("new-session", help="Start a fresh session")
class NewSession:
    pass


SessionChoice = ResumeSession | ContinueSession | NewSession


# mutating: spawns a real Claude Code subprocess, which reads and writes files
# under --cwd, runs shell commands, calls the network, spends money and
# persists a session transcript. It also appends to ~/.miniclaude/history.
@app.command("repl", effect="mutating", help="Start the interactive fullscreen REPL")
@strictcli.flag("profile", type=str, presence="required", help="claudewheel profile to use")
@strictcli.flag("model", type=str, presence="required", help="Model to use, e.g. sonnet, haiku")
@strictcli.flag(
    "permission-mode",
    type=str,
    presence="required",
    choices=[
        strictcli.Choice("default", help="ask before every edit and command"),
        strictcli.Choice("acceptEdits", help="accept file edits without asking"),
        strictcli.Choice("plan", help="plan only -- propose, never act"),
        strictcli.Choice("bypassPermissions", help="ask for nothing at all"),
        strictcli.Choice("dontAsk", help="act without asking, refusing what needs consent"),
        strictcli.Choice("auto", help="let Claude Code pick the mode"),
    ],
    help="Permission mode",
)
@strictcli.flag(
    "cwd",
    type=str,
    presence="optional",
    help="Working directory. Omitted, the REPL runs in the current directory.",
)
@strictcli.choice_flag(
    "session",
    help="Which session the REPL runs",
    elect_by="member-flags",
    choices=[ResumeSession, ContinueSession, NewSession],
    default=NewSession(),
)
def cmd_repl(
    ctx: Context,
    profile: str,
    model: str,
    permission_mode: str,
    session: SessionChoice,
    cwd: str | None = None,
) -> int | None:
    from claudestream import AsyncSession, SessionConfig, SessionResolution

    from miniclaude._dialogs import PromptToolkitInteraction
    from miniclaude._repl import Repl

    resolution = None
    resume_session_id = None
    match session:
        case ResumeSession(value=session_id):
            resume_session_id = session_id
        case ContinueSession():
            resolution = SessionResolution(
                name=None,
                session_id=None,
                resume_session_id=None,
                continue_last=True,
                fork=False,
            )
        case NewSession():
            pass
        case _:
            assert_never(session)

    config = SessionConfig(
        model=model,
        profile=profile,
        cwd=cwd,
        permission_mode=permission_mode,
        intercept_permissions=True,
        resume_session_id=resume_session_id,
        session_resolution=resolution,
    )

    def _printer(text: str) -> None:
        if text:
            sys.stdout.write(text)

    width = shutil.get_terminal_size((80, 24)).columns
    repl = Repl(
        session_factory=lambda: AsyncSession(config),
        interaction=PromptToolkitInteraction(),
        printer=_printer,
        width=width,
        model=model,
        permission_mode=permission_mode,
    )
    asyncio.run(repl.run())
    return None


def _resolve_seed(seed: str | None) -> int:
    """Resolve a --seed flag to an int. Absence picks a random seed.

    Raises ValueError when a supplied seed does not parse as an integer.
    """
    if seed is not None:
        return int(seed)
    import random

    return random.SystemRandom().randrange(2**31)


# mutating: the session is fake -- no Claude, no network, no spend -- but the
# REPL it drives is the real one, and the real one creates ~/.miniclaude/ and
# appends every prompt typed into it to ~/.miniclaude/history. A persistent
# write outside the working directory is a mutation whatever produced it.
@app.command(
    "mock",
    effect="mutating",
    help="Interactive REPL against a mock session for TUI testing — no Claude CLI needed.",
)
@strictcli.flag(
    "seed",
    type=str,
    presence="optional",
    help="Random seed (integer) for reproducible content. Omitted, one is picked at random.",
)
def cmd_mock(ctx: Context, seed: str | None = None) -> int | None:
    from miniclaude._dialogs import PromptToolkitInteraction
    from miniclaude._mock import MockSession
    from miniclaude._repl import Repl

    try:
        seed_int = _resolve_seed(seed)
    except ValueError:
        print(f"error: --seed must be an integer, got {seed!r}", file=sys.stderr)
        return 1

    def _printer(text: str) -> None:
        if text:
            sys.stdout.write(text)

    width = shutil.get_terminal_size((80, 24)).columns
    # The seed is emitted as the REPL's intro (first output block) so it is
    # visible inside the fullscreen app -- a pre-run print() would be swallowed
    # by the alternate screen. It stays reproducible via `miniclaude mock
    # --seed <n>`.
    repl = Repl(
        session_factory=lambda: MockSession(seed_int),
        interaction=PromptToolkitInteraction(),
        printer=_printer,
        width=width,
        model="claude-mock",
        permission_mode="default",
        intro=f"mock seed: {seed_int}",
    )
    asyncio.run(repl.run())
    return None


def main() -> None:
    app.run()
