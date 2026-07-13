"""Command-line interface entry point for miniclaude, a lean inline terminal client for Claude Code."""

from __future__ import annotations

import asyncio
import shutil
import sys

import strictcli

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
    help="A lean, snappy inline terminal client for Claude Code",
)


@app.command("version", help="Print the miniclaude version")
def cmd_version() -> None:
    print(__version__)


# --resume and --continue-session are both optional but mutually exclusive.
# strictcli's MutexGroup forces "exactly one required", which is wrong here, so
# the constraint is enforced manually in the handler.
@app.command("repl", help="Start the interactive inline REPL")
@strictcli.flag("profile", type=str, help="claudewheel profile to use (required)")
@strictcli.flag("model", type=str, help="Model to use, e.g. sonnet, haiku (required)")
@strictcli.flag(
    "permission-mode",
    type=str,
    choices=["default", "acceptEdits", "plan", "bypassPermissions", "dontAsk", "auto"],
    help="Permission mode (required)",
)
@strictcli.flag("cwd", type=str, default="", help="Working directory (default: current)")
@strictcli.flag("resume", type=str, default="", help="Resume a previous session by ID")
@strictcli.flag(
    "continue-session",
    type=bool,
    default=False,
    help="Continue the most recent session",
)
def cmd_repl(
    profile: str,
    model: str,
    permission_mode: str,
    cwd: str = "",
    resume: str = "",
    continue_session: bool = False,
) -> int | None:
    from claudestream import AsyncSession, SessionConfig, SessionResolution

    from miniclaude._dialogs import PromptToolkitInteraction
    from miniclaude._repl import Repl

    if resume and continue_session:
        print(
            "error: --resume and --continue-session are mutually exclusive",
            file=sys.stderr,
        )
        return 1

    resolution = None
    if continue_session:
        resolution = SessionResolution(
            name=None,
            session_id=None,
            resume_session_id=None,
            continue_last=True,
            fork=False,
        )

    config = SessionConfig(
        model=model,
        profile=profile,
        cwd=cwd or None,
        permission_mode=permission_mode,
        intercept_permissions=True,
        resume_session_id=resume or None,
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
        version=_get_version(),
        width=width,
    )
    asyncio.run(repl.run())
    return None


def main() -> None:
    app.run()
