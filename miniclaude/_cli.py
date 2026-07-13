"""Command-line interface entry point for miniclaude, a lean inline terminal client for Claude Code."""

from __future__ import annotations

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


def main() -> None:
    app.run()
