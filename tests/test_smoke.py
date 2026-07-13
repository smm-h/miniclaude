"""Smoke tests: the package imports and its version is internally consistent."""

from __future__ import annotations

import tomllib
from pathlib import Path

import miniclaude


def test_import_works() -> None:
    assert miniclaude.__version__


def test_version_matches_pyproject() -> None:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert miniclaude.__version__ == declared
