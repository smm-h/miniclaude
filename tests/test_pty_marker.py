"""Sanity guard for the ``pty`` marker gate in conftest.

The ``pty`` marker must gate ONLY on pty availability -- never on the live
claude CLI, a claudewheel profile, or network access. This trivial test carries
the marker and does nothing but confirm a pty can be allocated. If the conftest
gate ever regresses to requiring live-CLI prerequisites for pty-marked tests,
this test would start skipping in environments that have a pty but no profile,
surfacing the regression.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.pty


def test_pty_marker_runs_without_live_prereqs():
    master_fd, slave_fd = os.openpty()
    try:
        assert master_fd >= 0
        assert slave_fd >= 0
    finally:
        os.close(master_fd)
        os.close(slave_fd)
