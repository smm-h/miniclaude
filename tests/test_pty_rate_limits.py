"""Offline pty test: the status bar renders populated rate-limit rows.

Spawns ``miniclaude mock --seed N`` inside a pseudo-terminal, waits for the
input frame, and confirms the howmuchleft status bar starts out showing
the ``?%`` placeholder (no rate-limit data yet). After one mock turn -- which
emits seed-derived RateLimit events -- the placeholder disappears from the
visible status-bar region, proving the events reached howmuchleft's stdin.

Needs the ``howmuchleft`` binary (the status bar's renderer) to observe the
placeholder at all; skipped cleanly when it is absent. Marked ``pty`` so it
gates only on pty availability (see conftest).
"""

from __future__ import annotations

import shutil
import time

import pytest

from tests._pty_harness import PtySession

pytestmark = [
    pytest.mark.pty,
    pytest.mark.skipif(
        shutil.which("howmuchleft") is None,
        reason="howmuchleft binary not on PATH (status bar renderer unavailable)",
    ),
]

_STARTUP_TIMEOUT = 30.0
_TURN_TIMEOUT = 30.0
_QUIT_TIMEOUT = 20.0
_MARKER = "rl-MARKER"
_PLACEHOLDER = "?%"


@pytest.mark.timeout(120)
def test_status_bar_rate_limits_populated():
    argv = ["uv", "run", "miniclaude", "mock", "--seed", "11"]
    with PtySession(argv, rows=24, cols=80) as pty:
        # The input frame plus the placeholder rate-limit rows must appear
        # first: with no RateLimit events yet, howmuchleft renders "?%".
        assert pty.read_until("┌", _STARTUP_TIMEOUT), (
            f"Input frame never appeared. Output so far:\n{pty.raw_text}"
        )
        assert pty.read_until(_PLACEHOLDER, _STARTUP_TIMEOUT), (
            f"Rate-limit placeholder never appeared. Output:\n{pty.raw_text}"
        )

        # One mock turn emits RateLimit events that populate the status bar.
        pty.send(f"md {_MARKER}\r".encode())
        assert pty.read_until(_MARKER, _TURN_TIMEOUT), (
            f"Mock never streamed the marker back. Output:\n{pty.raw_text}"
        )

        # The status bar refreshes ~4x/s; poll the reconstructed visible frame
        # until the ?% placeholder is gone from the status-bar region.
        deadline = time.monotonic() + 15.0
        gone = False
        while time.monotonic() < deadline:
            pty._pump(timeout_slice=0.3)
            if _PLACEHOLDER not in "\n".join(pty.frame()):
                gone = True
                break
        assert gone, (
            "Status bar still shows the ?% placeholder after a turn; rate-limit "
            f"events did not populate it. Frame:\n" + "\n".join(pty.frame())
        )

        assert pty.quit(_QUIT_TIMEOUT), "Mock REPL did not exit after /quit"
