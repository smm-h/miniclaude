"""Offline pty smoke test for the miniclaude REPL, driven by the mock session.

Spawns ``miniclaude mock --seed 7`` inside a pseudo-terminal via the shared
:class:`tests._pty_harness.PtySession`, waits for the input frame, sends
one ``md <markdown>`` turn (which the mock streams back verbatim), asserts the
marker text renders, and quits. Unlike the ``integration`` smoke test, this
needs zero live prerequisites -- no claude binary, no claudewheel profile, no
network -- so it runs in the default unit suite whenever a pty is available.

Marked ``pty`` so it gates only on pty availability (see conftest).
"""

from __future__ import annotations

import pytest

from tests._pty_harness import PtySession

pytestmark = pytest.mark.pty

_STARTUP_TIMEOUT = 30.0
_TURN_TIMEOUT = 30.0
_QUIT_TIMEOUT = 20.0

# A distinctive marker unlikely to appear in any startup/chrome output, so its
# presence proves the mock actually streamed our turn back.
_MARKER = "hello-MARKER"


@pytest.mark.timeout(120)
def test_mock_repl_pty_smoke():
    argv = ["uv", "run", "miniclaude", "mock", "--seed", "7"]
    with PtySession(argv, rows=24, cols=80) as pty:
        # Wait for the input frame (box-drawing char from the Frame widget).
        assert pty.read_until("┌", _STARTUP_TIMEOUT), (
            f"Input frame never appeared. Output so far:\n{pty.raw_text}"
        )

        # `md <markdown>` streams the given markdown back verbatim.
        pty.send(f"md {_MARKER}\r".encode())

        assert pty.read_until(_MARKER, _TURN_TIMEOUT), (
            f"Mock never streamed the marker back. Output:\n{pty.raw_text}"
        )

        assert pty.quit(_QUIT_TIMEOUT), "Mock REPL did not exit after /quit"
