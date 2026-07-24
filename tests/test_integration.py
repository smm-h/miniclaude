"""End-to-end pty smoke test for the miniclaude REPL.

Spawns the real CLI (`miniclaude repl`) inside a pseudo-terminal so
prompt_toolkit has a genuine TTY, drives one turn against a live model, and
verifies the response renders in the fullscreen Application's output region.
Marked ``integration`` so the default unit run (`pytest -m "not integration"`)
skips it.

The REPL runs fullscreen (alternate screen): output is modelled as blocks and
materialized into the scrollable output window, with no terminal-native
scrollback. The spawn / pty-size / CPR-answering / read-until / graceful-quit /
teardown mechanics all live in the reusable
:class:`tests._pty_harness.PtySession`; this test only expresses the scenario on
top of it.

Requires a working claudewheel profile ("personal") and network access. On a
sandboxed/offline machine this test will fail at the model turn -- that is a
real environment blocker, not a bug in the REPL.
"""

from __future__ import annotations

import pytest

from tests._pty_harness import PtySession

pytestmark = pytest.mark.integration

_STARTUP_TIMEOUT = 30.0
_TURN_TIMEOUT = 120.0
_QUIT_TIMEOUT = 20.0


@pytest.mark.timeout(240)
def test_repl_pty_smoke():
    argv = [
        "uv", "run", "miniclaude", "repl",
        "--profile", "personal",
        "--model", "haiku",
        "--permission-mode", "default",
    ]
    with PtySession(argv, rows=24, cols=80) as pty:
        # Wait for the inline app to render. The Frame widget around the input
        # area uses box-drawing chars -- seeing "┌" proves the Application is up
        # and the input area is visible.
        assert pty.read_until("┌", _STARTUP_TIMEOUT), (
            f"Inline frame never appeared. Output so far:\n{pty.raw_text}"
        )

        # The marker "MINIOK" is asked for as MINI + OK so the literal token
        # never appears in our echoed input line -- only in the model's reply.
        pty.send(b"Reply with only the word MINI joined to OK, no space.\r")

        got = pty.read_until("MINIOK", _TURN_TIMEOUT)
        assert got, f"Model never produced MINIOK. Output:\n{pty.raw_text}"

        assert pty.quit(_QUIT_TIMEOUT), "REPL did not exit after /quit"
