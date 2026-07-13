"""End-to-end pty smoke test for the miniclaude REPL.

Spawns the real CLI (`miniclaude repl`) inside a pseudo-terminal so
prompt_toolkit has a genuine TTY, drives one turn against a live model, and
verifies the response renders in scrollback. Marked ``integration`` so the
default unit run (`pytest -m "not integration"`) skips it.

Requires a working claudewheel profile ("personal") and network access. On a
sandboxed/offline machine this test will fail at the model turn -- that is a
real environment blocker, not a bug in the REPL.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import time

import pytest

pytestmark = pytest.mark.integration

_STARTUP_TIMEOUT = 30.0
_TURN_TIMEOUT = 120.0
_QUIT_TIMEOUT = 20.0


def _read_until(master_fd: int, needle: str, timeout: float, accum: list[str]) -> bool:
    """Read from the pty master until ``needle`` appears in accumulated output."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master_fd], [], [], 0.5)
        if not ready:
            continue
        try:
            chunk = os.read(master_fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        accum.append(chunk.decode("utf-8", errors="replace"))
        if needle in "".join(accum):
            return True
    return False


@pytest.mark.timeout(240)
def test_repl_pty_smoke():
    master_fd, slave_fd = os.openpty()
    env = dict(os.environ)
    env["UV_NO_SYNC"] = "1"  # keep the local claudestream overlay
    env["TERM"] = "xterm-256color"

    proc = subprocess.Popen(
        [
            "uv", "run", "miniclaude", "repl",
            "--profile", "personal",
            "--model", "haiku",
            "--permission-mode", "default",
        ],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)

    accum: list[str] = []
    try:
        # Wait for the input prompt to appear (REPL is ready for input).
        assert _read_until(master_fd, ">", _STARTUP_TIMEOUT, accum), (
            f"REPL prompt never appeared. Output so far:\n{''.join(accum)}"
        )

        # The marker "MINIOK" is asked for as MINI + OK so the literal token
        # never appears in our echoed input line -- only in the model's reply.
        os.write(master_fd, b"Reply with only the word MINI joined to OK, no space.\r")

        # The SystemInit startup line proves the turn actually started.
        assert _read_until(master_fd, "miniclaude", 60.0, accum), (
            f"startup line never rendered. Output:\n{''.join(accum)}"
        )
        got = _read_until(master_fd, "MINIOK", _TURN_TIMEOUT, accum)
        output = "".join(accum)
        assert got, f"Model never produced MINIOK. Output:\n{output}"

        os.write(master_fd, b"/quit\r")
        deadline = time.monotonic() + _QUIT_TIMEOUT
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        assert proc.poll() is not None, "REPL did not exit after /quit"
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGKILL)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.close(master_fd)
        except OSError:
            pass
