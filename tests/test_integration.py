"""End-to-end pty smoke test for the miniclaude REPL.

Spawns the real CLI (`miniclaude repl`) inside a pseudo-terminal so
prompt_toolkit has a genuine TTY, drives one turn against a live model, and
verifies the response renders in the inline Application's output region.
Marked ``integration`` so the default unit run (`pytest -m "not integration"`)
skips it.

The REPL uses inline mode (no alternate screen). Output flows through
patch_stdout into native terminal scrollback. The test reads the raw byte
stream from the pty master and checks for content within the inline-rendered
output (box-drawing chars from the Frame widget, model response).

Requires a working claudewheel profile ("personal") and network access. On a
sandboxed/offline machine this test will fail at the model turn -- that is a
real environment blocker, not a bug in the REPL.
"""

from __future__ import annotations

import fcntl
import os
import select
import signal
import struct
import subprocess
import termios
import time

import pytest

pytestmark = pytest.mark.integration

_STARTUP_TIMEOUT = 30.0
_TURN_TIMEOUT = 120.0
_QUIT_TIMEOUT = 20.0


def _read_until(master_fd: int, needle: str, timeout: float, accum: list[str]) -> bool:
    """Read from the pty master until ``needle`` appears in accumulated output.

    Also responds to CPR (Cursor Position Report) requests that prompt_toolkit
    sends in inline mode. Without a CPR response, prompt_toolkit falls back to
    a degraded mode and may not render the layout correctly.
    """
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
        text = chunk.decode("utf-8", errors="replace")
        accum.append(text)
        # Respond to CPR requests (\x1b[6n) with cursor at row 1, col 1.
        # prompt_toolkit sends this in inline mode to discover cursor position.
        if "\x1b[6n" in text:
            try:
                os.write(master_fd, b"\x1b[1;1R")
            except OSError:
                pass
        if needle in "".join(accum):
            return True
    return False


@pytest.mark.timeout(240)
def test_repl_pty_smoke():
    master_fd, slave_fd = os.openpty()

    # Set a reasonable pty size so prompt_toolkit can render the inline layout.
    # Default pty size is 0x0 on Linux, which prevents any rendering.
    _set_pty_size(slave_fd, rows=24, cols=80)

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
        # Wait for the inline app to render. The Frame widget around the input
        # area uses box-drawing chars -- seeing "┌" proves the Application is
        # up and the input area is visible.
        assert _read_until(master_fd, "┌", _STARTUP_TIMEOUT, accum), (
            f"Inline frame never appeared. Output so far:\n{''.join(accum)}"
        )

        # The marker "MINIOK" is asked for as MINI + OK so the literal token
        # never appears in our echoed input line -- only in the model's reply.
        os.write(master_fd, b"Reply with only the word MINI joined to OK, no space.\r")

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


def _set_pty_size(fd: int, rows: int, cols: int) -> None:
    """Set the terminal size on a pty file descriptor."""
    # struct winsize: unsigned short rows, cols, xpixel, ypixel
    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)
