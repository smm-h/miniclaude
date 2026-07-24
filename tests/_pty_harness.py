"""Reusable pseudo-terminal harness for driving any miniclaude command.

Generalizes the spawn / size / CPR-answering / read-until / graceful-quit /
teardown logic that the REPL integration smoke test used to inline. A
:class:`PtySession` launches an arbitrary argv inside a pty at a chosen
rows x cols, answers the Cursor Position Report (CPR) requests prompt_toolkit
emits at startup, and exposes both the raw accumulated output and a
pyte-reconstructed visible frame for assertions.

This is a test helper, not a test module -- the underscore-prefixed name keeps
pytest from collecting it.

Typical use::

    with PtySession(["uv", "run", "miniclaude", "mock", "--seed", "7"],
                    rows=24, cols=80) as pty:
        assert pty.read_until("┌", timeout=30)   # input frame appeared
        pty.send(b"md hello\\r")
        assert pty.read_until("hello", timeout=30)
        assert pty.quit(timeout=20)                    # /quit and exit cleanly
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

from tests._pty_screen import PtyScreen

# Repo root = parent of the tests/ directory that holds this file.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CPR request prompt_toolkit sends at startup to discover the cursor row/col.
_CPR_REQUEST = "\x1b[6n"
# Our canned reply: cursor at row 1, col 1. Enough to keep prompt_toolkit's
# renderer out of its degraded fallback mode.
_CPR_REPLY = b"\x1b[1;1R"


def _set_pty_size(fd: int, rows: int, cols: int) -> None:
    """Set the terminal window size on a pty file descriptor via TIOCSWINSZ."""
    # struct winsize: unsigned short rows, cols, xpixel, ypixel.
    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)


class PtySession:
    """Spawns ``argv`` in a pty at ``rows`` x ``cols`` and drives it.

    Use as a context manager so the child is always torn down (SIGKILL + fd
    close) even when an assertion fails mid-test.
    """

    def __init__(
        self,
        argv: list[str],
        *,
        rows: int,
        cols: int,
        env: dict | None = None,
        cwd: str | None = None,
    ) -> None:
        self.argv = list(argv)
        self.rows = rows
        self.cols = cols
        self._env = env
        self._cwd = cwd or _REPO_ROOT
        self._master_fd: int | None = None
        self._proc: subprocess.Popen | None = None
        self._chunks: list[str] = []

    # --- lifecycle ---

    def __enter__(self) -> "PtySession":
        master_fd, slave_fd = os.openpty()
        # A nonzero size is mandatory: the default 0x0 pty prevents any
        # rendering, so prompt_toolkit draws nothing.
        _set_pty_size(slave_fd, rows=self.rows, cols=self.cols)

        env = dict(os.environ if self._env is None else self._env)
        # Keep any local editable overlay (e.g. claudestream) instead of letting
        # `uv run` re-sync the environment on every spawn.
        env.setdefault("UV_NO_SYNC", "1")
        env.setdefault("TERM", "xterm-256color")

        self._proc = subprocess.Popen(
            self.argv,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=self._cwd,
            env=env,
            close_fds=True,
        )
        os.close(slave_fd)
        self._master_fd = master_fd
        return self

    def __exit__(self, *exc) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGKILL)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None

    # --- I/O ---

    def send(self, data: bytes) -> None:
        """Write raw bytes to the pty master (as if typed at the terminal)."""
        assert self._master_fd is not None, "session not started"
        os.write(self._master_fd, data)

    def resize(self, rows: int, cols: int) -> None:
        """Resize the pty via TIOCSWINSZ.

        Setting the window size on the terminal makes the kernel deliver
        SIGWINCH to the child's foreground process group -- the same mechanism a
        real terminal emulator uses when its window is resized.
        """
        assert self._master_fd is not None, "session not started"
        _set_pty_size(self._master_fd, rows=rows, cols=cols)
        self.rows = rows
        self.cols = cols

    def _pump(self, timeout_slice: float = 0.5) -> str | None:
        """Read one chunk from the master, answering CPR; return decoded text.

        Returns ``None`` on EOF/error/no-data-ready so callers can loop.
        """
        assert self._master_fd is not None, "session not started"
        ready, _, _ = select.select([self._master_fd], [], [], timeout_slice)
        if not ready:
            return None
        try:
            chunk = os.read(self._master_fd, 65536)
        except OSError:
            return None
        if not chunk:
            return None
        text = chunk.decode("utf-8", errors="replace")
        self._chunks.append(text)
        if _CPR_REQUEST in text:
            try:
                os.write(self._master_fd, _CPR_REPLY)
            except OSError:
                pass
        return text

    def read_until(self, needle: str, timeout: float) -> bool:
        """Read (answering CPR) until ``needle`` appears in accumulated output."""
        deadline = time.monotonic() + timeout
        if needle in self.raw_text:
            return True
        while time.monotonic() < deadline:
            self._pump()
            if needle in self.raw_text:
                return True
        return False

    # --- output views ---

    @property
    def raw_text(self) -> str:
        """All decoded output read so far, concatenated."""
        return "".join(self._chunks)

    def frame(self, rows: int | None = None, cols: int | None = None) -> list[str]:
        """Reconstruct the visible screen from all output read so far."""
        screen = PtyScreen(rows or self.rows, cols or self.cols)
        screen.feed(self.raw_text.encode("utf-8", errors="replace"))
        return screen.frame

    # --- shutdown ---

    def wait_exit(self, timeout: float) -> bool:
        """Wait for the child to exit, draining output so it never blocks."""
        assert self._proc is not None, "session not started"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                return True
            self._pump(timeout_slice=0.2)
        return self._proc.poll() is not None

    def quit(self, timeout: float, command: bytes = b"/quit\r") -> bool:
        """Send the quit command and wait for the child to exit within ``timeout``."""
        self.send(command)
        return self.wait_exit(timeout)
