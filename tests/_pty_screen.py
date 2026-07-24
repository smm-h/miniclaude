"""Reconstruct a terminal screen from raw pty output bytes, for pty test asserts.

A pty master yields a raw byte stream full of ANSI control sequences: cursor
moves, colors, clears, scrolls. Asserting against that raw stream directly is
brittle -- the same visible frame can be produced by countless different byte
sequences. This helper feeds the bytes through a :mod:`pyte` terminal emulator
and exposes the *reconstructed visible frame* (what a user would actually see),
which is what tests should assert against.

Usage:

    from tests._pty_screen import PtyScreen, render_frame

    frame = render_frame(raw_bytes, rows=24, cols=80)  # list[str], one per row
    assert "hello" in frame[0]

    # Or keep the screen around to inspect cell attributes (colors, bold):
    screen = PtyScreen(rows=24, cols=80).feed(raw_bytes)
    assert screen.cell(1, 2).fg == "red"

This module is a test helper, not a test module -- its underscore-prefixed name
keeps pytest from collecting it.
"""

from __future__ import annotations

import pyte


class PtyScreen:
    """A rows x cols terminal emulator fed incrementally with raw pty bytes."""

    def __init__(self, rows: int, cols: int) -> None:
        # pyte.Screen takes (columns, lines) in that order.
        self._screen = pyte.Screen(cols, rows)
        self._stream = pyte.ByteStream(self._screen)
        self.rows = rows
        self.cols = cols

    def feed(self, data: bytes) -> "PtyScreen":
        """Feed a chunk of raw pty bytes into the emulator; returns self."""
        self._stream.feed(data)
        return self

    @property
    def frame(self) -> list[str]:
        """The visible screen as a list of ``rows`` strings, each ``cols`` wide."""
        return list(self._screen.display)

    def text(self) -> str:
        """The visible screen joined into a single newline-separated string."""
        return "\n".join(self.frame)

    def cell(self, row: int, col: int):
        """The pyte ``Char`` at (row, col) -- has ``.data``/``.fg``/``.bg``/``.bold``."""
        return self._screen.buffer[row][col]


def render_frame(data: bytes, rows: int, cols: int) -> list[str]:
    """Feed ``data`` into a fresh rows x cols screen and return the visible frame."""
    return PtyScreen(rows, cols).feed(data).frame
