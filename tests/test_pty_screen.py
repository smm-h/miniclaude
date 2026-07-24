"""Unit tests for the pyte screen-reconstruction helper.

Feeds a hand-built ANSI byte stream (exercising clear, cursor movement, and
colored text) into :class:`tests._pty_screen.PtyScreen` and asserts the
reconstructed frame matches the known expected visible content. This pins down
the helper's behavior so the pty harness that depends on it can be trusted.
"""

from __future__ import annotations

from tests._pty_screen import PtyScreen, render_frame

# A canned stream: clear screen, write "Hello" at home, a red "RED" on row 2
# starting at column 3, and "end" on row 3. Uses absolute cursor addressing so
# the expected frame is fully determined regardless of initial cursor state.
_CANNED = (
    b"\x1b[2J"          # clear entire screen
    b"\x1b[H"           # cursor to row 1, col 1 (home)
    b"Hello"            # -> row 0 (0-indexed), cols 0..4
    b"\x1b[2;3H"        # cursor to row 2, col 3
    b"\x1b[31mRED\x1b[0m"  # red foreground "RED" -> row 1, cols 2..4, then reset
    b"\x1b[3;1H"        # cursor to row 3, col 1
    b"end"             # -> row 2, cols 0..2
)


def test_render_frame_reconstructs_text_and_layout():
    frame = render_frame(_CANNED, rows=5, cols=20)

    assert len(frame) == 5
    assert all(len(row) == 20 for row in frame)

    # Row 0: "Hello" left-aligned, rest blank.
    assert frame[0].rstrip() == "Hello"
    # Row 1: "RED" starting at column index 2 (col 3, 1-indexed).
    assert frame[1][:5] == "  RED"
    assert frame[1].rstrip() == "  RED"
    # Row 2: "end".
    assert frame[2].rstrip() == "end"
    # Rows 3 and 4: cleared / blank.
    assert frame[3].strip() == ""
    assert frame[4].strip() == ""


def test_pty_screen_exposes_colors():
    screen = PtyScreen(rows=5, cols=20).feed(_CANNED)

    # The "R" of the red "RED" sits at row 1, col 2 with a red foreground.
    red_cell = screen.cell(1, 2)
    assert red_cell.data == "R"
    assert red_cell.fg == "red"

    # A plain cell (the "H" of "Hello") carries the default foreground.
    plain_cell = screen.cell(0, 0)
    assert plain_cell.data == "H"
    assert plain_cell.fg == "default"


def test_feed_is_incremental():
    # Feeding in two chunks yields the same frame as feeding all at once.
    split = len(_CANNED) // 2
    screen = PtyScreen(rows=5, cols=20)
    screen.feed(_CANNED[:split]).feed(_CANNED[split:])
    assert screen.frame == render_frame(_CANNED, rows=5, cols=20)
