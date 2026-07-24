"""Offline pty tests for the fullscreen REPL (Phase 3).

Spawns ``miniclaude mock`` inside a pseudo-terminal and drives it through the
fullscreen (alternate-screen) TUI:

- The layout anchors the input frame to the bottom with the 3-row howmuchleft
  status bar beneath it and no dead rows below (3.2).
- The seed line is emitted as the intro and is visible in the session (3.2).
- ``table``/``help``/``demo`` turns render without "Window too small" (3.2).
- A ``dialogs`` turn suspends the app for the permission prompt and returns to
  fullscreen after the prompt is answered (3.2).

Marked ``pty`` so it gates only on pty availability (see conftest). Dimensions
are rows x cols.
"""

from __future__ import annotations

import time

import pytest

from tests._pty_harness import PtySession

pytestmark = pytest.mark.pty

_STARTUP_TIMEOUT = 30.0
_TURN_TIMEOUT = 30.0
_QUIT_TIMEOUT = 20.0

# The two terminal geometries the layout must hold at (rows x cols).
_SIZES = [(40, 120), (50, 210)]


def _argv():
    return ["uv", "run", "miniclaude", "mock", "--seed", "7"]


def _settle(pty: PtySession, seconds: float = 1.0) -> None:
    """Pump output for a while so the reconstructed frame is up to date."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pty._pump(0.2)


def _lowest_frame_border_row(frame: list[str]) -> int:
    """Row index of the lowest line containing an input-frame bottom border."""
    return max((i for i, line in enumerate(frame) if "└" in line), default=-1)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("rows,cols", _SIZES)
def test_layout_anchored_to_bottom(rows: int, cols: int):
    """The input frame sits just above the 3-row status bar, with no dead rows
    below it, and the seed line is visible."""
    with PtySession(_argv(), rows=rows, cols=cols) as pty:
        assert pty.read_until("┌", _STARTUP_TIMEOUT), (
            f"Input frame never appeared. Output:\n{pty.raw_text}"
        )
        assert pty.read_until("mock seed:", _STARTUP_TIMEOUT), (
            f"Seed intro never appeared. Output:\n{pty.raw_text}"
        )
        _settle(pty)
        frame = pty.frame()
        assert len(frame) == rows

        # The input frame's bottom border must be exactly 3 rows above the last
        # row: those 3 rows are the howmuchleft status bar, and nothing lies
        # below it (no dead rows).
        assert _lowest_frame_border_row(frame) == rows - 4, (
            "Input frame not anchored to the bottom. Frame:\n" + "\n".join(frame)
        )

        # The seed line is visible in the output region.
        assert any("mock seed:" in line for line in frame), (
            "Seed line not visible in the frame:\n" + "\n".join(frame)
        )

        assert "Window too small" not in "\n".join(frame)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("rows,cols", _SIZES)
def test_table_and_help_turns_never_too_small(rows: int, cols: int):
    """A wide table turn and a help turn both render without the layout
    collapsing into prompt_toolkit's "Window too small" fallback."""
    with PtySession(_argv(), rows=rows, cols=cols) as pty:
        assert pty.read_until("┌", _STARTUP_TIMEOUT)

        pty.send(b"table\r")
        # The table's bottom border proves it rendered inside the output window.
        assert pty.read_until("┴", _TURN_TIMEOUT), (
            f"Table never rendered. Output:\n{pty.raw_text}"
        )
        _settle(pty)
        assert "Window too small" not in pty.raw_text
        assert _lowest_frame_border_row(pty.frame()) == rows - 4

        pty.send(b"help\r")
        assert pty.read_until("Mock commands", _TURN_TIMEOUT), (
            f"Help never rendered. Output:\n{pty.raw_text}"
        )
        _settle(pty)
        assert "Window too small" not in pty.raw_text
        assert _lowest_frame_border_row(pty.frame()) == rows - 4

        assert pty.quit(_QUIT_TIMEOUT)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("rows,cols", _SIZES)
def test_demo_turn_never_too_small(rows: int, cols: int):
    """The ``demo`` turn runs every section (including its two dialogs) without
    ever showing "Window too small"; answering the dialogs lets it finish."""
    with PtySession(_argv(), rows=rows, cols=cols) as pty:
        assert pty.read_until("┌", _STARTUP_TIMEOUT)

        pty.send(b"demo\r")
        # The demo's Dialogs section blocks on a permission prompt, then an
        # AskUserQuestion. Answer both by accepting the default option.
        assert pty.read_until("Allow once", _TURN_TIMEOUT), (
            f"Demo permission prompt never appeared. Output:\n{pty.raw_text}"
        )
        pty.send(b"\r")
        assert pty.read_until("Red", _TURN_TIMEOUT), (
            f"Demo question prompt never appeared. Output:\n{pty.raw_text}"
        )
        pty.send(b"\r")
        # The closing Result line marks the end of the turn.
        assert pty.read_until("turn(s)", _TURN_TIMEOUT), (
            f"Demo turn never completed. Output:\n{pty.raw_text}"
        )
        _settle(pty)
        assert "Window too small" not in pty.raw_text
        assert _lowest_frame_border_row(pty.frame()) == rows - 4

        assert pty.quit(_QUIT_TIMEOUT)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("rows,cols", _SIZES)
def test_dialogs_suspend_and_restore(rows: int, cols: int):
    """A dialogs turn shows the permission prompt (the app suspends via
    in_terminal); answering it returns to the fullscreen layout."""
    with PtySession(_argv(), rows=rows, cols=cols) as pty:
        assert pty.read_until("┌", _STARTUP_TIMEOUT)

        pty.send(b"dialogs\r")
        assert pty.read_until("Allow once", _TURN_TIMEOUT), (
            f"Permission prompt never appeared. Output:\n{pty.raw_text}"
        )
        pty.send(b"\r")  # accept "Allow once"
        assert pty.read_until("Red", _TURN_TIMEOUT), (
            f"AskUserQuestion prompt never appeared. Output:\n{pty.raw_text}"
        )
        pty.send(b"\r")  # pick the first option
        assert pty.read_until("turn(s)", _TURN_TIMEOUT), (
            f"Dialogs turn never completed. Output:\n{pty.raw_text}"
        )

        # Back in fullscreen: the layout is intact and the child is still alive.
        _settle(pty)
        assert _lowest_frame_border_row(pty.frame()) == rows - 4
        assert pty._proc is not None and pty._proc.poll() is None

        assert pty.quit(_QUIT_TIMEOUT)


@pytest.mark.timeout(120)
def test_resize_re_renders_table_at_new_width():
    """Resizing the pty (TIOCSWINSZ + kernel SIGWINCH) re-renders a table at the
    new width with the layout intact -- automatically, with no resize handler."""
    start_rows, start_cols = 40, 120
    new_rows, new_cols = 50, 210
    with PtySession(_argv(), rows=start_rows, cols=start_cols) as pty:
        assert pty.read_until("┌", _STARTUP_TIMEOUT)

        pty.send(b"table\r")
        assert pty.read_until("┴", _TURN_TIMEOUT), (
            f"Table never rendered at the initial size. Output:\n{pty.raw_text}"
        )
        _settle(pty)

        # Widest table row before the resize (bounded by the old 120-col width).
        before = pty.frame()
        width_before = max(
            (len(line.rstrip()) for line in before if "│" in line), default=0
        )
        assert _lowest_frame_border_row(before) == start_rows - 4

        pre_len = len(pty.raw_text)
        pty.resize(new_rows, new_cols)

        # The SIGWINCH-driven repaint re-materializes the table at the new
        # width: a fresh table bottom border appears after the resize point.
        deadline = time.monotonic() + 15.0
        re_rendered = False
        while time.monotonic() < deadline:
            pty._pump(0.3)
            if "┴" in pty.raw_text[pre_len:]:
                re_rendered = True
                break
        assert re_rendered, (
            "Table was not re-rendered after resize. Output since resize:\n"
            + pty.raw_text[pre_len:]
        )

        _settle(pty)
        after = pty.frame(rows=new_rows, cols=new_cols)
        assert "Window too small" not in "\n".join(after)
        assert any("┴" in line for line in after), (
            "Table missing after resize. Frame:\n" + "\n".join(after)
        )
        # Layout stays intact: frame anchored to the (new) bottom.
        assert _lowest_frame_border_row(after) == new_rows - 4
        # The table re-flowed to use the wider terminal.
        width_after = max(
            (len(line.rstrip()) for line in after if "│" in line), default=0
        )
        assert width_after > width_before, (
            f"Table did not widen on resize: {width_before} -> {width_after}"
        )

        assert pty.quit(_QUIT_TIMEOUT)
