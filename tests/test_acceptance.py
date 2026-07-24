"""Acceptance suite: one test per original user complaint about the TUI.

Before the fullscreen overhaul the REPL had five concrete, user-reported
defects. Each test below reproduces the scenario that surfaced one of those
complaints and asserts the fixed behaviour, so a regression in any of them
fails loudly. Every test is independent and seeded (``miniclaude mock
--seed``), needs zero live prerequisites (no claude binary, no profile, no
network), and drives the real fullscreen TUI inside a pseudo-terminal.

The five complaints, in order:

1. The input box drifted upward, leaving dead rows between it and the bottom
   status bar (:func:`test_input_anchored_to_bottom`).
2. The status bar stayed stuck on ``?%`` placeholders and never showed real
   rate-limit data (:func:`test_status_bar_populated_after_turn`).
3. Ordinary turns collapsed into prompt_toolkit's "Window too small" fallback
   (:func:`test_no_window_too_small`).
4. Tables rendered without their row separators
   (:func:`test_table_separators_present`).
5. The mouse wheel could not hold the view still -- output kept snapping back
   to the bottom (:func:`test_scroll_lock_end_to_end`).

Marked ``pty`` so it gates only on pty availability (see conftest). Dimensions
are rows x cols.
"""

from __future__ import annotations

import shutil
import time

import pytest

from tests._pty_harness import PtySession

pytestmark = pytest.mark.pty

_STARTUP_TIMEOUT = 30.0
_TURN_TIMEOUT = 30.0
_QUIT_TIMEOUT = 20.0

# The two terminal geometries the layout must hold at (rows x cols).
_SIZES = [(40, 120), (50, 210)]

# SGR-encoded mouse wheel events aimed inside the output window (1-based col;row
# near the top of the screen, well above the input frame and status bar).
# Button 64 = wheel up, 65 = wheel down; the trailing "M" marks a press.
_WHEEL_UP = b"\x1b[<64;5;2M"
_WHEEL_DOWN = b"\x1b[<65;5;2M"


def _argv(seed: int = 7) -> list[str]:
    return ["uv", "run", "miniclaude", "mock", "--seed", str(seed)]


def _settle(pty: PtySession, seconds: float = 1.5) -> None:
    """Pump output for a while so the reconstructed frame is up to date."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pty._pump(0.2)


def _lowest_input_border_row(frame: list[str]) -> int:
    """Row index of the lowest line containing an input-frame bottom border."""
    return max((i for i, line in enumerate(frame) if "└" in line), default=-1)


def _paste(text: str) -> bytes:
    """Wrap ``text`` in a bracketed-paste sequence.

    prompt_toolkit inserts bracketed-paste content into the input buffer
    literally -- including newlines -- without triggering the accept handler, so
    this is how a multi-line prompt (e.g. a markdown table) is entered before a
    final Enter submits it.
    """
    return b"\x1b[200~" + text.encode() + b"\x1b[201~"


# --- Complaint 1: input box drifts up, leaving dead rows --------------------


@pytest.mark.timeout(120)
@pytest.mark.parametrize("rows,cols", _SIZES)
def test_input_anchored_to_bottom(rows: int, cols: int):
    """Complaint 1: the input box floated upward with blank dead rows beneath it.

    The input frame's bottom border must sit directly above the 3-row status
    bar, which occupies the last three rows -- no blank rows between the input
    and the status bar, and nothing below the status bar.
    """
    with PtySession(_argv(), rows=rows, cols=cols) as pty:
        assert pty.read_until("┌", _STARTUP_TIMEOUT), (
            f"Input frame never appeared. Output:\n{pty.raw_text}"
        )
        _settle(pty)
        frame = pty.frame()
        assert len(frame) == rows

        # The input frame's bottom border is exactly 3 rows above the last row:
        # those 3 rows are the status bar, and nothing lies below it.
        assert _lowest_input_border_row(frame) == rows - 4, (
            "Input frame not anchored to the bottom (dead rows below it). Frame:\n"
            + "\n".join(frame)
        )
        # The last three rows are the status bar and must be non-blank -- i.e.
        # real content occupies them, not dead/blank rows.
        for i in range(rows - 3, rows):
            assert frame[i].strip(), (
                f"Status-bar row {i} is blank (dead row below input). Frame:\n"
                + "\n".join(frame)
            )
        # No blank rows between the input top border and its bottom border.
        top = next((i for i, line in enumerate(frame) if "┌" in line), -1)
        assert 0 <= top < rows - 4
        for i in range(top, rows - 3):
            assert frame[i].strip(), (
                f"Blank row {i} inside/around the input frame. Frame:\n"
                + "\n".join(frame)
            )


# --- Complaint 2: status bar stuck on ?% placeholders -----------------------


@pytest.mark.timeout(120)
@pytest.mark.skipif(
    shutil.which("howmuchleft") is None,
    reason="howmuchleft binary not on PATH (status bar renderer unavailable)",
)
def test_status_bar_populated_after_turn():
    """Complaint 2: the status bar never left its ``?%`` placeholder state.

    A single mock turn emits seed-derived RateLimit events; after it, the ``?%``
    placeholder must be gone from the status-bar region (the last three rows),
    proving real rate-limit data reached the howmuchleft status bar.
    """
    with PtySession(_argv(seed=11), rows=24, cols=80) as pty:
        assert pty.read_until("┌", _STARTUP_TIMEOUT)
        # With no RateLimit events yet, the bar starts on the placeholder.
        assert pty.read_until("?%", _STARTUP_TIMEOUT), (
            f"Placeholder never appeared. Output:\n{pty.raw_text}"
        )

        pty.send(b"md pop-MARKER\r")
        assert pty.read_until("pop-MARKER", _TURN_TIMEOUT), (
            f"Mock never streamed the marker back. Output:\n{pty.raw_text}"
        )

        # Poll the reconstructed status region until the placeholder clears.
        deadline = time.monotonic() + 15.0
        gone = False
        while time.monotonic() < deadline:
            pty._pump(0.3)
            status = "\n".join(pty.frame()[-3:])
            if "?%" not in status:
                gone = True
                break
        assert gone, (
            "Status bar still shows ?% after a turn; rate-limit events did not "
            "populate it. Frame:\n" + "\n".join(pty.frame())
        )


# --- Complaint 3: turns collapse into "Window too small" --------------------


def _safe_frame_text(pty: PtySession) -> str | None:
    """Reconstruct the visible frame, tolerating pyte's mid-stream fragility.

    ``frame()`` re-feeds the entire accumulated byte stream to pyte on every
    call; when the tail is cut mid-wide-character (an emoji/CJK sample still
    arriving), pyte's ``display`` raises ``IndexError`` on the half-formed stub
    cell. That is a transient reconstruction artifact, not a real render, so we
    return ``None`` for that poll and let the next (well-formed) frame carry the
    assertion. The "Window too small" fallback is a persistent state, so it is
    still caught on the many polls that reconstruct cleanly -- and the final
    cumulative ``raw_text`` check is definitive.
    """
    try:
        return "\n".join(pty.frame())
    except IndexError:
        return None


def _drive_and_poll_no_too_small(
    pty: PtySession, completion_needle: str, timeout: float
) -> None:
    """Pump until ``completion_needle`` appears, asserting "Window too small"
    never shows up in any reconstructed frame along the way."""
    deadline = time.monotonic() + timeout
    done = False
    while time.monotonic() < deadline:
        pty._pump(0.2)
        text = _safe_frame_text(pty)
        if text is not None:
            assert "Window too small" not in text, (
                "Layout collapsed to 'Window too small' mid-turn. Frame:\n" + text
            )
        if completion_needle in pty.raw_text:
            done = True
            break
    assert done, (
        f"Turn never reached {completion_needle!r}. Output:\n{pty.raw_text}"
    )
    # Nothing transient slipped through the cumulative stream either.
    assert "Window too small" not in pty.raw_text


@pytest.mark.timeout(180)
@pytest.mark.parametrize("rows,cols", _SIZES)
def test_no_window_too_small(rows: int, cols: int):
    """Complaint 3: table/help/demo turns fell into the "Window too small"
    fallback. None of them may ever render that string, polled throughout each
    turn (not just at the end)."""
    with PtySession(_argv(), rows=rows, cols=cols) as pty:
        assert pty.read_until("┌", _STARTUP_TIMEOUT)

        pty.send(b"table\r")
        _drive_and_poll_no_too_small(pty, "┴", _TURN_TIMEOUT)

        pty.send(b"help\r")
        _drive_and_poll_no_too_small(pty, "Mock commands", _TURN_TIMEOUT)

        # The demo blocks on two dialogs (a permission prompt, then a question);
        # accept the defaults so it runs to completion.
        pty.send(b"demo\r")
        _drive_and_poll_no_too_small(pty, "Allow once", _TURN_TIMEOUT)
        pty.send(b"\r")
        _drive_and_poll_no_too_small(pty, "Red", _TURN_TIMEOUT)
        pty.send(b"\r")
        _drive_and_poll_no_too_small(pty, "turn(s)", _TURN_TIMEOUT)

        assert pty.quit(_QUIT_TIMEOUT)


# --- Complaint 4: tables render without row separators ----------------------


@pytest.mark.timeout(120)
def test_table_separators_present():
    """Complaint 4: rendered tables were missing their row separators.

    A fixed 3-row, 3-column markdown table (entered via ``md`` + bracketed
    paste, small enough to fit fully on screen) must render with the heavy
    header rule (``╞═╪═╡``) and exactly two light body rules (``├─┼─┤``) between
    its three body rows. Separately, a seeded ``table`` turn must show at least
    one light body rule -- proving separators appear on generated tables too.
    """
    fixed = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |\n| 7 | 8 | 9 |"
    with PtySession(_argv(), rows=40, cols=120) as pty:
        assert pty.read_until("┌", _STARTUP_TIMEOUT)
        pty.send(b"md " + _paste(fixed) + b"\r")
        assert pty.read_until("╡", _TURN_TIMEOUT), (
            f"Fixed table never rendered its heavy header rule. Output:\n{pty.raw_text}"
        )
        _settle(pty)
        text = "\n".join(pty.frame())
        assert "╞" in text and "╪" in text and "╡" in text, (
            "Heavy header rule (╞═╪═╡) missing from the fixed table. Frame:\n" + text
        )
        assert text.count("├") == 2, (
            f"Expected exactly 2 light body rules, found {text.count('├')}. Frame:\n"
            + text
        )

    # A seeded table turn on a screen tall enough to show body separators.
    with PtySession(_argv(), rows=60, cols=120) as pty:
        assert pty.read_until("┌", _STARTUP_TIMEOUT)
        pty.send(b"table\r")
        assert pty.read_until("┴", _TURN_TIMEOUT), (
            f"Seeded table never rendered. Output:\n{pty.raw_text}"
        )
        _settle(pty)
        text = "\n".join(pty.frame())
        assert "├" in text, (
            "Seeded table shows no light body rule. Frame:\n" + text
        )


# --- Complaint 5: mouse wheel cannot hold the view still ---------------------


@pytest.mark.timeout(150)
def test_scroll_lock_end_to_end():
    """Complaint 5: the mouse wheel could not lock the view -- output kept
    snapping back to the bottom.

    Fills the output past one screen with a tail marker on the last line. While
    following, the marker is visible. After wheel-up events engage scroll-lock,
    the view stops following and the marker leaves the frame. Wheel-down events
    return to the bottom, auto-follow resumes, and the marker reappears.

    Driven through real SGR mouse events injected into the pty (the same
    encoding a terminal emulator sends), so it exercises the full mouse path,
    not just the internal scroll-lock seam.
    """
    tail = "ZZTAILMARKERZZ"
    body = "\n".join(
        [f"line{i:03d} filler filler filler" for i in range(60)] + [tail]
    )
    with PtySession(_argv(), rows=24, cols=80) as pty:
        assert pty.read_until("┌", _STARTUP_TIMEOUT)
        pty.send(b"md " + _paste(body) + b"\r")
        assert pty.read_until(tail, _TURN_TIMEOUT), (
            f"Long body never streamed. Output:\n{pty.raw_text}"
        )
        _settle(pty)

        def marker_visible() -> bool:
            return any(tail in line for line in pty.frame())

        assert marker_visible(), (
            "Tail marker not visible while following. Frame:\n"
            + "\n".join(pty.frame())
        )

        # Wheel up well past the divisor (10 events == 1 real line): 400 events
        # scroll ~40 lines, pushing the tail marker off the visible frame.
        for _ in range(400):
            pty.send(_WHEEL_UP)
        _settle(pty)
        assert not marker_visible(), (
            "View kept following after scroll-lock should have engaged (tail "
            "marker still visible). Frame:\n" + "\n".join(pty.frame())
        )

        # Wheel down enough to return to the bottom; auto-follow resumes and the
        # tail marker becomes visible again.
        for _ in range(700):
            pty.send(_WHEEL_DOWN)
        _settle(pty)
        assert marker_visible(), (
            "Auto-follow did not resume at the bottom (tail marker still "
            "hidden). Frame:\n" + "\n".join(pty.frame())
        )

        assert pty.quit(_QUIT_TIMEOUT)
