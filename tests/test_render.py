"""Tests for the streaming renderer (miniclaude/_render.py).

The heart of this module is the chunking-invariance property: the emitted ANSI must be
byte-identical no matter how the input token stream is split across feed calls. That
property is tested hard here (char-by-char, pairs, and several irregular chunkings over a
battery of fixtures covering every feature), alongside golden tests for the concrete
styling of headers, inline markup, lists, code fences, thinking transitions, and tables.
"""

from __future__ import annotations

import pytest

from miniclaude._render import (
    BOLD,
    CYAN,
    DIM,
    ITALIC,
    RESET,
    THINKING,
    UNDERLINE,
    StreamRenderer,
    _strip_ansi,
)

# -- helpers ---------------------------------------------------------------

# A segment is (kind, text) where kind is "text" or "thinking".
Segment = tuple[str, str]


def _feed(renderer: StreamRenderer, kind: str, chunk: str) -> str:
    return renderer.feed_text(chunk) if kind == "text" else renderer.feed_thinking(chunk)


def _run(segments: list[Segment], width: int = 80, sizes: list[int] | None = None) -> str:
    """Feed all segments, chunking each segment's text by the cycling `sizes` (None=whole)."""
    r = StreamRenderer(width)
    out: list[str] = []
    for kind, text in segments:
        if sizes is None:
            out.append(_feed(r, kind, text))
        else:
            i = 0
            k = 0
            while i < len(text):
                sz = sizes[k % len(sizes)]
                out.append(_feed(r, kind, text[i:i + sz]))
                i += sz
                k += 1
    out.append(r.finish())
    return "".join(out)


# -- fixtures covering every feature ---------------------------------------

FIXTURES: dict[str, list[Segment]] = {
    "prose_inline": [
        ("text", "# Big Title\n\nSome **bold**, *italic*, _under_, `code`, and a "
                 "[link](http://example.com) here.\n\n## Subhead\n\nplain tail\n"),
    ],
    "lists": [
        ("text", "- one\n- two\n* star\n1. first\n2. second\nnot a list\n"),
    ],
    "code_fence": [
        ("text", "before\n\n```python\n**not bold** and `not code`\nx = [a](b)\n```\n\nafter\n"),
    ],
    "table_by_prose": [
        ("text", "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 5 |\nafter table\n"),
    ],
    "table_by_finish": [
        ("text", "| Left | Center | Right |\n| :--- | :---: | ---: |\n| a | bb | ccc |"),
    ],
    "table_styled_cells": [
        ("text", "| Feature | Note |\n| --- | --- |\n| **fast** | `x` |\nend\n"),
    ],
    "thinking_only": [
        ("thinking", "considering the options\nweighing tradeoffs\n"),
    ],
    "transitions": [
        ("text", "opening line\n"),
        ("thinking", "let me think\nmore thought\n"),
        ("text", "final answer\n"),
        ("thinking", "second thought\n"),
        ("text", "done\n"),
    ],
    "unclosed": [
        ("text", "**bold and *ital and `code and [link\nnext line\n"),
    ],
    "partial_no_newline": [
        ("text", "no trailing newline here"),
    ],
    "everything": [
        ("text", "# Title\n\nintro **b** *i* `c` [l](u)\n\n- a\n- b\n\n"),
        ("thinking", "hidden reasoning\nstep two\n"),
        ("text", "## Results\n\n```sh\necho **x**\n```\n\n| K | V |\n| --- | --- |\n| 1 | 2 |\ntail\n"),
    ],
    "table_emoji": [
        ("text", "| \U0001f3b8 | hello | \U0001f98b\U0001f33b |\n"
                 "| --- | --- | --- |\n"
                 "| a | world | x |\n"),
    ],
    "table_mixed_content": [
        ("text", "| **bold** | \U0001f3b8music |\n"
                 "| --- | --- |\n"
                 "| normal | data |\n"),
    ],
}

CHUNK_STRATEGIES: list[list[int]] = [
    [1],           # char-by-char
    [2],           # pairs
    [3],           # threes
    [1, 3, 2, 5],  # irregular
    [7, 1],        # irregular
]


@pytest.mark.parametrize("name", list(FIXTURES))
@pytest.mark.parametrize("sizes", CHUNK_STRATEGIES)
def test_chunking_invariance(name: str, sizes: list[int]) -> None:
    """Output must be byte-identical whether fed whole or in any chunking."""
    segments = FIXTURES[name]
    whole = _run(segments, sizes=None)
    chunked = _run(segments, sizes=sizes)
    assert chunked == whole, f"fixture {name!r} not invariant under chunking {sizes}"


@pytest.mark.parametrize("name", list(FIXTURES))
def test_chunking_invariance_narrow_width(name: str) -> None:
    """Invariance also holds at a narrow width (exercises table truncation paths)."""
    segments = FIXTURES[name]
    whole = _run(segments, width=16, sizes=None)
    chunked = _run(segments, width=16, sizes=[1])
    assert chunked == whole


# -- inline styling goldens ------------------------------------------------


def _single(text: str, width: int = 80) -> str:
    return _run([("text", text)], width=width)


def test_header_h1_bold_underline() -> None:
    assert _single("# Title\n") == BOLD + UNDERLINE + "Title" + RESET + "\n"


def test_header_h2_bold_only() -> None:
    assert _single("## Sub\n") == BOLD + "Sub" + RESET + "\n"


def test_header_h3_bold_only() -> None:
    assert _single("### Deep\n") == BOLD + "Deep" + RESET + "\n"


def test_bold_inline() -> None:
    assert _single("a **b** c\n") == "a " + BOLD + "b" + RESET + " c\n"


def test_italic_star_and_underscore() -> None:
    assert _single("a *b* _c_ d\n") == (
        "a " + ITALIC + "b" + RESET + " " + ITALIC + "c" + RESET + " d\n"
    )


def test_inline_code_cyan() -> None:
    assert _single("run `ls` now\n") == "run " + CYAN + "ls" + RESET + " now\n"


def test_link_underline_and_dim_url() -> None:
    assert _single("see [docs](http://x) ok\n") == (
        "see " + UNDERLINE + "docs" + RESET + " (" + DIM + "http://x" + RESET + ")" + " ok\n"
    )


def test_bullet_dash_and_star() -> None:
    assert _single("- item\n") == DIM + "• " + RESET + "item\n"
    assert _single("* item\n") == DIM + "• " + RESET + "item\n"


def test_numbered_list_unchanged() -> None:
    assert _single("1. item\n") == "1. item\n"


def test_bullet_item_keeps_inline_styling() -> None:
    assert _single("- has **bold**\n") == (
        DIM + "• " + RESET + "has " + BOLD + "bold" + RESET + "\n"
    )


@pytest.mark.parametrize(
    "text",
    [
        "**bold\n",
        "*ital\n",
        "_under\n",
        "`code\n",
        "[x](y\n",
        "[x]y\n",
    ],
)
def test_unclosed_markers_render_literally(text: str) -> None:
    # No ANSI is introduced; the marker survives verbatim.
    assert _single(text) == text


# -- code fence goldens ----------------------------------------------------


def test_code_fence_markdown_not_parsed() -> None:
    out = _single("```python\n**x** `y`\n```\n")
    assert out == (
        DIM + "```python" + RESET + "\n"
        + "  **x** `y`\n"
        + DIM + "```" + RESET + "\n"
    )


def test_code_fence_body_two_space_indent() -> None:
    out = _single("```\nline1\n  line2\n```\n")
    assert out == (
        DIM + "```" + RESET + "\n"
        + "  line1\n"
        + "    line2\n"
        + DIM + "```" + RESET + "\n"
    )


# -- thinking transitions --------------------------------------------------


def test_thinking_start_emits_header() -> None:
    r = StreamRenderer(80)
    out = r.feed_thinking("first thought\n")
    assert out == (
        DIM + "✻ thinking" + RESET + "\n"
        + THINKING + "first thought" + RESET + "\n"
    )


def test_thinking_to_prose_emits_blank_line() -> None:
    r = StreamRenderer(80)
    r.feed_thinking("thinking\n")
    out = r.feed_text("prose\n")
    assert out == "\n" + "prose\n"


def test_thinking_no_markdown_parsing() -> None:
    r = StreamRenderer(80)
    out = r.feed_thinking("**not bold** `no code`\n")
    assert out == (
        DIM + "✻ thinking" + RESET + "\n"
        + THINKING + "**not bold** `no code`" + RESET + "\n"
    )


def test_finish_flushes_partial_line() -> None:
    r = StreamRenderer(80)
    assert r.feed_text("partial") == ""
    assert r.finish() == "partial\n"


# -- table goldens (compared on the visible layout) ------------------------


def test_table_basic_alignment() -> None:
    out = _single("| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 5 |\n")
    plain = _strip_ansi(out)
    assert plain == (
        "| Name  | Age |\n"
        "|───────|─────|\n"
        "| Alice | 30  |\n"
        "| Bob   | 5   |\n"
    )
    # Header cells are bold.
    assert BOLD + "Name" + RESET in out


def test_table_column_alignment_colons() -> None:
    out = _single("| Left | Center | Right |\n| :--- | :---: | ---: |\n| a | bb | ccc |\n")
    plain = _strip_ansi(out)
    assert plain == (
        "| Left | Center | Right |\n"
        "|──────|────────|───────|\n"
        "| a    |   bb   |   ccc |\n"
    )


def test_table_terminated_by_prose_then_processes_line() -> None:
    out = _single("| A | B |\n| --- | --- |\n| 1 | 2 |\nafter table\n")
    plain = _strip_ansi(out)
    assert plain.endswith("after table\n")
    assert "| A " in plain
    # The prose line is rendered after the table's box-drawing separator.
    assert plain.index("─") < plain.index("after table")


def test_table_terminated_by_finish() -> None:
    r = StreamRenderer(80)
    acc = r.feed_text("| A | B |\n| --- | --- |\n| 1 | 2 |")
    acc += r.finish()
    plain = _strip_ansi(acc)
    assert plain == (
        "| A | B |\n"
        "|───|───|\n"
        "| 1 | 2 |\n"
    )


def test_table_inline_styles_in_cells() -> None:
    out = _single("| Feature | Note |\n| --- | --- |\n| **fast** | `x` |\n")
    plain = _strip_ansi(out)
    # Display width ignores the markdown markers (bold -> "fast", code -> "x").
    assert plain == (
        "| Feature | Note |\n"
        "|─────────|──────|\n"
        "| fast    | x    |\n"
    )
    # And the styling is actually applied inside the cells.
    assert BOLD + "fast" + RESET in out
    assert CYAN + "x" + RESET in out


def test_table_truncates_widest_cells_to_fit_width() -> None:
    out = _single(
        "| Column One | Column Two |\n| --- | --- |\n| aaaa | bbbb |\n",
        width=20,
    )
    plain = _strip_ansi(out)
    assert plain == (
        "| Colum… | Column… |\n"
        "|────────|─────────|\n"
        "| aaaa   | bbbb    |\n"
    )
    # Every rendered row fits within the configured width.
    for line in plain.splitlines():
        assert len(line) <= 20


def test_table_header_bold_body_not() -> None:
    out = _single("| H |\n| --- |\n| b |\n")
    assert BOLD + "H" + RESET in out
    # Body cell has no bold wrapper.
    assert BOLD + "b" + RESET not in out


# -- blank lines and structure ---------------------------------------------


def test_blank_lines_preserved() -> None:
    assert _single("a\n\nb\n") == "a\n\nb\n"


# -- display-width counting (emoji / wide chars) ----------------------------


from miniclaude._render import _visible_len, _truncate_visible


def test__visible_len_ascii() -> None:
    assert _visible_len("hello") == 5


def test__visible_len_emoji() -> None:
    # Guitar emoji is East Asian Width = W, so 2 cells; "ab" = 2 cells => total 4.
    assert _visible_len("\U0001f3b8ab") == 4


def test__visible_len_ansi_stripped() -> None:
    # ANSI escapes wrapping an emoji must not affect the width count.
    styled = BOLD + "\U0001f3b8" + RESET + "x"
    assert _visible_len(styled) == 3  # emoji=2, x=1


def test__truncate_visible_emoji() -> None:
    # String: emoji (2 cols) + "ab" (2 cols) = 4 cols total.
    # Truncate to maxcols=3: only room for 2 visible cols before the ellipsis.
    # The emoji takes 2 cols, which equals maxcols-1=2, so it fits; then ellipsis.
    result = _truncate_visible("\U0001f3b8ab", 3)
    plain = _strip_ansi(result)
    assert plain == "\U0001f3b8…"
    # Truncate to maxcols=2: only 1 col before the ellipsis; emoji needs 2, so it
    # does not fit. Result is just the ellipsis.
    result2 = _truncate_visible("\U0001f3b8ab", 2)
    plain2 = _strip_ansi(result2)
    assert plain2 == "…"


# -- table rendering with emoji / wide chars --------------------------------


def test_table_emoji_alignment() -> None:
    """Emoji cells (width=2) must produce correctly-aligned pipes and padding.

    At width=16, columns shrink to target=2 each.  Truncating the emoji cell
    "\U0001f98b\U0001f33b" (4 visible cols) to target=2 produces just an ellipsis
    (1 col, because the 2-col emoji cannot fit in the 1-col budget before the
    ellipsis).  The renderer must pad the remaining 1 col so pipes stay aligned.
    """
    md = (
        "| \U0001f3b8 | hello | \U0001f98b\U0001f33b |\n"
        "| --- | --- | --- |\n"
        "| a | world | x |\n"
    )
    out = _single(md, width=16)
    plain = _strip_ansi(out)
    # Every row must be exactly 16 visible columns (pipes align).
    for line in plain.splitlines():
        assert _visible_len(line) == 16, f"misaligned: {line!r}"
    # Exact golden plain text.
    assert plain == (
        "| \U0001f3b8 | h… | …  |\n"
        "|────|────|────|\n"
        "| a  | w… | x  |\n"
    )
    # Header cells are bold.
    assert BOLD in out


def test_table_wide_truncation() -> None:
    """20-column table that exceeds 80 cols: cells truncated, pipes aligned.

    At width=82 the fitting algorithm leaves 19 columns at target=1 and one
    column at target=2.  The last column holds emoji content ("\U0001f3b8\U0001f3b8",
    4 visible cols) which, when truncated to target=2, produces a 1-col ellipsis
    (the 2-col emoji cannot fit in the 1-col budget before the ellipsis).  The
    renderer must pad the shortfall so all three rows are the same width.
    """
    headers = [chr(ord("A") + i) for i in range(20)]
    body_cells = ["ab"] * 19 + ["\U0001f3b8\U0001f3b8"]
    md = (
        "| " + " | ".join(headers) + " |\n"
        "| " + " | ".join(["-"] * 20) + " |\n"
        "| " + " | ".join(body_cells) + " |\n"
    )
    out = _single(md, width=82)
    plain = _strip_ansi(out)
    lines = plain.splitlines()
    assert len(lines) == 3
    # All rows must have the same visible width (pipes align).
    widths = [_visible_len(line) for line in lines]
    assert widths[0] == widths[1] == widths[2] == 82, f"misaligned: {widths}"
    # Body cells for columns 0-18 are truncated (just an ellipsis at target=1).
    # Column 19 body cell is truncated emoji, padded to target=2.
    expected_header = "| " + " | ".join(list("ABCDEFGHIJKLMNOPQRS") + ["T "]) + " |"
    expected_sep = "|" + "|".join(["───"] * 19 + ["────"]) + "|"
    expected_body = "| " + " | ".join(["…"] * 19 + ["… "]) + " |"
    assert lines[0] == expected_header
    assert lines[1] == expected_sep
    assert lines[2] == expected_body


def test_table_mixed_content() -> None:
    """Bold markdown + emoji in cells: alignment accounts for zero-width markers.

    "**bold**" renders as styled "bold" (4 visible cols, not 8 for the raw
    markdown).  At width=11 both columns shrink to target=2.  The emoji cell
    "\U0001f3b8music" truncates to a 1-col ellipsis (the 2-col emoji cannot fit
    in 1 col before the ellipsis), requiring 1 col of padding.
    """
    md = (
        "| **bold** | \U0001f3b8music |\n"
        "| --- | --- |\n"
        "| normal | data |\n"
    )
    out = _single(md, width=11)
    plain = _strip_ansi(out)
    # Every row must be exactly 11 visible columns.
    for line in plain.splitlines():
        assert _visible_len(line) == 11, f"misaligned: {line!r}"
    # Exact golden plain text.
    assert plain == (
        "| b… | …  |\n"
        "|────|────|\n"
        "| n… | d… |\n"
    )
    # Bold markers (**) are consumed, not visible.
    assert "**" not in plain
    # Bold styling is applied in the output.
    assert BOLD in out
