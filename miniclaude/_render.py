"""Streaming markdown renderer for assistant prose and thinking (pure, no I/O).

`StreamRenderer` converts an incoming token stream into ANSI-styled terminal output.
Every feed method returns the ANSI string to print (possibly empty). The renderer is
line-grain: it holds the partial trailing line and only emits complete lines, so the
output is byte-identical regardless of how the input stream is chunked (the central
invariant, enforced by tests). Tables are the one buffered-per-block exception so they
are never emitted mangled.

No terminal, no session, no side effects -- unit-testable in isolation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

# SGR codes.
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
CYAN = "\033[36m"
RESET = "\033[0m"
# Thinking content: dim + bright-black (gray).
THINKING = "\033[2;90m"

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")
_HEADER_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*][ \t]+(.*)$")
_SEP_CELL_RE = re.compile(r"^:?-+:?$")


@dataclass(frozen=True)
class TableData:
    """Raw markdown cell text for a table (not styled). Styling happens at render time."""

    header_rows: list[list[str]]
    body_rows: list[list[str]]
    aligns: list[str] = field(default_factory=list)


def _strip_ansi(s: str) -> str:
    """Return the visible text of an ANSI-styled string (for display-width math)."""
    return _ANSI_RE.sub("", s)


def _char_width(c: str) -> int:
    """Display width of a single character: 2 for Wide/Fullwidth, 1 otherwise."""
    eaw = unicodedata.east_asian_width(c)
    return 2 if eaw in ("W", "F") else 1


def _visible_len(styled: str) -> int:
    """Number of visible columns in an ANSI-styled string (wide chars = 2 cells)."""
    return sum(_char_width(c) for c in _strip_ansi(styled))


def _truncate_visible(styled: str, maxcols: int) -> str:
    """Truncate an ANSI-styled string to `maxcols` visible columns, ending with an ellipsis.

    ANSI escape sequences are preserved (never counted, never split); the result keeps
    visible chars that fit within maxcols-1 columns plus a trailing ellipsis, then a RESET.
    Characters that would cause the column count to exceed maxcols-1 are dropped.
    """
    if maxcols <= 0:
        return ""
    out: list[str] = []
    visible = 0
    i = 0
    n = len(styled)
    while i < n:
        m = _ANSI_RE.match(styled, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        cw = _char_width(styled[i])
        if visible + cw > maxcols - 1:
            break
        out.append(styled[i])
        visible += cw
        i += 1
    out.append("…")  # …
    out.append(RESET)
    return "".join(out)


# -- module-level helpers (pure, no instance state) --------------------------


def _style_inline(s: str) -> str:
    """Apply inline markdown styling; unclosed markers stay literal."""
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        # Inline code `...` (highest precedence: content is not re-parsed).
        if c == "`":
            end = s.find("`", i + 1)
            if end != -1:
                out.append(CYAN + s[i + 1:end] + RESET)
                i = end + 1
                continue
            out.append(c)
            i += 1
            continue
        # Link [text](url).
        if c == "[":
            close = s.find("]", i + 1)
            if close != -1 and close + 1 < n and s[close + 1] == "(":
                urlend = s.find(")", close + 2)
                if urlend != -1:
                    text = s[i + 1:close]
                    url = s[close + 2:urlend]
                    out.append(
                        UNDERLINE + text + RESET + " (" + DIM + url + RESET + ")"
                    )
                    i = urlend + 1
                    continue
            out.append(c)
            i += 1
            continue
        # Bold **...** (checked before single-* italic).
        if s.startswith("**", i):
            end = s.find("**", i + 2)
            if end != -1:
                out.append(BOLD + s[i + 2:end] + RESET)
                i = end + 2
                continue
            out.append(c)
            i += 1
            continue
        # Italic *...* or _..._.
        if c in "*_":
            end = s.find(c, i + 1)
            if end != -1 and end > i + 1:
                out.append(ITALIC + s[i + 1:end] + RESET)
                i = end + 1
                continue
            out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _split_row(line: str) -> list[str]:
    """Split a markdown table row into cell texts."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _parse_align(cell: str) -> str:
    """Determine column alignment from a separator cell (e.g. ':---:', '---:')."""
    c = cell.strip()
    left = c.startswith(":")
    right = c.endswith(":")
    if left and right:
        return "center"
    if right:
        return "right"
    return "left"


def _longest_word_width(styled: str) -> int:
    """Visible width of the longest unbreakable word/segment in styled text.

    Splits at spaces and hyphens (break-after: hyphen stays in the first segment).
    """
    plain = _strip_ansi(styled)
    if not plain:
        return 0
    max_w = 0
    cur_w = 0
    for c in plain:
        if c == " ":
            max_w = max(max_w, cur_w)
            cur_w = 0
        elif c == "-":
            cur_w += _char_width(c)
            max_w = max(max_w, cur_w)
            cur_w = 0
        else:
            cur_w += _char_width(c)
    return max(max_w, cur_w)


def _parse_sgr_params(escape: str) -> list[int]:
    """Parse parameter codes from an SGR escape sequence.

    ``\\x1b[2;90m`` -> ``[2, 90]``.  ``\\x1b[m`` -> ``[0]``.
    """
    inner = escape[2:-1]  # strip \\x1b[ and m
    if not inner:
        return [0]
    parts: list[int] = []
    for p in inner.split(";"):
        if p.isdigit():
            parts.append(int(p))
    return parts or [0]


def _update_sgr(state: set[int], escape: str) -> None:
    """Update active SGR state with a new escape sequence."""
    for p in _parse_sgr_params(escape):
        if p == 0:
            state.clear()
        elif p in (1, 2, 3, 4):
            state.add(p)
        elif p == 22:
            state.discard(1)
            state.discard(2)
        elif p == 23:
            state.discard(3)
        elif p == 24:
            state.discard(4)
        elif 30 <= p <= 37 or 90 <= p <= 97:
            state -= {c for c in state if 30 <= c <= 37 or 90 <= c <= 97}
            state.add(p)
        elif p == 39:
            state -= {c for c in state if 30 <= c <= 37 or 90 <= c <= 97}
        elif 40 <= p <= 47 or 100 <= p <= 107:
            state -= {c for c in state if 40 <= c <= 47 or 100 <= c <= 107}
            state.add(p)
        elif p == 49:
            state -= {c for c in state if 40 <= c <= 47 or 100 <= c <= 107}


def _reconstruct_sgr(state: set[int]) -> str:
    """Reconstruct a single SGR escape from the active state set."""
    if not state:
        return ""
    return "\x1b[" + ";".join(str(c) for c in sorted(state)) + "m"


def _wrap_cell(text: str, width: int) -> list[str]:
    """Wrap styled text into sub-lines of at most ``width`` visible columns.

    Breaks at word boundaries (spaces, hyphens) when possible; hard-breaks
    mid-word only when a single word exceeds the column width.  Tracks ANSI
    SGR state across breaks: each sub-line ends with a reset and the next
    starts with the re-emitted active SGR.
    """
    if width <= 0:
        return [""]
    if not text:
        return [""]

    # Tokenize: ANSI escapes and individual characters.
    Token = tuple  # ('ansi', str) | ('char', str, int)
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        m = _ANSI_RE.match(text, i)
        if m:
            tokens.append(("ansi", m.group(0)))
            i = m.end()
        else:
            tokens.append(("char", text[i], _char_width(text[i])))
            i += 1

    lines: list[str] = []
    cur: list[Token] = []
    cur_w = 0
    brk = -1        # index in cur AFTER last break-point char
    brk_sgr: set[int] = set()
    sgr: set[int] = set()

    def _emit(parts: list[Token], state: set[int]) -> str:
        s = "".join(p[1] for p in parts)
        if state:
            s += RESET
        return s

    def _sgr_prefix(state: set[int]) -> list[Token]:
        seq = _reconstruct_sgr(state)
        return [("ansi", seq)] if seq else []

    for tok in tokens:
        if tok[0] == "ansi":
            _update_sgr(sgr, tok[1])
            cur.append(tok)
            continue

        ch: str = tok[1]
        cw: int = tok[2]

        # If char overflows, break the line first.
        if cur_w + cw > width:
            if ch == " ":
                # Space at overflow point: emit current line, consume space.
                lines.append(_emit(cur, sgr))
                cur = _sgr_prefix(sgr)
                cur_w = 0
                brk = -1
                continue
            if brk >= 0:
                # Word-boundary break.
                before = cur[:brk]
                after = cur[brk:]
                lines.append(_emit(before, brk_sgr))
                after_w = sum(t[2] for t in after if t[0] == "char")
                cur = _sgr_prefix(brk_sgr) + after
                cur_w = after_w
                brk = -1
            elif cur_w > 0:
                # Hard break (only when cur has visible chars; ANSI-only
                # content has cur_w == 0 and must not be emitted as a line).
                lines.append(_emit(cur, sgr))
                cur = _sgr_prefix(sgr)
                cur_w = 0
                brk = -1

        # After the break, if still overflows (long leftover + current char),
        # hard-break again.
        if cur_w + cw > width and cur_w > 0:
            lines.append(_emit(cur, sgr))
            cur = _sgr_prefix(sgr)
            cur_w = 0
            brk = -1

        # If char doesn't fit on an empty line, add it anyway to avoid infinite loop
        # (happens when cw > width, e.g. wide char on a 1-col column).

        cur.append(tok)
        cur_w += cw

        if ch == " " or ch == "-":
            brk = len(cur)  # break AFTER this char
            brk_sgr = set(sgr)

    # Emit remaining.
    if cur or not lines:
        s = "".join(p[1] for p in cur)
        if sgr:
            s += RESET
        lines.append(s)

    return lines


def _fit_widths(
    natural_w: list[int], width: int, floors: list[int] | None = None
) -> list[int]:
    """Constraint-based column width allocation.

    Three tiers:
    1. Natural widths fit -- use them.
    2. Floors fit -- give each its floor, distribute remaining proportionally.
    3. Extreme -- proportional to floor.
    """
    num_cols = len(natural_w)
    overhead = 3 * num_cols + 1
    available = width - overhead

    if floors is None:
        floors = [1] * num_cols

    if available <= 0:
        return [max(f, 1) for f in floors]

    natural_total = sum(natural_w)

    # Tier 1: everything fits at natural width.
    if natural_total <= available:
        return list(natural_w)

    floor_total = sum(floors)

    # Tier 2: floors fit, distribute remaining proportionally to excess.
    if floor_total <= available:
        remaining = available - floor_total
        excess = [max(0, natural_w[j] - floors[j]) for j in range(num_cols)]
        total_excess = sum(excess)
        result = list(floors)
        if total_excess > 0:
            for j in range(num_cols):
                result[j] += remaining * excess[j] // total_excess
            allocated = sum(result) - floor_total
            leftover = remaining - allocated
            indices = sorted(range(num_cols), key=lambda j: excess[j], reverse=True)
            for k in range(min(leftover, len(indices))):
                result[indices[k]] += 1
        return result

    # Tier 3: extreme narrowing -- proportional to floor.
    if floor_total > 0:
        result = [max(1, available * floors[j] // floor_total) for j in range(num_cols)]
        allocated = sum(result)
        leftover = available - allocated
        indices = sorted(range(num_cols), key=lambda j: floors[j], reverse=True)
        for k in range(max(0, min(leftover, len(indices)))):
            result[indices[k]] += 1
        return result

    return [1] * num_cols


def _pad(styled: str, visible_width: int, target_width: int, align: str) -> str:
    """Pad (or truncate) a styled cell to ``target_width`` visible columns."""
    if visible_width > target_width:
        styled = _truncate_visible(styled, target_width)
        # Measure actual visible width after truncation; wide chars may
        # create a gap (e.g. a 2-col emoji that doesn't fit before the
        # ellipsis leaves the result shorter than target).
        visible_width = _visible_len(styled)
    pad = target_width - visible_width
    if align == "right":
        return " " * pad + styled
    if align == "center":
        lp = pad // 2
        return " " * lp + styled + " " * (pad - lp)
    return styled + " " * pad


def _render_top_border(col_w: list[int]) -> str:
    """Render top border: ``┌───┬───┐``."""
    return "┌" + "┬".join("─" * (w + 2) for w in col_w) + "┐\n"


def _render_bottom_border(col_w: list[int]) -> str:
    """Render bottom border: ``└───┴───┘``."""
    return "└" + "┴".join("─" * (w + 2) for w in col_w) + "┘\n"


def _render_separator(col_w: list[int]) -> str:
    """Render light between-row separator: ``├───┼───┤``."""
    return "├" + "┼".join("─" * (w + 2) for w in col_w) + "┤\n"


def _render_header_separator(col_w: list[int]) -> str:
    """Render heavy header/body separator: ``╞═══╪═══╡``."""
    return "╞" + "╪".join("═" * (w + 2) for w in col_w) + "╡\n"


def _render_wrapped_row(
    cells: list[tuple[str, int]],
    col_w: list[int],
    aligns: list[str],
) -> str:
    """Render one table row with cell wrapping, using box-drawing ``│`` separators.

    Each cell is wrapped into sub-lines.  All sub-lines for the row are emitted
    with ``│`` column separators aligned.  Shorter cells are padded with spaces.
    """
    num_cols = len(col_w)

    # Wrap each cell into sub-lines.
    wrapped: list[list[str]] = []
    for j, (styled, _) in enumerate(cells):
        wrapped.append(_wrap_cell(styled, col_w[j]))

    # Pad to num_cols if fewer cells than columns.
    while len(wrapped) < num_cols:
        wrapped.append([""])

    row_height = max(len(w) for w in wrapped) if wrapped else 1

    out: list[str] = []
    for i in range(row_height):
        parts: list[str] = []
        for j in range(num_cols):
            if i < len(wrapped[j]):
                subline = wrapped[j][i]
                vis_w = _visible_len(subline)
            else:
                subline = ""
                vis_w = 0
            parts.append(_pad(subline, vis_w, col_w[j], aligns[j]))
        out.append("│ " + " │ ".join(parts) + " │\n")

    return "".join(out)


# -- render_table: single rendering codepath for tables ----------------------


def render_table(data: TableData, width: int) -> str:
    """Render a TableData into an ANSI-styled box-drawing table at the given width.

    This is the SINGLE rendering codepath for tables, used by:
    - _flush_table (streaming path, when no callback)
    - The on_table callback in _PromptController (streaming path, with callback)
    - materialize_blocks (resize path)
    """
    num_cols = max(
        (max(len(r) for r in data.header_rows) if data.header_rows else 0),
        (max(len(r) for r in data.body_rows) if data.body_rows else 0),
        1,
    )
    aligns = list(data.aligns) if data.aligns else ["left"] * num_cols
    while len(aligns) < num_cols:
        aligns.append("left")

    def make_cell(text: str, is_header: bool) -> tuple[str, int]:
        styled = _style_inline(text)
        if is_header:
            styled = BOLD + styled + RESET
        return styled, _visible_len(styled)

    def build(rows: list[list[str]], is_header: bool) -> list[list[tuple[str, int]]]:
        grid: list[list[tuple[str, int]]] = []
        for r in rows:
            grid.append(
                [make_cell(r[j] if j < len(r) else "", is_header) for j in range(num_cols)]
            )
        return grid

    header_grid = build(data.header_rows, True)
    body_grid = build(data.body_rows, False)

    # Column width solver: compute natural widths, header widths, word widths.
    natural_w = [0] * num_cols
    header_w = [0] * num_cols
    word_w = [0] * num_cols

    for cells in header_grid:
        for j, (styled, vis_w) in enumerate(cells):
            natural_w[j] = max(natural_w[j], vis_w)
            header_w[j] = max(header_w[j], vis_w)
            word_w[j] = max(word_w[j], _longest_word_width(styled))

    for cells in body_grid:
        for j, (styled, vis_w) in enumerate(cells):
            natural_w[j] = max(natural_w[j], vis_w)
            word_w[j] = max(word_w[j], _longest_word_width(styled))

    floors = [max(header_w[j], word_w[j], 1) for j in range(num_cols)]
    col_w = _fit_widths(natural_w, width, floors)

    # Render with box-drawing borders and cell wrapping.
    lines: list[str] = []
    lines.append(_render_top_border(col_w))
    for cells in header_grid:
        lines.append(_render_wrapped_row(cells, col_w, aligns))
    if header_grid:
        lines.append(_render_header_separator(col_w))
    for i, cells in enumerate(body_grid):
        if i > 0:
            # Light rule between adjacent logical body rows (never between the
            # wrapped sub-lines of a single logical row).
            lines.append(_render_separator(col_w))
        lines.append(_render_wrapped_row(cells, col_w, aligns))
    lines.append(_render_bottom_border(col_w))
    return "".join(lines)


class StreamRenderer:
    """Line-grain streaming renderer for assistant prose and thinking blocks."""

    def __init__(self, width: int) -> None:
        self.width = width
        # Current mode: None (nothing yet), "text", or "thinking".
        self._mode: str | None = None
        # Held partial (trailing, newline-less) content for the current mode.
        self._buf = ""
        # Whether we are inside a fenced code block (text mode only).
        self._in_code = False
        # Buffered table rows (raw line strings), text mode only.
        self._table: list[str] = []
        # Optional callback for table output (set by Repl when _PromptController
        # is active). When set, _flush_table fires it with a TableData and
        # returns "". When not set, _flush_table renders the table directly.
        self.on_table: Callable[[TableData], None] | None = None

    # -- public feed API -------------------------------------------------

    def feed_text(self, chunk: str) -> str:
        """Feed a chunk of assistant prose (from a text_delta)."""
        if chunk == "":
            return ""
        out = ""
        if self._mode != "text":
            out += self._switch_to_text()
        self._buf += chunk
        out += self._drain()
        return out

    def feed_thinking(self, chunk: str) -> str:
        """Feed a chunk of thinking content (from a thinking_delta)."""
        if chunk == "":
            return ""
        out = ""
        if self._mode != "thinking":
            out += self._switch_to_thinking()
        self._buf += chunk
        out += self._drain()
        return out

    def finish(self) -> str:
        """Flush any held partial line, open table, or open code fence."""
        out = ""
        if self._buf:
            line = self._buf
            self._buf = ""
            if self._mode == "thinking":
                out += self._render_thinking_line(line)
            else:
                out += self._process_text_line(line)
        out += self._flush_table()
        return out

    # -- mode transitions ------------------------------------------------

    def _switch_to_text(self) -> str:
        out = ""
        if self._mode == "thinking":
            if self._buf:
                out += self._render_thinking_line(self._buf)
                self._buf = ""
            # Transition back to prose emits a blank line.
            out += "\n"
        self._mode = "text"
        return out

    def _switch_to_thinking(self) -> str:
        out = ""
        if self._mode == "text":
            if self._buf:
                out += self._process_text_line(self._buf)
                self._buf = ""
            out += self._flush_table()
        out += DIM + "✻ thinking" + RESET + "\n"
        self._mode = "thinking"
        return out

    # -- line draining ---------------------------------------------------

    def _drain(self) -> str:
        """Emit every complete line currently in the buffer, keeping the partial tail."""
        out: list[str] = []
        while True:
            idx = self._buf.find("\n")
            if idx < 0:
                break
            line = self._buf[:idx]
            self._buf = self._buf[idx + 1:]
            if self._mode == "thinking":
                out.append(self._render_thinking_line(line))
            else:
                out.append(self._process_text_line(line))
        return "".join(out)

    # -- thinking rendering ----------------------------------------------

    def _render_thinking_line(self, line: str) -> str:
        # No markdown parsing; whole line dim gray.
        return THINKING + line + RESET + "\n"

    # -- text rendering --------------------------------------------------

    def _process_text_line(self, line: str) -> str:
        stripped = line.lstrip()
        if self._in_code:
            if stripped.startswith("```"):
                # Fence close: flush any table buffered inside the code block,
                # then render the closing fence itself.
                out = self._flush_table()
                self._in_code = False
                return out + DIM + line + RESET + "\n"
            # Table rows inside code fences are rendered as aligned ASCII
            # tables (LLMs frequently wrap markdown tables in ``` fences).
            if stripped.startswith("|"):
                self._table.append(line)
                return ""
            # Code body: two-space indent, no markdown, no recolor.
            return "  " + line + "\n"
        if stripped.startswith("```"):
            out = self._flush_table()
            self._in_code = True
            return out + DIM + line + RESET + "\n"
        if stripped.startswith("|"):
            self._table.append(line)
            return ""
        # Ordinary prose line: any open table ends here first.
        out = self._flush_table()
        out += self._render_prose_line(line)
        return out

    def _render_prose_line(self, line: str) -> str:
        m = _HEADER_RE.match(line)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            if level == 1:
                return BOLD + UNDERLINE + text + RESET + "\n"
            return BOLD + text + RESET + "\n"
        m = _BULLET_RE.match(line)
        if m:
            indent, rest = m.group(1), m.group(2)
            return indent + DIM + "• " + RESET + _style_inline(rest) + "\n"
        return _style_inline(line) + "\n"

    # -- table rendering -------------------------------------------------

    def _flush_table(self) -> str:
        if not self._table:
            return ""
        rows_raw = self._table
        self._table = []
        parsed = [_split_row(line) for line in rows_raw]

        sep_idx: int | None = None
        for i, cells in enumerate(parsed):
            if cells and all(_SEP_CELL_RE.match(c) for c in cells):
                sep_idx = i
                break

        num_cols = max(len(c) for c in parsed)

        aligns = ["left"] * num_cols
        if sep_idx is not None:
            for j, c in enumerate(parsed[sep_idx]):
                if j < num_cols:
                    aligns[j] = _parse_align(c)

        if sep_idx is not None:
            header_rows = parsed[:sep_idx]
            body_rows = parsed[sep_idx + 1:]
        else:
            header_rows = parsed[:1]
            body_rows = parsed[1:]

        data = TableData(header_rows=header_rows, body_rows=body_rows, aligns=aligns)

        if self.on_table is not None:
            self.on_table(data)
            return ""

        return render_table(data, self.width)
