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
                self._in_code = False
                return DIM + line + RESET + "\n"
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
            return indent + DIM + "• " + RESET + self._style_inline(rest) + "\n"
        return self._style_inline(line) + "\n"

    def _style_inline(self, s: str) -> str:
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

    # -- table rendering -------------------------------------------------

    def _flush_table(self) -> str:
        if not self._table:
            return ""
        rows_raw = self._table
        self._table = []
        parsed = [self._split_row(l) for l in rows_raw]

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
                    aligns[j] = self._parse_align(c)

        if sep_idx is not None:
            header_rows = parsed[:sep_idx]
            body_rows = parsed[sep_idx + 1:]
        else:
            header_rows = parsed[:1]
            body_rows = parsed[1:]

        def make_cell(text: str, is_header: bool) -> tuple[str, int]:
            styled = self._style_inline(text)
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

        header_grid = build(header_rows, True)
        body_grid = build(body_rows, False)

        col_w = [0] * num_cols
        for grid in (header_grid, body_grid):
            for cells in grid:
                for j, (_s, w) in enumerate(cells):
                    col_w[j] = max(col_w[j], w)
        col_w = self._fit_widths(col_w)

        lines: list[str] = []
        for cells in header_grid:
            lines.append(self._render_table_row(cells, col_w, aligns))
        if header_grid:
            lines.append(self._render_separator(col_w))
        for cells in body_grid:
            lines.append(self._render_table_row(cells, col_w, aligns))
        return "".join(lines)

    def _split_row(self, line: str) -> list[str]:
        s = line.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        return [c.strip() for c in s.split("|")]

    def _parse_align(self, cell: str) -> str:
        c = cell.strip()
        left = c.startswith(":")
        right = c.endswith(":")
        if left and right:
            return "center"
        if right:
            return "right"
        return "left"

    def _fit_widths(self, col_w: list[int]) -> list[int]:
        num_cols = len(col_w)

        def total(cw: list[int]) -> int:
            return sum(cw) + 3 * num_cols + 1

        cw = list(col_w)
        while total(cw) > self.width:
            j = max(range(num_cols), key=lambda k: cw[k])
            if cw[j] <= 1:
                break
            cw[j] -= 1
        return cw

    def _render_table_row(
        self,
        cells: list[tuple[str, int]],
        col_w: list[int],
        aligns: list[str],
    ) -> str:
        parts = [self._pad(styled, w, col_w[j], aligns[j]) for j, (styled, w) in enumerate(cells)]
        return "| " + " | ".join(parts) + " |\n"

    def _render_separator(self, col_w: list[int]) -> str:
        return "|" + "|".join("─" * (w + 2) for w in col_w) + "|\n"

    def _pad(self, styled: str, w: int, target: int, align: str) -> str:
        if w > target:
            styled = _truncate_visible(styled, target)
            w = target
        pad = target - w
        if align == "right":
            return " " * pad + styled
        if align == "center":
            lp = pad // 2
            return " " * lp + styled + " " * (pad - lp)
        return styled + " " * pad
