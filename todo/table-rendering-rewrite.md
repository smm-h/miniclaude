# Table rendering rewrite: box-drawing tables with resize support

The centerpiece feature of miniclaude. Tables must render with full box-drawing
characters, wrap cell content instead of truncating, and re-render at the correct
width when the terminal is resized.

## Architecture

Output is modeled as an ordered list of typed blocks: `ProseBlock` (pre-rendered
ANSI text) and `TableBlock` (structured cell data that can be rendered at any
width). On resize, the visible area is cleared and reprinted from the block list,
with tables re-rendered at the new width. Scrollback above the visible area stays
at its original width (inherent to native terminal scrollback).

The application uses prompt_toolkit in inline mode (no alternate screen). Output
flows into native terminal scrollback via `patch_stdout(raw=True)`. Scrolling is
the terminal's native smooth scroll. The screen is cleared on launch (prior
terminal content is not shown).

## Phase 0 -- Structured output model

### 0a. Data types and printer split

Define in `_render.py` (or a new `_types.py` if cleaner):

- `TableData`: raw markdown cell text (not styled), header_rows (`list[list[str]]`),
  body_rows (`list[list[str]]`), aligns (`list[str]`). Styling (`_style_inline`)
  happens at render time, not at parse time, so the same data can be re-styled at
  different widths when wrapping breaks fall at different positions.
- `OutputBlock`: union type -- either `ProseBlock(ansi_text: str)` or
  `TableBlock(data: TableData)`.

In `_PromptController` (`_repl.py`):

- Add `_output_blocks: list[OutputBlock]`.
- Split the printer into two functions:
  - `_raw_write(text: str)`: just `sys.stdout.write(text)` under `patch_stdout`.
    No block tracking. Used by the `on_table` callback for table display.
  - `printer(text: str)`: calls `_raw_write(text)`, then appends
    `ProseBlock(text)` to `_output_blocks`, then increments
    `_output_newline_count`.

This split ensures tables never create a ProseBlock (the `on_table` callback uses
`_raw_write`, not `printer`), eliminating double-counting.

Verification: all existing tests pass unchanged (tests use `Repl` directly, not
`_PromptController`, so the printer split is invisible to them).

### 0b. Table data extraction and single render path

Refactor `_flush_table` in `_render.py`:

1. Parse raw rows, build `TableData` (header_rows, body_rows, aligns) -- these
   are already computed as local variables in the current `_flush_table`.
2. If `self.on_table` callback is set, fire it with the `TableData`, then return
   `""` (empty string -- the callback handles display).
3. If no callback, call `render_table(data, self.width)` and return the result.

Extract `render_table(data: TableData, width: int) -> str` as a standalone
module-level function. This is the SINGLE rendering codepath for tables, used by:
- `_flush_table` (streaming path, when no callback)
- The `on_table` callback in `_PromptController` (streaming path, with callback)
- The materialization function (resize path, Phase 3)

The `on_table` callback (wired in `Repl._run_turn` each time a new
`StreamRenderer` is created) does three things:
1. Appends `TableBlock(data)` to `_output_blocks`.
2. Calls `_raw_write(render_table(data, self._width))` for immediate display.
3. Increments `_output_newline_count` by the rendered table's line count.

Wire the callback in `_run_turn` at the point where `StreamRenderer` is created
(currently line 357). The callback closure captures `self` (the Repl) for access
to `_input._output_blocks`, `_input._raw_write`, `_width`, and
`_input._output_newline_count`. Tables flushed by `finish()` in the `finally`
block of `_run_turn` also go through this callback because the StreamRenderer
(and its callback) is still alive at that point.

In tests, `self._input` is None and no callback is wired, so `_flush_table`
returns the rendered string directly. Tests remain unaffected.

Verification: output is visually identical at the original width. All tests pass.

### 0c. Materialization function

Add a function (on `_PromptController` or standalone):

```
def materialize_blocks(blocks: list[OutputBlock], width: int) -> str
```

Walks `_output_blocks`: ProseBlocks contribute their `ansi_text` as-is,
TableBlocks call `render_table(data, width)`. Returns the concatenated string.

This is used by the SIGWINCH repaint (Phase 3) to reprint the visible area at the
new width. Width is passed explicitly (the new terminal width).

Verification: `materialize_blocks(blocks, original_width)` produces identical
output to what was originally printed.

## Phase 1 -- Full box-drawing tables with cell wrapping

### 1a. Box-drawing character upgrade

In `render_table` (the single rendering function from 0b):

- Replace `|` (ASCII pipe U+007C) with `│` (U+2502 BOX DRAWINGS LIGHT VERTICAL)
  for column separators.
- Add `_render_top_border(col_widths) -> str`: `┌` + `─`*w + `┬` between
  columns + `┐` + newline. Uses U+250C, U+2500, U+252C, U+2510.
- Add `_render_bottom_border(col_widths) -> str`: `└` + `─`*w + `┴` between
  columns + `┘` + newline. Uses U+2514, U+2500, U+2534, U+2518.
- Change `_render_separator` from `|───|` to `├─┼─┤`. Uses U+251C, U+2500,
  U+253C, U+2524.
- Column width calculations must account for the border characters: each column
  contributes `│ ` + content + ` ` (3 chars overhead per column) plus a trailing
  `│` (1 char). Total: `sum(col_widths) + 3 * num_cols + 1`.

### 1b. Cell wrapping

Replace cell truncation (`_truncate_visible` + `…`) with cell wrapping.

`_wrap_cell(text: str, width: int) -> list[str]`:
- Splits styled text into physical lines of at most `width` visible characters.
- Breaks at word boundaries (spaces, hyphens) when possible; hard-breaks
  mid-word only when a single word exceeds the column width.
- Wide characters (emoji, CJK): if a 2-cell char doesn't fit at the end of a
  line, insert a 1-cell padding space and wrap the char to the next line.
- ANSI state preservation: track active SGR attributes (bold, italic, colors)
  as a state machine. At the end of each physical sub-line, emit `\x1b[0m`
  (reset). At the start of the next sub-line, re-emit the active SGR sequence.
  This ensures each sub-line is self-contained and correct even if displayed
  independently.

Row rendering with wrapping:
- For each logical row, wrap every cell: `wrapped[col] = _wrap_cell(styled, col_width)`.
- Row height = `max(len(wrapped[col]) for col in range(num_cols))`.
- For each physical line index `0..row_height`:
  - Emit `│ `.
  - For each column: if this cell has a sub-line at this index, emit it
    (padded to `col_width`). If not, emit `col_width` spaces.
  - Emit ` │` after each column.
  - Emit newline.
- The separator row (`├─┼─┤`) appears between logical rows, NOT between
  physical sub-lines of the same row.

Height is unconstrained: tables can be as tall as needed. The user scrolls
through native terminal scrollback.

### 1c. Column width solver

Replace the greedy shrink-widest loop in `_fit_widths` with a constraint-based
allocator:

1. Compute per-column: `natural_width` (widest cell), `header_width` (header
   cell width), `min_word_width` (longest unbreakable word/token across all
   cells in the column).
2. Floor per column: `max(header_width, min_word_width, 1)`.
3. If `sum(natural_widths) + overhead <= terminal_width`: use natural widths
   (no wrapping needed).
4. Else if `sum(floors) + overhead <= terminal_width`: give each column its
   floor, distribute remaining width proportionally to
   `natural_width - floor` per column.
5. Else (extreme narrowing): proportional to floor. Some mid-word breaks
   unavoidable.

Header priority: headers are the last content to wrap because the floor
includes `header_width`. A column like "Rating" (6 chars) keeps its header
intact even when body content wraps.

### 1d. Tests

Rewrite all 11 existing table golden tests for the new box-drawing format:
- `test_table_basic_alignment`
- `test_table_column_alignment_colons`
- `test_table_terminated_by_prose_then_processes_line`
- `test_table_terminated_by_finish`
- `test_table_inline_styles_in_cells`
- `test_table_truncates_widest_cells_to_fit_width` (rename: no longer truncates,
  now wraps)
- `test_table_header_bold_body_not`
- `test_table_emoji_alignment`
- `test_table_wide_truncation` (rename: now wraps instead)
- `test_table_mixed_content`
- `test_table_inside_code_fence`

Add new tests:
- `test_table_cell_wrapping_multiline`: cell content wraps to 2-3 sub-lines,
  `│` aligned on every sub-line, shorter cells padded.
- `test_table_wrap_word_boundary`: breaks at spaces, not mid-word.
- `test_table_wrap_ansi_preservation`: bold text wrapping preserves bold across
  sub-lines (reset at end, reopen at start).
- `test_table_wrap_wide_char_padding`: emoji at column boundary gets padding.
- `test_table_header_priority_sizing`: header stays unwrapped while body wraps.
- `test_table_extreme_narrow`: width=30, table still renders correctly with
  aggressive wrapping.
- `test_table_render_at_different_widths`: same `TableData` rendered at width 80
  and width 40 produces different (both correct) output.

Verify chunking invariance still holds for all table fixtures.

## Phase 2 -- Switch to inline mode

### 2a. Remove fullscreen, add patch_stdout

In `_PromptController._build_app()`:
- Change `Application(full_screen=True)` to `Application(full_screen=False)`.
- Remove `mouse_support=True` (native terminal scroll handles this).
- Keep `color_depth=ColorDepth.DEPTH_24_BIT` (for howmuchleft truecolor).
- Keep `refresh_interval=0.25` (for howmuchleft updates).

Remove scroll-lock infrastructure (no longer needed -- native scroll):
- `_user_scrolled` flag
- `_patched_scroll_up` / `_patched_scroll_down` methods
- `_get_cursor_position` method
- `scroll_offsets`, `allow_scroll_beyond_bottom` on output Window

Remove the output `Window` and its `FormattedTextControl` entirely. Output no
longer goes through a prompt_toolkit Window -- it goes through `patch_stdout`.

In `_PromptController.run()`: wrap the `app.run_async()` call in
`with patch_stdout(raw=True):`. All `_raw_write` / `printer` calls now flow
through `patch_stdout`, appearing above the prompt in native scrollback.

### 2b. Clear screen on launch

At the start of `Repl.run()`, before creating the session: write `\x1b[2J\x1b[H`
to clear the visible screen and home the cursor. Prior terminal content (shell
prompts, claudewheel output) scrolls into scrollback history and is no longer
visible. This is the desired behavior per user requirement.

### 2c. Fix dead-space gap

Set bounded heights on all remaining layout children:
- Frame input Window: `Dimension(min=1, max=10)`, `dont_extend_height=True`.
- howmuchleft Window: `height=3`, `dont_extend_height=True`.
- Outer HSplit: `height=Dimension(max=16)` (frame borders 2 + max input 10 +
  howmuchleft 3 + margin 1).

This prevents `_min_available_height` from inflating the layout and creating
dead space. The layout children pack together at the top of the allocated
region.

### 2d. howmuchleft rendering

Keep `FormattedTextControl(lambda: ANSI(hml_output))` as a layout child (below
the Frame). `ANSI()` parses truecolor escapes; `DEPTH_24_BIT` re-emits them
losslessly. Refresh via the `refresh_interval=0.25` on the Application.

### 2e. Verify modals

Modals (permission prompts, AskUserQuestion) use `in_terminal` /
`run_in_terminal` to suspend the inline Application, run the dialog widgets, and
resume. This is the same mechanism as the current fullscreen mode. Verify it works
in inline mode by testing with a real permission prompt.

### 2f. Printer pathway

- Non-table output: `printer(text)` → `_raw_write(text)` (goes through
  `patch_stdout` into scrollback) + append `ProseBlock` to `_output_blocks`.
- Table output: `on_table` callback → `_raw_write(render_table(...))` (into
  scrollback) + append `TableBlock` to `_output_blocks`. No ProseBlock created.

## Phase 3 -- SIGWINCH table repaint

### 3a. Width propagation

On terminal resize (SIGWINCH), update `self._width` on the Repl so future
`StreamRenderer(self._width)` instances use the new width. Also update the
current StreamRenderer's width if a turn is active.

Hook into the resize: either install a SIGWINCH handler via `signal.signal`, or
hook into prompt_toolkit's Application `_on_resize` callback (which fires on
SIGWINCH). The hook must be safe to call from a signal context (schedule the
actual work via `asyncio.get_event_loop().call_soon_threadsafe`).

### 3b. Visible-area repaint

On resize, after updating the width:

1. Use `in_terminal` to suspend the inline Application.
2. Clear the visible screen: `\x1b[2J\x1b[H`.
3. Compute how many blocks to reprint: walk `_output_blocks` backwards,
   computing physical line counts at the NEW width, until the accumulated
   count exceeds terminal height. This is the "last screenful."
   - ProseBlock line count: `text.count('\n')` (fixed, prose is pre-rendered).
   - TableBlock line count: render at new width, count newlines. Use a cache
     (width-keyed) so repeated resizes don't re-render unnecessarily.
4. Reprint those blocks in order: ProseBlocks re-emitted via `_raw_write`,
   TableBlocks rendered via `render_table(data, new_width)` and written via
   `_raw_write`.
5. Resume the Application.

Content above the reprinted region (deeper in scrollback) stays at whatever
width it was originally rendered. This is inherent to native terminal scrollback
and is acceptable -- the visible area is always correct.

Flash mitigation: wrap the clear+reprint in synchronized output sequences
(`\x1b[?2026h` before, `\x1b[?2026l` after) so the terminal batches the update
into a single frame. Most modern terminals support this (kitty, WezTerm, VTE,
iTerm2). Terminals that don't support it show a brief flash.

### 3c. ProseBlock line count note

ProseBlocks store pre-rendered ANSI text. Their line count at any terminal width
is just `text.count('\n')` because the text was already formatted by the
StreamRenderer at the original width. On resize, the terminal reflows these lines
natively. The reprinted version is identical bytes to the original -- the
terminal handles the reflow the same way. This is NOT a problem: the purpose of
the reprint is to re-render TABLES at the new width. Prose is reprinted unchanged
because it must fill the space between tables.

If we ever want prose to also reflow on resize (currently not a goal), ProseBlocks
would need to store source markdown and be re-rendered. That is a future
enhancement, not part of this plan.

## Phase 4 -- Tests and integration

### 4a. Update unit tests

Remove tests that no longer apply:
- Scroll-lock tests (`test_*scroll*` if any exist in test_repl.py) -- scroll is
  now native terminal, no application-side logic.
- Any test referencing the output `Window` or `FormattedTextControl` -- removed
  in Phase 2.

Add new tests:
- `test_output_block_accumulation`: verify that printer creates ProseBlocks and
  on_table callback creates TableBlocks (no double-counting).
- `test_materialize_blocks_renders_tables_at_width`: same blocks materialized at
  width=80 and width=40 produce different table renderings.
- `test_on_table_callback_increments_newline_count`: callback correctly tracks
  line count.
- `test_render_table_width_independence`: `render_table(data, 80)` and
  `render_table(data, 40)` both produce valid box-drawing tables.

Keep all repl dispatch tests (they use FakeSession directly, unaffected by
_PromptController changes).

### 4b. Integration smoke test

Update `tests/test_integration.py` for inline mode:
- Remove any assertions about fullscreen behavior.
- The test should verify: miniclaude launches (no crash), output appears, the
  boxed input and howmuchleft render. May need adjusted timeouts since inline
  mode has different startup characteristics.

### 4c. Commits and changelog

One commit per logical unit:
- 0a: data types + printer split
- 0b: _flush_table refactor + render_table extraction
- 0c: materialize_blocks function
- 1a: box-drawing characters
- 1b: cell wrapping
- 1c: column width solver
- 1d: test rewrites + new tests
- 2a-2f: inline mode switch (may be one or two commits)
- 3a-3b: SIGWINCH repaint
- 4a-4b: test updates

Changelog entries:
- "Tables now render with full box-drawing characters" (feature)
- "Table cells wrap instead of truncating" (feature)
- "Column widths prioritize header readability" (feature)
- "Switched to inline mode with native terminal scroll" (breaking)
- "Tables re-render on terminal resize" (feature)

## Dependencies

- 0a -> 0b -> 0c (sequential, each builds on the last)
- Phase 1 depends on 0b (needs `render_table` and `TableData`)
- Phase 2 depends on 0a (needs `_output_blocks` and printer split)
- Phases 1 and 2 can run IN PARALLEL after 0b completes (they touch
  `_render.py` and `_repl.py` respectively, with no conflicts)
- Phase 3 depends on Phases 1 + 2 (needs box-drawing tables in inline mode
  with the block model)
- Phase 4 depends on everything

## Affected files

- `miniclaude/_render.py`: TableData, render_table, _flush_table refactor,
  box-drawing chars, _wrap_cell, column solver, _render_top_border,
  _render_bottom_border, updated _render_separator, updated _render_table_row
- `miniclaude/_repl.py`: OutputBlock/ProseBlock/TableBlock types,
  _output_blocks, printer split (_raw_write vs printer), on_table callback
  wiring in _run_turn, _PromptController rewrite for inline mode, SIGWINCH
  handler, materialize_blocks, screen clear on launch
- `miniclaude/_cli.py`: minimal changes (width is already read once; SIGWINCH
  handler updates it dynamically)
- `tests/test_render.py`: all 11 table golden tests rewritten, 7+ new tests
- `tests/test_repl.py`: scroll tests removed, output-block tests added
- `tests/test_integration.py`: updated for inline mode

## Effort estimate

Phase 0: ~200 lines of new/changed code, moderate complexity.
Phase 1: ~300 lines, high complexity (cell wrapping with ANSI state is the
hardest single piece).
Phase 2: ~150 lines changed, moderate complexity (prompt_toolkit layout work).
Phase 3: ~100 lines, moderate complexity (SIGWINCH + clear/reprint).
Phase 4: ~200 lines of test code.

Total: ~950 lines across 5 files. The cell wrapping ANSI state machine (1b)
is the critical path.
