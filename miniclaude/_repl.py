"""The REPL loop: turn orchestration, event dispatch, and type-ahead input.

The design splits cleanly into pure orchestration (unit-testable with a
``FakeSession``) and production prompt_toolkit wiring:

- :class:`Repl` owns the turn loop. It is constructed with
  ``(session_factory, interaction, printer)`` so tests inject fakes. Event
  dispatch, slash-command parsing, the interrupt guard, and the result/status
  lines are all driven through those injected surfaces -- no terminal required.
- :class:`_PromptController` is the production input surface: a fullscreen
  prompt_toolkit ``Application`` (alternate screen) with a three-region
  layout (scrollable output, boxed input, howmuchleft status bar). Output is
  modelled as a list of :class:`OutputBlock`s (pre-rendered prose and
  structured tables); the output window's content callback materializes those
  blocks to ANSI at the live terminal width every render, so a resize simply
  re-materializes tables at the new width -- there is no SIGWINCH handler.
  User input arrives via the ``Buffer``'s ``accept_handler``. Modals
  (permission prompts, AskUserQuestion) use ``in_terminal`` to suspend the
  fullscreen app, run the existing _dialogs.py code unchanged, and return.

Layout:
  HSplit [
    Window(FormattedTextControl)   # scrollable output (weight=1)
    Frame(Window(BufferControl))   # boxed input (dynamic, 1..10 lines)
    Window(FormattedTextControl)   # howmuchleft (height=3)
  ]

The Application uses ``full_screen=True`` (alternate screen -- output scrolls
in-app with the mouse wheel, no terminal-native scrollback) and
``color_depth=DEPTH_24_BIT`` (truecolor lossless -- no quantization of
howmuchleft's RGB escapes).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Awaitable, Callable

from claudestream import (
    ApiRetry,
    AssistantText,
    BudgetThreshold,
    ClaudeStreamError,
    CompactBoundary,
    FileEdit,
    FileWrite,
    PermissionRequest,
    RateLimit,
    Result,
    StreamDelta,
    SystemInit,
    Thinking,
    ToolResult,
    ToolUse,
    UserDialogRequest,
)

from miniclaude._dialogs import (
    Interaction,
    Printer,
    run_dialog_notice,
    run_permission_flow,
    run_question_flow,
)
from miniclaude._render import StreamRenderer, TableData, render_table
from miniclaude._toolline import format_tool_result, format_tool_use

# --- Output block types -------------------------------------------------------

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ProseBlock:
    """Pre-rendered ANSI text block (immutable after creation)."""

    ansi_text: str


@dataclass(frozen=True)
class TableBlock:
    """Structured table data that can be rendered at any width."""

    data: TableData


OutputBlock = Union[ProseBlock, TableBlock]


def materialize_blocks(blocks: list[OutputBlock], width: int) -> str:
    """Render a list of output blocks into a single ANSI string.

    ProseBlocks contribute their pre-rendered ansi_text as-is. TableBlocks are
    rendered at the given width via render_table. This is the materialization
    path behind the output window's content callback: it runs at the live
    terminal width every render, so tables re-flow automatically on resize.
    """
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, ProseBlock):
            parts.append(block.ansi_text)
        elif isinstance(block, TableBlock):
            parts.append(render_table(block.data, width))
    return "".join(parts)


# --- ANSI helpers (raw SGR) ---------------------------------------------------

RESET = "\x1b[0m"


def _sgr(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}{RESET}"


def _dim(text: str) -> str:
    return _sgr("2", text)


def _red(text: str) -> str:
    return _sgr("31", text)


def _yellow(text: str) -> str:
    return _sgr("33", text)


def _fmt_duration(ms: float) -> str:
    """Render a millisecond duration compactly (e.g. ``2.3s``)."""
    return f"{ms / 1000.0:.1f}s"


def _ctx_pct(model_usage: dict, model_name: str | None) -> int | None:
    """Compute context-window utilization percentage from Result.model_usage.

    Prefers the entry keyed by ``model_name``; otherwise the first entry that
    reports a ``contextWindow``. Returns None when no usable entry exists.
    Utilization = (input + cacheRead + cacheCreation) / contextWindow.
    """
    if not model_usage:
        return None
    entry: Any = None
    if model_name and model_name in model_usage:
        entry = model_usage[model_name]
    if not isinstance(entry, dict) or not entry.get("contextWindow"):
        entry = None
        for value in model_usage.values():
            if isinstance(value, dict) and value.get("contextWindow"):
                entry = value
                break
    if not isinstance(entry, dict):
        return None
    context_window = entry.get("contextWindow")
    if not context_window:
        return None
    used = (
        entry.get("inputTokens", 0)
        + entry.get("cacheReadInputTokens", 0)
        + entry.get("cacheCreationInputTokens", 0)
    )
    return round(100 * used / context_window)


_HELP_LINES = [
    "/model <name>   switch the model (empty: show current)",
    "/mode <mode>    change permission mode (empty: show current)",
    "/context        show context-window usage",
    "/cost           show session cost and token totals",
    "/quit, /exit    leave the REPL (also Ctrl+D)",
    "/help           show this list",
]


# --- howmuchleft integration -------------------------------------------------

_HOWMUCHLEFT_NOT_FOUND = (
    "howmuchleft not installed\nhowmuchleft not installed\nhowmuchleft not installed"
)

_HML_TIMEOUT = 0.5  # seconds


def render_howmuchleft(
    model: str,
    ctx_pct: float | None,
    cwd: str,
    cost_usd: float,
    rate_limits: dict | None = None,
    session_id: str = "",
) -> str:
    """Spawn howmuchleft and return 3 lines of ANSI status text.

    Caches the result for ``cache_ttl`` seconds. On timeout or error, returns
    the last successful output (or a fallback message).
    """
    binary = shutil.which("howmuchleft")
    if binary is None:
        return _HOWMUCHLEFT_NOT_FOUND

    stdin_data: dict[str, Any] = {
        "model": model or "?",
        "cwd": cwd or os.getcwd(),
        "cost": {"total_cost_usd": cost_usd},
    }
    if session_id:
        stdin_data["session_id"] = session_id
    if ctx_pct is not None:
        stdin_data["context_window"] = {"used_percentage": float(ctx_pct)}
    else:
        stdin_data["context_window"] = {}
    if rate_limits:
        stdin_data["rate_limits"] = rate_limits

    try:
        proc = subprocess.run(
            [binary],
            input=json.dumps(stdin_data),
            capture_output=True,
            text=True,
            timeout=_HML_TIMEOUT,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.rstrip("\n")
    except (subprocess.TimeoutExpired, OSError):
        pass
    return _HOWMUCHLEFT_NOT_FOUND


class _HowMuchLeftCache:
    """Time-based cache for howmuchleft output. Thread/coroutine safe enough
    for single-writer (the refresh loop) usage."""

    def __init__(self) -> None:
        self._cached: str = ""
        self._last_time: float = 0.0

    def get(
        self,
        model: str,
        ctx_pct: float | None,
        cwd: str,
        cost_usd: float,
        rate_limits: dict | None,
        ttl: float,
        session_id: str = "",
    ) -> str:
        now = time.monotonic()
        if self._cached and (now - self._last_time) < ttl:
            return self._cached
        result = render_howmuchleft(
            model, ctx_pct, cwd, cost_usd, rate_limits, session_id=session_id
        )
        if result != _HOWMUCHLEFT_NOT_FOUND or not self._cached:
            self._cached = result
            self._last_time = now
        return self._cached or result


class Repl:
    """The turn loop. Pure orchestration over injected session/interaction/printer.

    ``session_factory`` is a zero-argument callable returning an async context
    manager that yields a session (production: ``lambda: AsyncSession(config)``).
    ``interaction`` drives the modal UI (``ask_choice``/``ask_text``).
    ``printer`` writes pre-rendered output.
    """

    def __init__(
        self,
        session_factory: Callable[[], Any],
        interaction: Interaction,
        printer: Printer,
        *,
        width: int = 80,
        model: str = "",
        permission_mode: str = "",
        intro: str = "",
    ) -> None:
        self._session_factory = session_factory
        self._interaction = interaction
        self._printer = printer
        self._width = width
        self._intro = intro

        self._queue: asyncio.Queue = asyncio.Queue()
        self._session: Any = None
        self._input: _PromptController | None = None
        self._renderer = StreamRenderer(width)

        self._turn_active = False
        self._interrupt_pending = False
        self._interrupt_task: asyncio.Task | None = None
        self._exit = False

        # State tracked for howmuchleft
        self._model: str = model
        self._mode: str = permission_mode
        self._ctx_pct: float | None = None
        self._cost_usd: float = 0.0
        self._cwd: str = ""
        self._session_id: str = ""
        self._rate_limits: dict | None = None
        self._hml_cache = _HowMuchLeftCache()

    # --- Public state used by the input controller ---

    @property
    def turn_active(self) -> bool:
        return self._turn_active

    def notice_queued(self, text: str) -> None:
        """Print the dim ``queued: ...`` notice when a line is typed mid-turn."""
        snippet = " ".join(text.strip().split())[:40]
        self._printer(_dim(f"queued: {snippet}") + "\n")
        self._invalidate()

    def request_exit(self) -> None:
        """Ask the main loop to stop; unblock it with a sentinel."""
        self._exit = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        # In production, also exit the fullscreen Application.
        if self._input and self._input._app and self._input._app.is_running:
            self._input._app.exit()

    def request_interrupt(self) -> None:
        """Interrupt the running turn (fire-and-forget, guarded against double-fire)."""
        if self._interrupt_pending or not self._turn_active:
            return
        self._interrupt_pending = True
        self._printer(_dim("interrupting…") + "\n")
        self._interrupt_task = asyncio.ensure_future(self._do_interrupt())

    async def _do_interrupt(self) -> None:
        try:
            await self._session.interrupt()
        except Exception:
            pass

    # --- Production entry point ---

    def _invalidate(self) -> None:
        if self._input and self._input._app and self._input._app.is_running:
            self._input._app.invalidate()

    def _get_toolbar(self) -> str:
        """Return howmuchleft output for the status bar (ANSI pass-through)."""
        ttl = 0.25 if self._turn_active else 1.0
        return self._hml_cache.get(
            self._model,
            self._ctx_pct,
            self._cwd or os.getcwd(),
            self._cost_usd,
            self._rate_limits,
            ttl=ttl,
            session_id=self._session_id,
        )

    async def _refresh_loop(self) -> None:
        """Invalidate the app ~4x/s while a turn is active to refresh howmuchleft."""
        while self._turn_active:
            self._invalidate()
            await asyncio.sleep(0.25)

    async def run(self) -> None:
        """Production run: fullscreen Application with output/input/status regions."""
        async with self._session_factory() as session:
            self._session = session
            self._input = _PromptController(self)
            # Route the printer into the output-block model for the duration of
            # the fullscreen app. The original printer is restored afterward so
            # the post-exit cost summary prints to the normal terminal.
            orig_printer = self._printer
            self._printer = self._input.printer
            # Emit the intro as the first output block, now that the controller
            # is wired. The alternate screen would swallow anything printed
            # before this point.
            if self._intro:
                self._printer(_dim(self._intro) + "\n")
            try:
                await self._input.run(self._main_loop, session)
            finally:
                self._exit = True
                self._printer = orig_printer
            # Print cost summary after the fullscreen app exits.
            sys.stdout.write(self._format_cost(session))

    # --- Core loop ---

    async def _main_loop(self, session: Any) -> None:
        while not self._exit:
            line = await self._queue.get()
            if line is None:
                break
            if await self._handle_line(session, line):
                continue
            await self._run_turn(session, line)

    async def _run_turn(self, session: Any, prompt: str) -> None:
        self._renderer = StreamRenderer(self._width)
        self._turn_active = True
        self._interrupt_pending = False
        refresh_task: asyncio.Task | None = None
        if self._input:
            refresh_task = asyncio.ensure_future(self._refresh_loop())
            # Wire the on_table callback so tables become TableBlocks. The
            # output window materializes them at the live width every render,
            # so no explicit rendering happens here.
            ctrl = self._input

            def _on_table(data: TableData) -> None:
                ctrl._output_blocks.append(TableBlock(data))
                self._invalidate()

            self._renderer.on_table = _on_table
        # Echo the user's input into the output region
        lines = prompt.strip().splitlines()
        echo = lines[0][:100] + ("…" if len(lines) > 1 or len(lines[0]) > 100 else "")
        self._printer(_dim(f"> {echo}") + "\n")
        try:
            async for event in session.send(prompt):
                await self._dispatch(session, event)
        except ClaudeStreamError as exc:
            self._printer(_red(f"error: {exc}") + "\n")
        except Exception as exc:  # noqa: BLE001 -- turn errors must not kill the REPL
            self._printer(_red(f"error: {exc}") + "\n")
        finally:
            self._printer(self._renderer.finish())
            self._turn_active = False
            self._interrupt_pending = False
            if refresh_task:
                refresh_task.cancel()
                try:
                    await refresh_task
                except BaseException:
                    pass
            self._invalidate()

    # --- Event dispatch ---

    async def _dispatch(self, session: Any, event: Any) -> None:
        p = self._printer
        r = self._renderer

        if isinstance(event, StreamDelta):
            if event.parent_tool_use_id is not None:
                return
            delta_type = event.delta_type
            if delta_type == "text_delta":
                p(r.feed_text(event.text or ""))
            elif delta_type == "thinking_delta":
                delta = event.event.get("delta", {}) if isinstance(event.event, dict) else {}
                p(r.feed_thinking(delta.get("thinking", "") or ""))
            return

        # Flattened partial-message events already rendered by deltas -> ignore.
        if isinstance(event, (AssistantText, Thinking)):
            return

        if isinstance(event, ToolUse):
            p(r.finish())
            p(format_tool_use(event.name, event.input, event.parent_tool_use_id) + "\n")
            return

        if isinstance(event, ToolResult):
            p(
                format_tool_result(
                    event.tool_name,
                    event.content,
                    getattr(event, "is_error", False),
                    event.parent_tool_use_id,
                )
                + "\n"
            )
            return

        # File-tracking events are already covered by the ToolUse line.
        if isinstance(event, (FileWrite, FileEdit)):
            return

        if isinstance(event, PermissionRequest):
            if event.tool_name == "AskUserQuestion":
                await self._with_prompt_suspended(
                    run_question_flow(event, session, self._interaction, p)
                )
            else:
                await self._with_prompt_suspended(
                    run_permission_flow(event, session, self._interaction, p)
                )
            return

        if isinstance(event, UserDialogRequest):
            await run_dialog_notice(event, session, p)
            return

        if isinstance(event, Result):
            p(r.finish())
            self._interrupt_pending = False
            p(self._format_result_line(session, event))
            self._cost_usd = session.total_cost_usd
            pct = _ctx_pct(event.model_usage, getattr(session, "model_name", None))
            if pct is not None:
                self._ctx_pct = pct
            self._invalidate()
            return

        if isinstance(event, ApiRetry):
            p(_dim(f"retry {event.attempt}/{event.max_retries}…") + "\n")
            return

        if isinstance(event, RateLimit):
            # Accumulate rate-limit state for the howmuchleft status bar; no
            # visible output (howmuchleft renders the limit rows itself).
            #
            # Mapping grounded on fixture evidence, not live traffic: a live
            # probe (2026-07-24) observed ZERO RateLimit events (the account was
            # not near any threshold). The keys/values come from claudestream's
            # own test fixture (rate_limit_type "five_hour") and howmuchleft's
            # stdin schema (a rate_limits dict keyed five_hour/seven_day/
            # seven_day_overage_included, each {used_percentage, resets_at}).
            # The discriminator is passed through verbatim: whatever
            # rate_limit_type arrives becomes the key, later events overwrite.
            if self._rate_limits is None:
                self._rate_limits = {}
            entry: dict[str, Any] = {"used_percentage": event.utilization * 100}
            if event.resets_at is not None:
                entry["resets_at"] = event.resets_at
            self._rate_limits[event.rate_limit_type] = entry
            return

        if isinstance(event, BudgetThreshold):
            p(
                _yellow(
                    f"budget: {event.metric} crossed {event.threshold} "
                    f"(now {event.current_value})"
                )
                + "\n"
            )
            return

        if isinstance(event, CompactBoundary):
            p(_dim("── compacted ──") + "\n")
            return

        if isinstance(event, SystemInit):
            self._model = event.model or self._model
            self._mode = event.permission_mode or self._mode
            if event.cwd:
                self._cwd = event.cwd
            self._session_id = event.session_id or ""
            self._invalidate()
            return

        # UnknownEvent / ControlResponse / anything else -> ignored.

    async def _with_prompt_suspended(self, coro: Awaitable[Any]) -> Any:
        """Run a modal coroutine with the fullscreen app temporarily suspended.

        In tests (no input controller) the coroutine simply runs. In production
        ``in_terminal`` suspends the fullscreen Application so the modal's own
        widgets (ChoiceInput / PromptSession) can run without conflict.
        """
        if self._input is None:
            return await coro
        return await self._input.run_modal(coro)

    # --- Slash commands ---

    async def _handle_line(self, session: Any, line: str) -> bool:
        """Dispatch a client-side slash command. Returns True if fully handled.

        Unknown ``/`` commands return False so they are sent to Claude verbatim
        (server-side skill commands still work).
        """
        stripped = line.strip()
        if not stripped.startswith("/"):
            return False
        parts = stripped.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/model":
            if arg:
                await session.set_model(arg)
            else:
                self._printer(_dim(f"model: {session.model_name}") + "\n")
            return True
        if cmd == "/mode":
            if arg:
                await session.set_permission_mode(arg)
            else:
                self._printer(_dim(f"mode: {session.permission_mode}") + "\n")
            return True
        if cmd == "/context":
            usage = await session.get_context_usage()
            self._printer(self._format_context(usage))
            return True
        if cmd == "/cost":
            self._printer(self._format_cost(session))
            return True
        if cmd in ("/quit", "/exit"):
            self.request_exit()
            return True
        if cmd == "/help":
            self._printer("".join(_dim(line) + "\n" for line in _HELP_LINES))
            return True

        return False

    # --- Status-line formatting ---

    def _format_result_line(self, session: Any, event: Result) -> str:
        parts = [
            f"${event.total_cost_usd:.4f}",
            _fmt_duration(event.duration_ms),
            f"{event.num_turns} turn(s)",
        ]
        line = "── " + " · ".join(parts)
        pct = _ctx_pct(event.model_usage, getattr(session, "model_name", None))
        if pct is not None:
            line += f" · ctx {pct}%"
        return _dim(line) + "\n"

    def _format_context(self, usage: Any) -> str:
        categories = list(getattr(usage, "categories", []) or [])
        width = max((len(c.name) for c in categories), default=0)
        lines = [f"  {c.name.ljust(width)}  {c.tokens}" for c in categories]
        total = usage.total_tokens
        maximum = usage.max_tokens
        pct = round(100 * total / maximum) if maximum else 0
        lines.append(f"  total {total}/{maximum} ({pct}%)")
        return "".join(_dim(line) + "\n" for line in lines)

    def _format_cost(self, session: Any) -> str:
        return (
            _dim(
                f"── ${session.total_cost_usd:.4f} · "
                f"{session.total_tokens} tokens · {session.turn_count} turn(s)"
            )
            + "\n"
        )


# --- Production input controller (fullscreen prompt_toolkit Application) ------

# Mouse-wheel coalescing divisor: the output window scrolls one real line only
# every Nth wheel event, so scrolling is slow and controllable.
_SCROLL_DIVISOR = 10


def _render_info_at_bottom(info: Any) -> bool:
    """True when the last content line is visible (the view is at the bottom).

    Uses the window's render_info, which counts *wrapped* screen rows via
    ``content_height`` -- so this is correct under wrapped long lines, unlike a
    naive comparison against the logical newline count. Returns True when there
    is no render_info yet (nothing scrolled == at bottom).
    """
    if info is None:
        return True
    try:
        content_height = info.content_height
        last = info.last_visible_line()
    except Exception:  # noqa: BLE001 -- any render_info shape issue => treat as bottom
        return True
    if content_height <= 0:
        return True
    return last >= content_height - 1


class _PromptController:
    """Fullscreen Application with output/input/status regions.

    The layout is an HSplit of three regions: a scrollable output window (top,
    weight=1), a boxed input area (Frame around BufferControl, height 1..10),
    and a howmuchleft status bar (fixed height=3).

    Output is a list of :class:`OutputBlock`s. The output window's content
    callback materializes them to ANSI at the *live* terminal width every
    render (memoized incrementally; full recompute only on width change), so a
    resize automatically re-flows tables -- there is no SIGWINCH handler.

    The Application runs with ``full_screen=True`` (alternate screen),
    ``mouse_support=True``, and ``color_depth=DEPTH_24_BIT`` (truecolor, no
    quantization). Submitted lines flow into ``repl._queue``. Modals use
    ``in_terminal`` to suspend the app and run the existing _dialogs.py widgets.
    """

    def __init__(self, repl: Repl) -> None:
        self._repl = repl
        self._app: Any | None = None
        # Structured output blocks -- the single source of truth for the
        # scrollable output region.
        self._output_blocks: list[OutputBlock] = []
        # Incremental materialization memo.
        self._memo_text: str = ""
        self._memo_count: int = 0
        self._memo_width: int = -1
        self._last_materialized_text: str = ""
        # Number of blocks materialize_blocks was called with on the last
        # _materialize() (exposed for tests of the incremental memo).
        self._last_render_block_count: int = 0
        # Scroll-lock + mouse-wheel coalescing state.
        self._user_scrolled: bool = False
        self._up_count: int = 0
        self._down_count: int = 0
        self._last_scroll_dir: str = ""

    # -- Output materialization (block-backed, width-live, memoized) -----------

    def _current_width(self) -> int:
        """The live terminal width, read from the running app when available."""
        if self._app is not None:
            try:
                cols = self._app.output.get_size().columns
                if cols > 0:
                    return cols
            except Exception:  # noqa: BLE001
                pass
        return self._repl._width

    def _materialize(self, width: int) -> str:
        """Materialize the output blocks to ANSI at ``width``, incrementally.

        Blocks are baked into ``_memo_text`` once they are "sealed" (every
        block except a trailing ProseBlock, which the printer may still coalesce
        into). Only newly-sealed blocks are rendered on each same-width call;
        the unsealed trailing prose block is rendered fresh (cheap, single
        block). A width change forces a full recompute.
        """
        blocks = self._output_blocks
        n = len(blocks)
        if width != self._memo_width:
            self._memo_width = width
            self._memo_text = ""
            self._memo_count = 0
        sealed = n
        if n > 0 and isinstance(blocks[-1], ProseBlock):
            sealed = n - 1
        rendered = 0
        if self._memo_count < sealed:
            self._memo_text += materialize_blocks(blocks[self._memo_count:sealed], width)
            rendered += sealed - self._memo_count
            self._memo_count = sealed
        if sealed < n:
            tail = materialize_blocks(blocks[sealed:], width)
            rendered += n - sealed
        else:
            tail = ""
        self._last_render_block_count = rendered
        self._last_materialized_text = self._memo_text + tail
        return self._last_materialized_text

    def _content_line_count(self) -> int:
        """Logical (newline) line count of the materialized output."""
        return self._materialize(self._current_width()).count("\n")

    def _cursor_y(self, tail_line: int, vertical_scroll: int) -> int:
        """Cursor row for the output window: tail when following, else pinned."""
        return vertical_scroll if self._user_scrolled else tail_line

    # -- Mouse-wheel coalescing + scroll-lock ----------------------------------

    def _on_scroll_up(self, do_move: Callable[[], None]) -> None:
        """Handle one upward wheel event with divisor coalescing + scroll-lock."""
        if self._last_scroll_dir != "up":
            # Direction change: reset the opposite counter so a later reversal
            # responds within one divisor's worth of events.
            self._down_count = 0
            self._last_scroll_dir = "up"
        # Engage scroll-lock unconditionally on the first up-event of a burst.
        self._user_scrolled = True
        self._up_count += 1
        if self._up_count >= _SCROLL_DIVISOR:
            self._up_count = 0
            do_move()

    def _on_scroll_down(
        self, do_move: Callable[[], None], at_bottom: Callable[[], bool]
    ) -> None:
        """Handle one downward wheel event; release the lock at the bottom."""
        if self._last_scroll_dir != "down":
            self._up_count = 0
            self._last_scroll_dir = "down"
        self._down_count += 1
        if self._down_count >= _SCROLL_DIVISOR:
            self._down_count = 0
            do_move()
            # Re-evaluate lock release after each REAL downward movement.
            if at_bottom():
                self._user_scrolled = False

    def _reset_scroll(self) -> None:
        """Re-engage auto-follow and clear coalescing state (on input submit)."""
        self._user_scrolled = False
        self._up_count = 0
        self._down_count = 0
        self._last_scroll_dir = ""

    def _build_app(self):
        from pathlib import Path

        from prompt_toolkit.application import Application
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.data_structures import Point
        from prompt_toolkit.formatted_text import ANSI
        from prompt_toolkit.history import FileHistory, ThreadedHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import (
            BufferControl,
            Dimension,
            FormattedTextControl,
            HSplit,
            Layout,
            ScrollOffsets,
            Window,
        )
        from prompt_toolkit.output.color_depth import ColorDepth
        from prompt_toolkit.widgets import Frame

        history_dir = Path.home() / ".miniclaude"
        history_dir.mkdir(parents=True, exist_ok=True)
        history = ThreadedHistory(FileHistory(str(history_dir / "history")))

        # --- Output region (scrollable, top) ---
        def _get_output_text():
            width = self._current_width()
            # Keep the Repl/renderer width in sync so new prose wraps correctly
            # after a resize -- no SIGWINCH handler needed.
            if width != self._repl._width:
                self._repl._width = width
                self._repl._renderer.width = width
            return ANSI(self._materialize(width))

        def _get_cursor_position() -> Point:
            tail_line = self._content_line_count()
            return Point(x=0, y=self._cursor_y(tail_line, output_window.vertical_scroll))

        output_control = FormattedTextControl(
            text=_get_output_text,
            show_cursor=False,
            get_cursor_position=_get_cursor_position,
        )
        output_window = Window(
            content=output_control,
            wrap_lines=True,
            scroll_offsets=ScrollOffsets(bottom=0),
            allow_scroll_beyond_bottom=True,
        )

        # Wrap the per-instance scroll methods with mouse-wheel coalescing and
        # scroll-lock (see _on_scroll_up/_on_scroll_down).
        _orig_scroll_up = output_window._scroll_up
        _orig_scroll_down = output_window._scroll_down

        def _wheel_up() -> None:
            self._on_scroll_up(_orig_scroll_up)

        def _wheel_down() -> None:
            self._on_scroll_down(
                _orig_scroll_down,
                lambda: _render_info_at_bottom(output_window.render_info),
            )

        output_window._scroll_up = _wheel_up
        output_window._scroll_down = _wheel_down

        # --- Input region (boxed, bottom) ---
        def _accept(buf: Buffer) -> bool:
            text = buf.text
            if not text.strip():
                return False
            # Submitting input re-engages auto-follow.
            self._reset_scroll()
            if self._repl.turn_active:
                self._repl.notice_queued(text)
            self._repl._queue.put_nowait(text)
            # Return False so the buffer is NOT kept (reset after accept).
            return False

        input_buffer = Buffer(
            multiline=True,
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            accept_handler=_accept,
            name="input",
        )
        input_control = BufferControl(buffer=input_buffer, focusable=True)
        input_window = Window(
            content=input_control,
            height=Dimension(min=1, max=10),
            dont_extend_height=True,
            wrap_lines=True,
        )
        framed_input = Frame(body=input_window, title="")

        # --- howmuchleft region (status bar, bottom) ---
        def _get_hml_text():
            return ANSI(self._repl._get_toolbar())

        hml_control = FormattedTextControl(text=_get_hml_text)
        hml_window = Window(content=hml_control, height=3)

        # --- Key bindings ---
        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event) -> None:
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")
        def _newline(event) -> None:
            event.current_buffer.insert_text("\n")

        @kb.add("c-c")
        def _ctrl_c(event) -> None:
            if self._repl.turn_active:
                self._repl.request_interrupt()
            else:
                event.current_buffer.reset()

        @kb.add("escape")
        def _escape(event) -> None:
            if self._repl.turn_active:
                self._repl.request_interrupt()

        @kb.add("c-d")
        def _ctrl_d(event) -> None:
            if not event.current_buffer.text:
                self._repl.request_exit()
                event.app.exit()

        # --- Layout ---
        root = HSplit([output_window, framed_input, hml_window])
        layout = Layout(root, focused_element=input_buffer)

        app: Application = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            mouse_support=True,
            color_depth=ColorDepth.DEPTH_24_BIT,
            # Refresh at ~4Hz so howmuchleft updates during turns and the
            # output window picks up new printer() content without explicit
            # invalidate() calls.
            refresh_interval=0.25,
        )
        # Snappy lone-Esc resolution (default is 0.5s).
        app.ttimeoutlen = 0.05
        return app

    def printer(self, text: str) -> None:
        """Append prose text to the output blocks (called as the Repl's printer).

        Consecutive prose appends coalesce into the tail ProseBlock so that
        thousands of small streamed chunks do not become thousands of blocks.
        """
        if not text:
            return
        blocks = self._output_blocks
        if blocks and isinstance(blocks[-1], ProseBlock):
            blocks[-1] = ProseBlock(blocks[-1].ansi_text + text)
        else:
            blocks.append(ProseBlock(text))

    async def run(
        self,
        main_loop: Callable[[Any], Awaitable[None]],
        session: Any,
    ) -> None:
        """Build the Application, run the main loop as a background task, and
        run the app. When either finishes (exit or EOFError), clean up."""
        self._app = self._build_app()

        async def _loop_wrapper() -> None:
            try:
                await main_loop(session)
            finally:
                # Main loop finished (e.g. /quit) -- make the app exit too.
                if self._app and self._app.is_running:
                    self._app.exit()

        loop_task = asyncio.ensure_future(_loop_wrapper())

        try:
            await self._app.run_async()
        except EOFError:
            self._repl.request_exit()
        finally:
            self._repl._exit = True
            loop_task.cancel()
            try:
                await loop_task
            except BaseException:
                pass

    async def run_modal(self, coro: Awaitable[Any]) -> Any:
        """Briefly exit fullscreen, run the modal coroutine, return to fullscreen.

        Uses ``in_terminal`` which suspends the Application's rendering,
        restores the normal terminal, runs the modal (which may create its own
        inline Application via ChoiceInput/PromptSession), then resumes.
        """
        from prompt_toolkit.application import in_terminal

        async with in_terminal():
            return await coro
