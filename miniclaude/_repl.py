"""The REPL loop: turn orchestration, event dispatch, and type-ahead input.

The design splits cleanly into pure orchestration (unit-testable with a
``FakeSession``) and production prompt_toolkit wiring:

- :class:`Repl` owns the turn loop. It is constructed with
  ``(session_factory, interaction, printer)`` so tests inject fakes. Event
  dispatch, slash-command parsing, the interrupt guard, and the result/status
  lines are all driven through those injected surfaces -- no terminal required.
- :class:`_PromptController` is the production input surface: a fullscreen
  prompt_toolkit ``Application`` with a three-region layout (scrollable output,
  boxed input, howmuchleft status bar). It runs continuously; user input arrives
  via the ``Buffer``'s ``accept_handler``. Modals (permission prompts,
  AskUserQuestion) use ``in_terminal`` to briefly exit fullscreen, run the
  existing _dialogs.py code unchanged, and return.

Layout:
  HSplit [
    Window(FormattedTextControl)   # scrollable output (weight=1)
    Frame(Window(BufferControl))   # boxed input (dynamic, 1..10 lines)
    Window(FormattedTextControl)   # howmuchleft (height=3)
  ]

The Application uses ``full_screen=True`` (alternate screen -- preserves
scrollback on exit) and ``color_depth=DEPTH_24_BIT`` (truecolor lossless --
no quantization of howmuchleft's RGB escapes).
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
from miniclaude._render import StreamRenderer
from miniclaude._toolline import format_tool_result, format_tool_use

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
    ) -> str:
        now = time.monotonic()
        if self._cached and (now - self._last_time) < ttl:
            return self._cached
        result = render_howmuchleft(model, ctx_pct, cwd, cost_usd, rate_limits)
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
    ) -> None:
        self._session_factory = session_factory
        self._interaction = interaction
        self._printer = printer
        self._width = width

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
            # Redirect the printer to the output region for the duration of
            # the fullscreen app. The original printer is restored afterward
            # so the cost summary prints to the real terminal.
            orig_printer = self._printer
            self._printer = self._input.printer
            try:
                await self._input.run(self._main_loop, session)
            finally:
                self._exit = True
                self._printer = orig_printer
            # Print cost summary after exiting fullscreen (back to normal terminal).
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
        # Echo the user's input into scrollback
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
            if event.status != "allowed" or getattr(event, "utilization", 0) >= 0.8:
                parts = [f"rate limit: {event.status}"]
                if event.rate_limit_type:
                    parts.append(event.rate_limit_type)
                util = getattr(event, "utilization", None)
                if util is not None and util > 0:
                    parts.append(f"{util:.0%}")
                p(_yellow(" · ".join(parts)) + "\n")
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
        ``in_terminal`` briefly exits fullscreen so the modal's own inline
        Application (ChoiceInput / PromptSession) can run without conflict.
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


class _PromptController:
    """Fullscreen Application with output/input/status regions.

    The layout is an HSplit of three regions: a scrollable output window (top,
    weight=1), a boxed input area (Frame around BufferControl, fixed height),
    and a howmuchleft status bar (fixed height=3).

    The Application runs continuously with ``full_screen=True`` (alternate
    screen) and ``color_depth=DEPTH_24_BIT`` (truecolor, no quantization).

    Submitted lines flow into ``repl._queue``. Modals use ``in_terminal``
    to briefly exit fullscreen and run the existing _dialogs.py inline widgets.
    """

    def __init__(self, repl: Repl) -> None:
        self._repl = repl
        # Accumulated output text (raw ANSI). Appended by the printer.
        self._output_lines: list[str] = []
        self._app: Any | None = None
        # Scroll-lock: track output size and user scroll state.
        self._output_newline_count: int = 0
        self._user_scrolled: bool = False

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
            return ANSI("".join(self._output_lines))

        def _get_cursor_position() -> Point:
            if self._user_scrolled:
                # Pin the view where the user scrolled to.
                return Point(x=0, y=output_window.vertical_scroll)
            # Auto-follow: cursor at the last line so scroll keeps up.
            return Point(x=0, y=self._output_newline_count)

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

        # Intercept scroll-up to engage scroll-lock (auto-follow pauses
        # until the user submits input).
        _original_scroll_up = output_window._scroll_up

        def _patched_scroll_up() -> None:
            self._user_scrolled = True
            _original_scroll_up()

        output_window._scroll_up = _patched_scroll_up

        # --- Input region (boxed, bottom) ---
        def _accept(buf: Buffer) -> bool:
            text = buf.text
            if not text.strip():
                return False
            # Resume auto-follow on input submission.
            self._user_scrolled = False
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
        """Append text to the output region (called as the Repl's printer)."""
        if text:
            self._output_lines.append(text)
            self._output_newline_count += text.count("\n")

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
