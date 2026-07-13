"""The REPL loop: turn orchestration, event dispatch, and type-ahead input.

The design splits cleanly into pure orchestration (unit-testable with a
``FakeSession``) and production prompt_toolkit wiring:

- :class:`Repl` owns the turn loop. It is constructed with
  ``(session_factory, interaction, printer)`` so tests inject fakes. Event
  dispatch, slash-command parsing, the interrupt guard, and the result/status
  lines are all driven through those injected surfaces -- no terminal required.
- :class:`_PromptController` is the production input surface: one reused
  prompt_toolkit ``PromptSession`` running as a background asyncio task,
  feeding submitted lines into an ``asyncio.Queue``. It implements the
  type-ahead concurrency model (queue while a turn is active) and the
  one-Application-at-a-time discipline: before a modal it suspends the prompt
  (``app.exit`` with a sentinel, buffer text preserved) and resumes afterward.

Concurrency invariant: exactly one prompt_toolkit ``Application`` runs at any
moment. The background prompt is suspended around every permission/dialog modal
via :meth:`Repl._with_prompt_suspended`.
"""

from __future__ import annotations

import asyncio
import sys
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

# --- ANSI helpers (raw SGR for patch_stdout(raw=True)) -----------------------

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
        version: str = "",
        width: int = 80,
    ) -> None:
        self._session_factory = session_factory
        self._interaction = interaction
        self._printer = printer
        self._version = version
        self._width = width

        self._queue: asyncio.Queue = asyncio.Queue()
        self._session: Any = None
        self._input: _PromptController | None = None
        self._renderer = StreamRenderer(width)

        self._turn_active = False
        self._interrupt_pending = False
        self._interrupt_task: asyncio.Task | None = None
        self._startup_printed = False
        self._exit = False

    # --- Public state used by the input controller ---

    @property
    def turn_active(self) -> bool:
        return self._turn_active

    def notice_queued(self, text: str) -> None:
        """Print the dim ``queued: ...`` notice when a line is typed mid-turn."""
        snippet = " ".join(text.strip().split())[:40]
        self._printer(_dim(f"queued: {snippet}") + "\n")

    def request_exit(self) -> None:
        """Ask the main loop to stop; unblock it with a sentinel."""
        self._exit = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

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

    async def run(self) -> None:
        """Production run: real session, patch_stdout, concurrent prompt task."""
        from prompt_toolkit.patch_stdout import patch_stdout

        async with self._session_factory() as session:
            self._session = session
            self._input = _PromptController(self)
            with patch_stdout(raw=True):
                loop_task = asyncio.ensure_future(self._input.run_loop())
                try:
                    await self._main_loop(session)
                finally:
                    self._exit = True
                    loop_task.cancel()
                    try:
                        await loop_task
                    except BaseException:
                        pass
            self._printer(self._format_cost(session))

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
                    event.tool_name, event.content, False, event.parent_tool_use_id
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
            return

        if isinstance(event, ApiRetry):
            p(_dim(f"retry {event.attempt}/{event.max_retries}…") + "\n")
            return

        if isinstance(event, RateLimit):
            p(_yellow(f"rate limit: {event.status} ({event.rate_limit_type})") + "\n")
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
            if not self._startup_printed:
                self._startup_printed = True
                p(
                    _dim(
                        f"miniclaude {self._version} · {event.model} · "
                        f"{event.permission_mode} · {event.cwd} · "
                        f"session {event.session_id or ''}"
                    )
                    + "\n"
                )
            return

        # UnknownEvent / ControlResponse / anything else -> ignored.

    async def _with_prompt_suspended(self, coro: Awaitable[Any]) -> Any:
        """Run a modal coroutine with the background prompt suspended.

        In tests (no input controller) the coroutine simply runs. In production
        the prompt is exited first so exactly one Application is live.
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


# --- Production input controller (prompt_toolkit; inline, never full_screen) ---

_SUSPEND = object()  # sentinel returned from prompt_async when parked for a modal


class _PromptController:
    """One reused inline PromptSession run as a background task.

    Submitted lines flow into ``repl._queue``. While a turn is active a
    submitted line is queued with a dim notice. Around a modal the prompt is
    suspended (``app.exit(result=_SUSPEND)``) and resumed with the in-progress
    buffer text preserved, guaranteeing one Application at a time.
    """

    def __init__(self, repl: Repl) -> None:
        self._repl = repl
        self._running = True
        self._preserved = ""
        self._suspended = asyncio.Event()
        self._resume = asyncio.Event()
        self._session = self._build_session()

    def _build_session(self):
        from pathlib import Path

        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.history import FileHistory, ThreadedHistory
        from prompt_toolkit.key_binding import KeyBindings

        history_dir = Path.home() / ".miniclaude"
        history_dir.mkdir(parents=True, exist_ok=True)
        history = ThreadedHistory(FileHistory(str(history_dir / "history")))

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
                event.app.exit(exception=EOFError())

        session = PromptSession(
            multiline=True,
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=kb,
        )
        # Snappy lone-Esc resolution (default is 0.5s).
        session.app.ttimeoutlen = 0.05
        return session

    async def run_loop(self) -> None:
        from prompt_toolkit.formatted_text import HTML

        prompt = HTML("<b>&gt; </b>")
        continuation = ". "
        while self._running:
            try:
                text = await self._session.prompt_async(
                    prompt,
                    default=self._preserved,
                    prompt_continuation=continuation,
                )
            except EOFError:
                self._repl.request_exit()
                return
            except KeyboardInterrupt:
                self._preserved = ""
                continue

            if text is _SUSPEND:
                self._suspended.set()
                await self._resume.wait()
                self._resume.clear()
                continue

            self._preserved = ""
            if not text.strip():
                continue
            if self._repl.turn_active:
                self._repl.notice_queued(text)
            await self._repl._queue.put(text)

    async def run_modal(self, coro: Awaitable[Any]) -> Any:
        """Suspend the prompt, run ``coro``, then resume with buffer preserved."""
        app = self._session.app
        if app.is_running:
            self._preserved = app.current_buffer.text
            self._suspended.clear()
            app.exit(result=_SUSPEND)
            await self._suspended.wait()
        try:
            return await coro
        finally:
            self._resume.set()
