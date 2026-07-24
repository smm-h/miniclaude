"""The canonical fake session: both the unit-test double and the live mock mode.

:class:`MockSession` implements the exact session surface the REPL duck-types
(``send``/``interrupt``/``set_model``/``respond_*``/``get_context_usage`` plus the
``model_name``/``permission_mode``/``total_cost_usd``/``total_tokens``/``turn_count``
properties). It runs in two modes selected by the constructor:

- **Scripted mode** (``MockSession()`` or ``MockSession(turns=[...])``): ``send()``
  replays pre-built event lists, one list per turn. This is the driver for the
  ``_repl`` unit tests -- no randomness, no timing, deterministic events.
- **Live mode** (``MockSession(seed)`` with an integer seed): ``send()`` parses a
  mock command from the prompt and fabricates a realistic event stream -- text
  deltas, tool activity, permission dialogs, status events, and a closing
  ``Result``. All randomness flows through a single seeded ``random.Random`` so a
  given seed reproduces byte-identical output. This powers ``miniclaude mock``,
  the offline TUI test harness.

The event-constructor helpers (:func:`text_delta`, :func:`thinking_delta`,
:func:`result_event`, :func:`perm_request`) are shared by both the scripted tests
and this module's live generators.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any

from claudestream import (
    ApiRetry,
    BudgetThreshold,
    CompactBoundary,
    ContextCategory,
    ContextUsage,
    PermissionRequest,
    RateLimit,
    Result,
    StreamDelta,
    SystemInit,
    ToolResult,
    ToolUse,
)

# --- Event constructors (shared by scripted tests and live generators) --------


def text_delta(text: str) -> StreamDelta:
    """A top-level ``text_delta`` streaming event carrying ``text``."""
    return StreamDelta(
        type="stream_event", event={"delta": {"type": "text_delta", "text": text}}
    )


def thinking_delta(text: str) -> StreamDelta:
    """A top-level ``thinking_delta`` streaming event carrying ``text``."""
    return StreamDelta(
        type="stream_event", event={"delta": {"type": "thinking_delta", "thinking": text}}
    )


def result_event(**kw: Any) -> Result:
    """A minimal successful ``Result`` with overridable fields."""
    base = dict(type="result", subtype="success", total_cost_usd=0.0, num_turns=1)
    base.update(kw)
    return Result(**base)


def perm_request(**kw: Any) -> PermissionRequest:
    """A ``PermissionRequest`` defaulting to a Bash ``ls`` with no suggestions."""
    base = dict(
        type="control_request",
        request_id="p1",
        tool_name="Bash",
        tool_input={"command": "ls"},
        permission_suggestions=[],
    )
    base.update(kw)
    return PermissionRequest(**base)


# --- Mock content vocabularies -----------------------------------------------

# Lorem-style filler drawn on for table cells, list items, and prose bodies.
_LOREM = [
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
    "sed", "eiusmod", "tempor", "incididunt", "labore", "magna", "aliqua", "enim",
    "minim", "veniam", "quis", "nostrud", "ullamco", "laboris", "aliquip", "commodo",
    "consequat", "duis", "aute", "irure", "reprehenderit", "voluptate", "velit",
    "esse", "cillum", "fugiat", "nulla", "pariatur", "excepteur", "occaecat",
    "cupidatat", "proident", "culpa", "officia", "deserunt", "mollit", "animus",
]

# Display-width stress samples: emoji (wide), CJK (wide), and combining sequences.
_WIDE_SAMPLES = [
    "😀", "🚀", "🎉", "🧪", "🌍",
    "日本語", "中文字", "한국어", "繁體",
    "é", "ä", "ñ", "ôü",
]

# The mock command catalogue, listed by `help` and on unknown input.
_MOCK_COMMANDS = [
    ("table", "markdown table with random dimensions and cell contents"),
    ("wide", "table salted with emoji/CJK/combining chars (width stress test)"),
    ("text", "multi-paragraph markdown: headers, lists, code, bold, inline code"),
    ("thinking", "a few thinking deltas followed by a short answer"),
    ("tools", "tool use/result pairs incl. an error and a subagent case"),
    ("dialogs", "a permission prompt and an AskUserQuestion, awaiting your answer"),
    ("status", "ApiRetry, BudgetThreshold, and CompactBoundary status events"),
    ("slow", "a long, slow stream for interrupt / type-ahead / resize testing"),
    ("md <markdown>", "stream the given markdown back verbatim"),
    ("demo", "run every section above in one turn"),
    ("help", "list these commands"),
]

# Fabricated context window for the live-mode Result.model_usage entry.
_MOCK_CONTEXT_WINDOW = 200_000

# Fixed offset that seeds the DEDICATED rate-limit RNG independently of the
# content RNG, so rate-limit jitter never perturbs generated content.
_RL_RNG_OFFSET = 0x5A17

# Fixed plausible-future base (year 2027) for seed-derived resets_at values.
# Fully deterministic -- no wall-clock dependence, so tests reproduce per seed.
_RL_EPOCH = 1_800_000_000

# Per-rate-limit-type rise parameters: (type, base, per-turn step, reset window).
# five_hour rises faster than seven_day. reset window bounds the seed-derived
# offset so five_hour resets sooner than seven_day.
_RL_SPECS = (
    ("five_hour", 0.10, 0.06, 18_000),
    ("seven_day", 0.05, 0.02, 604_800),
)


class MockSession:
    """Fake session implementing the full REPL-facing surface.

    Construct with an integer ``seed`` for live mock mode, or with no seed (and an
    optional ``turns`` list) for scripted test mode. See the module docstring.
    """

    def __init__(self, seed: int | None = None, *, turns: list | None = None) -> None:
        self._live = seed is not None
        self._seed = seed if self._live else 0
        self._rng = random.Random(self._seed)
        # Dedicated RNG for rate-limit jitter -- kept separate from the content
        # RNG so emitting rate limits never shifts generated content.
        self._rl_rng = random.Random(self._seed + _RL_RNG_OFFSET)
        self._turns = list(turns or [])

        self._first_send = True
        self._interrupted = False
        self._context_tokens = 0
        self._pending_response: asyncio.Future | None = None

        # Recorded for test assertions (scripted mode) and harmless in live mode.
        self.sent: list[str] = []
        self.calls: list[tuple] = []
        self.interrupt_count = 0

        self.permission_mode = "default"
        if self._live:
            self.model_name = "claude-mock"
            self.total_cost_usd = 0.0
            self.total_tokens = 0
            self.turn_count = 0
        else:
            # Defaults the _repl unit tests assert against.
            self.model_name = "haiku"
            self.total_cost_usd = 0.5
            self.total_tokens = 1234
            self.turn_count = 3

    # --- Async context manager ---

    async def __aenter__(self) -> "MockSession":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    # --- Turn entry point ---

    def send(self, prompt: str, *, raw: bool = False):
        """Return an async iterator of events for this turn.

        Scripted mode replays the next scripted event list; live mode parses a
        mock command and fabricates a stream.
        """
        self.sent.append(prompt)
        if not self._live:
            events = self._turns.pop(0) if self._turns else []

            async def _replay():
                for event in events:
                    yield event

            return _replay()
        return self._live_send(prompt)

    async def _live_send(self, prompt: str):
        start = time.monotonic()
        self._interrupted = False
        if self._first_send:
            self._first_send = False
            yield SystemInit(
                type="system",
                cwd=os.getcwd(),
                model="claude-mock",
                permission_mode="default",
                session_id=f"mock-{self._seed}",
            )
        parts = (prompt or "").split(maxsplit=1)
        cmd = parts[0] if parts else ""
        arg = parts[1] if len(parts) > 1 else ""
        async for event in self._dispatch(cmd, arg):
            yield event
        for rl in self._rate_limit_events():
            yield rl
        yield self._make_result(start)

    def _rate_limit_events(self) -> list:
        """Two seed-derived RateLimit events emitted before each turn's Result.

        Utilization rises slowly with the turn count (five_hour faster than
        seven_day) plus small jitter drawn from the DEDICATED rate-limit RNG --
        never the content RNG -- so a given seed reproduces the same rate-limit
        sequence AND the same content. resets_at is fully seed+turn derived, so
        there is no wall-clock dependence. status crosses to "allowed_warning"
        past 0.8 utilization.
        """
        turn = self.turn_count
        events = []
        for rate_type, base, step, window in _RL_SPECS:
            jitter = self._rl_rng.uniform(-0.01, 0.01)
            util = max(0.0, min(0.99, base + turn * step + jitter))
            status = "allowed_warning" if util >= 0.8 else "allowed"
            resets_at = _RL_EPOCH + (self._seed % window) + turn * 300
            events.append(
                RateLimit(
                    type="system",
                    status=status,
                    rate_limit_type=rate_type,
                    utilization=util,
                    resets_at=resets_at,
                )
            )
        return events

    def _dispatch(self, cmd: str, arg: str):
        """Map a mock command word to its event generator (case-sensitive)."""
        if cmd == "table":
            return self._gen_table(salt=False)
        if cmd == "wide":
            return self._gen_table(salt=True)
        if cmd == "text":
            return self._gen_text()
        if cmd == "thinking":
            return self._gen_thinking()
        if cmd == "tools":
            return self._gen_tools()
        if cmd == "dialogs":
            return self._gen_dialogs()
        if cmd == "status":
            return self._gen_status()
        if cmd == "slow":
            return self._gen_slow()
        if cmd == "md":
            return self._gen_md(arg)
        if cmd == "demo":
            return self._gen_demo()
        if cmd == "help":
            return self._gen_help()
        return self._gen_unknown(cmd)

    # --- Session surface (state mutators + responders) ---

    async def interrupt(self, *, timeout: float = 30.0) -> list:
        self.interrupt_count += 1
        self._interrupted = True
        return []

    async def set_model(self, model: str) -> None:
        self.calls.append(("set_model", model))
        self.model_name = model

    async def set_permission_mode(self, mode: str) -> None:
        self.calls.append(("set_mode", mode))
        self.permission_mode = mode

    async def get_context_usage(self, *, timeout: float = 30.0) -> ContextUsage:
        self.calls.append(("get_context_usage",))
        return ContextUsage(
            total_tokens=100,
            max_tokens=1000,
            percentage=10.0,
            categories=[
                ContextCategory(name="system", tokens=60),
                ContextCategory(name="messages", tokens=40),
            ],
        )

    async def respond_allow(
        self, request_id: str, updated_input: dict, *, updated_permissions: list | None = None
    ) -> None:
        resp = ("allow", request_id, updated_input, updated_permissions)
        self.calls.append(resp)
        self._resolve(resp)

    async def respond_deny(self, request_id: str, message: str = "Denied by user") -> None:
        resp = ("deny", request_id, message)
        self.calls.append(resp)
        self._resolve(resp)

    async def respond_dialog_cancelled(self, request_id: str) -> None:
        resp = ("cancelled", request_id)
        self.calls.append(resp)
        self._resolve(resp)

    def _resolve(self, resp: tuple) -> None:
        """Unblock the dialogs generator awaiting a user response, if any."""
        fut = self._pending_response
        if fut is not None and not fut.done():
            fut.set_result(resp)

    # --- Result fabrication (live mode) ---

    def _make_result(self, start: float) -> Result:
        """Fabricate the closing Result, accumulating cost/tokens/context."""
        self.turn_count += 1
        self.total_cost_usd = round(self.total_cost_usd + self._rng.uniform(0.001, 0.01), 6)
        self.total_tokens += self._rng.randint(200, 2000)
        self._context_tokens += self._rng.randint(1500, 6000)
        return Result(
            type="result",
            subtype="success",
            total_cost_usd=self.total_cost_usd,
            duration_ms=(time.monotonic() - start) * 1000.0,
            num_turns=self.turn_count,
            model_usage={
                "claude-mock": {
                    "contextWindow": _MOCK_CONTEXT_WINDOW,
                    "inputTokens": self._context_tokens,
                    "cacheReadInputTokens": 0,
                    "cacheCreationInputTokens": 0,
                }
            },
        )

    # --- Text streaming ---

    async def _stream_text(self, text: str, delay: float = 0.01):
        """Emit ``text`` as text deltas split at random 8-32 char boundaries.

        Checks the interrupt flag between chunks so :meth:`interrupt` genuinely
        stops the stream mid-turn.
        """
        i, n = 0, len(text)
        while i < n:
            if self._interrupted:
                return
            size = self._rng.randint(8, 32)
            chunk = text[i : i + size]
            i += size
            yield text_delta(chunk)
            await asyncio.sleep(delay)

    # --- Content builders (pure, seeded) ---

    def _word(self) -> str:
        return self._rng.choice(_LOREM)

    def _cell(self) -> str:
        """One table cell: either a single word or a 4-12 word sentence."""
        if self._rng.random() < 0.5:
            return self._word()
        n = self._rng.randint(4, 12)
        return " ".join(self._word() for _ in range(n))

    def _salt(self, cell: str) -> str:
        """Prepend or append a display-width stress sample to a cell."""
        extra = self._rng.choice(_WIDE_SAMPLES)
        if self._rng.random() < 0.5:
            return f"{extra} {cell}"
        return f"{cell} {extra}"

    def _make_table(self, salt: bool) -> str:
        """Build a markdown table: 5-10 columns by 5-10 rows."""
        cols = self._rng.randint(5, 10)
        rows = self._rng.randint(5, 10)
        header = [self._word() for _ in range(cols)]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in range(cols)) + " |",
        ]
        for _ in range(rows):
            row = []
            for _ in range(cols):
                cell = self._cell()
                row.append(self._salt(cell) if salt else cell)
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines) + "\n"

    def _text_body(self) -> str:
        w = self._word
        return (
            f"# {w().capitalize()} {w()}\n\n"
            f"A paragraph with **{w()} {w()}** and inline `code_{w()}` spans.\n\n"
            f"- {w()} {w()}\n- {w()} {w()}\n- {w()} {w()}\n\n"
            f"1. {w()} {w()}\n2. {w()} {w()}\n3. {w()} {w()}\n\n"
            f"```python\ndef {w()}():\n    return \"{w()}\"\n```\n"
        )

    def _command_list_md(self) -> str:
        body = "".join(f"- `{name}` — {desc}\n" for name, desc in _MOCK_COMMANDS)
        return "## Mock commands\n\n" + body

    def _ack_text(self, resp: tuple) -> str:
        """A short acknowledgment reflecting the user's dialog choice."""
        kind = resp[0]
        if kind == "allow":
            updated = resp[2] if len(resp) > 2 else {}
            if isinstance(updated, dict) and "answers" in updated:
                return f"You answered: {updated['answers']}\n"
            return "Approved — running the command.\n"
        if kind == "deny":
            return f"Denied: {resp[2] if len(resp) > 2 else ''}\n"
        return "Dialog cancelled.\n"

    # --- Per-command generators ---

    async def _gen_table(self, salt: bool):
        async for event in self._stream_text(self._make_table(salt=salt)):
            yield event

    async def _gen_text(self):
        async for event in self._stream_text(self._text_body()):
            yield event

    async def _gen_thinking(self):
        for _ in range(self._rng.randint(3, 5)):
            n = self._rng.randint(6, 12)
            yield thinking_delta(" ".join(self._word() for _ in range(n)) + "\n")
            await asyncio.sleep(0.01)
        async for event in self._stream_text("Done thinking — here is a short answer.\n"):
            yield event

    async def _gen_tools(self):
        yield ToolUse(
            type="assistant", tool_use_id="mock-t1", name="Read",
            input={"file_path": "/mock/example.py"},
        )
        yield ToolResult(
            type="user", tool_use_id="mock-t1", content="file read ok",
            tool_name="Read", is_error=False,
        )
        async for event in self._stream_text("Read succeeded; running a command.\n"):
            yield event
        yield ToolUse(
            type="assistant", tool_use_id="mock-t2", name="Bash",
            input={"command": "false"},
        )
        yield ToolResult(
            type="user", tool_use_id="mock-t2", content="command failed",
            tool_name="Bash", is_error=True,
        )
        async for event in self._stream_text("That failed; delegating to a subagent.\n"):
            yield event
        yield ToolUse(
            type="assistant", tool_use_id="mock-t3", name="Task",
            input={"description": "sub task", "prompt": "do sub work"},
            parent_tool_use_id="mock-parent",
        )
        yield ToolResult(
            type="user", tool_use_id="mock-t3", content="subagent done",
            tool_name="Task", is_error=False, parent_tool_use_id="mock-parent",
        )

    async def _gen_status(self):
        async for event in self._stream_text("Retrying the API call.\n"):
            yield event
        yield ApiRetry(type="api_retry", attempt=2, max_retries=10)
        async for event in self._stream_text("Crossing a budget threshold.\n"):
            yield event
        yield BudgetThreshold(metric="cost", threshold=1.0, current_value=1.25)
        async for event in self._stream_text("Compacting the conversation.\n"):
            yield event
        yield CompactBoundary(type="system")
        async for event in self._stream_text("Recovered; continuing.\n"):
            yield event

    async def _gen_slow(self, lines: int | None = None):
        if lines is None:
            lines = self._rng.randint(15, 25)
        paras = []
        for i in range(lines):
            n = self._rng.randint(8, 16)
            paras.append(f"{i + 1}. " + " ".join(self._word() for _ in range(n)))
        async for event in self._stream_text("\n".join(paras) + "\n", delay=0.08):
            yield event

    async def _gen_md(self, markdown: str):
        async for event in self._stream_text(markdown):
            yield event

    async def _gen_help(self):
        async for event in self._stream_text(self._command_list_md()):
            yield event

    async def _gen_unknown(self, cmd: str):
        text = f"Unknown command: `{cmd}`\n\n" + self._command_list_md()
        async for event in self._stream_text(text):
            yield event

    async def _arm_response(self) -> asyncio.Future:
        """Install a fresh future the next respond_* call will resolve."""
        fut = asyncio.get_running_loop().create_future()
        self._pending_response = fut
        return fut

    async def _gen_dialogs(self):
        # A Bash permission prompt: yield it, then block on the user's response.
        perm = perm_request(
            request_id="mock-perm-1",
            tool_name="Bash",
            tool_input={"command": "echo mock"},
            permission_suggestions=[
                {"rules": [{"toolName": "Bash", "ruleContent": "echo mock"}]}
            ],
        )
        fut = await self._arm_response()
        yield perm
        resp = await fut
        self._pending_response = None
        async for event in self._stream_text(self._ack_text(resp)):
            yield event

        # An AskUserQuestion (arrives as a PermissionRequest on the wire).
        ask = perm_request(
            request_id="mock-ask-1",
            tool_name="AskUserQuestion",
            tool_input={
                "questions": [
                    {
                        "question": "Which color do you prefer?",
                        "options": [
                            {"label": "Red"},
                            {"label": "Blue"},
                            {"label": "Green"},
                        ],
                        "multiSelect": False,
                    }
                ]
            },
            permission_suggestions=[],
        )
        fut2 = await self._arm_response()
        yield ask
        resp2 = await fut2
        self._pending_response = None
        async for event in self._stream_text(self._ack_text(resp2)):
            yield event

    async def _gen_demo(self):
        sections = [
            ("Text", self._gen_text()),
            ("Thinking", self._gen_thinking()),
            ("Table", self._gen_table(salt=False)),
            ("Wide", self._gen_table(salt=True)),
            ("Tools", self._gen_tools()),
            ("Status", self._gen_status()),
        ]
        for title, gen in sections:
            async for event in self._stream_text(f"\n## {title}\n\n"):
                yield event
            async for event in gen:
                yield event
        # Dialogs section (inline yield + await, like _gen_dialogs).
        async for event in self._stream_text("\n## Dialogs\n\n"):
            yield event
        async for event in self._gen_dialogs():
            yield event
        # A shortened slow segment (~3s at 0.08s/chunk).
        async for event in self._stream_text("\n## Slow\n\n"):
            yield event
        async for event in self._gen_slow(lines=8):
            yield event
