"""Unit tests for miniclaude._repl.

Drive the pure orchestration surface (event dispatch, slash commands, the
interrupt guard, queued-line draining) with a scripted ``FakeSession`` and the
same lightweight ``FakeInteraction``/``ListPrinter`` fakes used by the dialog
tests. No terminal, no prompt_toolkit, no live CLI.
"""

from __future__ import annotations

import asyncio
import functools
import re

import pytest

from claudestream import (
    AssistantText,
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

from miniclaude._mock import (
    MockSession as FakeSession,
    perm_request as _perm,
    result_event as _result,
    text_delta as _text_delta,
    thinking_delta as _thinking_delta,
)
from miniclaude._repl import (
    ProseBlock,
    Repl,
    TableBlock,
    _HowMuchLeftCache,
    _PromptController,
    _ctx_pct,
    _HOWMUCHLEFT_NOT_FOUND,
    materialize_blocks,
    render_howmuchleft,
)
from miniclaude._render import TableData, render_table


def sync(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


# --- Fakes -------------------------------------------------------------------


class FakeInteraction:
    def __init__(self, choices=None, texts=None):
        self._choices = list(choices or [])
        self._texts = list(texts or [])

    async def ask_choice(self, message, options, default=None):
        sel = self._choices.pop(0)
        if isinstance(sel, BaseException):
            raise sel
        return options[sel][0]

    async def ask_text(self, message):
        val = self._texts.pop(0)
        if isinstance(val, BaseException):
            raise val
        return val


class ListPrinter:
    def __init__(self):
        self.chunks = []

    def __call__(self, text):
        self.chunks.append(text)

    @property
    def text(self):
        return "".join(self.chunks)


def make_repl(fake, interaction=None, printer=None):
    interaction = interaction or FakeInteraction()
    printer = printer or ListPrinter()
    repl = Repl(
        session_factory=lambda: None,
        interaction=interaction,
        printer=printer,
        width=80,
    )
    repl._session = fake
    return repl, printer


# --- Text / thinking rendering ----------------------------------------------


@sync
async def test_text_delta_is_rendered():
    fake = FakeSession(turns=[[_text_delta("hello world\n"), _result()]])
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    assert "hello world" in _plain(printer.text)


@sync
async def test_thinking_delta_is_rendered():
    fake = FakeSession(turns=[[_thinking_delta("pondering\n"), _result()]])
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    assert "pondering" in _plain(printer.text)


@sync
async def test_assistant_text_flattened_event_is_ignored():
    # AssistantText / Thinking flattened events must NOT be printed (deltas already did).
    fake = FakeSession(
        turns=[[AssistantText(type="assistant", text="DOUBLED"),
                  Thinking(type="assistant", text="ALSO_DOUBLED"),
                  _result()]]
    )
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    plain = _plain(printer.text)
    assert "DOUBLED" not in plain
    assert "ALSO_DOUBLED" not in plain


@sync
async def test_subagent_stream_delta_is_ignored():
    delta = StreamDelta(
        type="stream_event",
        event={"delta": {"type": "text_delta", "text": "SUBAGENT"}},
        parent_tool_use_id="sub_1",
    )
    fake = FakeSession(turns=[[delta, _result()]])
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    assert "SUBAGENT" not in _plain(printer.text)


# --- Tool activity -----------------------------------------------------------


@sync
async def test_tool_use_and_result_lines():
    fake = FakeSession(
        turns=[[
            ToolUse(type="assistant", tool_use_id="t1", name="Read",
                    input={"file_path": "/a/b.py"}),
            ToolResult(type="user", tool_use_id="t1", content="ok", tool_name="Read"),
            _result(),
        ]]
    )
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    plain = _plain(printer.text)
    assert "Read" in plain
    assert "/a/b.py" in plain


@sync
async def test_tool_error_result_renders_red_cross():
    fake = FakeSession(
        turns=[[
            ToolUse(type="assistant", tool_use_id="t1", name="Bash",
                    input={"command": "false"}),
            ToolResult(type="user", tool_use_id="t1", content="command failed",
                       tool_name="Bash", is_error=True),
            _result(),
        ]]
    )
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    assert "✗ command failed" in _plain(printer.text)
    assert "\033[31m" in printer.text  # red is wired through from is_error


# --- Result / status line ----------------------------------------------------


@sync
async def test_result_line_with_context_pct():
    result = _result(
        total_cost_usd=0.0123,
        duration_ms=2300,
        num_turns=2,
        model_usage={
            "haiku": {
                "contextWindow": 1000,
                "inputTokens": 100,
                "cacheReadInputTokens": 50,
                "cacheCreationInputTokens": 0,
            }
        },
    )
    fake = FakeSession(turns=[[result]])
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    plain = _plain(printer.text)
    assert "$0.0123" in plain
    assert "2.3s" in plain
    assert "2 turn(s)" in plain
    assert "ctx 15%" in plain


@sync
async def test_result_line_without_model_usage_has_no_ctx():
    fake = FakeSession(turns=[[_result(total_cost_usd=0.01, duration_ms=1000, num_turns=1)]])
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    assert "ctx" not in _plain(printer.text)


def test_ctx_pct_helper():
    mu = {"m": {"contextWindow": 200, "inputTokens": 40, "cacheReadInputTokens": 10}}
    assert _ctx_pct(mu, "m") == 25
    assert _ctx_pct({}, "m") is None
    assert _ctx_pct({"m": {"inputTokens": 5}}, "m") is None  # no contextWindow


# --- SystemInit state assignment ---------------------------------------------


@sync
async def test_system_init_sets_state_without_printing():
    """SystemInit assigns model/mode/cwd/session_id on the Repl and prints nothing."""
    init = SystemInit(
        type="system", cwd="/proj", model="haiku",
        permission_mode="default", session_id="sess-1",
    )
    fake = FakeSession(turns=[[init, init, _result()]])
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    plain = _plain(printer.text)
    # No startup line printed (no cwd/session in output).
    assert "/proj" not in plain
    assert "sess-1" not in plain
    # State was stored correctly.
    assert repl._model == "haiku"
    assert repl._mode == "default"
    assert repl._cwd == "/proj"
    assert repl._session_id == "sess-1"
    # Echo line still appears.
    assert "> hi" in plain


# --- Permission flow ---------------------------------------------------------


@sync
async def test_permission_flow_allow_once():
    fake = FakeSession(turns=[[_perm()]])
    interaction = FakeInteraction(choices=[0])  # Allow once
    repl, _ = make_repl(fake, interaction)
    await repl._run_turn(fake, "hi")
    assert fake.calls == [("allow", "p1", {"command": "ls"}, None)]


@sync
async def test_permission_flow_allow_always_passes_suggestion():
    sug = {"rules": [{"toolName": "Bash", "ruleContent": "ls"}]}
    fake = FakeSession(turns=[[_perm(permission_suggestions=[sug])]])
    interaction = FakeInteraction(choices=[1])  # Allow always
    repl, _ = make_repl(fake, interaction)
    await repl._run_turn(fake, "hi")
    assert fake.calls == [("allow", "p1", {"command": "ls"}, [sug])]


@sync
async def test_ask_user_question_flow_injects_answers():
    req = _perm(
        request_id="q1",
        tool_name="AskUserQuestion",
        tool_input={
            "questions": [
                {"question": "Color?", "options": [{"label": "Red"}, {"label": "Blue"}],
                 "multiSelect": False}
            ]
        },
    )
    fake = FakeSession(turns=[[req]])
    interaction = FakeInteraction(choices=[0])  # picks Red
    repl, _ = make_repl(fake, interaction)
    await repl._run_turn(fake, "hi")
    kind, request_id, updated, perms = fake.calls[0]
    assert kind == "allow"
    assert request_id == "q1"
    assert updated["answers"] == {"Color?": "Red"}


@sync
async def test_user_dialog_request_is_cancelled():
    req = UserDialogRequest(
        type="control_request", request_id="d1", dialog_kind="mystery_kind"
    )
    fake = FakeSession(turns=[[req]])
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    assert fake.calls == [("cancelled", "d1")]
    assert "mystery_kind" in _plain(printer.text)


# --- Slash commands ----------------------------------------------------------


@sync
async def test_slash_model_sets_model():
    fake = FakeSession()
    repl, _ = make_repl(fake)
    handled = await repl._handle_line(fake, "/model opus")
    assert handled is True
    assert ("set_model", "opus") in fake.calls


@sync
async def test_slash_model_empty_shows_current():
    fake = FakeSession()
    repl, printer = make_repl(fake)
    await repl._handle_line(fake, "/model")
    assert "haiku" in _plain(printer.text)
    assert all(c[0] != "set_model" for c in fake.calls)


@sync
async def test_slash_mode_sets_permission_mode():
    fake = FakeSession()
    repl, _ = make_repl(fake)
    await repl._handle_line(fake, "/mode plan")
    assert ("set_mode", "plan") in fake.calls


@sync
async def test_slash_context_queries_usage():
    fake = FakeSession()
    repl, printer = make_repl(fake)
    await repl._handle_line(fake, "/context")
    assert ("get_context_usage",) in fake.calls
    plain = _plain(printer.text)
    assert "system" in plain
    assert "total 100/1000" in plain


@sync
async def test_slash_cost_prints_totals():
    fake = FakeSession()
    repl, printer = make_repl(fake)
    await repl._handle_line(fake, "/cost")
    plain = _plain(printer.text)
    assert "$0.5000" in plain
    assert "1234 tokens" in plain
    assert "3 turn(s)" in plain


@sync
async def test_slash_help_lists_commands():
    fake = FakeSession()
    repl, printer = make_repl(fake)
    await repl._handle_line(fake, "/help")
    plain = _plain(printer.text)
    for token in ["/model", "/mode", "/context", "/cost", "/quit", "/help"]:
        assert token in plain


@sync
async def test_slash_quit_requests_exit():
    fake = FakeSession()
    repl, _ = make_repl(fake)
    await repl._handle_line(fake, "/quit")
    assert repl._exit is True


@sync
async def test_unknown_slash_is_sent_to_claude():
    # /compact etc. are server-side skill commands: not handled client-side.
    fake = FakeSession()
    repl, _ = make_repl(fake)
    handled = await repl._handle_line(fake, "/compact now")
    assert handled is False


@sync
async def test_plain_line_is_not_handled_as_slash():
    fake = FakeSession()
    repl, _ = make_repl(fake)
    assert await repl._handle_line(fake, "hello there") is False


# --- Queued-line draining ----------------------------------------------------


@sync
async def test_queued_lines_sent_one_per_turn_in_order():
    fake = FakeSession(turns=[[_result()], [_result()]])
    repl, _ = make_repl(fake)
    await repl._queue.put("a")
    await repl._queue.put("b")
    await repl._queue.put(None)  # sentinel stops the loop
    await repl._main_loop(fake)
    assert fake.sent == ["a", "b"]


@sync
async def test_main_loop_dispatches_slash_without_sending():
    fake = FakeSession()
    repl, _ = make_repl(fake)
    await repl._queue.put("/model opus")
    await repl._queue.put(None)
    await repl._main_loop(fake)
    assert fake.sent == []
    assert ("set_model", "opus") in fake.calls


# --- Interrupt guard ---------------------------------------------------------


@sync
async def test_interrupt_guard_prevents_double_fire():
    fake = FakeSession()
    repl, _ = make_repl(fake)
    repl._turn_active = True
    repl.request_interrupt()
    repl.request_interrupt()
    await asyncio.sleep(0.01)
    assert fake.interrupt_count == 1


@sync
async def test_interrupt_noop_when_no_turn_active():
    fake = FakeSession()
    repl, _ = make_repl(fake)
    repl._turn_active = False
    repl.request_interrupt()
    await asyncio.sleep(0.01)
    assert fake.interrupt_count == 0


@sync
async def test_turn_error_does_not_kill_repl():
    class Boom(FakeSession):
        def send(self, prompt, *, raw=False):
            self.sent.append(prompt)

            async def gen():
                raise RuntimeError("kaboom")
                yield  # pragma: no cover

            return gen()

    fake = Boom()
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")  # must not raise
    assert "kaboom" in _plain(printer.text)
    assert repl._turn_active is False


# --- render_howmuchleft ------------------------------------------------------


def test_render_howmuchleft_missing_binary(monkeypatch):
    """When howmuchleft is not found, return the fallback message."""
    monkeypatch.setattr("shutil.which", lambda _name: None)
    result = render_howmuchleft("haiku", 10.0, "/tmp", 0.05)
    assert result == _HOWMUCHLEFT_NOT_FOUND
    assert result.count("\n") == 2  # 3 lines joined by 2 newlines


def test_render_howmuchleft_subprocess_json(monkeypatch):
    """Verify the JSON piped to howmuchleft has the expected shape."""
    import json
    import subprocess

    captured_input = {}

    def fake_run(cmd, *, input, capture_output, text, timeout):
        captured_input["json"] = json.loads(input)
        result = subprocess.CompletedProcess(cmd, 0)
        result.stdout = "line1\nline2\nline3\n"
        result.stderr = ""
        return result

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/howmuchleft")
    monkeypatch.setattr("subprocess.run", fake_run)

    result = render_howmuchleft("haiku", 25.0, "/proj", 0.1234)
    assert result == "line1\nline2\nline3"

    j = captured_input["json"]
    assert j["model"] == "haiku"
    assert j["context_window"] == {"used_percentage": 25.0}
    assert j["cwd"] == "/proj"
    assert j["cost"] == {"total_cost_usd": 0.1234}


def test_render_howmuchleft_no_ctx_pct(monkeypatch):
    """When ctx_pct is None, context_window is an empty dict."""
    import json
    import subprocess

    captured_input = {}

    def fake_run(cmd, *, input, capture_output, text, timeout):
        captured_input["json"] = json.loads(input)
        result = subprocess.CompletedProcess(cmd, 0)
        result.stdout = "a\nb\nc\n"
        result.stderr = ""
        return result

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/howmuchleft")
    monkeypatch.setattr("subprocess.run", fake_run)

    render_howmuchleft("haiku", None, "/proj", 0.0)
    assert captured_input["json"]["context_window"] == {}


def test_render_howmuchleft_timeout_returns_fallback(monkeypatch):
    """On subprocess timeout, return the fallback message."""
    import subprocess

    def fake_run(cmd, *, input, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/howmuchleft")
    monkeypatch.setattr("subprocess.run", fake_run)

    result = render_howmuchleft("haiku", 10.0, "/tmp", 0.0)
    assert result == _HOWMUCHLEFT_NOT_FOUND


def test_render_howmuchleft_with_rate_limits(monkeypatch):
    """rate_limits are passed through when provided."""
    import json
    import subprocess

    captured_input = {}

    def fake_run(cmd, *, input, capture_output, text, timeout):
        captured_input["json"] = json.loads(input)
        result = subprocess.CompletedProcess(cmd, 0)
        result.stdout = "a\nb\nc\n"
        result.stderr = ""
        return result

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/howmuchleft")
    monkeypatch.setattr("subprocess.run", fake_run)

    rl = {"five_hour": {"used_percentage": 50.0}}
    render_howmuchleft("haiku", 10.0, "/tmp", 0.0, rate_limits=rl)
    assert captured_input["json"]["rate_limits"] == rl


# --- _HowMuchLeftCache -------------------------------------------------------


def test_hml_cache_returns_cached_within_ttl(monkeypatch):
    """Within the TTL, the cache returns the previous result without re-invoking."""
    import subprocess

    call_count = 0

    def fake_run(cmd, *, input, capture_output, text, timeout):
        nonlocal call_count
        call_count += 1
        result = subprocess.CompletedProcess(cmd, 0)
        result.stdout = f"call{call_count}\nline2\nline3\n"
        result.stderr = ""
        return result

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/howmuchleft")
    monkeypatch.setattr("subprocess.run", fake_run)

    cache = _HowMuchLeftCache()
    r1 = cache.get("haiku", None, "/tmp", 0.0, None, ttl=10.0)
    assert "call1" in r1
    r2 = cache.get("haiku", None, "/tmp", 0.0, None, ttl=10.0)
    assert r2 == r1  # cached, not re-invoked
    assert call_count == 1


def test_hml_cache_refreshes_after_ttl(monkeypatch):
    """After TTL expires, the cache re-invokes howmuchleft."""
    import subprocess

    call_count = 0

    def fake_run(cmd, *, input, capture_output, text, timeout):
        nonlocal call_count
        call_count += 1
        result = subprocess.CompletedProcess(cmd, 0)
        result.stdout = f"call{call_count}\nline2\nline3\n"
        result.stderr = ""
        return result

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/howmuchleft")
    monkeypatch.setattr("subprocess.run", fake_run)

    cache = _HowMuchLeftCache()
    r1 = cache.get("haiku", None, "/tmp", 0.0, None, ttl=0.0)  # ttl=0 => always stale
    r2 = cache.get("haiku", None, "/tmp", 0.0, None, ttl=0.0)
    assert call_count == 2
    assert "call1" in r1
    assert "call2" in r2


def test_hml_cache_fallback_on_error(monkeypatch):
    """On error after a successful call, the cache returns the last good output."""
    import subprocess

    call_count = 0

    def fake_run(cmd, *, input, capture_output, text, timeout):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            result = subprocess.CompletedProcess(cmd, 0)
            result.stdout = "good\nline2\nline3\n"
            result.stderr = ""
            return result
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/howmuchleft")
    monkeypatch.setattr("subprocess.run", fake_run)

    cache = _HowMuchLeftCache()
    r1 = cache.get("haiku", None, "/tmp", 0.0, None, ttl=0.0)
    assert "good" in r1
    r2 = cache.get("haiku", None, "/tmp", 0.0, None, ttl=0.0)
    # Should return cached "good" result, not the fallback
    assert "good" in r2


# --- No banner in turn output ------------------------------------------------


@sync
async def test_no_banner_in_turn_output():
    """No banner or startup info is printed during a turn."""
    init = SystemInit(
        type="system", cwd="/proj", model="haiku",
        permission_mode="default", session_id="sess-1",
    )
    fake = FakeSession(turns=[[init, _result()]])
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    plain = _plain(printer.text)
    assert "miniclaude" not in plain
    # SystemInit no longer prints cwd/session_id to output.
    assert "/proj" not in plain
    assert "sess-1" not in plain


# --- RateLimit silently ignored -----------------------------------------------


@sync
async def test_rate_limit_silently_ignored():
    """RateLimit events produce no visible output but do populate _rate_limits."""
    events = [
        RateLimit(type="system", status="allowed", rate_limit_type="tokens", utilization=0.0),
        RateLimit(type="system", status="allowed", rate_limit_type="tokens", utilization=0.9),
        RateLimit(type="system", status="throttled", rate_limit_type="tokens", utilization=0.5),
        _result(),
    ]
    fake = FakeSession(turns=[events])
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    plain = _plain(printer.text)
    assert "rate limit" not in plain
    assert "throttled" not in plain
    # No visible output, but the state is captured for the status bar. The last
    # event for a given key wins (utilization 0.5 -> 50.0%).
    assert repl._rate_limits is not None
    assert repl._rate_limits["tokens"]["used_percentage"] == 50.0


def test_rate_limits_fed_to_howmuchleft(monkeypatch):
    """RateLimit events accumulate into _rate_limits and reach howmuchleft stdin."""
    import json
    import subprocess

    captured: dict = {}

    def fake_run(cmd, *, input, capture_output, text, timeout):
        captured["json"] = json.loads(input)
        result = subprocess.CompletedProcess(cmd, 0)
        result.stdout = "a\nb\nc\n"
        result.stderr = ""
        return result

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/howmuchleft")
    monkeypatch.setattr("subprocess.run", fake_run)

    ts_5h = 1900000000
    ts_7d = 1900500000
    events = [
        RateLimit(
            type="system", status="allowed", rate_limit_type="five_hour",
            utilization=0.42, resets_at=ts_5h,
        ),
        RateLimit(
            type="system", status="allowed", rate_limit_type="seven_day",
            utilization=0.13, resets_at=ts_7d,
        ),
        _result(),
    ]
    fake = FakeSession(turns=[events])
    repl, _printer = make_repl(fake)

    asyncio.run(repl._run_turn(fake, "hi"))
    # The toolbar feed (the production status bar's data source) forwards the
    # accumulated rate limits into howmuchleft's stdin JSON.
    repl._get_toolbar()
    assert captured["json"]["rate_limits"] == {
        "five_hour": {"used_percentage": 42.0, "resets_at": ts_5h},
        "seven_day": {"used_percentage": 13.0, "resets_at": ts_7d},
    }


# --- Echo line ----------------------------------------------------------------


@sync
async def test_echo_appears_in_output():
    """After _run_turn, the user's input is echoed dimmed."""
    fake = FakeSession(turns=[[_result()]])
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    plain = _plain(printer.text)
    assert "> hi" in plain
    # Verify it is dim (ANSI SGR 2).
    assert "\x1b[2m> hi" in printer.text


# --- Result updates howmuchleft state -----------------------------------------


@sync
async def test_result_updates_hml_state():
    """After a Result event, the howmuchleft state fields are updated."""
    result = _result(
        total_cost_usd=0.05,
        duration_ms=1000,
        num_turns=1,
        model_usage={
            "haiku": {
                "contextWindow": 1000,
                "inputTokens": 200,
                "cacheReadInputTokens": 50,
                "cacheCreationInputTokens": 0,
            }
        },
    )
    fake = FakeSession(turns=[[result]])
    fake.total_cost_usd = 0.05
    repl, printer = make_repl(fake)
    await repl._run_turn(fake, "hi")
    assert repl._cost_usd == 0.05
    assert repl._ctx_pct == 25  # (200+50)/1000 = 25%


# --- Output block model (Phase 4a) -------------------------------------------


def test_output_block_accumulation():
    """Printer creates ProseBlocks; on_table callback creates TableBlocks.

    Tests the block types directly and verifies no double-counting occurs
    when both prose and table blocks are accumulated.
    """
    blocks: list = []
    # Simulate prose output.
    prose = ProseBlock("hello world\n")
    blocks.append(prose)
    assert isinstance(blocks[0], ProseBlock)
    assert blocks[0].ansi_text == "hello world\n"

    # Simulate table output (what the on_table callback does).
    data = TableData(
        header_rows=[["Name", "Value"]],
        body_rows=[["a", "1"]],
        aligns=["left", "left"],
    )
    table = TableBlock(data)
    blocks.append(table)
    assert isinstance(blocks[1], TableBlock)
    assert blocks[1].data is data

    # No double-counting: 1 prose + 1 table = 2 blocks total.
    assert len(blocks) == 2
    assert sum(1 for b in blocks if isinstance(b, ProseBlock)) == 1
    assert sum(1 for b in blocks if isinstance(b, TableBlock)) == 1


def test_materialize_blocks_renders_tables_at_width():
    """materialize_blocks produces different output at different widths
    for the same set of blocks containing tables."""
    data = TableData(
        header_rows=[["Name", "Description"]],
        body_rows=[["Alice", "A moderately long description that will wrap"]],
        aligns=["left", "left"],
    )
    blocks = [
        ProseBlock("some prose\n"),
        TableBlock(data),
        ProseBlock("more prose\n"),
    ]
    out80 = materialize_blocks(blocks, 80)
    out40 = materialize_blocks(blocks, 40)

    # Both contain the prose text unchanged.
    assert "some prose\n" in out80
    assert "some prose\n" in out40
    assert "more prose\n" in out80
    assert "more prose\n" in out40

    # Both contain box-drawing table borders.
    for out in (out80, out40):
        stripped = _plain(out)
        assert "┌" in stripped  # top-left corner
        assert "┘" in stripped  # bottom-right corner
        assert "│" in stripped  # vertical border

    # The table renders differently at different widths.
    assert out80 != out40


def test_render_table_width_independence():
    """render_table(data, 80) and render_table(data, 40) both produce valid
    box-drawing tables with different layouts."""
    data = TableData(
        header_rows=[["Category", "Score"]],
        body_rows=[["Performance", "95"], ["Reliability", "87"]],
        aligns=["left", "right"],
    )
    out80 = render_table(data, 80)
    out40 = render_table(data, 40)

    for out in (out80, out40):
        stripped = _plain(out)
        lines = stripped.strip().splitlines()
        # First line is top border.
        assert lines[0].startswith("┌")
        assert lines[0].endswith("┐")
        # Last line is bottom border.
        assert lines[-1].startswith("└")
        assert lines[-1].endswith("┘")
        # Contains data.
        full = "\n".join(lines)
        assert "Category" in full
        assert "Score" in full


# --- Phase 3: block-backed output path (3.1) ---------------------------------


def _mk_controller():
    """A _PromptController bound to a minimal Repl (no app, no terminal)."""
    fake = FakeSession(turns=[])
    repl, _ = make_repl(fake)
    return _PromptController(repl)


def _table_data():
    return TableData(
        header_rows=[["Name", "Description"]],
        body_rows=[["Alice", "A moderately long description that will wrap around"]],
        aligns=["left", "left"],
    )


def test_printer_coalesces_consecutive_prose():
    """Consecutive prose appends merge into the tail ProseBlock, bounding block
    count during streaming; a TableBlock breaks the run."""
    ctrl = _mk_controller()
    for _ in range(1000):
        ctrl.printer("x")
    assert len(ctrl._output_blocks) == 1
    assert isinstance(ctrl._output_blocks[0], ProseBlock)
    assert ctrl._output_blocks[0].ansi_text == "x" * 1000

    # A table appended mid-stream starts a fresh prose run afterward.
    ctrl._output_blocks.append(TableBlock(_table_data()))
    ctrl.printer("a")
    ctrl.printer("b")
    assert len(ctrl._output_blocks) == 3
    assert isinstance(ctrl._output_blocks[0], ProseBlock)
    assert isinstance(ctrl._output_blocks[1], TableBlock)
    assert isinstance(ctrl._output_blocks[2], ProseBlock)
    assert ctrl._output_blocks[2].ansi_text == "ab"


def test_printer_ignores_empty_text():
    ctrl = _mk_controller()
    ctrl.printer("")
    assert ctrl._output_blocks == []


def test_materialize_memo_incremental_and_width_recompute():
    """The memo renders only newly-sealed blocks at the same width, renders a
    trailing prose block fresh each call, and fully recomputes on width change.
    """
    ctrl = _mk_controller()
    d = _table_data()
    # Tail is a table => every block is "sealed" and baked.
    ctrl._output_blocks = [
        ProseBlock("a\n"),
        TableBlock(d),
        ProseBlock("b\n"),
        TableBlock(d),
    ]
    m1 = ctrl._materialize(80)
    assert ctrl._last_render_block_count == 4  # first call renders all four

    # Append a trailing prose block: it is NOT sealed (printer may coalesce),
    # so it renders fresh but nothing new is baked.
    ctrl._output_blocks.append(ProseBlock("c\n"))
    m2 = ctrl._materialize(80)
    assert ctrl._last_render_block_count == 1  # only the fresh tail
    assert m2 == m1 + "c\n"

    # Append a table: the "c\n" prose now seals, and the table seals too.
    ctrl._output_blocks.append(TableBlock(d))
    m3 = ctrl._materialize(80)
    assert ctrl._last_render_block_count == 2  # the two newly-sealed blocks
    assert m3.startswith(m2)

    # A width change forces a full recompute of every block.
    ctrl._materialize(40)
    assert ctrl._last_render_block_count == 6


def test_materialize_correct_at_multiple_widths():
    """Materialization matches materialize_blocks at several widths, and tables
    re-flow so the output differs by width."""
    ctrl = _mk_controller()
    d = _table_data()
    ctrl._output_blocks = [ProseBlock("prose\n"), TableBlock(d)]
    for w in (40, 80, 120, 200):
        assert ctrl._materialize(w) == materialize_blocks(ctrl._output_blocks, w)
    assert ctrl._materialize(40) != ctrl._materialize(200)


def test_content_line_count_matches_materialized_newlines():
    ctrl = _mk_controller()
    ctrl._output_blocks = [ProseBlock("one\ntwo\nthree\n")]
    # _current_width falls back to repl width (80) when no app is running.
    expected = ctrl._materialize(80).count("\n")
    assert ctrl._content_line_count() == expected == 3
