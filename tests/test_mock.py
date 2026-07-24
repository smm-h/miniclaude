"""Unit tests for miniclaude._mock live mock mode.

These drive :class:`MockSession` in live mode (with an integer seed) directly,
iterating the ``send()`` stream and inspecting the fabricated events. Seeded
randomness must be reproducible: the same seed produces byte-identical content.
"""

from __future__ import annotations

import asyncio
import functools

import pytest

from claudestream import PermissionRequest, RateLimit, Result, StreamDelta

from miniclaude._cli import _resolve_seed
from miniclaude._dialogs import build_question_answers
from miniclaude._mock import _MOCK_COMMANDS, MockSession


def sync(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


@pytest.fixture(autouse=True)
def _instant_sleep(monkeypatch):
    """Neutralize the mock's inter-chunk sleeps so tests run fast.

    Event order is preserved by the generators' ``yield`` points, so interrupt
    semantics (flag checked between chunks) are unaffected -- only the wall-clock
    delay disappears.
    """

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr("asyncio.sleep", _noop)


def _concat_text(events) -> str:
    """Concatenate all top-level text_delta payloads from an event list."""
    return "".join(
        ev.text or ""
        for ev in events
        if isinstance(ev, StreamDelta) and ev.delta_type == "text_delta"
    )


async def _drive(sess: MockSession, prompt: str) -> list:
    """Iterate a live turn to completion, auto-answering any dialog request."""
    events = []
    async for ev in sess.send(prompt):
        events.append(ev)
        if isinstance(ev, PermissionRequest):
            if ev.tool_name == "AskUserQuestion":
                await sess.respond_allow(
                    ev.request_id, build_question_answers(ev.tool_input, ["Red"])
                )
            else:
                await sess.respond_allow(ev.request_id, ev.tool_input)
    return events


def _parse_table(text: str) -> tuple[int, int]:
    """Return (columns, body_rows) for the first markdown table in ``text``."""
    rows = [line for line in text.splitlines() if line.strip().startswith("|")]
    assert len(rows) >= 3  # header, separator, at least one body row
    cols = rows[0].count("|") - 1
    body_rows = len(rows) - 2  # exclude header + separator
    return cols, body_rows


# --- Determinism -------------------------------------------------------------


@sync
async def test_same_seed_identical_table():
    a = await _drive(MockSession(7), "table")
    b = await _drive(MockSession(7), "table")
    assert _concat_text(a) == _concat_text(b)


@sync
async def test_different_seed_different_table():
    a = await _drive(MockSession(1), "table")
    b = await _drive(MockSession(999), "table")
    assert _concat_text(a) != _concat_text(b)


@sync
async def test_table_dimensions_within_bounds():
    for seed in range(20):
        events = await _drive(MockSession(seed), "table")
        cols, rows = _parse_table(_concat_text(events))
        assert 5 <= cols <= 10, f"cols={cols} seed={seed}"
        assert 5 <= rows <= 10, f"rows={rows} seed={seed}"


@sync
@pytest.mark.parametrize(
    "prompt",
    ["table", "text", "thinking", "tools", "status", "help", "demo", "bogus"],
)
async def test_mock_generated_content_is_ascii(prompt):
    """Mock-generated content is ASCII-only: emoji/CJK render at unpredictable
    widths in real terminals and would break table alignment. (Only text the
    mock fabricates is checked; user-supplied `md` payloads are exempt.)"""
    events = await _drive(MockSession(3), prompt)
    text = _concat_text(events)
    assert text.isascii(), f"non-ASCII in {prompt!r} content: {text!r}"


# --- Rate-limit emission -----------------------------------------------------


def _rate_limits(events):
    """The RateLimit events from a turn, as (type, util, status, resets_at)."""
    return [
        (e.rate_limit_type, e.utilization, e.status, e.resets_at)
        for e in events
        if isinstance(e, RateLimit)
    ]


@sync
async def test_each_turn_emits_two_rate_limits():
    events = await _drive(MockSession(7), "table")
    rls = [e for e in events if isinstance(e, RateLimit)]
    assert [e.rate_limit_type for e in rls] == ["five_hour", "seven_day"]


@sync
async def test_same_seed_identical_rate_limits_and_content():
    a = await _drive(MockSession(7), "table")
    b = await _drive(MockSession(7), "table")
    # Rate-limit sequence is reproducible per seed...
    assert _rate_limits(a) == _rate_limits(b)
    # ...and the content (table text) is identical too, proving the dedicated
    # rate-limit RNG is independent of the content RNG stream.
    assert _concat_text(a) == _concat_text(b)


@sync
async def test_different_seed_different_rate_limits():
    a = await _drive(MockSession(1), "table")
    b = await _drive(MockSession(999), "table")
    assert _rate_limits(a) != _rate_limits(b)


@sync
async def test_rate_limit_utilization_rises_with_turns():
    sess = MockSession(3)
    first = _rate_limits(await _drive(sess, "table"))
    second = _rate_limits(await _drive(sess, "table"))
    # five_hour is the first entry; it rises faster and must grow turn-over-turn.
    assert second[0][1] > first[0][1]


@sync
async def test_rate_limit_status_warns_past_threshold():
    # Drive enough turns that five_hour utilization crosses 0.8.
    sess = MockSession(5)
    saw_warning = False
    for _ in range(20):
        rls = _rate_limits(await _drive(sess, "table"))
        for _type, util, status, _resets in rls:
            assert status == ("allowed_warning" if util >= 0.8 else "allowed")
            if status == "allowed_warning":
                saw_warning = True
    assert saw_warning


# --- md round-trip -----------------------------------------------------------


@sync
async def test_md_roundtrip():
    md = "# Title\n\nSome **bold** and `code`.\n\n- a\n- b\n\n```py\nx = 1\n```\n"
    events = await _drive(MockSession(4), "md " + md)
    assert _concat_text(events) == md


# --- Unknown command ---------------------------------------------------------


@sync
async def test_unknown_command_lists_commands():
    events = await _drive(MockSession(5), "bogus")
    text = _concat_text(events)
    assert "Unknown command" in text
    for name, _desc in _MOCK_COMMANDS:
        # _MOCK_COMMANDS lists "md <markdown>"; the bare command word must appear.
        assert name.split()[0] in text


# --- Result per turn ---------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    ["table", "text", "thinking", "tools", "status", "md hello", "help", "dialogs", "bogus"],
)
@sync
async def test_each_command_ends_with_one_result(prompt):
    events = await _drive(MockSession(5), prompt)
    results = [ev for ev in events if isinstance(ev, Result)]
    assert len(results) == 1


@sync
async def test_demo_ends_with_one_result():
    events = await _drive(MockSession(5), "demo")
    results = [ev for ev in events if isinstance(ev, Result)]
    assert len(results) == 1


@sync
async def test_error_command_emits_non_streamed_text_and_error_result():
    from claudestream import AssistantText

    events = await _drive(MockSession(5), "error")
    # The explanation arrives as an AssistantText with NO text deltas preceding it.
    assert _concat_text(events) == ""
    texts = [ev for ev in events if isinstance(ev, AssistantText)]
    assert len(texts) == 1
    assert "credit balance" in texts[0].text
    # The closing Result is an error and carries the explanation.
    results = [ev for ev in events if isinstance(ev, Result)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert "credit balance" in results[0].result
    # A hard-failure turn emits no rate-limit events.
    assert not [ev for ev in events if isinstance(ev, RateLimit)]


@sync
async def test_first_send_yields_system_init():
    from claudestream import SystemInit

    events = await _drive(MockSession(5), "help")
    assert isinstance(events[0], SystemInit)
    assert events[0].model == "claude-mock"
    # The same session's second turn must NOT re-emit SystemInit.
    sess = MockSession(5)
    await _drive(sess, "help")
    second = await _drive(sess, "help")
    assert not any(isinstance(ev, SystemInit) for ev in second)


@sync
async def test_result_context_pct_grows():
    from miniclaude._repl import _ctx_pct

    sess = MockSession(5)
    first = await _drive(sess, "text")
    second = await _drive(sess, "text")
    r1 = next(ev for ev in first if isinstance(ev, Result))
    r2 = next(ev for ev in second if isinstance(ev, Result))
    p1 = _ctx_pct(r1.model_usage, "claude-mock")
    p2 = _ctx_pct(r2.model_usage, "claude-mock")
    assert p1 is not None and p1 > 0
    assert p2 > p1
    # Cost accumulates across turns.
    assert r2.total_cost_usd > r1.total_cost_usd


# --- Dialogs blocking + acknowledgment ---------------------------------------


@sync
async def test_dialogs_awaits_response_before_continuing():
    sess = MockSession(2)
    agen = sess.send("dialogs")
    ev = await agen.__anext__()
    while not isinstance(ev, PermissionRequest):
        ev = await agen.__anext__()
    assert ev.tool_name == "Bash"
    # The stream must be blocked until we respond: the ack only flows afterwards.
    await sess.respond_allow(ev.request_id, ev.tool_input)
    nxt = await agen.__anext__()
    assert isinstance(nxt, StreamDelta)
    await agen.aclose()


@sync
async def test_dialogs_acknowledgment_reflects_choice():
    sess = MockSession(2)
    events = []
    saw_bash = False
    saw_ask = False
    async for ev in sess.send("dialogs"):
        events.append(ev)
        if isinstance(ev, PermissionRequest) and ev.tool_name == "Bash":
            saw_bash = True
            await sess.respond_deny(ev.request_id, "no thanks")
        elif isinstance(ev, PermissionRequest) and ev.tool_name == "AskUserQuestion":
            saw_ask = True
            await sess.respond_allow(
                ev.request_id, build_question_answers(ev.tool_input, ["Blue"])
            )
    text = _concat_text(events)
    assert saw_bash and saw_ask
    assert "no thanks" in text  # deny message reflected
    assert "Blue" in text  # chosen answer reflected


# --- interrupt stops a slow stream -------------------------------------------


@sync
async def test_interrupt_stops_slow_stream():
    sess = MockSession(1)
    deltas = 0
    has_result = False
    async for ev in sess.send("slow"):
        if isinstance(ev, StreamDelta) and ev.delta_type == "text_delta":
            deltas += 1
            if deltas == 3:
                await sess.interrupt()
        if isinstance(ev, Result):
            has_result = True
    assert deltas <= 5  # stopped early, far short of a full slow stream
    assert has_result
    assert sess.interrupt_count == 1


# --- Seed resolution ---------------------------------------------------------


def test_resolve_seed_parses_int():
    assert _resolve_seed("123") == 123
    assert _resolve_seed("0") == 0


def test_resolve_seed_empty_is_random_in_range():
    for _ in range(5):
        s = _resolve_seed("")
        assert 0 <= s < 2**31


def test_resolve_seed_bad_raises():
    with pytest.raises(ValueError):
        _resolve_seed("not-an-int")
