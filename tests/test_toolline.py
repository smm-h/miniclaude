"""Tests for the one-liner tool-activity formatters (pure, no terminal needed)."""

from __future__ import annotations

import re

from miniclaude._toolline import format_tool_result, format_tool_use

_ANSI = re.compile(r"\033\[[0-9;]*m")


def strip(text: str) -> str:
    """Remove ANSI SGR codes for content assertions."""
    return _ANSI.sub("", text)


# ---------------------------------------------------------------------------
# format_tool_use: per-tool arg summaries
# ---------------------------------------------------------------------------


def test_read_shows_path() -> None:
    line = format_tool_use("Read", {"file_path": "/home/m/foo.py"}, None)
    assert strip(line) == "▸ Read /home/m/foo.py"


def test_write_shows_path() -> None:
    line = format_tool_use("Write", {"file_path": "/tmp/out.txt"}, None)
    assert strip(line) == "▸ Write /tmp/out.txt"


def test_edit_shows_path() -> None:
    line = format_tool_use("Edit", {"file_path": "/tmp/x.py", "old_string": "a"}, None)
    assert strip(line) == "▸ Edit /tmp/x.py"


def test_bash_shows_first_line_of_command() -> None:
    line = format_tool_use("Bash", {"command": "git status\ngit log"}, None)
    assert strip(line) == "▸ Bash git status"


def test_glob_shows_pattern() -> None:
    line = format_tool_use("Glob", {"pattern": "**/*.py"}, None)
    assert strip(line) == "▸ Glob **/*.py"


def test_grep_shows_pattern_and_path() -> None:
    line = format_tool_use("Grep", {"pattern": "TODO", "path": "src/"}, None)
    assert strip(line) == "▸ Grep TODO in src/"


def test_grep_pattern_only_when_no_path() -> None:
    line = format_tool_use("Grep", {"pattern": "TODO"}, None)
    assert strip(line) == "▸ Grep TODO"


def test_webfetch_shows_url() -> None:
    line = format_tool_use("WebFetch", {"url": "https://example.com"}, None)
    assert strip(line) == "▸ WebFetch https://example.com"


def test_websearch_shows_query() -> None:
    line = format_tool_use("WebSearch", {"query": "python asyncio"}, None)
    assert strip(line) == "▸ WebSearch python asyncio"


def test_agent_shows_description() -> None:
    line = format_tool_use("Agent", {"description": "refactor module"}, None)
    assert strip(line) == "▸ Agent refactor module"


def test_task_shows_description() -> None:
    line = format_tool_use("Task", {"description": "run the audit"}, None)
    assert strip(line) == "▸ Task run the audit"


def test_todowrite_shows_item_count_plural() -> None:
    line = format_tool_use("TodoWrite", {"todos": [{"a": 1}, {"b": 2}, {"c": 3}]}, None)
    assert strip(line) == "▸ TodoWrite 3 items"


def test_todowrite_singular() -> None:
    line = format_tool_use("TodoWrite", {"todos": [{"a": 1}]}, None)
    assert strip(line) == "▸ TodoWrite 1 item"


def test_todowrite_zero() -> None:
    line = format_tool_use("TodoWrite", {"todos": []}, None)
    assert strip(line) == "▸ TodoWrite 0 items"


def test_mcp_tool_shows_server_colon_tool_and_first_arg() -> None:
    line = format_tool_use("mcp__github__create_issue", {"title": "Bug report"}, None)
    assert strip(line) == "▸ github:create_issue Bug report"


def test_mcp_tool_no_args() -> None:
    line = format_tool_use("mcp__github__list_repos", {}, None)
    assert strip(line) == "▸ github:list_repos"


def test_unknown_tool_shows_kv_pairs() -> None:
    line = format_tool_use("FrobnicateWidget", {"count": 5, "mode": "fast"}, None)
    assert strip(line) == "▸ FrobnicateWidget count=5 mode=fast"


def test_unknown_tool_no_args() -> None:
    line = format_tool_use("MysteryTool", {}, None)
    assert strip(line) == "▸ MysteryTool"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_bash_command_truncated_to_80() -> None:
    command = "echo " + "x" * 200
    line = format_tool_use("Bash", {"command": command}, None)
    summary = strip(line).removeprefix("▸ Bash ")
    assert summary.endswith("…")
    assert len(summary) == 81  # 80 chars + ellipsis


def test_kv_value_truncated_to_40() -> None:
    line = format_tool_use("Weird", {"blob": "y" * 100}, None)
    summary = strip(line).removeprefix("▸ Weird ")
    assert summary.startswith("blob=")
    assert summary.endswith("…")


def test_truncation_collapses_newlines() -> None:
    line = format_tool_use("WebSearch", {"query": "line1\nline2"}, None)
    assert "\n" not in line
    assert strip(line) == "▸ WebSearch line1 line2"


# ---------------------------------------------------------------------------
# Styling: glyph color and bold name
# ---------------------------------------------------------------------------


def test_tool_use_has_bold_name_and_colored_glyph() -> None:
    line = format_tool_use("Read", {"file_path": "/x"}, None)
    assert "\033[1m" in line  # bold
    assert "\033[36m" in line  # glyph color
    assert "\033[0m" in line  # reset


# ---------------------------------------------------------------------------
# Subagent indent
# ---------------------------------------------------------------------------


def test_subagent_use_is_indented_two_spaces() -> None:
    line = format_tool_use("Read", {"file_path": "/x"}, "parent-123")
    assert line.startswith("  ")
    assert strip(line) == "  ▸ Read /x"


def test_top_level_use_is_not_indented() -> None:
    line = format_tool_use("Read", {"file_path": "/x"}, None)
    assert not line.startswith(" ")


# ---------------------------------------------------------------------------
# format_tool_result
# ---------------------------------------------------------------------------


def test_success_result_dim_check() -> None:
    line = format_tool_result("Read", "file contents", False, None)
    assert strip(line) == "✓ file contents"
    assert "\033[2m" in line  # dim


def test_success_result_no_red() -> None:
    line = format_tool_result("Read", "ok", False, None)
    assert "\033[31m" not in line


def test_error_result_uses_red_cross() -> None:
    line = format_tool_result("Bash", "command failed", True, None)
    assert strip(line) == "✗ command failed"
    assert "\033[31m" in line  # red


def test_multiline_result_shows_extra_line_count() -> None:
    line = format_tool_result("Read", "line1\nline2\nline3", False, None)
    assert strip(line) == "✓ line1 (+2 lines)"


def test_multiline_error_shows_extra_line_count() -> None:
    line = format_tool_result("Bash", "boom\ntrace1\ntrace2", True, None)
    assert strip(line) == "✗ boom (+2 lines)"


def test_trailing_newlines_do_not_count_as_extra_lines() -> None:
    line = format_tool_result("Read", "only line\n\n", False, None)
    assert strip(line) == "✓ only line"


def test_result_first_line_truncated_to_100() -> None:
    line = format_tool_result("Read", "z" * 250, False, None)
    body = strip(line).removeprefix("✓ ")
    assert body.endswith("…")
    assert len(body) == 101  # 100 chars + ellipsis


def test_result_content_as_list_of_blocks() -> None:
    content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
    line = format_tool_result("Read", content, False, None)
    assert strip(line) == "✓ hello (+1 lines)"


def test_result_content_none() -> None:
    line = format_tool_result("Read", None, False, None)
    assert strip(line) == "✓ "


def test_subagent_result_is_indented() -> None:
    line = format_tool_result("Read", "done", False, "parent-9")
    assert line.startswith("  ")
    assert strip(line) == "  ✓ done"
