"""Shared test fixtures for miniclaude."""

from __future__ import annotations

import json
import os
import shutil

import pytest


def _pty_unavailable_reason() -> str | None:
    """Return a skip reason if a pseudo-terminal cannot be allocated, else None.

    Allocating and immediately closing a pty is the cheapest way to probe whether
    the environment supports the terminal-driven tests at all. Shared by both the
    ``pty`` and ``integration`` gates so the pty capability check lives in exactly
    one place.
    """
    try:
        master_fd, slave_fd = os.openpty()
    except OSError as exc:
        return f"pty allocation unavailable: {exc}"
    os.close(master_fd)
    os.close(slave_fd)
    return None


def _missing_real_cli_prereqs(profile: str) -> str | None:
    """Return a skip reason if real-CLI prerequisites are absent, else None.

    Integration tests spawn the real ``miniclaude repl`` inside a pseudo-terminal
    and drive a live model turn. They need three external prerequisites that are
    unavailable in credential-less/headless environments (CI runners, fresh
    checkouts):

    - the ``claude`` binary on PATH,
    - a resolvable claudewheel profile whose launch env can actually
      authenticate a session, and
    - the ability to allocate a pseudo-terminal (pty).

    claudewheel's ``resolve_profile`` raises (``ValueError`` for an unknown
    profile, ``TokenStoreError`` for corrupt tokens) instead of failing soft, so
    the resolution is wrapped broadly and any failure -- or an empty result --
    is treated as "prerequisites missing" and turns the test into a skip.
    """
    if shutil.which("claude") is None:
        return "claude CLI not found on PATH"
    try:
        from claudewheel.profile import resolve_profile

        env = resolve_profile(profile)
    except Exception as exc:  # noqa: BLE001 -- any resolution failure means skip
        return f"claudewheel profile {profile!r} unavailable: {exc}"
    if not env:
        return f"claudewheel profile {profile!r} resolved to empty env"
    credential_reason = _missing_claude_credentials(profile, env)
    if credential_reason:
        return credential_reason
    return _pty_unavailable_reason()


def _missing_claude_credentials(profile: str, env: dict[str, str]) -> str | None:
    """Return a skip reason when the resolved profile cannot log a session in.

    A profile that resolves cleanly can still be signed out: the config
    directory exists and ``resolve_profile`` hands back its path, but the
    credentials file carries no ``claudeAiOauth`` block. The spawned CLI then
    starts, renders, and answers the first prompt with ``Not logged in ·
    Please run /login`` -- which reaches the test as "the model never said
    what we asked for", a failure that reads like a product defect and is
    not one. An ``ANTHROPIC_API_KEY`` in the resolved env authenticates on its
    own, so it satisfies the requirement without a credentials file.
    """
    if env.get("ANTHROPIC_API_KEY"):
        return None
    config_dir = env.get("CLAUDE_CONFIG_DIR")
    if not config_dir:
        return None
    credentials = os.path.join(config_dir, ".credentials.json")
    if not os.path.exists(credentials):
        return f"claudewheel profile {profile!r} has no credentials file"
    try:
        with open(credentials) as handle:
            stored = json.load(handle)
    except (OSError, ValueError) as exc:
        return f"claudewheel profile {profile!r} has unreadable credentials: {exc}"
    if not stored.get("claudeAiOauth"):
        return f"claudewheel profile {profile!r} is signed out of the Claude CLI"
    return None


@pytest.fixture(autouse=True)
def _skip_without_prereqs(request):
    """Skip pty/integration-marked tests when their prerequisites are absent.

    Two independent gates share a single pty-availability probe:

    - ``@pytest.mark.integration`` keeps the full three-way gate (claude binary +
      claudewheel profile + pty), since it drives a live model turn.
    - ``@pytest.mark.pty`` gates ONLY on pty availability -- these tests spawn a
      miniclaude command against a fake/mock session and need no live CLI,
      profile, or network.

    Where the prerequisites exist (developer machines with a real terminal, and
    for integration a configured profile), the tests run normally.
    """
    if request.node.get_closest_marker("integration") is not None:
        profile = getattr(request.module, "PROFILE", "personal")
        reason = _missing_real_cli_prereqs(profile)
        if reason:
            pytest.skip(reason)
        return
    if request.node.get_closest_marker("pty") is not None:
        reason = _pty_unavailable_reason()
        if reason:
            pytest.skip(reason)
