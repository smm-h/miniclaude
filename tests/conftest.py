"""Shared test fixtures for miniclaude."""

from __future__ import annotations

import os
import shutil

import pytest


def _missing_real_cli_prereqs(profile: str) -> str | None:
    """Return a skip reason if real-CLI prerequisites are absent, else None.

    Integration tests spawn the real ``miniclaude repl`` inside a pseudo-terminal
    and drive a live model turn. They need three external prerequisites that are
    unavailable in credential-less/headless environments (CI runners, fresh
    checkouts):

    - the ``claude`` binary on PATH,
    - a resolvable claudewheel profile with usable launch env, and
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
    try:
        master_fd, slave_fd = os.openpty()
    except OSError as exc:
        return f"pty allocation unavailable: {exc}"
    else:
        os.close(master_fd)
        os.close(slave_fd)
    return None


@pytest.fixture(autouse=True)
def _skip_without_real_cli(request):
    """Skip integration-marked tests when the real claude CLI/profile/pty is absent.

    Applies only to tests carrying ``@pytest.mark.integration`` (including the
    module-level ``pytestmark``). Where the prerequisites exist (developer
    machines with a configured profile and a real terminal), the tests run
    normally.
    """
    if request.node.get_closest_marker("integration") is None:
        return
    profile = getattr(request.module, "PROFILE", "personal")
    reason = _missing_real_cli_prereqs(profile)
    if reason:
        pytest.skip(reason)
