#!/usr/bin/env python3
"""Probe the Claude CLI for RateLimit event fields.

Sends a trivial prompt and prints every RateLimit event's fields verbatim,
plus raw JSON lines containing rate_limit data for full protocol inspection.
Used to discover what rate_limit_type values the CLI actually sends.
"""

import asyncio
import json
import os
import subprocess
import sys

from claudestream import AsyncSession, SessionConfig, RateLimit, Result, UnknownEvent
from claudewheel.profile import resolve_profile


async def probe_via_session(profile: str, model: str) -> None:
    """Probe using claudestream's AsyncSession (parsed events)."""
    print(f"\n=== Session probe profile={profile!r} model={model!r} ===\n")
    config = SessionConfig(
        model=model,
        profile=profile,
        intercept_permissions=True,
    )

    rate_limit_events: list[RateLimit] = []

    try:
        async with AsyncSession(config) as session:
            async for event in session.send("Say hi", raw=True):
                if isinstance(event, RateLimit):
                    rate_limit_events.append(event)
                    print(f"  [RateLimit] status={event.status!r} "
                          f"rate_limit_type={event.rate_limit_type!r} "
                          f"utilization={event.utilization!r} "
                          f"resets_at={event.resets_at!r}")
                elif isinstance(event, Result):
                    print(f"  [Result] cost=${event.total_cost_usd:.6f} "
                          f"turns={event.num_turns}")
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")

    if not rate_limit_events:
        print("  (no RateLimit events observed)")
    else:
        print(f"\n  Total RateLimit events: {len(rate_limit_events)}")
        types_seen = sorted(set(e.rate_limit_type for e in rate_limit_events))
        print(f"  Distinct rate_limit_type values: {types_seen}")


def probe_raw_ndjson(profile: str, model: str) -> None:
    """Run claude CLI directly and capture raw NDJSON for rate_limit events."""
    print(f"\n=== Raw NDJSON probe profile={profile!r} model={model!r} ===\n")
    env = os.environ.copy()
    env.update(resolve_profile(profile))
    try:
        proc = subprocess.run(
            ["claude", "--output-format", "stream-json", "--model", model,
             "--verbose", "-p", "Say hi"],
            capture_output=True, text=True, timeout=60, env=env,
        )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_str = json.dumps(data)
            if "rate" in raw_str.lower():
                print(f"  {json.dumps(data, indent=2)}")
    except subprocess.TimeoutExpired:
        print("  (timed out)")
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")


async def main() -> None:
    configs = [
        ("personal", "haiku"),
        ("personal", "sonnet"),
        ("emergency", "opus-4.6"),
    ]
    for profile, model in configs:
        await probe_via_session(profile, model)
        probe_raw_ndjson(profile, model)


if __name__ == "__main__":
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=300))
    except asyncio.TimeoutError:
        print("\n[TIMEOUT] Probe timed out after 300s")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)
