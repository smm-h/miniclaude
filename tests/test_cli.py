"""CLI-level tests for the strictcli app registration.

strictcli validates every command handler's signature against its declared
flags at decoration time (i.e. at ``import miniclaude._cli``). Since strictcli
0.29 the first handler parameter is unconditionally the injected context slot
and is excluded from flag matching, so every handler must lead with a ``ctx``
parameter. Importing the module and inspecting the registered commands is
enough to catch a handler-contract skew: under the bug, the import itself
raises ``ValueError: handler missing parameter ...``.
"""

from __future__ import annotations


def test_cli_registers_all_commands() -> None:
    import miniclaude._cli as cli

    registered = set(cli.app._collect_all_command_paths())
    assert {"repl", "mock", "version"} <= registered
