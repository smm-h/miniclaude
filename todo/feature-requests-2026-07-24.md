# Feature requests batch (2026-07-24)

Seven user-requested features, filed together after the v0.2.0 fullscreen release.
Each section is independently actionable; split this file when items complete.

## 1. Spawn subagents by model version, not just family

**Context.** The Agent tool's `model` parameter accepts family names ("opus",
"sonnet", "haiku") and CC resolves them to its current default version. The user
wants to pin a specific version (e.g. Opus 4.6 instead of 4.8) per spawn.

**Problem.** Model resolution happens inside CC's Agent tool, not in miniclaude.
miniclaude never sees the spawn except as a ToolUse event -- by then the model
is chosen. Unknown whether CC accepts full model IDs (e.g.
"claude-opus-4-6") in the Agent `model` param; vanilla CC UI only documents the
family names.

**Solutions.**
- (a) Permission-interception rewrite: miniclaude already runs with
  `intercept_permissions=True`; the permission flow can return an updated
  `tool_input` via `respond_allow(request_id, updated_input)`. If CC accepts
  full model IDs in `Agent.model`, miniclaude can rewrite "opus-4.6" -> the full
  ID on the way through, or offer a picker in the permission dialog.
  Pros: purely client-side, no CC changes. Cons: only works when the Agent call
  actually triggers a permission prompt (bypassPermissions modes skip it);
  depends on undocumented acceptance of full IDs -- needs a probe first.
- (b) Upstream feature request to CC. Pros: correct place. Cons: not ours to
  schedule.
- (c) CLAUDE.md convention instructing the main agent to pass full IDs (works
  today IF CC accepts them; zero code). Probe first, then pick.

**Affected files.** `miniclaude/_repl.py` (permission flow), `_dialogs.py`
(picker UI if any), claudestream (only if `respond_allow` shape needs work).

**Effort.** Small probe + small-to-medium implementation, dominated by the
unknown of what CC accepts.

## 2. Pause/resume mode (freeze all agents non-destructively)

**Context.** When CC waits for a permission answer, the whole session (including
subagents at their own permission points) freezes cleanly and resumes exactly
where it stopped -- no cache damage, no interrupt semantics. The user wants that
freeze as a first-class mode: a pause command/keybind that halts all work at the
next natural stopping point, and a resume that continues.

**Problem.** miniclaude has no "hold" concept; the only stop today is
`interrupt()`, which is destructive to in-flight turns.

**Solutions.**
- (a) Hold at interception points: while paused, miniclaude defers answering
  `canUseTool` permission requests (and any dialog requests) instead of
  auto-responding/prompting. Every agent freezes at its next tool call exactly
  like a permission wait; resume answers the held requests. Pros: reuses the
  precise mechanism the user likes; cache-safe by construction. Cons: agents
  pause at their NEXT tool call, not instantly; turns that never call tools
  don't pause until the turn ends; in bypass/accept-edits modes CC may not
  consult the client per call -- needs verification of which modes still emit
  canUseTool.
- (b) Additionally hold the queue: pause also stops miniclaude from submitting
  queued user turns (trivial, complements (a)).

**Affected files.** `miniclaude/_repl.py` (pause state, keybind, queue gating,
status-bar indicator), `_dialogs.py` (held-request bookkeeping), claudestream
(verify deferred responses don't time out).

**Effort.** Medium. The verification of CC's timeout behavior on unanswered
canUseTool is the risk item.

## 3. Refresh howmuchleft on demand

**Context.** The status bar renders `howmuchleft` output through
`_HowMuchLeftCache` (TTL 0.25s during turns, 1.0s idle); the underlying binary
also caches its own upstream data. Sometimes the user wants a hard refresh now.

**Problem.** No user-facing way to force a re-render/refetch.

**Solutions.** A `/hml` slash command and/or keybind that invalidates
`_HowMuchLeftCache` and invokes the binary immediately (check whether
howmuchleft itself has a force-refresh flag for its own upstream cache; pass it
if so). Pros: trivial, isolated. Cons: none.

**Affected files.** `miniclaude/_repl.py` (`_HowMuchLeftCache`, `_handle_line`,
key bindings, `_HELP_LINES`).

**Effort.** Small.

## 4. Object selection (tables) with arrow keys + ephemeral HTML view

**Context.** Output is already block-structured (`ProseBlock`/`TableBlock` in
`_repl.py`); tables exist as structured `TableData`, not just text. The user
wants to select such objects with arrow keys and open the selected one in a
throwaway HTML view via a keybind.

**Problem.** No selection concept in the output window; no HTML renderer.

**Solutions.**
- Selection mode: a keybind enters "object select"; up/down moves a highlight
  across selectable blocks (style transform applied during materialization --
  the memoized materializer needs a highlight-aware variant or a post-pass);
  Esc leaves. Enter/`o` renders the selected `TableData` to a standalone HTML
  file in a temp dir and opens it (`xdg-open`); file cleaned up on session exit.
  Pros: builds directly on the block model; HTML tables trivially beat terminal
  rendering for wide content. Cons: touches the materialization memo (highlight
  must not thrash the cache); scrolling must follow the selection.
- Scope note: start with tables only; the block model makes other object kinds
  (code fences?) a natural follow-up.

**Affected files.** `miniclaude/_repl.py` (selection state, keybinds,
materialization highlight, scroll-to-selection), new `miniclaude/_htmlview.py`
(TableData -> HTML, temp-file lifecycle), `_render.py` (expose any needed cell
metadata), tests.

**Effort.** Medium-large. The selection/highlight plumbing is the bulk; HTML
generation is small.

## 5. Control which tools agents see; provide native (non-MCP) tools

**Context.** The user wants to disable disliked CC tools entirely and provide
some tools natively rather than through MCP.

**Problem/state.** Disabling already exists upstream: CC honors
`disallowedTools` / permission `deny` rules in profile settings (claudewheel
profiles each carry a `permissions` object). What's missing is a comfortable
surface. Adding tools has no CC-native client API -- MCP is the only injection
path for a frontend.

**Solutions.**
- Disable: (a) document/curate deny lists in claudewheel profiles (zero code);
  (b) miniclaude flag or config (e.g. `--disallow-tools`) mapped onto the
  session's settings via claudestream if it exposes it. Pros of (b): per-session
  control without editing profiles. Cons: another config surface.
- Provide: an in-process MCP server owned by miniclaude (stdio, spawned/managed
  by the session) exposing the user's "native" tools -- agents see ordinary
  tools; the user writes plain Python handlers in a miniclaude config module.
  Pros: only viable path; feels native; no external server management.
  Cons: it IS MCP under the hood (CC restart quirk applies -- see item 6);
  handler API design needed.

**Affected files.** `miniclaude/_cli.py`, claustream SessionConfig passthrough,
new MCP-server module if (Provide) is pursued, claudewheel profile docs.

**Effort.** Disable: small. Provide: large (own design round before
implementation).

## 6. MCP config changes without restarting CC

**Context.** CC only reads MCP config at startup; adding a server mid-session
requires restarting CC. The user wants instant pickup.

**Problem.** The restart requirement is inside CC; a frontend cannot make the
running process re-read config.

**Solutions.** Transparent process bounce: a `/reload` command that gracefully
stops the underlying claude process and respawns it with `--resume` on the same
session id (claudestream already supports `resume_session_id`). The
conversation continues where it was; the user never leaves miniclaude; the only
cost is process startup latency and whatever server-side prompt cache has
expired (content-keyed, so typically warm). Pros: solves the quirk entirely
client-side; also useful after settings/permission edits generally.
Cons: in-flight turn must complete or be interrupted first; edge cases around
pending permission requests need defined behavior.

**Affected files.** `miniclaude/_repl.py` (`/reload`, lifecycle),
claudestream (clean stop + respawn/resume path).

**Effort.** Medium.

## 7. Session JSONL logs on disk

**Context/question.** Are miniclaude sessions writing the same JSONL transcript
logs CC writes, or must we mimic them?

**Answer (to verify, believed true).** They are written automatically: miniclaude
drives the real `claude` process, and CC persists its session transcripts
itself, independent of frontend, under the active profile's config dir (with
claudewheel: the shared projects store that profiles symlink). Observed
indirectly this session via transcript files for spawned agents.

**Work.**
- Verify: run a miniclaude session, locate its JSONL under the profile's
  projects dir, confirm completeness (turns, tool calls, subagents).
- Surface: optionally a `/transcript` command printing the live session's JSONL
  path (session_id is already tracked from SystemInit).
- Document in README.

**Affected files.** `miniclaude/_repl.py` (optional command), README.

**Effort.** Small (verification + optional tiny command).
