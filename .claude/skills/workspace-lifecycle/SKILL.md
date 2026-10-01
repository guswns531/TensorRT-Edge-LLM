---
name: workspace-lifecycle
description: Where research work goes in this repository and how it moves through its lifecycle — work items (notes/work/W###), conclusion notes (notes/NNN), scratch under .local/scratch/W###-slug/, retained results with manifests, script promotion into benchmarks/ or scripts/, and reference-driven .local cleanup. Use when starting, resuming, or finishing an investigation; creating a script, result directory, or note; or cleaning up .local.
---

# Workspace Lifecycle

Keep durable work memory separate from execution artifacts, and never break a path that a note,
manifest, or script already cites.

`AGENTS.md` (section "Local Research Workflow") is authoritative. This skill explains how to apply it;
if the two disagree, follow `AGENTS.md` and fix this skill.

## Map

| Concept | Location | Mutability |
|---|---|---|
| Work item (ongoing line of work) | `notes/work/W###-<slug>.md`, indexed in `notes/work/README.md` | Rewritten in place |
| Conclusion | `notes/NNN-<topic>-YYYYMMDD.md` | Fixed once written (errata allowed) |
| Scratch | `.local/scratch/W###-<slug>/` | Disposable |
| Retained result | `.local/results/<campaign>-YYYYMMDD/` + `manifest.json` | Path is frozen once cited |
| Model, engine, ONNX | `.local/artifacts/`, reached through `.local/current/<family>/` | Protected |
| Comparison baseline | `.local/baselines/`, `.local/worktrees/` | Protected / read-only |
| Build tree | `.local/builds/` | Regenerate, never move |
| Reusable tool | `benchmarks/phase_serving/`, `scripts/` | Tracked |
| Retention policy | `.local/registry/retention.json`, `current.json` | Tracked by hand |

Do not create repository-root `tools/`, `results/`, or `src/`, new top-level `.local/` directories, or
non-Markdown files in `notes/`.

## Work items

A work item is a line of work that spans several sessions and conclusion notes (for example
"Gemma long-prefill variance" or "v0.11.0 port"), not one session or one experiment.

- Before starting, read `notes/work/README.md`. Continue an existing item when the goal is substantially
  the same; create a new one only for a new goal.
- IDs are `W###`, assigned sequentially, never reused. The slug is short kebab-case.
- Use the same `W###-<slug>` for the scratch directory.
- Rewrite `Current state` instead of appending history. Execution detail goes in scratch logs; results go in
  conclusion notes.
- Change status in the frontmatter; never move the file. Update the index row in the same commit.

Template:

```markdown
---
id: W012
status: active        # active | waiting | blocked | done
updated: YYYY-MM-DD
notes: [364, 365, 368] # conclusion notes in this line, oldest first
---

# Title

## Goal
What this line of work is trying to establish or ship.

## Current state
The latest compressed understanding, with the governing numbers. Cite the note that establishes each claim.

## Conclusions
- NNN — one line: what it established. Mark superseded entries `(superseded by MMM)`.

## Open questions
- Concrete remaining questions or gates, with the evidence that would close them.

## Artifacts
Retained results, engines, or `current` pointers this line depends on (paths as cited by notes).
```

`done` requires: the outcome is stated in `Current state`, open questions are empty or explicitly handed
to another work item, and the artifact list matches the retention state.

## Conclusion notes

- Number sequentially after the highest existing `notes/NNN-*`, including notes on other development
  worktrees under `.local/worktrees/` (they share one sequence). One conclusion per note: outcome first, then
  evidence, method, and retained paths.
- A later finding gets a new note that names the note it revises; it does not rewrite history.
- Correcting a factual error in an existing note (wrong cause, wrong number) is allowed in place, with a
  `docs: Correct ...` commit.
- Every new conclusion note must also be added to its work item (`notes:` list and `Conclusions`).

## Scratch

- Any intermediate script, log, transformed trace, or plot starts in `.local/scratch/W###-<slug>/`.
- Scratch is not evidence. Never cite a scratch path from a conclusion note; promote first.
- Before writing a new script, search `benchmarks/` and `scripts/` for an existing one.

## Promotion

Promotion changes a status or adds a record. It never moves a path that is already cited.

**Results.** Write a campaign into `.local/results/<campaign>-YYYYMMDD/` with a `manifest.json` that records
`state`, the command and config, source commit and dirty state, binary/runner/plugin and engine identity,
workload, repeat count, and summary paths (see an existing manifest for the schema). Write the manifest when the
campaign starts, not afterwards, and copy the driver script into the result directory (for example `run.sh`): a
driver that exists only in scratch is lost at the next scratch cleanup. A result directory without a manifest is
treated as `scratch`. States follow
`.local/registry/retention.json`: `scratch` → `diagnostic` → `validation` → `citable`. Promoting a result
means raising `state` and, if it is part of the current reference set, adding a link under
`.local/results/current/` or `.local/current/<family>/`.

**Scripts.** Move a script from scratch to `benchmarks/phase_serving/` or `scripts/` when a second work
item needs it, or when a `validation`/`citable` manifest depends on it. Before moving it, make all paths
arguments, give it a descriptive name and a `--help`, and remove work-item-specific constants. A script
that is checked into the repository needs the license header.

**Relocation.** If a retained directory really must move, move it with a compatibility symlink at the old
path, and add that path to `compatibility_links` in `retention.json`. Never relocate `.local/worktrees/`
(use `git worktree move`), CMake build trees, or Python virtual environments (their absolute paths are
built in; regenerate them instead).

## Finishing or pausing

1. Classify each scratch artifact as **promote** (into a result or tracked script), **keep** (the work
   item is still active), or **discard**.
2. Update the work item: `Current state`, `Conclusions`, `Open questions`, `status`, `updated`.
3. If the store layout or the `current` set changed, update `.local/README.md` and `registry/current.json`.

## Cleanup

Cleanup is driven by references, not age, and is always a dry run first.

1. Build the keep set: `current` link targets, protected paths in `retention.json`, `.local/...` paths
   cited by tracked code and `registry/current.json`, paths cited by notes of active work items, and the
   transitive references in kept manifests.
2. Write the plan (keep set with reasons, delete list with sizes) to `.local/registry/` and show it to
   the user.
3. Delete only after explicit approval, including inside `scratch/`. Record what was deleted.
4. `validation` and `citable` results, models, engines, and worktrees need an explicit allowlist and a
   `notes/` reference audit. A cleanup note records which older notes now cite deleted data.
5. After any cleanup or relocation, check for dangling links across the stores, not only `current/`:
   `find .local \( -path .local/worktrees -o -path .local/cache \) -prune -o -type l ! -exec test -e {} \; -print`
   (`cache/` venvs link to container-only paths).
   Fix or remove each one, and update the index files (`.local/README.md`, `results/current/`) that point at them.

Note 370 (`.local` latest-only cleanup) is the reference implementation.
