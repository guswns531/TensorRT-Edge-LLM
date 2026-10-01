---
id: W015
status: active
updated: 2026-10-01
notes: [309, 370]
---

# .local retention, cleanup, and workspace lifecycle

## Goal
Keep `.local` bounded by reference- and state-driven retention (never age), while protecting source worktrees, models, promoted engines, citable results and frozen vLLM references, and give research work a defined lifecycle.

## Current state
- Note 309 (2026-09-15): disk was 100% full from engine variants; about 98 GiB in `.local` (51 GiB v0.10.1 forward-port, 7.6 GiB results, 2.76 GiB repeated `gateway.log`). Defined retention classes (source worktree, model, ONNX, promoted engine, citable / validation / diagnostic / scratch result), applied an allowlist cleanup (manifest `.local/results/cleanup-20260915/manifest.json`, since deleted) and added per-model `current` pointers (`current/cosmos`, `current/gemma4`; `current/active` selects the default).
- Note 370 (2026-09-30), the reference implementation: latest-only cleanup before the v0.11.0 port; 426 retention units deleted, free disk 26 -> 61 GB, `.local` 65 GB; plan and apply script are retained under `.local/registry/`. Keep rules: `current` targets and protected stores, paths cited by tracked code and `registry/current.json`, paths cited by notes 364-370, transitive manifest references, `vllm` results, recent scratch, and `builds/upstream-v0110`. Notes 363 and earlier may cite deleted campaigns; their numbers stand but are not reproducible from raw data.
- 2026-10-01: workspace-lifecycle skill (`.claude/skills/workspace-lifecycle/SKILL.md`) and work items W001-W016
  backfilled for notes 236-371 (`notes/work/README.md`), each fact-checked against its member notes. Layout repair
  (log: `.local/registry/relocation-20261001.tsv`; no data deleted):
  - Manifests: 28 of 48 retained campaigns had none (mostly 2026-09-28/29 A/B runs driven from scratch `.sh`);
    each now has a backfilled `manifest.json` (state `diagnostic`, `backfilled` block naming its sources, unknown
    identity fields null), and 11 scratch drivers were copied to `results/<campaign>/run.sh`.
  - Dangling links (targets deleted by note 370; 370 checked `current/` only): the five `results/current/*` links
    and `results/baselines/vllm-frozen-12x3` were replaced by links to the reference set in `registry/current.json`;
    six links in `cosmos-reason2-2b/v010-onnx-fp16-packed-tied-atomic1024/llm/` were removed and the directory
    marked `INCOMPLETE.md`. Zero dangling links outside `cache/` (venv links to container paths are expected).
  - Scratch grouped into `W010/W013/W014/W015-*`; `balanced-vllm-crossover-20260828` → `results/legacy/` and
    `atomic-packed-vision-runtime-20260824` → `artifacts/legacy/` with compatibility links in `retention.json`;
    `notes/*.csv` → `results/legacy/notes-csv/`.
  - `.local/README.md` rewritten for the current store layout.
- Note 371 cites `.local/scratch/v0110-upstream-serve-20260930/`; `v0110-*` scratch was left in place because the
  v0.11.0 merge in `.local/worktrees/v0110-port` was still writing there.

## Conclusions
- 309 — Retention classes and protected active artifacts defined; engines (not logs) were the disk driver; first allowlist cleanup and per-model `current` pointers.
- 370 — Reference-driven latest-only cleanup with keep rules, plan, apply script and deletion record; older notes may cite deleted data.

## Open questions
- After the v0.11.0 merge settles: move `scratch/v0110-*` into `scratch/W016-upstream-v0110-port/` and promote
  `scratch/v0110-upstream-serve-20260930` (cited by note 371) into `.local/results/`.
- Shared manifest writer in `scripts/` so ad-hoc campaign drivers record identity at start (skill rule; not built).
- `results/v0101-forward-port/` (164 v0.10.1-era runs, no per-run manifests) and the incomplete
  `cosmos-reason2-2b/v010-onnx-fp16-packed-tied-atomic1024/`: keep or delete needs a user decision.
- Worktree `.local/worktrees/v0110-port` is a second development worktree, which `AGENTS.md` does not list.

## Artifacts
- `.local/registry/cleanup-plan-20260930.json` (present)
- `.local/registry/cleanup-apply-20260930.py` (present)
- `.local/registry/cleanup-deleted-20260930.txt` (present)
- `.local/registry/current.json` and `.local/registry/retention.json` (present)
- `.local/current/active` (present; resolves to `.local/current/gemma4`)
- `.local/README.md` (present; rewritten 2026-10-01)
- `.local/registry/relocation-20261001.tsv` (present)
- `notes/work/README.md`, `.claude/skills/workspace-lifecycle/SKILL.md` (present)
- `.local/results/cleanup-20260915` (deleted; note 370)
