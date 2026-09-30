# `.local` Latest-Only Cleanup

## Outcome

On 2026-09-30 `.local` was reduced to the latest lineage before the v0.11.0 port: 426 retention units deleted, free
disk 26 -> 61 GB, `.local` 65 GB. All `current` links resolve and the benchmark traces, calibration, replay tools,
and frozen vLLM references remain.

Plan: `.local/registry/cleanup-plan-20260930.json` (keep set with reasons, delete list). Applied by
`.local/registry/cleanup-apply-20260930.py --yes`; deleted units: `.local/registry/cleanup-deleted-20260930.txt`.

## Keep rules

A retention unit (second level of each store; third/fourth level under `artifacts/v0101-forward-port`) was kept when
any of these referenced it or something inside it:

1. `.local/current` symlink targets, and `worktrees`, `registry`, `cache`, `artifacts/models`, the HF token.
2. `.local/...` paths in tracked `benchmarks/`, `scripts/`, `examples/`, `tests/`, and in `registry/current.json`.
3. Notes 364-370, by `.local/...` path or by bare campaign/baseline directory name.
4. Transitively, `.local/...` paths in manifests (up to four levels deep) of kept results and baselines.
5. Any result whose name contains `vllm` (frozen vLLM references), recent scratch scripts (2026-09-29 onward), and
   the in-progress `builds/upstream-v0110` tree.

## What was removed

- About 190 legacy flat top-level campaign directories from 2026-08-24 to 2026-09-05 (14.7 GB).
- Superseded Gemma engine/ONNX variants: eight `engine-asym-*`, `engine-profiled-p8x512-p8x128-d24-kv2048-p96`,
  `onnx-int4-awq-p128`, `onnx-int4-awq-packed-p512`, visual `e6`, `e8`, `tiered-e1-e4` (14.6 GB).
- About 150 results referenced only by notes before 364 (9.0 GB), 14 superseded binaries (startup-*, async-decode
  trials), `builds/v0101-v3`, and pre-2026-09-29 scratch.

Notes 363 and earlier may cite deleted campaigns; their recorded numbers and conclusions stand, but those runs are no
longer reproducible from retained raw data without regeneration.
