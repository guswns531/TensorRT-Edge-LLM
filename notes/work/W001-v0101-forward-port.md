---
id: W001
status: done
updated: 2026-10-01
notes: [236, 237, 238, 239, 240, 241, 242, 243]
---

# v0.10.1 phase forward port

## Goal
Carry the V0/V1/V2 phase-serving runtime (independent E/P/D contexts, stable paged-KV ownership, packed/atomic prefill,
continuous admission) onto the upstream v0.10.1 layout and recover the v0.10.0 memory and throughput frontier.

## Current state
- Base: upstream v0.10.1 `e8b2952`; the v0.10.1 paged-KV pool aligns with the old KV substrate but does not replace stable
  leases, E/P/D queues or online selection, so a staged semantic port was used (notes 237, 238).
- Port complete for the declared scope: public `llm_inference --phaseServing --phasePolicy=exact|scalar|scalar-transition`,
  12-workload V0/V1/V2 HTTP matrix, one/two-image semantic VLM (note 239).
- Memory regressions were not KV: duplicated tied LM head (593.5 MiB) and lost packed-attention early return (512 MiB)
  explained the text/decode arena growth; fixed workspaces match v0.10.0 (P 79.7 MB, D 21.5 MB) (note 240). Dropping the
  1,024-token vision-prefill profile cut ready memory 9,281 -> 9,061 MiB (note 242); restoring atomic P1024 costs +218 MiB
  and gives V1 +4.24% geomean over P128 (note 243).
- Throughput deltas in notes 242/243 against v0.10.0 (-11.22%, -5.64%) were measured on an unoptimized build with invalid
  VLM warmup assets; the corrected Release/validated comparison is -0.44% (V0), -1.43% (V1), -0.39% (V2) geomean (note 275,
  via 259/261/262). Frozen-vLLM margins quoted in 242/243 (+1.68%, +8.06%) are superseded by +11.5%/+11.7% for V1/V2 (note 275).
- Exact-output promotion was never reached in this line: V1 token hashes differ on four traces (note 243). V1 is the
  repeated-measurement reference; V2 is not a universal winner (notes 241-243).

## Conclusions
- 236 — Scheduler consolidated to V0 exact / V1 scalar / V2 scalar+transition (-14.5k net lines); V2 +1.81% vs V0, not production-promoted.
- 237 — Upstream v0.10.1 shares merge base v0.10.0 with the fork and adds a paged KV pool plus a RuntimeCoordinator split; blind merge rejected, staged port required.
- 238 — Compile-only checkpoint, architecture mapping and promotion gates (superseded by 239).
- 239 — Port functional: 12 workloads V0/V1/V2, direct encoder output binding, E4/D32 and D64 frontiers; V1 default candidate; three E4 cross-policy divergences open.
- 240 — Tied-head and workspace fixes restore v0.10.0 arenas and independent E/P/D contexts; KV unchanged.
- 241 — Independent E/P/D gate: V1 +0.82% vs V0, V2 +0.34%; remaining gap is VLM path (partly corrected by 242/243/262).
- 242 — Dedicated vision-prefill profile dominated memory; no-VP engine recovers 220 MiB and V1 +5.13%; corrected admission still -11.22% vs old (superseded by 275).
- 243 — Atomic P1024 plus residual P+D Scalar eligibility restored; V1 +4.24% vs P128; hash mismatch on four traces keeps promotion failed (parity numbers corrected by 262/275).

## Open questions
- Four-trace numerical/ownership cause of V1 hash divergence -> W003 (encoder payload, notes 247-251) and W002.
- Balanced and historical-v0.10 parity decomposition -> W002.
- Selector/formation behavior on VLM traces -> W004.

## Artifacts
- `.local/worktrees/upstream-v0101` (present)
- `.local/results/v0101-forward-port/atomic-vp1024-final-full12-r3` (present; also `-v0`, `-v2`)
- `.local/results/v0101-forward-port/atomic-final-v0-v1-v2.csv` (present)
- `.local/results/v0101-forward-port/http-full12-independent-fixed-v0-v1-v2.json` (present)
- `.local/results/v0101-forward-port/no-vp-v010-contract-v0-v1-v2.json` (present)
- `.local/current/engine-vp` (present)
- `.local/v0101-forward-build-make` (present)
- `.local/results/baselines/vllm-frozen-12x3` (dangling symlink to `.local/profile-free-global-20260827/r4-vllm-12x3`, target absent; note 370 cleanup)
