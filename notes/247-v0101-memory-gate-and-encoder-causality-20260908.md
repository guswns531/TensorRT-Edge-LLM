# 247. v0.10.1 memory gate, encoder causality, and corrected HTTP comparison

Date: 2026-09-08. Development branch: `codex/v0101-phase-forward-port`.
Parent source: `4f40cde`; diagnostic implementation commit: `5577e49`.
Status: **bounded-memory instrumentation passed; numerical origin localized for one request;
production exact-output and all-workload promotion are NOT complete**.

## 1. Scope and identity

Continues notes245/246, including the startup deepstack binding capacity fix in `79e8b12`.
No model quantization, KV reduction, allocator replacement, engine replacement, workload-specific
scheduler rule, or default graph policy change was made in this investigation.

- Cosmos-Reason2-2B FP16; P8/D64/E4; stable slots80; KV256 pages / 3584 MiB;
  context2048; text chunk128; atomic vision-prefill engine supports1024.
- Engine: `.local/v0101-forward-artifacts/cosmos-reason2-2b/engine-p8-d64-kv256-p128-vp1024-atomic`.
- Engine SHA256: `084d039248e08dc192a57ebf38d521ee1fdca2f037a865f55ac0b0b904214b0b`.
- Final diagnostic-capable binary SHA256: `8dfd17b241ef996d2248bdbd0a05ebf95fe2fe233af668556389c97869b43217`.
- GPU: RTX3080 10GB; host driver610.43.02; TensorRT26.06 container / TRT11 / CUDA13.3.
- All new artifacts: `.local/results/v0101-forward-port/closure-gates/`.
- Final full12 uses generic calibration: 239 text / 319 VLM requests, one fresh process per run.
  All encoder/logit diagnostic flags are disabled for performance measurement.

Successful memory instrumentation preceded the final embedding replay addition; retain the separate
commands, logs, and source progression rather than treating every diagnostic as one identical binary.
The full12 and fixed-embedding replay use the final binary identified above.

## 2. Full VLM memcheck: what passed, and what did not

The unbounded CUDA13.3 Compute Sanitizer2026.2 run still terminated with host signal11.
A diagnostic host backtrace places the fault in `__pthread_rwlock_rdlock` called from
`libsanitizer-collection.so`, through sanitizer callbacks / `cuLaunchKernel` / TensorRT.
This is evidence of a sanitizer-side failure path, not sufficient proof that every possible runtime
error is excluded. Explicitly loading the host CUDA driver instead of the container compatibility
library did not remove it. Both driver libraries report610.43.02.

The retained CUDA13.1 sanitizer2025.4 alternative emitted768 internal errors, including failure to
allocate instrumentation memory and untracked launches. **That run is invalid as a memory-safety gate.**

The successful configuration uses:

```text
compute-sanitizer --tool memcheck --force-synchronization-limit 1 --error-exitcode 99
TRT_EDGELLM_SEMANTIC_ONLY=1
zero_start, same engine, same KV256, actual multi-image HTTP requests
```

Use `memcheck-bounded/commands.json` for the authoritative resolved full command; the spelling of runtime
environment options is retained there. Semantic-only skips the synthetic startup benchmark, not
the actual encoder/prefill/decode serving work. The limit periodically drains launches to bound
instrumentation resource usage; it does not reduce the KV pool or filter kernels.

| Check | Result |
|---|---|
| Target exit | success |
| Sanitizer summary | **0 errors** |
| HTTP requests | 5 completed |
| Prompt/output tokens | 3075 / 160 |
| Captured output token IDs | 160 / 160 |
| Encoder requests started/completed | 5 / 5 |
| Encoder batch executions / max batch | 3 / 3 |
| Final available KV pages | 256 |
| Final downstream vision ownership | 0 |
| Peak sampled GPU memory | 9651 MiB |
| Instrumented duration | 413.250 s |

Artifacts: `memcheck-bounded.log`, `bounded-runner.log`,
`memcheck-bounded/zero_start/multi-image/worker-4/`.
Token trace SHA256: `9f801809d957d8a56fb0ec8f6e11cf3b70975b9b7e8bd9648875bd084fd28742`.

**This is a passing actual-serving memory-access test under bounded synchronization.**
It is not a throughput measurement, an unrestricted concurrent-overlap test, a racecheck result,
or full coverage of the synthetic startup microbenchmark. The roughly7-minute duration also means
the previous apparent sanitizer hang cannot be classified as a runtime deadlock from elapsed time alone.

## 3. Numerical cause: encoder payload rather than observed P/D rebatching

### 3.1 Opt-in implementation

`examples/llm/llm_phase_context_smoke.cpp` adds a measurement-epoch-only payload capture beside
the existing first64-step logits diagnostic:

- `TRT_EDGELLM_DIAGNOSTIC_LOGIT_DIR=<fresh directory>`.
- `TRT_EDGELLM_DIAGNOSTIC_REQUEST_ID=2`.
- `TRT_EDGELLM_DIAGNOSTIC_VISION_INPUTS=1` captures the final embedding and three deepstack tensors.
- `TRT_EDGELLM_DIAGNOSTIC_REPLAY_VISION_EMBEDDING=<FP16 file>` optionally replaces only that
  request's final embedding, after validating the byte length.

Capture/replay uses pinned host tensors and the explicit prefill stream. Diagnostic stream
synchronization protects the CPU read and pinned-buffer lifetime. Disabled mode adds no device
allocation or synchronization. Replay is **a causal intervention, not a serving optimization**;
it must never be enabled for production, semantic accuracy claims, or performance comparisons.

### 3.2 Three unfixed-payload runs

The multi-image workload has5 requests. Target request2 is its **single-image giant-panda request**;
the workload name does not mean this particular target has two images.

Each captured final embedding has991232 FP16 elements (484 x 2048). The three deepstack features
are byte-identical across all3 runs. They are finite, nonzero feature tensors, not zero-filled placeholders
(991214/991215/991216 nonzero elements respectively). The final embedding is not:

| Pair | Different elements | Max absolute delta | Mean absolute delta | RMS delta |
|---|---:|---:|---:|---:|
| 001 / 002 | 427539 | 0.015625 | 0.0003118212 | 0.0006509455 |
| 001 / 003 | 424166 | 0.015625 | 0.0003067186 | 0.0006374526 |

Final embedding hash prefixes: `e14e9039d2436520`, `a5afa13b05b0b7c6`, `ab8fa1b6f12d9ec8`.
Deepstack hash prefixes: `d857110e2ad9a69f`, `79e48b4c4307d96b`, `e54a2afb00b675c6`.
In pair001/002, the first greedy divergence is step26. Its later steps cannot be compared as
same-prefix numerical tests. The earlier note246 divergence at step14 belongs to another trajectory.

This visual model has24 blocks and deepstack taps at5/11/17. Equal intermediate taps and unequal
final outputs localize the variation beyond the observed taps, but do **not** identify the final
merger alone: blocks18–23 and the merger are still unseparated. Shape-dependent FP16/tactic effects
are a hypothesis; input/batch/tactic-controlled inspection is still necessary.

Artifacts: `payload-comparison/`, `payload-dumps/001..003`, `payload-pair12.json`.

### 3.3 Fixed-final-embedding causal replay

Replayed run001's final embedding for request2 in three fresh calibrated executions. The deepstack
features were not overwritten; they were already identical in the capture experiment.

| Comparison | Same-prefix steps | Equal greedy steps | Maximum logit absolute error |
|---|---:|---:|---:|
| Replay001 / 002 | 32 / 32 | 32 / 32 | **0** |
| Replay001 / 003 | 32 / 32 | 32 / 32 | **0** |

This remains true despite different batch membership and/or row positions, recorded per step in
`replay-pair12.json` and `replay-pair13.json`. Artifacts: `fixed-embedding-replay/`, `replay-dumps/`.

For this request, freezing the encoder payload removes the downstream numerical divergence.
That is stronger evidence than simply observing a near-tie. It supports the P/D stable ownership
path under these tested rebatchings, but is **not a proof for every request or KV access**.
It does not fix production encoder determinism. An E1-only fallback or frozen embedding would
change the execution contract and cannot be silently substituted for an E4 correctness gate.

## 4. Corrected full12 performance screen

Final is one new run per workload after the startup capacity fix; baseline is the prior Current
three-run median; frozen vLLM is three runs. All latency columns are client-send-relative ms;
client admission waiting is not included in those latency fields. Do not reinterpret them as
scheduled-arrival-relative latency. The four cap-mismatched text cases require the fresh comparison
below. Repeating only flagged workloads later does not turn the full12 screen into a three-run suite.

All1513 requests completed with188872 requested/generated/captured output token IDs. Peak sampled
memory ranged9279–9443 MiB, leaving at least797 MiB on a10240 MiB GPU. These are sampled peaks,
not an allocator high-water proof. One-run `token_trace_deterministic=true` is vacuous; it cannot
establish cross-repeat identity. Bimodal throughput fell3.31% against the prior median and vision-heavy
TPOTp95 increased37.38->45.87ms; they require repeated checks before any promotion decision.

| Workload | Variant | token/s | TTFT mean | p95 | TPOT mean | p95 | E2E mean | p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | baseline-r3 | 4164.17 | 71.21 | 174.29 | 13.28 | 15.03 | 1201.16 | 1882.52 |
| balanced | final-r1 | 4153.50 | 65.92 | 175.62 | 13.36 | 15.66 | 1202.12 | 1909.34 |
| balanced | frozen-vllm-r3 | 4319.89 | 146.79 | 365.02 | 15.12 | 17.40 | 1437.43 | 2244.11 |
| bimodal | baseline-r3 | 1869.52 | 1987.13 | 4537.04 | 18.16 | 26.00 | 4504.74 | 9628.46 |
| bimodal | final-r1 | 1807.55 | 2048.70 | 4091.54 | 18.98 | 28.53 | 4678.19 | 9700.55 |
| bimodal | frozen-vllm-r3 | 1868.29 | 2509.06 | 4291.79 | 23.67 | 42.84 | 5622.52 | 10328.21 |
| decode-heavy | baseline-r3 | 4898.64 | 71.79 | 207.18 | 11.37 | 12.24 | 3009.52 | 4639.49 |
| decode-heavy | final-r1 | 4922.52 | 70.35 | 208.22 | 11.34 | 12.20 | 2996.48 | 4606.38 |
| decode-heavy | frozen-vllm-r3 | 4854.34 | 153.43 | 394.23 | 14.00 | 15.03 | 3772.64 | 5812.44 |
| late-vision | baseline-r3 | 2399.25 | 130.03 | 493.15 | 9.81 | 9.86 | 1535.31 | 1922.13 |
| late-vision | final-r1 | 2410.17 | 128.94 | 491.02 | 9.77 | 9.82 | 1528.31 | 1913.35 |
| late-vision | frozen-vllm-r3 | 2359.23 | 153.40 | 631.51 | 9.90 | 9.93 | 1576.03 | 1954.46 |
| long-prefill | baseline-r3 | 1223.75 | 2068.31 | 2727.59 | 25.34 | 29.45 | 4221.91 | 5870.92 |
| long-prefill | final-r1 | 1227.16 | 2066.48 | 2677.17 | 25.29 | 29.64 | 4215.17 | 5673.73 |
| long-prefill | frozen-vllm-r3 | 1120.88 | 2944.21 | 4257.24 | 32.43 | 37.65 | 5696.02 | 7860.82 |
| mixed | baseline-r3 | 1078.53 | 714.19 | 2213.64 | 33.16 | 40.55 | 2273.13 | 2649.30 |
| mixed | final-r1 | 1092.62 | 706.04 | 2099.57 | 33.51 | 41.28 | 2282.96 | 2614.04 |
| mixed | frozen-vllm-r3 | 921.48 | 874.56 | 2541.43 | 46.97 | 84.02 | 3008.38 | 3140.85 |
| multi-image | baseline-r3 | 298.40 | 233.13 | 317.70 | 9.40 | 12.39 | 524.70 | 535.91 |
| multi-image | final-r1 | 299.68 | 232.24 | 315.51 | 9.36 | 12.32 | 522.42 | 533.46 |
| multi-image | frozen-vllm-r3 | 244.52 | 259.81 | 402.58 | 12.42 | 16.32 | 644.30 | 653.90 |
| poisson | baseline-r3 | 1878.52 | 218.41 | 764.79 | 22.81 | 41.01 | 1660.85 | 2125.02 |
| poisson | final-r1 | 1883.14 | 226.89 | 748.52 | 22.57 | 41.48 | 1665.74 | 2138.86 |
| poisson | frozen-vllm-r3 | 1800.07 | 438.11 | 902.68 | 22.19 | 45.67 | 1800.22 | 2266.55 |
| short | baseline-r3 | 2347.45 | 109.05 | 188.46 | 12.92 | 22.27 | 352.18 | 434.29 |
| short | final-r1 | 2363.04 | 107.67 | 186.18 | 12.87 | 22.67 | 349.43 | 431.54 |
| short | frozen-vllm-r3 | 1983.53 | 174.92 | 263.97 | 13.36 | 24.88 | 426.71 | 503.71 |
| text-heavy | baseline-r3 | 1872.49 | 314.58 | 1107.21 | 26.65 | 39.82 | 1687.72 | 1773.05 |
| text-heavy | final-r1 | 1838.21 | 323.14 | 1108.25 | 27.11 | 40.60 | 1720.32 | 1805.80 |
| text-heavy | frozen-vllm-r3 | 1634.76 | 421.58 | 1232.20 | 29.21 | 47.36 | 1943.42 | 2037.81 |
| vision-heavy | baseline-r3 | 668.94 | 1342.22 | 3167.36 | 30.52 | 37.38 | 2532.63 | 3606.61 |
| vision-heavy | final-r1 | 670.59 | 1354.02 | 3200.69 | 31.83 | 45.87 | 2622.63 | 3611.05 |
| vision-heavy | frozen-vllm-r3 | 579.20 | 1710.70 | 3691.37 | 63.70 | 119.58 | 4119.14 | 4229.37 |
| wave-drain | baseline-r3 | 96.31 | 248.30 | 398.29 | 9.89 | 13.90 | 555.92 | 615.70 |
| wave-drain | final-r1 | 96.34 | 246.06 | 394.36 | 9.88 | 13.70 | 552.43 | 611.95 |
| wave-drain | frozen-vllm-r3 | 95.85 | 252.76 | 418.60 | 12.43 | 17.26 | 637.86 | 649.42 |

## 5. Equal-client-admission vLLM comparison

The frozen client used80 concurrent requests in balanced/decode-heavy/long-prefill/bimodal,
whereas Current used64. This can materially change request latency; compare the fresh cap64 rows
before claiming that the old latency gap is a runtime improvement.

The reproducible `benchmarks/phase_serving/run_container_http_matrix.py` starts one fresh detached
Docker server per case, waits for health (including transient connection resets), runs the retained
HTTP client, saves exact commands and server/client logs, then stops/removes only that created
container. Existing result directories are rejected. Unit tests cover cleanup after a client failure,
retry after reset, and refusal to overwrite results.

Fresh vLLM image ID: `sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2`;
version0.27.1, same local HF checkpoint, FP16 weights/KV, KV budget3758096384 bytes,
max model length2048, max sequences80, max batched tokens8192, chunked prefill enabled,
prefix caching disabled, max2 images, processor pixels516096, processor cache0.
Client cap64, fixed requested output length / ignore-EOS. vLLM client warmup64 requests x32 tokens;
Current generic policy calibration is different and is explicitly not claimed identical training.
Fresh results are one run each, not confidence intervals. Exact commands are in
`closure-gates/vllm-equal-cap-commands.json` and each retained case directory.

| Workload | Variant | token/s | TTFT mean | p95 | TPOT mean | p95 | E2E mean | p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | Current64 | 4153.50 | 65.92 | 175.62 | 13.36 | 15.66 | 1202.12 | 1909.34 |
| balanced | vLLM64 | 4318.34 | 110.37 | 247.20 | 12.23 | 13.71 | 1154.26 | 1771.51 |
| bimodal | Current64 | 1807.55 | 2048.70 | 4091.54 | 18.98 | 28.53 | 4678.19 | 9700.55 |
| bimodal | vLLM64 | 1852.31 | 1573.46 | 2609.84 | 22.46 | 36.31 | 4689.19 | 9283.93 |
| decode-heavy | Current64 | 4922.52 | 70.35 | 208.22 | 11.34 | 12.20 | 2996.48 | 4606.38 |
| decode-heavy | vLLM64 | 4943.41 | 113.96 | 310.66 | 11.03 | 11.60 | 2969.50 | 4486.02 |
| long-prefill | Current64 | 1227.16 | 2066.48 | 2677.17 | 25.29 | 29.64 | 4215.17 | 5673.73 |
| long-prefill | vLLM64 | 1127.42 | 1925.36 | 3027.37 | 31.92 | 36.89 | 4643.91 | 6579.62 |

This comparison removes the earlier all-latency-win interpretation for the four text cases:

- **Balanced:** Current throughput about3.82% lower; TTFT better, but TPOT and E2E worse.
- **Decode-heavy:** Current throughput about0.42% lower; TTFT better; TPOT/E2E slightly worse.
- **Long-prefill:** Current throughput about8.85% higher and E2E better, but TTFT mean worse
  (2066.48 vs1925.36ms). TTFTp95 is better, so mean and tail cannot be conflated.
- **Bimodal:** Current throughput about2.42% lower; TTFT substantially worse, TPOT better;
  E2E mean approximately tied while E2Ep95 is worse. Recheck the Current run before attributing
  the full change to implementation, but the frozen cap80 latency advantage is not an equal-cap claim.

Fresh vLLM artifacts: `vllm-equal-cap-r1/<workload>/client/aggregate.json`, per-request CSV and server.log.
All four trace SHA256 values,288 requests per case, and generated token totals match Current's
corresponding run (balanced24960, bimodal44160, decode-heavy74880, long-prefill24960).
This validates trace/length equality, not exact cross-framework greedy token identity; vLLM's
retained HTTP client reports no captured token IDs in these runs.

| Workload | Current peak MiB | Fresh vLLM peak MiB | Current minus vLLM |
|---|---:|---:|---:|
| balanced | 9279 | 8959 | +320 |
| bimodal | 9279 | 9365 | -86 |
| decode-heavy | 9279 | 8967 | +312 |
| long-prefill | 9279 | 9245 | +34 |

These are sampled whole-process GPU usage values under the same configured KV byte budget, not
a decomposition of context/workspace/graph memory. The differences must not be attributed to KV
representation alone without allocator-level accounting.

The first `vllm-equal-cap/` attempt failed on a startup connection reset before measurement; it is
retained as harness failure evidence and excluded from all performance tables.

## 6. Promotion and remaining work

| Gate | Status / required next evidence |
|---|---|
| Startup deepstack binding capacity | Fixed previously; actual serving bounded memcheck now passes |
| Full unrestricted sanitizer | Open; sanitizer-side failure persists without bounded synchronization |
| Multi-image numerical localization | Encoder final payload causally implicated for request2 |
| Production cross-batch greedy identity | Open; no default precision/tactic correction established |
| Corrected full12 | 12/12 execution complete; single-run performance screen, not universal promotion |
| CUDA graph default | Not promoted; note246 long-prefill/mixed regressions remain |
| Compact telemetry | Diagnostic-only; prior observer-effect gate not waived |
| Same-snapshot policy replay | Not replaced by embedding replay; still a separate causal gate |
| V0/V1/V2 repeated final comparison | Earlier data retained; not newly repeated for all variants here |

Next numerical experiment must hold encoder inputs and batch packing fixed, then expose late-block
and merger outputs or compare precision/tactic variants through export -> build -> inference.
Do not round logits, substitute cached answers, reduce KV, or silently force E1 to satisfy exact output.
After a genuine correction, repeat multi-image/scaled-wave and full12 with the same runtime limits.
Do not add further policy heuristics while these mechanism and comparison gates remain open.

## 7. Validation

- Diagnostic-capable smoke binary built successfully.
- Changed-file pre-commit passed.
- Python tests: **22 passed** (HTTP container matrix, logits comparison, warmup matrix/sweep).
- Runtime suite: **619 passed, 1 NCCL-environment skip**;620 tests across51 suites.
  Full log: `closure-gates/runtime-tests.log`.

## 8. Two flagged workloads: additional repeat check

Two more fresh generic-calibrated runs use the same binary/engine/config and no diagnostic flags.
The following combines the original full12 run with those two additional runs. Median columns are
medians of the individual run statistics, not a pooled per-request p95 or a confidence interval.

| Workload | Repeat | token/s | TTFT mean | p95 | TPOT mean | p95 | E2E mean | p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bimodal | 1 | 1807.55 | 2048.70 | 4091.54 | 18.98 | 28.53 | 4678.19 | 9700.55 |
| bimodal | 2 | 1862.42 | 1955.44 | 3851.31 | 18.24 | 27.02 | 4492.86 | 9188.41 |
| bimodal | 3 | 1856.30 | 1965.59 | 4606.43 | 18.10 | 25.23 | 4505.49 | 9774.17 |
| bimodal | median | 1856.30 | 1965.59 | 4091.54 | 18.24 | 27.02 | 4505.49 | 9700.55 |
| vision-heavy | 1 | 670.59 | 1354.02 | 3200.69 | 31.83 | 45.87 | 2622.63 | 3611.05 |
| vision-heavy | 2 | 669.09 | 1341.04 | 3153.89 | 31.59 | 37.44 | 2564.84 | 3595.72 |
| vision-heavy | 3 | 670.50 | 1338.04 | 3147.40 | 31.50 | 37.38 | 2558.80 | 3588.07 |
| vision-heavy | median | 670.50 | 1341.04 | 3153.89 | 31.59 | 37.44 | 2564.84 | 3595.72 |

Bimodal's three-run median1856.30 token/s is0.71% below the previous1869.52, rather than the
first screen's3.31%. The three-run median TPOTp9527.02ms remains about3.9% above the old26.00ms;
this is not an all-metric3% promotion pass. Fresh vLLM1852.31 token/s is effectively tied in throughput,
but its TTFT remains lower. Current E2E mean4505.49 is lower than vLLM4689.19; E2Ep959700.55 is
higher than vLLM9283.93. It is a latency trade-off, not universal superiority.

Vision-heavy TPOTp95 returns to37.44/37.38ms in the additional runs. The three-run median37.44 is
close to the prior37.38; the original45.87ms spike was not reproduced twice. Its TPOT mean median31.59
is still about3.5% above the prior30.52, while E2E mean2564.84 is about1.3% higher. Retain the outlying
run rather than deleting it. Tail variability and residual mean differences remain visible.

All new source changes are opt-in diagnostics or external harness code; there is no intentional normal
policy behavior change to explain these fluctuations. That fact supports repeated measurement, not
an assertion that any observed difference must be noise.

This report deliberately separates completed diagnosis from an implemented production fix. It does
not assert that all remaining work has been solved.
