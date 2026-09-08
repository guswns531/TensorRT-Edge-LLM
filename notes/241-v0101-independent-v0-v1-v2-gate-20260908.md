# v0.10.1 Independent E/P/D V0/V1/V2 Gate

## 1. Outcome

The tied-head and profile-local workspace fixes recover the intended v0.10.1
memory frontier. The repaired engine runs all 12 retained HTTP traces with
independent E/P/D TensorRT contexts on the RTX 3080 10 GiB device. Ready memory
is 9,281 MiB and the largest measured peak is 9,333 MiB, leaving approximately
907 MiB of device headroom.

The scheduler result is mixed rather than a universal V2 promotion:

- V1 Scalar is the best aggregate policy at `+0.82%` geometric-mean request and
  token throughput over V0, with 8/12 throughput wins.
- V2 Scalar+Transition is `+0.34%` over V0, with 6/12 wins. It is materially
  better on long-prefill and mixed, but worse on balanced and decode-heavy.
- Cross-policy greedy identity is not yet a production pass. V1 matches V0 on
  8/12 traces and V2 matches V0 on 10/12.
- The repaired v0.10.1 system is still behind the retained v0.10.0 system and
  frozen vLLM table on the VLM-heavy traces. This remaining gap is not caused
  by a larger KV pool.

## 2. Reproducibility contract

All V0/V1/V2 runs use the same:

- branch and binary: `codex/v0101-phase-forward-port` and
  `.local/v0101-forward-build-make`;
- model: `nvidia/Cosmos-Reason2-2B`, FP16;
- engine: E4/P8/D64, fixed P128, 256 KV pages, vision-prefill profile 4;
- stable indexed-paged KV ownership, 80 logical slots, and 2,048 token slot
  capacity;
- generic workload-independent calibration;
- request traces, output lengths, request order, admission, memory limits, and
  CUDA graph setting;
- independent E/P/D TensorRT contexts in one CUDA context.

The old command manifest forced
`TRT_EDGELLM_SHARED_VISION_DECODE_CONTEXT_MEMORY=1`. The new matrix explicitly
removes that environment variable. The harness now supports repeatable
`--drop-backend-env NAME` so inherited mechanism flags cannot silently alter an
ablation.

Retained engine identity:

```text
path    .local/v0101-forward-artifacts/cosmos-reason2-2b/
        engine-p8-d64-vp4-kv256-p128/llm.engine
size    3,081,676,788 bytes
SHA256  50493717b98f2c3b7dcbac4bb41b5bc395bd70ecfa5f9c5f577bdd20de49dad1
```

This dedicated-P1024 plan was superseded by the shared-P128 engine in note 242
and removed after the replacement passed the same 12-workload gate. The raw
results and engine hash remain retained as the reproducibility record.

Raw retained results:

```text
.local/results/v0101-forward-port/http-full12-independent-fixed-v0
.local/results/v0101-forward-port/http-full12-independent-fixed-v1
.local/results/v0101-forward-port/http-full12-independent-fixed-v2
.local/results/v0101-forward-port/http-full12-independent-fixed-v0-v1-v2.json
.local/results/v0101-forward-port/http-full12-independent-fixed-v0-v1-v2.csv
```

These are one-run engineering measurements. The decisive text points are also
reported as three-run medians in Section 5.

## 3. Complete 12-workload result

Latency values are milliseconds. `hash=V0` reports exact captured greedy-token
trace identity against V0. The comparison tool remains strict by default; the
new opt-in mismatch mode records fidelity explicitly instead of discarding the
otherwise useful performance comparison.

| Workload | Variant | req/s | tok/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | Peak MiB | hash=V0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| short | V0 | 103.061 | 2233.0 | 115.0/198.4 | 13.98/25.69 | 377.3/459.9 | 9281 | yes |
|  | V1 | 107.635 | 2332.1 | 107.5/190.0 | 13.25/22.53 | 355.3/437.3 | 9281 | yes |
|  | V2 | 106.134 | 2299.6 | 104.0/184.6 | 13.70/23.66 | 362.0/445.8 | 9281 | yes |
| balanced | V0 | 48.327 | 4188.3 | 73.7/172.7 | 13.18/14.77 | 1196.6/1872.1 | 9281 | yes |
|  | V1 | 48.488 | 4202.3 | 68.7/172.0 | 13.14/15.24 | 1188.4/1897.1 | 9281 | yes |
|  | V2 | 46.713 | 4048.5 | 69.4/170.6 | 13.74/15.60 | 1238.4/1922.7 | 9281 | yes |
| decode-heavy | V0 | 19.111 | 4968.9 | 72.0/195.7 | 11.21/11.99 | 2971.5/4553.5 | 9281 | yes |
|  | V1 | 19.109 | 4968.3 | 73.4/195.7 | 11.22/11.96 | 2969.8/4584.7 | 9281 | yes |
|  | V2 | 18.194 | 4730.5 | 77.0/218.6 | 11.81/12.43 | 3126.9/4777.6 | 9281 | yes |
| long-prefill | V0 | 13.083 | 1133.8 | 2147.4/2986.9 | 28.44/33.14 | 4579.8/6680.3 | 9281 | yes |
|  | V1 | 12.902 | 1118.2 | 2208.0/2791.0 | 28.39/33.52 | 4640.2/6677.0 | 9281 | yes |
|  | V2 | 14.331 | 1242.0 | 2056.2/2698.7 | 24.71/29.27 | 4160.2/5763.1 | 9281 | yes |
| bimodal | V0 | 12.211 | 1872.3 | 1966.7/4250.9 | 18.18/27.83 | 4502.9/9483.4 | 9281 | yes |
|  | V1 | 12.562 | 1926.1 | 1911.1/3999.3 | 17.68/27.57 | 4347.9/8978.0 | 9281 | yes |
|  | V2 | 12.297 | 1885.5 | 1906.4/4534.0 | 18.08/27.89 | 4404.1/9392.5 | 9281 | yes |
| text-heavy | V0 | 23.397 | 1240.0 | 458.6/2065.2 | 17.96/23.92 | 1412.9/2612.7 | 9325 | yes |
|  | V1 | 23.534 | 1247.3 | 466.0/2088.1 | 17.47/22.38 | 1400.0/2556.3 | 9333 | no |
|  | V2 | 22.672 | 1201.6 | 450.1/2124.6 | 19.66/25.06 | 1497.8/2694.5 | 9319 | no |
| mixed | V0 | 11.910 | 544.9 | 1375.4/4699.0 | 17.67/23.44 | 2193.8/5238.4 | 9325 | yes |
|  | V1 | 12.093 | 553.3 | 1307.7/4572.8 | 16.65/20.01 | 2060.3/5160.0 | 9325 | no |
|  | V2 | 12.731 | 582.4 | 1241.3/4343.0 | 15.20/19.17 | 1945.3/4856.2 | 9333 | no |
| poisson | V0 | 19.403 | 1416.4 | 390.9/2108.8 | 22.05/27.73 | 2018.1/2956.5 | 9333 | yes |
|  | V1 | 20.435 | 1491.7 | 396.6/1987.5 | 20.27/27.92 | 1892.3/2780.5 | 9333 | no |
|  | V2 | 19.464 | 1420.9 | 401.1/2126.3 | 20.79/25.68 | 1945.9/2874.1 | 9325 | yes |
| vision-heavy | V0 | 8.214 | 316.2 | 2915.2/7091.2 | 16.52/20.21 | 3547.4/7617.5 | 9317 | yes |
|  | V1 | 8.340 | 321.1 | 2818.0/7038.7 | 15.96/18.90 | 3422.9/7508.6 | 9325 | no |
|  | V2 | 8.054 | 310.1 | 2894.4/7252.6 | 17.04/19.31 | 3539.1/7809.8 | 9317 | yes |
| wave-drain | V0 | 2.916 | 93.3 | 329.4/640.2 | 8.47/12.50 | 591.9/848.4 | 9325 | yes |
|  | V1 | 2.912 | 93.2 | 332.5/680.1 | 9.08/12.57 | 613.9/886.7 | 9325 | yes |
|  | V2 | 2.916 | 93.3 | 326.3/630.0 | 8.94/12.68 | 603.3/837.7 | 9325 | yes |
| late-vision | V0 | 15.369 | 2216.9 | 179.5/771.0 | 10.64/10.69 | 1704.3/2080.2 | 9325 | yes |
|  | V1 | 15.703 | 2265.2 | 175.7/737.0 | 10.42/10.46 | 1667.8/2035.9 | 9325 | yes |
|  | V2 | 15.519 | 2238.6 | 175.9/745.5 | 10.56/10.63 | 1689.5/2060.0 | 9325 | yes |
| multi-image | V0 | 5.637 | 180.4 | 364.8/606.2 | 10.62/12.51 | 694.1/849.4 | 9317 | yes |
|  | V1 | 5.247 | 167.9 | 370.2/681.3 | 10.86/12.52 | 706.8/926.0 | 9325 | yes |
|  | V2 | 5.457 | 174.6 | 383.0/636.5 | 11.06/13.67 | 725.8/879.5 | 9317 | yes |

## 4. Aggregate policy comparison

Geometric means below are relative to V0. Positive means better. Latency wins
count a lower value as a win.

| Metric | V1 | V1 wins | V2 | V2 wins |
|---|---:|---:|---:|---:|
| request/token throughput | +0.82% | 8/12 | +0.34% | 6/12 |
| TTFT mean | +1.41% | 6/12 | +2.07% | 9/12 |
| TTFT p95 | +1.05% | 8/12 | +0.25% | 6/12 |
| TPOT mean | +1.83% | 9/12 | +0.61% | 6/12 |
| TPOT p95 | +3.29% | 7/12 | +2.42% | 6/12 |
| E2E mean | +1.88% | 9/12 | +1.00% | 7/12 |
| E2E p95 | +0.74% | 8/12 | +1.24% | 7/12 |

V1 is the current aggregate candidate. V2 is not removed: its `+6.89%` mixed
and `+9.54%` one-run long-prefill throughput gains show that deterministic
transition reasoning has real useful regions. Its balanced and decode-heavy
regressions show that the current terminal value or invocation frontier is not
yet sufficiently conservative for decode-dominant states.

## 5. Three-run confirmation

The three selected traces use fresh processes and repeat the full generic
calibration before each measured run. Values are medians of three runs.

| workload | V0 req/s | V1 req/s | V2 req/s | V1 vs V0 | V2 vs V0 |
|---|---:|---:|---:|---:|---:|
| long-prefill | 13.132 | 12.867 | 14.341 | -2.01% | +9.21% |
| balanced | 49.591 | 48.929 | 46.619 | -1.33% | -5.99% |
| decode-heavy | 18.892 | 19.180 | 18.238 | +1.52% | -3.46% |

The signs are stable enough to reject a single-run-noise explanation. V2's
long-prefill gain and decode-dominant regressions are both reproducible.

## 6. What was recovered relative to the earlier v0.10.1 port

Compared with the pre-fix v0.10.1 E4/P8/D64 matrix in note 239:

- text-only token-throughput geometric means change by `+1.78%`, `-0.12%`,
  and `+5.31%` for V0, V1, and V2 respectively;
- peak memory falls from roughly 9,819--9,873 MiB to 9,281--9,333 MiB;
- the formerly mandatory shared E/D arena can be removed, so E/P/D are again
  separately enqueueable;
- every retained workload runs without OOM with more than the 512 MiB
  headroom gate.

Thus the v0.10.1 `+582 MiB` regression and its local text-engine frontier are
resolved. This statement is relative to the earlier v0.10.1 port, not to the
older v0.10.0 performance champion.

## 7. Remaining v0.10.0 and vLLM gap

Relative to the retained v0.10.0 V0/V1/V2 token-throughput tables, the repaired
v0.10.1 variants are approximately `-24.64%/-24.95%/-25.73%` geometric mean
over all 12 traces. Restricting the comparison to the five pure-text traces
reduces the gap to `-4.76%/-5.25%/-5.97%`. The large all-trace gap is therefore
primarily a VLM-path problem.

Against the unchanged frozen vLLM token-throughput table:

| Variant | geometric mean | wins |
|---|---:|---:|
| V0 | -14.75% | 4/12 |
| V1 | -14.05% | 3/12 |
| V2 | -14.46% | 3/12 |

This is not a fresh vLLM run. Reuse is valid for directional comparison because
the model, precision, requests, output lengths, and vLLM configuration are
unchanged. It must not be presented as a fresh confidence-interval result.

## 8. Why this is not a KV-cache regression

The repaired and pre-fix v0.10.1 systems use the same stable indexed-paged KV
ownership and the same 256-page physical pool. The current pool remains about
3,584 MiB. Slot/page allocation, release, and no-compaction semantics did not
change in this step.

The changed memory is compiled engine and execution-context memory:

```text
before
  large v0.10.1 text plan
  P workspace about 197 MiB
  D workspace about 539 MiB
  forced shared E/D arena

after
  tied-head external weight
  profile-local P workspace about 302 MiB for the VLM profile
  D workspace about 21 MiB
  independent E/P/D context arenas
```

The KV pool is a large fixed resident consumer, but it is not the source of the
recovered approximately 0.55 GiB or the remaining VLM throughput loss.

## 9. Evidence for the remaining VLM mechanism gap

The old pre-fix runs forced shared E/D context memory and consequently serialized
every encoder batch against decode. The repaired matrix removes that constraint.
The comparison is not a pure policy A/B because the engine plan also changed,
but its runtime telemetry identifies a concrete formation difference.

For V0 vision-heavy:

| property | old shared E/D | repaired independent |
|---|---:|---:|
| encoder requests | 48 | 48 |
| encoder batches | 41 | 45 |
| max E batch | 3 | 3 |
| exclusive E batches | 41 | 0 |
| explicit global overlaps | 0 | 0 |
| token throughput | about 383.6 tok/s | 316.2 tok/s |

Removing a physical exclusion boundary changes when E is allowed to drain, and
therefore changes future E cohort formation even when the policy reports no
explicit overlap action. This is execution--formation coupling at the mechanism
boundary. The next experiment must measure actual stream intervals, not infer
concurrency from action labels alone.

Generic VLM calibration currently reports zero E+P and E+D calibration probes
despite containing vision requests. P+D obtains 20--30 probes. Consequently V1
and V2 have little authority to replace the old implicit E/P opportunity with
explicit, measured E-pair actions. This is the leading scheduler-side target.

## 10. Preserved v0.10.0 engine reuse experiment

The smaller retained v0.10.0 P128/D64 plan is 2,839,069,516 bytes, about 242 MiB
smaller than the repaired v0.10.1 plan. Directly loading it in the v0.10.1
runtime is not valid. TensorRT deserializes the plan, but the first packed decode
fails in the v0.10.1 attention plugin with no matching SM86 GQA cubin key and
the process exits with code 139. The old serialized plan and new plugin/runtime
therefore cannot be used as a fair shortcut. A comparable v0.10.1 plan must be
rebuilt from v0.10.1 ONNX and validated through build and inference.

## 11. Next implementation and validation order

1. Capture actual E/P/D/Copy activity for V0 vision-heavy and mixed under the
   repaired engine, once independent and once with current-engine E/D exclusion.
   Compare E batch sequence, stream overlap masks, queue wait, and total E/P/D
   GPU work.
2. Fix generic VLM calibration so at least representative E+P and E+D physical
   samples reach the contextual models without workload labels.
3. Keep V1 as production candidate and run E-pair learning in shadow first.
   Activate only when held-out physical error, action fidelity, and SLO guards
   pass.
4. Add a bounded formation value for E dispatch timing so independence does not
   eagerly fragment E cohorts. It must use ready state and outstanding events,
   not a vision-heavy workload rule.
5. Re-run the four VLM diagnostics first: text-heavy, mixed, vision-heavy, and
   multi-image. Require no regression on the five pure-text controls.
6. Resolve V1's four and V2's two token-hash divergences before production
   promotion.
7. Only after the above, rerun the full 12-workload 3x gate and a fresh vLLM
   matrix.

The immediate architectural goal is not to restore forced sharing. It is to
retain independent TensorRT contexts while making the global selector reproduce
profitable E/P overlap, reject harmful E/D interference, and preserve future E
cohort formation from observable state.
