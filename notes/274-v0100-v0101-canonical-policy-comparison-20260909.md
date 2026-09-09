# 274. v0.10.0 and v0.10.1 canonical policy comparison

> This initial four-workload comparison is superseded for aggregate conclusions
> by the canonical 12-workload result in
> `notes/275-v0100-v0101-canonical-full12-20260909.md`.

## Outcome

The retained v0.10.0 V0/V1/V2 runtime was rerun against the same text engine,
vision engine, model, request traces, generic calibration, fixed-output
contract, KV capacity, and batch limits used by the canonical v0.10.1 gate.
Four workloads were measured three times per policy. The missing v0.10.1 V0
and V1 rows were also freshly measured; the canonical v0.10.1 V2 control from
P4 was reused because its contract is identical.

The comparison separates three effects:

- The V0 runtime baseline is at parity: v0.10.1 is -0.10% by geometric mean
  across the four workloads relative to v0.10.0.
- The V1 Scalar port is weaker: v0.10.1 is -2.36% relative to v0.10.0 V1.
- The V2 Scalar+Transition port is stronger: v0.10.1 is +1.87% relative to
  v0.10.0 V2, although neither V2 dominates its own V0 across this four-case
  subset.

The retained GPU-ready allocation is 9,581 MiB for v0.10.0 and 9,399 MiB for
v0.10.1. This 182 MiB reduction is not a smaller KV cache: both processes use
the same 256-page, 128-token/page FP16 KV pool and the same engine.

## Compared policies

| ID | Runtime policy | Learned value | Transition reasoning |
|---|---|---|---|
| V0 | Exact | exact CUDA registry and conservative fallback | no learned transition value |
| V1 | Scalar | contextual scalar overlap advantage | immediate action |
| V2 | Scalar+Transition | contextual scalar overlap advantage | bounded deterministic request/DAG transition |

Service-normalized authority is disabled. No workload-specific exception is
enabled. Generic calibration runs before every measured process, but measured
workload evidence is not carried into the next process.

## Reproduction contract

### Runtime identities

| Item | v0.10.0 | v0.10.1 |
|---|---|---|
| source commit | `f4eb53e899dd1a2319c0a75d4f06b47403648533` | `9898a6bc79ce464abefeb815a4f02b45f69536c0` |
| build | `.local/v010-forward-build` | `.local/builds/v0101-release` |
| executable SHA256 | `9e48a1253aa8d5dd17cbc1158bed1602d98ba3e252ba17ab8e26172f5acf0ea1` | `9959b6097df5dba67a531b901c6308d1594c4aa38fbaef6ac82e3b8798cc5598` |
| executable bytes | 22,133,672 | 22,160,752 |
| plugin bytes | 32,508,632 | 18,694,384 |

Both are CUDA 13.3, SM86, Release builds running in
`nvcr.io/nvidia/tensorrt:26.06-py3` on the same RTX 3080 10GB.

### Shared assets and limits

| Contract item | Value |
|---|---|
| model | `nvidia/Cosmos-Reason2-2B`, FP16 |
| text engine SHA256 | `084d039248e08dc192a57ebf38d521ee1fdca2f037a865f55ac0b0b904214b0b` |
| vision engine SHA256 | `e6afa43a7b6a915f4d4c1f820af89b579a3b4cb05dd57f8687f8ec158a4c7f40` |
| max P / D / E batch | 8 / 64 / 4 |
| prefill chunk | 128 tokens |
| active slots | 80 |
| KV pool | 256 pages x 128 tokens, FP16, 3,584 MiB payload |
| output contract | fixed requested length, EOS ignored |
| repetitions | 3 per policy and workload |

The v0.10.0 runtime needs older tied-engine metadata. The retained compatibility
view changes only `config.json`; `llm.engine`, embedding, tokenizer, and chat
template resolve to the same v0.10.1 files. Byte comparison of `llm.engine`
passes.

The replay command family is retained in:

```text
.local/results/v0101-forward-port/
  v0100-canonical-fourcase-20260909/
  v0101-canonical-fourcase-v0-v1-20260909/
  p0-p4-20260909/p4-screen-v2-control/
```

The replay harness accepts `--text-engine-dir` so a compatibility view can be
selected without editing the trusted command manifest.

## Throughput

All values are median generated token/s across three runs. The frozen vLLM
result is shown for context and was not rerun because the model, traces, output
length, and capacity contract did not change.

| Workload | frozen vLLM | 0.10.0 V0 | 0.10.0 V1 | 0.10.0 V2 | 0.10.1 V0 | 0.10.1 V1 | 0.10.1 V2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 4318.34 | **4520.18** | 4517.55 | 4511.72 | **4514.46** | 4477.28 | 4454.53 |
| text-heavy | 1634.76 | 2007.70 | **2032.39** | 1955.40 | **2013.48** | 1946.78 | 2001.06 |
| mixed | 921.48 | **1141.88** | 1109.52 | 1072.84 | **1129.79** | 1112.94 | 1111.62 |
| multi-image | 244.52 | 300.72 | **308.09** | 298.62 | 302.22 | 294.07 | **307.14** |

Best-in-version throughput remains above frozen vLLM by +4.67%, +24.32%,
+23.92%, and +26.00% for v0.10.0, and +4.54%, +23.17%, +22.61%, and +25.61%
for v0.10.1.

### Version delta at fixed policy

Positive means v0.10.1 is faster.

| Workload | V0 | V1 | V2 |
|---|---:|---:|---:|
| balanced | -0.13% | -0.89% | -1.27% |
| text-heavy | +0.29% | -4.21% | +2.34% |
| mixed | -1.06% | +0.31% | +3.61% |
| multi-image | +0.50% | -4.55% | +2.85% |
| four-case geometric mean | **-0.10%** | **-2.36%** | **+1.87%** |

### Policy delta inside each version

The geometric mean is normalized per workload to that version's V0.

| Version | V1 vs V0 | V2 vs V0 | throughput wins over V0 |
|---|---:|---:|---:|
| v0.10.0 | +0.18% | -2.41% | V1 2/4, V2 0/4 |
| v0.10.1 | -2.09% | -0.49% | V1 0/4, V2 1/4 |

This four-case result does not justify replacing V0 with V2 solely for raw
throughput. V2 remains useful because the wider 12-workload evaluation and
transition correctness cover behavior that this subset does not, and because
it stabilizes the multi-image runs described below.

## Request latency

Each cell is mean/p95 in milliseconds. Lower is better.

| Workload | Variant | TTFT | TPOT | E2E |
|---|---|---:|---:|---:|
| balanced | 0.10.0 V0 | 64.41/167.03 | 12.23/13.70 | 1106.37/1713.16 |
| balanced | 0.10.0 V1 | 61.45/166.80 | 12.26/13.98 | 1103.00/1721.61 |
| balanced | 0.10.0 V2 | 62.96/170.81 | 12.21/13.57 | 1103.06/1714.56 |
| balanced | 0.10.1 V0 | 61.57/164.47 | 12.25/13.56 | 1106.01/1716.73 |
| balanced | 0.10.1 V1 | 65.32/168.97 | 12.30/13.64 | 1112.52/1736.41 |
| balanced | 0.10.1 V2 | 65.23/167.68 | 12.33/13.77 | 1115.33/1721.23 |
| text-heavy | 0.10.0 V0 | 293.44/1061.26 | 24.64/38.80 | 1572.88/1678.29 |
| text-heavy | 0.10.0 V1 | 288.47/1023.25 | 24.92/38.71 | 1551.30/1656.89 |
| text-heavy | 0.10.0 V2 | 376.15/1037.37 | 24.17/37.06 | 1628.48/1719.33 |
| text-heavy | 0.10.1 V0 | 281.44/903.61 | 25.17/39.24 | 1565.89/1674.23 |
| text-heavy | 0.10.1 V1 | 332.42/1004.56 | 24.75/38.13 | 1617.71/1714.09 |
| text-heavy | 0.10.1 V2 | 334.75/1135.57 | 24.00/35.17 | 1571.54/1670.33 |
| mixed | 0.10.0 V0 | 677.56/1949.67 | 39.01/64.00 | 2405.08/2555.61 |
| mixed | 0.10.0 V1 | 708.46/2208.87 | 33.01/48.24 | 2263.92/2576.63 |
| mixed | 0.10.0 V2 | 740.45/2027.29 | 38.25/57.96 | 2527.41/2678.75 |
| mixed | 0.10.1 V0 | 625.46/1970.79 | 41.11/66.86 | 2434.96/2583.68 |
| mixed | 0.10.1 V1 | 729.21/2200.82 | 34.44/42.53 | 2343.92/2599.82 |
| mixed | 0.10.1 V2 | 718.40/2087.65 | 38.06/53.42 | 2450.86/2625.15 |
| multi-image | 0.10.0 V0 | 283.82/321.86 | 7.82/8.75 | 526.33/531.57 |
| multi-image | 0.10.0 V1 | 277.29/309.06 | 7.99/9.67 | 512.08/518.77 |
| multi-image | 0.10.0 V2 | 287.24/326.16 | 7.83/9.25 | 529.86/535.35 |
| multi-image | 0.10.1 V0 | 277.07/317.45 | 8.36/10.68 | 523.33/529.00 |
| multi-image | 0.10.1 V1 | 287.43/331.84 | 8.12/9.59 | 538.26/543.69 |
| multi-image | 0.10.1 V2 | 277.72/308.83 | 7.81/9.50 | 516.26/520.59 |

V2 old-to-new improves text-heavy and multi-image E2E p95 by 2.93% and 2.84%,
and mixed E2E p95 by 2.04%. Balanced V2 instead regresses mean TTFT by 3.47%
and mean E2E by 1.10%. There is no single latency winner across all metrics.

## Stability and output fidelity

All 24 policy/workload combinations are token deterministic across their three
runs. For each workload, all six version/policy combinations also have the same
token-trace SHA256. Request counts, generated-token counts, and trace SHA256
match across versions.

Multi-image exposes an important cold-run effect:

| Variant | token/s range across three runs |
|---|---:|
| 0.10.0 V0 | 239.96--312.39 |
| 0.10.0 V1 | 241.44--314.92 |
| 0.10.0 V2 | 295.77--302.25 |
| 0.10.1 V0 | 239.04--309.35 |
| 0.10.1 V1 | 239.45--297.38 |
| 0.10.1 V2 | 299.81--309.55 |

V0 and V1 have a large cold outlier in both releases. V2 is much tighter, so
the V2 multi-image result is not merely a favorable median. The source of the
cold outlier is not identified by this non-telemetry run and should be analyzed
as initialization/vision realization behavior rather than fitted with a
workload rule.

## Memory

| Runtime | GPU ready/peak MiB | 10 GiB headroom | Delta |
|---|---:|---:|---:|
| v0.10.0 V0/V1/V2 | 9,581 | 659 | reference |
| v0.10.1 V0/V1/V2 | 9,399 | 841 | -182 MiB (-1.90%) |

The equality across V0/V1/V2 within each release also shows that RLS and the
bounded transition evaluator are not material GPU-memory consumers. The
v0.10.1 plugin file is 13.8 MiB smaller, but this alone does not prove where all
182 MiB of device allocation was removed. A CUDA allocation ledger would be
needed for attribution. It is safe to conclude only that the reduction is not
caused by shrinking the KV pool.

## Architectural interpretation

1. **The base execution substrate is recovered.** V0 differs by only -0.10%
   geometrically, so the earlier large v0.10.1 text regression is absent under
   the canonical same-engine contract.
2. **V1 is the remaining weak port.** Its -2.36% version delta cannot be blamed
   on the text engine or KV capacity because V0 is at parity.
3. **V2 fixes part of the V1 behavior, not all V0 gaps.** It improves over old
   V2 on three workloads and makes multi-image stable, but it still trails the
   new V0 geometric mean by 0.49% on this subset.
4. **Overlap count is not a sufficient explanation.** In a representative
   repeat, old/new V2 overlap counts are 98/95 for balanced, 10/12 for
   text-heavy, 17/16 for mixed, and 2/1 for multi-image. The performance change
   therefore likely depends on action placement, completion order, WAIT, and
   resulting formation rather than simply enabling more overlap.
5. **Frozen vLLM remains a secondary reference.** Every best-in-release row
   beats frozen vLLM throughput, but the direct release comparison is stronger
   evidence because it uses byte-identical TensorRT engines and token traces.

## Decision and next work

- Keep v0.10.1 as the development baseline: V0 performance is at parity while
  device usage is 182 MiB lower.
- Do not claim that current V2 universally beats V0. On this four-case gate it
  wins raw throughput only for multi-image.
- Retain V0/V1/V2. V0 is the mechanism control, V1 isolates immediate learned
  action value, and V2 isolates bounded transition reasoning.
- Next, run paired full telemetry for old/new V1 and V2 on text-heavy, mixed,
  and multi-image. Compare canonical action hashes, requested/actual skew,
  completion order, WAIT placement, cohort sequence, and host submission gaps.
- Investigate the shared V0/V1 multi-image cold outlier before changing policy.
  A workload-specific warm-up or batch-size exception is not an acceptable fix.
