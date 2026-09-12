# Gemma 4 E2B asymmetric batch frontier

## Outcome

The original P2/D4/E2 Gemma 4 E2B AWQ engine was not the largest configuration supported by the RTX 3080. This
campaign built and executed P2/D8, P2/D16, P4/D8, P4/D16, P4/D32, and E4 profiles with the same v0.10.1 phase
runtime. The useful operating frontier moved to P4/D16/E2, while P4/D16/E4 is a high-throughput VLM point with only
541 MiB of measured headroom. P4/D32 executes correctly but misses the 512 MiB headroom gate.

This is a one-repeat diagnostic campaign. It proves capability and exposes the latency/throughput frontier; it does
not yet promote a new default.

## Fixed contract

- source: `codex/v0101-phase-forward-port` at `32045ed5e94b7ac90dae66188e39481a204c3266`
- model: Gemma 4 E2B IT, INT4-AWQ backbone, FP16 PLE/embedding/LM head/KV
- runtime: three independent TensorRT E/P/D contexts, stable indexed-paged ownership, 128-token KV pages
- policy: `service-scaled-transition`, workload-independent generic calibration, no trace-derived calibration
- text engine: max input 1,024, max KV 2,048, dense prefill
- vision engine: 280 output tokens per image
- GPU: RTX 3080, 10,240 MiB reported total memory
- generated token counts were complete in every retained run

The generic calibration trace was regenerated for each supported P/D/E capacity. This avoids invalid shapes, but it
also means the cap sweep compares equal calibration methodology rather than an identical set of calibration
requests.

## Built engine profiles

| Profile | KV pages | P activation | D activation | Build/run |
|---|---:|---:|---:|---|
| P2/D8 | 64 | 367,291,392 B | 34,643,456 B | pass/pass |
| P2/D16 | 64 | 367,291,392 B | 69,283,328 B | pass/pass |
| P4/D8 | 64 | 734,579,712 B | 34,643,456 B | pass/pass |
| P4/D16 | 64 | 734,579,712 B | 69,283,328 B | pass/pass |
| P4/D32 | 96 | 734,579,712 B | 138,563,584 B | pass/pass |
| E4 | n/a | 313,528,320 B | n/a | pass/pass |

The P workspace approximately doubles from P2 to P4. The D workspace grows much more slowly, which is why raising
D is the most memory-efficient way to expand throughput. The previous E2 runtime needed about 145 MiB of visual
context workspace; the E4 engine needs about 299 MiB, a 192 MiB increase in measured ready/peak memory after all
runtime allocations are included.

Large external sidecars were hard-linked to the canonical ONNX export instead of copied into every engine. Combined
P/D/E artifact roots are symbolic-link compositions and consume only one filesystem block for the links.

## Same-engine P/D cap sweep

The following points all use the same P4/D16/E2 engine, 16 stable owners, 16 server-side active requests, the same
64-request balanced HTTP trace, and the same binary. Only the runtime P/D cap and matching generic calibration
coverage change.

| Runtime cap | tok/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms | Peak MiB |
|---|---:|---:|---:|---:|---:|
| P2/D4 | 497.2 | 2,101/3,353 | 30.10/32.34 | 4,614/6,962 | 9,493 |
| P2/D8 | 535.7 | 2,208/3,446 | 25.04/29.89 | 4,280/6,711 | 9,497 |
| P2/D16 | 856.8 | 1,196/1,814 | 16.41/17.93 | 2,567/3,739 | 9,505 |
| P4/D4 | 480.8 | 2,174/3,356 | 31.44/33.92 | 4,805/7,136 | 9,493 |
| P4/D8 | 530.8 | 2,584/3,980 | 20.47/29.85 | 4,279/6,613 | 9,497 |
| P4/D16 | **886.4** | **1,139/1,780** | **16.17/17.00** | **2,491/3,613** | 9,507 |

This is the cleanest result in the campaign:

1. D16 is the dominant improvement. Relative to P2/D8, P2/D16 raises throughput by 60.0%, lowers TTFT mean by
   45.8%, lowers TPOT mean by 34.5%, and lowers E2E mean by 40.0%.
2. P4 is not universally beneficial. At D4 and D8 it slightly reduces throughput because larger P work competes with
   decode without enough D amortization. At D16 it raises throughput by 3.5% and reduces TTFT/E2E.
3. A small D cap does not guarantee low TPOT when 16 requests remain active. D4 must rotate four cohorts, producing
   a 30--31 ms mean TPOT. Active-owner capacity and active decode batch must therefore be controlled together.
4. The engine maximum should remain a capability. The scheduler needs to choose the active frontier from observed
   service cost instead of treating the maximum as the fixed operating point.

The old P2/D4 engine result in note 290 used only four stable owners and is not a causal baseline for this table. It
had lower TPOT because it admitted fewer resident decode sequences. The table above intentionally holds ownership
capacity at 16 to isolate runtime batch caps.

## Workload sentinels across compiled profiles

These rows answer different capability questions and must not be interpreted as one uniform workload comparison.

| Engine/profile | Workload | tok/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms | Peak MiB |
|---|---|---:|---:|---:|---:|---:|
| old P2/D4/E2 | short | 371.7 | 590/737 | 10.02/11.42 | 795/1,027 | 9,045 |
| P2/D8/E2 | short | 408.6 | 387/562 | 17.84/20.18 | 750/999 | 9,065 |
| old P2/D4/E2 | decode-heavy | 504.9 | 5,183/6,273 | 7.63/7.77 | 7,120/8,895 | 9,045 |
| P2/D16/E2 | decode-heavy | 977.6 | 2,941/4,510 | 14.82/15.21 | 6,691/9,540 | 9,109 |
| P4/D32/E2, 96 pages | decode-heavy | 1,439.6 | 2,626/6,454 | 15.41/16.12 | 6,520/10,853 | 9,761 |
| old P2/D4/E2 | long-prefill | 289.2 | 3,203/4,013 | 11.64/14.58 | 4,168/5,134 | 9,045 |
| P4/D8/E2 | long-prefill | 358.7 | 1,967/2,629 | 18.25/20.35 | 3,498/4,592 | 9,465 |
| old P2/D4/E2 | balanced | 480.7 | 1,838/2,190 | 7.93/8.27 | 2,503/3,141 | 9,045 |
| P4/D16/E2 | balanced | 886.4 | 1,139/1,780 | 16.17/17.00 | 2,491/3,613 | 9,507 |

Increasing concurrency changes both capacity and the latency trade-off. P2/D16 nearly doubles decode-heavy
throughput over the four-owner engine, while P4/D32 reaches 1,439.6 tok/s. Neither should be called a universal
speedup: TPOT rises because more resident sequences share decode service, and P4/D32 has worse tail TTFT/E2E than
P2/D16.

## Encoder capacity

The same vision-heavy 16-text/48-vision HTTP trace was used for the following comparison.

| P/D/E configuration | tok/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms | Peak MiB |
|---|---:|---:|---:|---:|---:|
| old P2/D4/E2 | 246.7 | 1,854/2,434 | 8.80/9.98 | 2,182/2,730 | 9,045 |
| P4/D16/E2 engine | 271.1 | 2,803/5,185 | 11.37/18.37 | 3,267/5,471 | 9,507 |
| P4/D16/E4 engine, runtime E2 cap | 271.9 | 2,796/5,172 | 11.34/18.56 | 3,259/5,457 | 9,695 |
| P4/D16/E4 engine, runtime E4 cap | **388.4** | **1,860/3,234** | 13.65/19.43 | **2,401/3,625** | 9,699 |
| frozen vLLM diagnostic | 284.7 | 1,052/1,687 | 25.06/27.18 | 1,980/2,552 | 8,143--8,227 |

The E2 cap on the E4 engine reproduces the E2-engine performance within 0.3%, while consuming 188 MiB more. The
E4-cap improvement is therefore caused by increased encoded ownership/formation, not by a faster visual tactic.
Relative to P4/D16/E2, E4 raises throughput by 43.3% and lowers TTFT/E2E substantially. Relative to the old engine it
raises throughput by 57.4%; relative to the retained vLLM diagnostic it is 36.4% higher in token throughput.

The trace formed an actual maximum encoder batch of three, not four. Capacity four still reduced downstream
serialization enough to produce the gain. This point does not beat vLLM in TTFT or E2E tail, and its 541 MiB
headroom is only 29 MiB above the project gate. It remains a characterization point pending repeats and transient
memory stress.

## Memory and KV interpretation

KV is not the reason the previous engine stopped at P2/D4/E2. The indexed-paged allocator decouples stable logical
owners from active phase rows, and the 64-page pool completed P4/D16/E4 without reservation waits. D32 used 96 pages
and also completed without page pressure.

The limiting allocation is phase context workspace:

```text
fixed model + FP16 embedding + Gemma PLE
                 |
       independent TRT contexts
       /          |          \
  P workspace  D workspace  E workspace
   P4: 735 MB   D16: 69 MB   E4: 314 MB
                 |
          indexed-paged KV pool
```

Measured headroom is:

| Point | Peak MiB | Headroom from 10,240 MiB | Gate |
|---|---:|---:|---|
| P4/D16/E2 | 9,507 | 733 | pass |
| P4/D16/E4 | 9,699 | 541 | pass, narrow |
| P4/D32/E2 | 9,761 | 479 | fail |

P8 would add roughly another P4-sized workspace, taking the independent-context process beyond 10 GiB. It should
not be built as a production candidate until packed/ragged Gemma prefill, shared/tiered context memory, or a smaller
PLE/weight residency scheme removes at least 0.7--1.0 GiB.

## Correctness and limitations

- All retained HTTP runs returned the requested token count and no KV page waits, invalid slot, OOB, or OOM errors.
- Token hashes differ across batch profiles. This can be numerical row-order/tactic sensitivity, but exact
  cross-engine greedy identity has not passed and remains a promotion blocker.
- The retained values are one run each. They are diagnostics, not confidence intervals.
- Some cross-profile rows also change stable-owner/client-inflight capacity. Use the same-engine balanced table for
  causal P/D-cap conclusions.
- The frozen vLLM values use a different server lifecycle and are context only. Fresh equal-contract comparison is
  still required for a citable claim.

## Next gates

1. Use the P4/D16/E4-capable artifacts but sweep active owners and D cap together: `(owners,D) = (4,4), (8,8),
   (16,8), (16,16)`. Select from online service measurements, not a model/workload name.
2. Repeat P4/D16/E2 and P4/D16/E4 sentinels three times. Require at least 512 MiB headroom under transient vision
   allocation, deterministic per-configuration token hashes, and no ownership/page errors.
3. Run the twelve-workload gate on the two finalists. Report throughput, TTFT, TPOT, E2E, client admission delay,
   phase activity, and memory separately.
4. Add canonical row-order and cross-engine logit/token diagnosis before promotion.
5. Replace the fixed maximum-active choice with an endogenous controller over the measured `(active owners, P, D,
   E)` service surface. The engine maximum remains capability; the controller protects decode continuity without a
   workload label or explicit user SLO.
6. Revisit P8 only after reducing P workspace or sharing context memory. D64 is lower priority because D32 already
   violates the headroom gate and does not improve tail latency.

## Retained artifacts

| Artifact | Path |
|---|---|
| build logs and exact replay scripts | `.local/results/gemma4-e2b-awq-batch-frontier-20260911` |
| P2/D8 engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-asym-p2-d8-kv2048-soft280` |
| P2/D16 engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-asym-p2-d16-kv2048-soft280` |
| P4/D8 engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-asym-p4-d8-kv2048-soft280` |
| P4/D16 engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-asym-p4-d16-kv2048-soft280` |
| P4/D32 engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-asym-p4-d32-kv2048-p96-soft280` |
| E4 engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/visual-e4-soft280` |
| campaign manifest | `.local/results/gemma4-e2b-awq-batch-frontier-20260911/manifest.json` |
