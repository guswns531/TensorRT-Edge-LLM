# Gemma 4 model-port heuristic audit and full12 diagnostic

## Outcome

Changing the served model from Cosmos Reason2 2B to Gemma 4 E2B exposed two different classes of configuration that
must not be conflated:

1. model and engine capabilities, which define legal actions; and
2. scheduler heuristics, which choose among legal actions.

The corrected Gemma path passes one production HTTP/SSE diagnostic run for every retained workload role. Against a
fresh optimized vLLM 0.28 run on the same textual/image requests, TensorRT wins generated-token throughput in 9 of
12 workloads, TPOT mean and p95 in 12 of 12, and E2E mean in 10 of 12. It loses throughput in `vision-heavy`,
`wave-drain`, and scaled `multi-image`. TTFT mean wins only 2 of 12, so this is not a promotion result.

All numbers in this note are single-run diagnostics. They locate model-port problems; they are not citable confidence
intervals.

## What changed with the model

### Model contracts, not heuristics

| Item | Cosmos reference | Gemma 4 E2B | Classification |
|---|---:|---:|---|
| image soft tokens per image | model-specific | 280 | model-declared shape contract |
| text position encoding | one model path | dual M-RoPE, d256/d512 | model semantics |
| embedding preparation | normal embedding | 35 PLE inputs, hidden 256 | model semantics |
| KV geometry | model-specific | 35 layers, 1 KV head, head dim 256 | model semantics |
| maximum input/KV length | engine-specific | 1,024 / 2,048 | compiled engine capability |

The 280-token fix must remain. Treating it as a tunable serving parameter would recreate the original incorrect VLM
placement.

### Manually selected engine and memory capabilities

| Item | Retained Cosmos full12 | Gemma diagnostic | Why it changed |
|---|---:|---:|---|
| P capacity | 8 | 2 | smallest three-context engine that fits the RTX 3080 |
| D capacity | 64 | 4 | same memory constraint |
| E capacity | 4 | 2 requests / 2 media items | visual engine profile has 560 total tokens |
| stable KV owners | 80 | 4 | LLM engine and context-memory budget |
| KV pages | engine-specific large pool | 64 | compiled engine capability |
| active backend requests | 64 | 4 | stable-owner capacity |
| packed prefill | enabled | disabled | current Gemma engine is dense P2 |

These values strongly affect performance but are capability choices rather than policy rules. They still require a
memory/capacity sweep because the present three-context allocation consumes about 9,045 MiB and leaves too little
room for a larger frontier.

### Static policy values that remain

| Static value | Current source/command | Model-port risk |
|---|---|---|
| P chunk and overlap cap = 128 tokens | `phaseSchedulerOptions.inc` | retained compiled/runtime granularity; dense Gemma does not have Cosmos packed-P semantics |
| P completion bonus = 128 tokens | `phaseSchedulerOptions.inc` | policy value tied to the former chunk scale |
| max P cohort turns = 8 | `phaseSchedulerOptions.inc` | bounded mechanism with a static horizon |
| P/D queue targets = 5/2 ms | `phaseSchedulerOptions.inc` | time constants are not derived from current phase service time |
| decode batch cost table | `phaseSchedulerOptions.inc` | the 6.2--9.8 ms table was measured on the Cosmos P8/D64 path and is not a Gemma prior |
| E formation wait = 25 ms | benchmark contract | retained guard after its automatic replacement failed the earlier VLM gate |
| encoded downstream cap = 2 | benchmark contract/default | conservative ownership limit; likely constrains Gemma VLM TTFT |
| E initial cost/margin = 50/5 ms | `PhaseThreeCoordinatorConfig` | cold-start physical prior, not measured from Gemma |
| E text guard age = 250 ms | `PhaseThreeCoordinatorConfig` | fixed urgency boundary |
| E decode-pressure limit = 0.9 | `PhaseThreeCoordinatorConfig` | fixed normalized boundary |
| E max defer = 500 ms | `PhaseThreeCoordinatorConfig` | fixed starvation guard |
| prefix-before-vision minimum = 128 tokens | `PhaseThreeCoordinatorConfig` | fixed launch-amortization threshold |
| formation attribution = 4 dispatches | benchmark contract/default | fixed measurement horizon |
| hard prefill TTFT guard | benchmark contract | enabled even without an external SLO |
| encoder arbiter | benchmark contract | enabled; placement mechanism also contains fixed guards above |

The logged 20 ms decode fallback and 2.5 s vision TTFT fallback are present, but the V3 no-explicit-SLO path reports
`tpot_budget_us=0` and does not activate the former recovery/hysteresis controller. They are dormant compatibility
defaults in this campaign, not the direct cause of the measured result.

### Evaluation choices that were previously confounded with policy

The first Gemma comparison used only eight simultaneous requests and eight trace-derived warmup requests. That lets
the workload under test train its own predictor and provides almost no E-shape coverage. The new gate uses one shared,
workload-independent 65-request calibration trace generated for P2/D4/E2. The trace contains 17 resident-decode,
24 long-prefill, and 24 vision requests.

The 65-request calibration still did not reach every runtime-declared key: individual runs calibrated 6--10 keys out
of 28--32 targets. Its role is a common generic starting point, not proof of learner convergence.

## Expanded workload contract

The retained twelve roles are preserved:

| Workload | Requests | Primary stress |
|---|---:|---|
| short | 48 | short-output overhead and tail |
| balanced | 64 | ordinary P/D continuous batching |
| decode-heavy | 64 | long resident D4 service |
| long-prefill | 64 | dense P2 pressure |
| bimodal | 64 | short/long fairness |
| text-heavy | 64 | 48 text + 16 vision coexistence |
| mixed | 64 | 32 text + 32 vision |
| vision-heavy | 64 | 16 text + 48 vision |
| poisson | 64 | nonuniform online arrival |
| wave-drain | 20 | separated encoder waves and drain |
| multi-image | 20 | the former five-request trace repeated into four waves |
| late-vision | 32 | late E placement against long decode |

The materializer keeps the original request content, output-length distribution, and arrival offsets. It only caps
the four 288-request text traces at 64 requests, repeats multi-image to 20 requests, and rewrites local image URLs to
their container-visible path. The resulting suite has 632 measured requests.

Compared with the Cosmos gate, this is capability-scaled rather than literally load-identical. The old gate used
P8/D64/E4, 80 owners, max HTTP in-flight 64, and 717/957 generic warmup requests. Reusing those values on a four-owner
Gemma engine would mostly measure admission queueing rather than the same scheduling phenomenon.

## One-run diagnostic results

`m/p95` columns are mean and p95 in milliseconds.

| Workload | TRT tok/s | vLLM tok/s | TRT advantage | TRT TTFT m/p95 | vLLM TTFT m/p95 | TRT TPOT m/p95 | vLLM TPOT m/p95 | TRT E2E m/p95 | vLLM E2E m/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 371.7 | 288.5 | +28.9% | 590/737 | 518/734 | 10.02/11.42 | 23.43/24.96 | 795/1,027 | 1,005/1,389 |
| balanced | 480.7 | 324.3 | +48.2% | 1,838/2,190 | 1,778/2,562 | 7.93/8.27 | 22.38/22.81 | 2,503/3,141 | 3,659/5,109 |
| decode-heavy | 504.9 | 331.4 | +52.3% | 5,183/6,273 | 5,039/6,555 | 7.63/7.77 | 22.19/22.30 | 7,120/8,895 | 10,679/15,000 |
| long-prefill | 289.2 | 261.4 | +10.6% | 3,203/4,013 | 2,328/3,164 | 11.64/14.58 | 27.39/29.03 | 4,168/5,134 | 4,622/6,296 |
| bimodal | 376.2 | 298.0 | +26.2% | 4,025/4,990 | 3,218/4,723 | 9.85/12.06 | 24.25/27.77 | 5,451/8,195 | 6,802/11,868 |
| text-heavy | 398.0 | 265.5 | +49.9% | 1,396/5,241 | 1,717/2,725 | 8.63/9.87 | 20.83/24.97 | 1,845/5,471 | 2,791/4,152 |
| mixed | 312.5 | 300.2 | +4.1% | 1,653/4,046 | 1,153/1,692 | 8.71/9.86 | 24.15/27.90 | 2,039/4,320 | 2,224/3,104 |
| vision-heavy | 246.7 | 284.7 | -13.3% | 1,854/2,434 | 1,052/1,687 | 8.80/9.98 | 25.06/27.18 | 2,182/2,730 | 1,980/2,552 |
| poisson | 428.4 | 308.6 | +38.8% | 1,827/6,963 | 1,516/2,304 | 8.48/9.60 | 23.00/23.69 | 2,434/7,207 | 3,176/4,554 |
| wave-drain | 91.8 | 92.4 | -0.7% | 339/709 | 163/213 | 8.00/9.35 | 23.00/24.38 | 587/942 | 876/895 |
| multi-image | 188.6 | 232.0 | -18.7% | 1,219/2,313 | 560/1,041 | 8.76/9.97 | 25.80/27.12 | 1,490/2,619 | 1,360/1,783 |
| late-vision | 516.8 | 352.8 | +46.5% | 3,109/4,409 | 3,305/4,489 | 7.44/7.65 | 22.03/22.06 | 4,174/5,770 | 6,461/8,572 |

### Interpretation

- The TensorRT D path is consistently strong: TPOT mean and p95 win 12/12.
- The weak path is E admission and first-token progress. `vision-heavy` and `multi-image` lose throughput and TTFT;
  `wave-drain` is throughput-neutral but has materially worse TTFT.
- `long-prefill` narrows the throughput lead to 10.6% and raises TensorRT TPOT p95 to 14.58 ms. This is the clearest
  current evidence that dense P2, rather than the V3 selector alone, limits the Gemma port.
- Text-heavy and Poisson p95 show very late vision requests. The enabled arbiter, encoded cap two, and fixed max-defer
  path require causal A/B tests before changing the learned selector.
- TensorRT uses about 9,045--9,047 MiB while vLLM uses 8,143--8,227 MiB. vLLM also supports eight sequences while
  TensorRT has four stable owners. The VLM losses are therefore coupled to independent-context memory overhead.

The HTTP request content is identical, but each runtime applies its native Gemma chat/tokenizer path. Prompt token
counts consequently differ slightly. Generated token counts are fixed and equal. The vLLM diagnostic reused one
already-loaded server across the twelve traces, while TensorRT used a fresh backend and repeated generic calibration
for each trace. This biases startup state in vLLM's favor and is acceptable only for this diagnostic.

## Test and implementation changes

- `build_model_port_workload_gate.py` materializes a strict twelve-role gate from a retained campaign and records the
  SHA of every source and generated trace.
- `build_generic_policy_calibration_trace.py` now accepts P, D, and E engine capacities plus the prefill length. Its
  previous hard-coded P8/D64/E4 coverage was itself a model-port error.
- Unit tests cover default backward compatibility, P2/D4/E2 calibration coverage, deterministic multi-wave replay,
  and container-visible media paths.
- All twelve TensorRT and twelve vLLM request-level HTTP runs completed without OOM or invalid request/engine shape.

## Next gates

1. Repeat the current V3 and vLLM full12 three times with a fresh server per repeat and report confidence intervals.
2. Run V0 exact, V1 scalar, V2 scalar-transition, and V3 service-scaled-transition with the same Gemma binary, engine,
   generic calibration, memory limits, and twelve traces. Only the estimator/reasoning mode may differ.
3. A/B the inherited Cosmos decode cost table against measured-only cold start. Do not tune new Gemma table entries.
4. On `vision-heavy`, `wave-drain`, `multi-image`, `text-heavy`, and `poisson`, isolate encoded capacity 2 versus 4,
   E wait 25 ms versus event-derived waiting, and arbiter on/off. Treat these as causal tests, not workload profiles.
5. Build a memory-reduced or shared-workspace Gemma engine so P4/D8 and at least eight stable owners can be tested.
6. Add load tiers at low, knee, and overload plus output lengths 64/128/256. Report service latency separately from
   client-side admission delay.
7. Promote only after output identity/semantic checks and three-repeat TTFT, TPOT, E2E, throughput, memory, activity,
   and SLO-surface gates pass.

## Retained artifacts

| Artifact | Path |
|---|---|
| materialized full12 and provenance | `.local/results/gemma4-e2b-awq-full12-20260911/inputs` |
| capability-aware generic calibration | `.local/results/gemma4-e2b-awq-full12-20260911/inputs/generic-gemma-p2-d4-e2.json` |
| TensorRT V3 1x | `.local/results/gemma4-e2b-awq-full12-20260911/trt-v3-generic/full12-1x` |
| optimized vLLM 1x | `.local/results/gemma4-e2b-awq-full12-20260911/vllm-optimized/full12-1x` |
| machine-readable comparison | `.local/results/gemma4-e2b-awq-full12-20260911/comparison-1x.json` |
| manifest | `.local/results/gemma4-e2b-awq-full12-20260911/manifest.json` |
