# Phase-Serving Experiment Methodology

Scope: the `benchmarks/phase_serving/` campaign that measures TensorRT Edge-LLM's phase-IPC serving runtime
(`llm_phase_context_smoke` behind an OpenAI-compatible gateway) against frozen vLLM and upstream baselines, using
12 fixed-length request traces per model, repeated three times. All facts below cite their source file or note; items
that could not be determined from code, retained artifacts, or notes are marked **Unknown**.

## 1. Purpose and summary

The campaign quantifies decode throughput, latency (TTFT/TPOT/E2E), and GPU memory of the fork's independent
encoder/prefill/decode (E/P/D) serving runtime under 12 synthetic-but-request-shaped workloads, for two models
(Gemma-4-E2B-it INT4-AWQ and Cosmos-Reason2-2B), against a frozen vLLM v0.28.0 reference and (for the v0.11.0 port)
an upstream clean-build baseline — all on one GPU, with byte-identical traces, a fixed-output (`ignore_eos`)
contract, three repeats, and a cryptographic identity record (source commit, binary, plugin, engine, trace hashes)
written to `manifest.json` for every run (`benchmarks/phase_serving/run_lifetime_encoded_admission.py`). The
runner's own `summary.json` aggregates the three repeat cells' rates with the arithmetic mean
(`statistics.mean`, despite several of its field names containing "median" — see §6); the per-workload tables
published in this document instead take the **median** of the three cell rates (§9/Results). These are two
different reducers over the same three values, not a restatement of the same number.

## 2. Hardware and software

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA RTX 3080, 10 GiB | `notes/371-upstream-v0110-clean-baseline-attempt-20260930.md:1,5` |
| SM | SM86 (Ampere) | `notes/371...md:17`; `.local/baselines/v0110-port-*/manifest.json` (`build_contract`) |
| Container (runner) | `nvcr.io/nvidia/tensorrt@sha256:7cd94ee...` | `run_lifetime_encoded_admission.py:17` (`IMAGE`) |
| TensorRT container (port build) | "TensorRT 26.06 container" | `.local/baselines/v0110-port-38dac48d-20261001/manifest.json` (`build_contract`) |
| CUDA | 13.3 | `.local/baselines/v0110-port-*/manifest.json` (`build_contract`) |
| Build flags | `-DENABLE_CUTE_DSL=ALL -DCUTE_DSL_ARTIFACT_TAG=sm_86` | same `manifest.json`; rationale in `notes/371...md` ("Silent INT4 V2 failure without `-DENABLE_CUTE_DSL=ALL`") |
| vLLM (frozen reference, Gemma) | v0.28.0 | `.local/results/gemma4-vllm-capacity-sweep-20260912/.../server.log:3` |
| vLLM (frozen reference, Cosmos) | v0.27.1 (all 35 retained successful raw origins) | `.local/results/review-correction-20260926/cosmos-vllm-frozen-raw-corrected.json` `raw_origins[].version` |
| Driver version | 610.57.04 (RTX 3080, 10240 MiB) | `nvidia-smi` on the measurement host, 2026-10-03 |

## 3. Models, engines and build contract

| Field | Gemma-4-E2B-it (AWQ INT4) | Cosmos-Reason2-2B |
|---|---|---|
| HF repo id | `google/gemma-4-e2b-it` | `nvidia/Cosmos-Reason2-2B` |
| Engine path (port) | `.local/artifacts/v0110-port/gemma-4-e2b-it-awq/engine` | `.local/artifacts/v0110-port/cosmos-reason2-2b/engine` |
| `max_batch_size` (decode-batch capacity) | 24 | 80 |
| `max_kv_pool_pages` | 192 | 256 |
| `max_input_len` | 1024 | 1024 |
| `in_flight` (client max concurrency) | 24 | 64 |
| `decode_batch` (`TRT_EDGELLM_MAX_DECODE_BATCH`) | 24 | 64 |
| `stable_slots` (`TRT_EDGELLM_MAX_STABLE_SLOTS`) | 24 | 80 |
| Vision batch size | 4 | 4 |
| Vision encoder max input tokens | 1120 (env override, default) | 8192 |
| Builder config extra keys | `num_swa_pages` present (sliding-window attention) | `max_vision_prefill_batch_size`, `max_vision_prefill_chunk_tokens`, `vision_prefill_profile` |

Source: `benchmarks/phase_serving/run_lifetime_encoded_admission.py` function `model_config` (lines ~90-160) and
`.local/artifacts/v0110-port/{gemma-4-e2b-it-awq,cosmos-reason2-2b}/engine/config.json` `builder_config` (read
directly: Gemma `max_kv_pool_pages=192, max_batch_size=24, max_input_len=1024`; Cosmos
`max_kv_pool_pages=256, max_batch_size=80, max_input_len=1024`). The runner asserts
`builder_config["max_kv_pool_pages"] == config["kv_pages"]` at campaign start and fails the run otherwise
(`run_lifetime_encoded_admission.py`, KV-page check in `main()`).

`num_swa_pages` in the Gemma builder config indicates sliding-window attention (SWA) is built into the engine;
no further SWA-specific serving flag was found in the runner beyond the engine's own window size. Fixed prefill
chunk is `TRT_EDGELLM_FIXED_PREFILL_CHUNK=128` tokens and `TRT_EDGELLM_MAX_PREFILL_BATCH_TOKENS=1024`
(`command_for`, same file). Packed prefill ("true packed prefill T=sum q") is a W016 port fix (F2) recorded in
`.local/baselines/v0110-port-9f911c16-20261001/manifest.json` `build_contract`; it is not a runner flag but a
runtime/engine change: commit `91feff6c` ("feat: Restore true packed prefill"), with the ordinary-path layout in
`c80fee7f` (note 374).

Vision engines for both models are the "production `.local/current` vision engines" per the `9f911c16`/`f6c2f094`/
`f0849333` baseline manifests, i.e. frozen from an earlier campaign rather than rebuilt per port — the manifest text
is the only evidence found; no separate vision-engine build log was read in this pass.

## 4. Serving system under test

Pipeline: HTTP client (`run_vllm_trace_bench.py`) → OpenAI-compatible gateway
(`run_phase_openai_gateway.py` / `scripts/phase_openai_gateway.py`, an `EventBroker` that line-pipes JSON requests
to the backend subprocess and demultiplexes `PHASE_EVENT` lines back to per-request queues) → backend binary
`examples/llm/llm_phase_context_smoke` running inside the pinned container with `--gpus all --network none
--read-only` (`command_for` in `run_lifetime_encoded_admission.py`).

Independent E/P/D phases and the "V3" policy are activated through environment variables passed to the backend
container (all in `command_for`):

| Flag | Meaning (from code) |
|---|---|
| `TRT_EDGELLM_MEASUREMENT_ENCODED_ADMISSION=lifetime` | Encoder-admission mode selected by `variant` ("independent" maps to `lifetime` admission) |
| `TRT_EDGELLM_PHASE_WORKSPACE_MODE` | Set to the variant name (`independent`, `shared_ep`, `tiered_ep`, `auto`) when the variant requests a specific E/P workspace sharing mode |
| `TRT_EDGELLM_PHASE_POLICY=service-scaled-transition` | The "V3" admission/transition policy name |
| `TRT_EDGELLM_ENABLE_TRANSITION_PREDICTOR` | On/off toggle for the transition predictor, driven by `--transition-predictors` |
| `TRT_EDGELLM_CAPTURE_PHASE_GRAPHS`, `TRT_EDGELLM_MAX_PREFILL_GRAPHS`, `TRT_EDGELLM_MAX_DECODE_GRAPHS` | CUDA graph capture on/off and graph-count ceilings, controlled by `--cuda-graphs` (default on) |
| `TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES` | Comma list of decode batch sizes to prime graphs for, `warmup_decode_batches(config["decode_batch"])` = `1,2,4,...` up to the model's decode batch (24 or 64) |
| `TRT_EDGELLM_DISABLE_SERVING_OVERLAP_PROBES` | Set only when `--serving-overlap-probes off`; "serving-overlap probes" are enabled by default |
| `TRT_EDGELLM_EMIT_PHASE_METRICS=1`, `TRT_EDGELLM_PHASE_TELEMETRY_LEVEL` (`full`|`dispatch`) | Dispatch telemetry emission and verbosity; default is `dispatch` (compact) |
| `TRT_EDGELLM_PHASE_ACTIVITY_PREFIX` | Per-run activity-log path, only set when `--activity-observer` is `full` (default) or `encoder` |
| `TRT_EDGELLM_PREFILL_TTFT_HARD_GUARD=1` | Always-on hard TTFT guard in the prefill scheduler |
| `TRT_EDGELLM_IGNORE_EOS=1` | Always-on: generation runs to `max_generate_length` regardless of EOS (fixed-output contract, see §6) |
| `TRT_EDGELLM_STARTUP_CALIBRATION`, `TRT_EDGELLM_STARTUP_BUDGET_MS`, `TRT_EDGELLM_MEASUREMENT_MEASURED_DECODE_COSTS`, `TRT_EDGELLM_STARTUP_PLAN_ONLY`, `TRT_EDGELLM_STARTUP_DECODE_SERVICE(_OBSERVE)` | Calibration-contract knobs selected per `--variants` value via `calibration_contract()`; the retained default variant is `independent`, which runs none of the `CALIBRATION_VARIANTS` ablations (`startup=False`) |

The full12 campaigns in `.local/results/` use `--variants independent` only — the other variant names
(`static-base`, `static-large`, `lifetime`, `ownership`, `chunked`, `e1`/`e2`, `shared_ep`, `tiered_ep`,
`unified_action`, and the nine `CALIBRATION_VARIANTS`) are explicit diagnostics/ablations, not part of the retained
serving measurement (`run_lifetime_encoded_admission.py` `parse_args`, `--variants` default `["independent"]`;
`notes/301-lifetime-encoded-admission-two-model-plan-20260913.md` for why "independent" encoder admission was
selected as the retained mode).

## 5. The 12 workloads

Names (`WORKLOADS` in `run_lifetime_encoded_admission.py`): `balanced, mixed, vision-heavy, multi-image,
long-prefill, bimodal, decode-heavy, short, text-heavy, poisson, wave-drain, late-vision`. `--full12` sets
`args.workloads = list(WORKLOADS)`.

Traces are pre-materialized JSON files (not regenerated per run); the runner only reads their path per model
(`model_config`, `traces` dict) and hashes them into the manifest. The two models use **different trace sets** with
the same 12 names: Gemma reads from `.local/results/gemma4-e2b-awq-full12-20260911/inputs/<workload>.json`; Cosmos
reads from the canonical commands recorded in
`.local/results/v0101-forward-port/heuristic-elimination-20260911/v3-profile-free-canonical-full12-3x/commands.json`
(each record's `--trace` argument), which point at differently-named materialized files under that directory's
`inputs/`. Measured directly from the trace JSON (`requests` list; `arrival_offset_us`; `max_generate_length`;
`messages[].content[].type == "image_url"` count per request):

### Gemma traces (`gemma4-e2b-awq-full12-20260911/inputs/*.json`)

| Workload | N | Classes (text / single-image / multi-image) | Output tokens (min/med/max) | Max images/req | Span (last−first arrival) |
|---|---:|---|---:|---:|---:|
| balanced | 64 | 64 / 0 / 0 | 32 / 96 / 128 | 0 | 0.1 s |
| mixed | 64 | 32 / 26 / 6 | 32 / 32 / 64 | 2 | 0.1 s |
| vision-heavy | 64 | 16 / 39 / 9 | 32 / 32 / 64 | 2 | 0.1 s |
| multi-image | 20 | 0 / 16 / 4 | 32 / 32 / 32 | 2 | 0.8 s |
| long-prefill | 64 | 64 / 0 / 0 | 32 / 96 / 128 | 0 | 0.1 s |
| bimodal | 64 | 64 / 0 / 0 | 16 / 104 / 384 | 0 | 0.1 s |
| decode-heavy | 64 | 64 / 0 / 0 | 96 / 288 / 384 | 0 | 0.1 s |
| short | 48 | 48 / 0 / 0 | 8 / 24 / 32 | 0 | 0.0 s |
| text-heavy | 64 | 48 / 13 / 3 | 32 / 64 / 64 | 2 | 0.1 s |
| poisson | 64 | 48 / 13 / 3 | 32 / 64 / 128 | 2 | 0.9 s |
| wave-drain | 20 | 0 / 16 / 4 | 32 / 32 / 32 | 2 | 6.0 s |
| late-vision | 32 | 24 / 8 / 0 | 1 / 192 / 192 | 1 | 0.5 s |

### Cosmos traces (per `commands.json`, trace file named per `--trace`)

| Workload | N | Classes (request_class field, where present) | Output tokens (min/med/max) | Max images/req | Span |
|---|---:|---|---:|---:|---:|
| short | 48 | unlabeled | 8 / 24 / 32 | 0 | 0.03 s |
| balanced | 288 | unlabeled | 32 / 96 / 128 | 0 | 0.3 s |
| decode-heavy | 288 | unlabeled | 96 / 288 / 384 | 0 | 0.28 s |
| long-prefill | 288 | unlabeled | 32 / 96 / 128 | 0 | 0.3 s |
| bimodal | 288 | unlabeled | 16 / 104 / 384 | 0 | 0.3 s |
| text-heavy | 64 | 48 text / 16 vision | 32 / 64 / 64 | 2 | 0.06 s |
| mixed | 64 | 32 text / 32 vision | 32 / 32 / 64 | 2 | 0.06 s |
| vision-heavy | 64 | 16 text / 48 vision | 32 / 32 / 64 | 2 | 0.06 s |
| poisson | 64 | 48 text / 16 vision | 32 / 64 / 128 | 2 | 0.87 s |
| wave-drain | 20 | 20 vision | 32 / 32 / 32 | 2 | 6.03 s |
| multi-image | 5 | 5 vision (4 single-image + 1 two-image) | 32 / 32 / 32 | 2 | ~0 s |
| late-vision | 32 | unlabeled | 1 / 192 / 192 | 1 | 0.5 s |

Notes: Cosmos's `balanced`/`decode-heavy`/`long-prefill`/`bimodal` traces are far larger (288 requests) than the
matching Gemma traces (64) because Cosmos's in-flight capacity is 64 vs Gemma's 24 (see "load level" below);
`multi-image` for Cosmos is only 5 requests — this is the materialized trace actually referenced by
`commands.json`'s `multi-image` case and was read directly, not inferred.

What each workload stresses (names are self-describing and consistent with the request-class/output-length mix
above): `short` = minimal decode (24-token median output, smallest trace) exercising TTFT-dominated small requests;
`balanced`/`text-heavy` = mixed text-biased prefill/decode; `long-prefill` = larger prompts with moderate output;
`bimodal`/`decode-heavy` = long-output decode-bound load (up to 384 tokens); `mixed`/`vision-heavy` = text+vision
request mixes with up to 2 images/request. They differ not only in text:vision ratio but also in output-length
histogram: Gemma `mixed` is 35 requests × 32 tokens, 3 × 48, 26 × 64; `vision-heavy` is 50 × 32, 2 × 48, 12 × 64
(measured from the trace JSON `max_generate_length` field). `multi-image` = requests that are mostly
**single**-image, not all multi-image — Gemma is 16 single-image + 4 two-image requests, Cosmos is 4 single-image
+ 1 two-image request (measured per-request `image_url` counts); `poisson`/`wave-drain` = the only traces with
non-trivial arrival spans (0.9 s and 6 s), i.e. genuine arrival-process stress rather than a near-simultaneous
burst; `late-vision` = **24 text requests producing 192-token output and 8 vision requests producing 1-token
output** (Gemma trace; measured directly) — there are no 192-token vision requests in this trace, so this stresses
late-arriving low-output vision admission against long-running text decode, not a 1-token-vs-192-token split
within either class.

**Generation of traces.** The runner never regenerates traces; it replays retained, content-hash-named files. Lineage
of the 24 files used by the v0.11.0 port campaigns:

| Step | Location | What changed |
|---|---|---|
| Cosmos originals | `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-v3-full12-3x/inputs/materialized-trace-*.json` | Retained materialized traces of the v0.10.1 forward-port campaigns; the synthetic generator run that first produced them predates the retained artifacts and is not recorded there |
| Cosmos canonical | `.local/results/v0101-forward-port/heuristic-elimination-20260911/v3-profile-free-canonical-full12-3x/inputs/` | Re-materialized with asset hashes and remaps recorded in `input-contract.json` (image assets: `examples/multimodal/pics/{woman_and_dog,red_panda,giant_panda,database_er}.jpeg`); content unchanged |
| Gemma capability-scaled | `.local/results/gemma4-e2b-awq-full12-20260911/inputs/{workload}.json`, built by `build_model_port_workload_gate.py` (note 290) | Request content, output-length distribution and arrival offsets kept; the four 288-request text traces capped at 64 requests; `multi-image` repeated from 5 to 20 requests (four waves); image URLs rewritten to container paths; per-case source/target SHA-256 and arrival span recorded in `inputs/manifest.json` |

Note 290 states the reason for scaling: reusing the Cosmos contract (P8/D64/E4, 80 owners, HTTP in-flight 64) on the
smaller Gemma engine would mostly measure admission queueing instead of the same scheduling phenomenon, so the Gemma
suite is capability-scaled rather than load-identical (632 measured requests). `materialize_load_sweep.py`
(`scale_trace`, divides `arrival_offset_us` by `--multiplier`) and `build_scaled_request_waves.py` (`build_waves`) are
available for load and wave ablations; they were not used to build the retained full12 set.

**Load level vs. capacity — concurrency ceiling, not achieved concurrency.** Each model's `in_flight` / client
concurrency setting equals its serving decode-batch capacity: Gemma `in_flight=24`, `decode_batch=24`,
`stable_slots=24` (confirmed: `model_config`, and the full24 `run.sh` comment "Port full24 x3"); Cosmos
`in_flight=64`, `decode_batch=64`, `stable_slots=80`. Both models' client concurrency
(`--max-workers`/`--max-in-flight`, derived from `config["in_flight"]`) is set to the full capacity limit, but
this is a **ceiling**, not the concurrency the workload actually reaches — a short or sparsely-arriving trace can
finish, or drain to near-zero in-flight requests, well before saturating that ceiling. Maximum observed HTTP
concurrency per workload, computed from each final cell's `requests.csv` (`send_us`/`completed_us`) in
`.local/results/v0110-port-final-full24-3x-20261003` (repeat 1):

| Model | Ceiling | Workloads at ceiling | Workloads below ceiling |
|---|---:|---|---|
| Gemma | 24 | balanced, bimodal, decode-heavy, late-vision, long-prefill, mixed, poisson, short, text-heavy, vision-heavy | multi-image 20/24, wave-drain 5/24 |
| Cosmos | 64 | balanced, bimodal, decode-heavy, long-prefill, mixed, poisson, text-heavy, vision-heavy | short 48/64, late-vision 32/64, multi-image 5/64, wave-drain 5/64 |

`multi-image`, `wave-drain` (both models), and Cosmos `short`/`late-vision` measure finite-trace completion or
spaced-wave behavior rather than saturated admission; results for those workloads should not be read as capacity
measurements. The TensorRT runs warm up with a generic calibration trace of `config["calibration_requests"]`
requests (49 Gemma, 239 Cosmos) unless `calibration["compact_http"]` selects a 1-request warmup. The "warmup 8 /
warmup 64" figures belong to the frozen vLLM and upstream-server contracts (Gemma in-flight 24 / warmup 8, Cosmos
in-flight 64 / warmup 64 for the upstream-server baseline; note 371), not to the TensorRT runner. The frozen vLLM
Cosmos warmup is not uniformly 64 either — see §7.

## 6. Measurement protocol

- **Warmup.** HTTP-level warmup before measurement: `--warmup-requests`, `--phase-calibration-round-requests`,
  `--phase-calibration-min-requests` are all set to `config["calibration_requests"]` (Gemma 49, Cosmos 239) against
  a separate calibration trace (`config["calibration"]`, a "generic" policy-calibration trace, e.g.
  `gemma4-packed-prefill-g4-20260912/generic-p8-d24-e4.json`), unless the policy-warmup variant requests a
  "compact_http" 1-request warmup (`command_for`, `calibration_contract`). `command_for` passes
  `run_vllm_trace_bench.py` as `--client-script` (`run_lifetime_encoded_admission.py:449`); it does **not** invoke
  `guarded_trace_client.py`, so that script's warmup-response validation and calibration-round convergence guard
  (`calibration_signature`, `PHASE_CALIBRATION_STABLE_ROUNDS`) are not in effect for these campaigns. Warmup is a
  fixed request budget: warmup responses are discarded and `completed_warmup` is incremented by the number
  submitted, regardless of their HTTP status or whether calibration converged. In the final retained artifacts
  (`.local/results/v0110-port-final-full24-3x-20261003`), **36/36 Gemma cells and 4/36 Cosmos cells** end warmup
  with `calibration_converged=false` in their `calibration.json`, and no cell contains a `warmup-validation.json`
  (verified by counting `calibration.json` files per model). Separately,
  `TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES` primes CUDA-graph capture for a fixed decode-batch schedule
  `(1,2,4,8,12,...)` up to the model's decode batch, independent of HTTP warmup
  (`WARMUP_DECODE_BATCHES`/`warmup_decode_batches`).
- **Client concurrency / in-flight limit.** `--max-workers` and `--max-in-flight` are both set to
  `config["in_flight"]` (Gemma 24, Cosmos 64) unless overridden by `--client-max-in-flight`.
- **Fixed-length output (ignore EOS).** `--ignore-eos` is always passed to the client, and
  `TRT_EDGELLM_IGNORE_EOS=1` is always set on the backend; every request runs to its trace's
  `max_generate_length` regardless of EOS. The runner asserts
  `aggregate["generated_tokens_per_run_min"] == aggregate["requested_output_tokens_per_run"]` and raises
  `"Incomplete fixed-output workload"` otherwise (`main()`, post-cell validation).
- **Repeats and aggregation.** `--repeats 3` for the full12 x3 campaigns (`run.sh`); each repeat is `--repeats 1`
  at the HTTP-client level inside a single cell directory `repeat-NNN`. Per-metric aggregation across the three
  repeat cells is `statistics.mean` (`token_repeatability`/`summary["mean_run_metrics"]` in
  `run_lifetime_encoded_admission.py`), despite several metric names containing "median" — those names originate
  one level down, inside a single run (see below), not across the three retained repeats. The repeat ordering
  alternates (`variants`/`predictors` reversed on even repeats) to avoid confounding repeat order with an untested
  warm-cache effect (`main()`, "variants = ... if repeat % 2 else list(reversed(...))").
- **Per-run metric construction (`run_vllm_trace_bench.py`).** Per request: `ttft_ms = (first_token_ns -
  send_ns)/1e6`; `e2e_ms = (done_ns - send_ns)/1e6`; `tpot_ms = (done_ns - first_token_ns)/1e6 /
  (output_tokens - 1)` (time-per-output-token, excluding the first token). A run's summary takes `mean`/`median`/
  `p95`/`p99` across successful requests for each of TTFT/TPOT/E2E, plus `generated_token_s` (aggregate decode
  throughput for that run). The cell-level `aggregate.json` metrics consumed by the campaign
  (`generated_token_s_median`, `ttft_mean_of_run_means_ms`, `ttft_p95_median_ms`, `tpot_mean_of_run_means_ms`,
  `tpot_p95_median_ms`, `e2e_mean_of_run_means_ms`, `e2e_p95_median_ms`, `gpu_memory_peak_mib_median`) are each a
  **median across the HTTP client's own internal repeats** of the corresponding per-run statistic (confirmed by
  `rederive_frozen_vllm.py`'s explicit correction note: "Historical mean_of_run_means fields were medians of
  per-run means; raw values are unchanged" — i.e., the field name is historical and the actual reducer is
  `statistics.median`, except `achieved_req_s`/`generated_token_s` style fields, which are medians of per-run
  medians).
- **GPU memory — TensorRT and vLLM peaks are not comparable.** `run_vllm_trace_bench.py` (used as the HTTP client
  for both TensorRT and vLLM) has its own `GpuMemorySampler`, but the TensorRT runner never passes
  `--sample-gpu-memory` to it, so that inner sampler is disabled for TensorRT cells
  (`command_for` sets `"sample_gpu_memory": args.sample_gpu_memory` to `False`). TensorRT's reported
  `gpu_memory_peak_mib` instead comes from the **outer** `GpuMemoryMonitor` in `run_phase_http_trace_bench.py`,
  which starts (default 100 ms interval) **before** launching the client subprocess — covering HTTP warmup,
  calibration, and measurement — and stops only after the client returns. vLLM's own client-side sampler, when
  enabled, starts only after warmup, inside the per-run measurement loop, with a default 50 ms interval. The two
  "peak" numbers therefore cover different windows (TensorRT: warmup+calibration+measurement at 100 ms; vLLM:
  measurement only at 50 ms) and must not be read as peaks over the same interval. The per-cell aggregate reduces
  to `gpu_memory_peak_mib_median` across repeats in both cases.
- **Token-trace determinism.** `token_repeatability()` collects `token_trace_sha256_per_run` hashes across the three
  repeat aggregates and reports `observed_equal` / `observed_different` / `not_tested` (fewer than 2 observations).
  This is recorded per `model/workload/variant` group in `summary.json` but is informational — it does not gate
  campaign completion.
- **Output / integrity audit.** `audit_phase_campaign_outputs.py` re-verifies, per completed cell: the trace file
  hash is unchanged since the campaign ran; exactly one `requests.csv` exists per cell; every request id in that
  CSV is unique, in range, and its observed `output_tokens` equals the trace's `max_generate_length` for that
  request; and EOS-token presence is checked against `model_eos()` (engine `config.json`'s `eos_token_id`, plus
  Gemma's hardcoded end-of-turn diagnostic tokens `{1, 50, 106}`). The audit does not claim semantic correctness
  (file docstring: "without running inference or claiming semantic correctness").
- **Identity/provenance.** Every campaign writes `manifest.json["identity"]` with: `source_commit`,
  `tracked_diff_sha256`, `source_status` (git porcelain), `binary_sha256`/`plugin_sha256` (re-checked before every
  cell — `"Runtime binary or plugin changed during campaign"` raises if they drift), per-model engine/vision/config
  SHA-256, per-workload trace SHA-256, and the replay-tool script hashes. `validate_cell_contract()` rejects reusing
  a cell directory whose `contract.json` does not match the current `{identity, record}` pair.

## 7. Comparison baselines and fairness contract

| Baseline | Version/build | Flags held identical to ours | What differs |
|---|---|---|---|
| Frozen vLLM (Gemma) | v0.28.0, server log `.local/results/gemma4-vllm-capacity-sweep-20260912/selected-seq24-kv480-p4096-g24-full12/server.log` | Same traces (byte-identical, hash-checked); same `--max-in-flight`/`--trace` per case via `build_vllm_equal_contract.py`; same `ignore_eos` presence (synced per-record); CUDA graphs enabled for sizes `[1,2,4,8,16,24]`, async scheduling on, seed 0, temperature 0 | vLLM: `enable_chunked_prefill=True`, `max_num_batched_tokens=4096`, `max_model_len=2048`, `kv_cache_memory_bytes=503316480` (480 MiB fixed KV), `enable_prefix_caching=False`; served model is a vLLM-0.28-compatibility-repacked checkpoint (`gemma-4-e2b-it-awq-vllm028-compat`) |
| Frozen vLLM (Cosmos) | v0.27.1 (all 35 retained successful raw origins); `.local/results/review-correction-20260926/cosmos-vllm-frozen-raw-corrected.json` | Same contract pattern as Gemma, per `build_vllm_equal_contract.py` | Warmup requests are **not uniform across workloads**: `short`/`balanced`/`decode-heavy`/`long-prefill`/`bimodal`/`text-heavy`/`mixed` use warmup 64; `vision-heavy`/`poisson`/`wave-drain`/`multi-image`/`late-vision` use warmup 16 (per-workload `contract.warmup_requests` in the raw-origins file). The upstream-server baseline (row below) instead used warmup 64 for every Cosmos workload, so the two baselines are not on an equal warmup contract per workload. `vision-heavy` has only 2 successful runs (1 failed attempt) versus 3 for the other workloads; Gemma's frozen baseline has 1 run per workload. |
| vLLM re-derivation | `rederive_frozen_vllm.py` | Requires the current canonical trace hash to equal the one recorded when the frozen baseline ran (`"Canonical workload has conflicting trace hashes"` otherwise); requires `repeats == 1` per raw aggregate and a complete set of previously-successful runs | It only re-aggregates already-captured raw per-request data ("no_new_gpu_measurement"); it does not rerun vLLM |
| v0.10.1 tip | **Not directly inspected in this pass** — referenced by the AGENTS.md workflow description ("v0.10.1 phase forward port") but no specific full12 campaign for it was opened here | — | — |
| Upstream v0.11.0 clean baseline | Published wheel `tensorrt-edgellm[server]==0.11.0`, commit `95515c2` for the from-source build | Same traces, same `ignore_eos` semantics (via `EDGELLM_IGNORE_EOS=1` on the server since the HTTP schema forbids the field), same warmup/in-flight limits as the frozen vLLM contract per workload (`notes/371...md`, "Upstream serving baseline" section) | Batch size: Gemma 4 vs fork's 24; Cosmos 8 vs fork's 64 (upstream OOMs above these on the 10 GiB card); upstream's in-flight batching requires equal `max_tokens` to join a batch, so heterogeneous-output traces stall; one run per cell (not x3) |

Fairness contract notes recorded directly in `rederive_frozen_vllm.py`'s `contract_notes`: "Every retained trace
hash matches the current canonical trace bytes," "Recorded successful and failed counts remain distinct; no failed
run is imputed," "Warmup count is retained per origin, not homogenized across workload classes," and "Trace
identity does not establish equal runtime configuration or identical generated output."

## 8. Output-quality gates

- **MMLU zero-shot serving vs. batch-1 reference.** `mmlu_serving_accuracy.py build` constructs one request per
  MMLU question (`cais/mmlu` parquet; 4-choice, zero-shot, `max_tokens=4`, `temperature=0`, `top_p=1`, `top_k=1`,
  `batch_size=1`), skipping prompts over `--max-prompt-chars` (default 3200). `score` parses the first standalone
  `[ABCD]` token in `output_prefix` (`predicted_letter`, regex `\b([ABCD])\b`) and compares predicted letters to the
  parquet's `answer` field, reporting accuracy and pairwise agreement (same prediction / same token IDs) across
  multiple `requests.csv` runs. Per note 368: 14,031/14,042 questions used (11 exceed the 1024-token engine input
  limit and are dropped); batch-1 reference 7065/14031 = 50.35%, serving repeat 1 = 7066/14031 = 50.36% — serving
  accuracy is statistically indistinguishable from the batch-1 reference despite token-level divergence (see next
  bullet). **Parser caveat.** `\b([ABCD])\b` accepts any standalone letter, including non-answer prefixes such as
  `Let $A$`, `Let $D$`, and `The area $A`. In the final MMLU gate artifacts
  (`.local/results/v0110-port-final-mmlu-20261003/serving` and `.local/results/v0110-port-mmlu-20261001/reference`),
  both runs contain the same 11 such non-answer outputs (8 `Let $A$`, 2 `Let $D$`, 1 `The area $A`); excluding them
  changes final serving from 7193 to 7191 correct (51.27% → 51.25%) and reference from 7194 to 7192 correct
  (51.27% → 51.26%) — the corrected figures (51.25% vs 51.26%) no longer round to the same value, though the gap
  remains small. The port's MMLU gate run is `.local/results/v0110-port-fix-mmlu-20261001/run.sh`, invoking the `short`
  workload slot with `--trace-file short=.local/artifacts/datasets/mmlu/mmlu-zero-shot-trace.json`, `--repeats 1`,
  same binary/engine/build-root as the full24 x3 campaign.
- **Batch-invariance / output audit.** Note 368 found Gemma's INT4 `Int4GroupwiseGemmPlugin` switches between a
  CUDA-core GEMV kernel (M≤6) and a tensor-core GEMM kernel (M>6), both FP16-accumulating in different operation
  orders, causing a batch-size-dependent output path; default-mode serving matched its own batch-1 reference on
  only ~60% of requests (1137-1140/1896 across 3 runs), while forcing GEMM for all M
  (`TRT_EDGELLM_INT4_GEMV_MAX_M=0`) raised exact agreement to 96.5% (610/632) — the remaining ~3.5% traced to the
  vision encoder's batched-image shape (encoder batch 4 vs per-request in the reference; forcing
  `TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE=1` as well raised agreement to 99.0%, 293/296 vision-bearing requests).
  Cosmos (FP16 TensorRT GEMMs, no INT4 plugin) was bit-exact (1513/1513 at tip per note 369). The note explicitly
  concludes bit-equality to a batch-1 reference "is not an achievable gate for this INT4 engine" and is used as a
  diagnostic/audit mode, not a serving-default correctness bar; it also measured vLLM's own run-to-run and
  batch-1 agreement (88.9% run-to-run, 81.6% vs its own batch-1) as a standard-of-comparison, finding TensorRT
  Edge-LLM's default-mode serving more run-to-run stable (93.0%) than vLLM.
- **Integrity audit (`audit_phase_campaign_outputs.py`).** See §6; applied per cell across the port's full24 x3
  campaign (note 369: "72/72 cells, 0 integrity issues, 0 first-EOS anomalies").

## 9. Result layout, manifests, and reproduction commands

Each campaign lives under `.local/results/<campaign-name>/` with: `manifest.json` (state, full identity record,
every planned command, calibration/startup contract, frozen-vLLM references, `completed`/`failures` lists,
`note_references`), `summary.json` (per `model/workload/variant` group: `runs` list of per-repeat `aggregate.json`
contents, `mean_run_metrics`, `token_repeatability`), `run.sh` (the exact invocation), `runner-source.py` and
`source.patch` (frozen copies of the runner script and any uncommitted diff at launch time), and per-cell
directories `<model>/<variant>/repeat-NNN/<workload>/` each holding `contract.json`, `driver.log`,
`aggregate.json`, and `run-001/client/run-001/requests.csv`. `state` is one of `scratch`, `diagnostic`,
`validation`, `citable` (workspace-lifecycle convention; campaigns read in this pass are `diagnostic` or
`validation`).

Reproduction commands, taken verbatim from the port's retained `run.sh` files:

**Full24 x3 (both models, all 12 workloads, 3 repeats):**
```bash
cd /home/sslab/TensorRT-Edge-LLM
root=.local/results/v0110-port-fix-full24-3x-20261001
python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py --models gemma cosmos --full12 --variants independent \
  --transition-predictors on --cuda-graphs on --serving-overlap-probes on --telemetry-level dispatch --repeats 3 \
  --build-root .local/baselines/v0110-port-9f911c16-20261001/bin --binary-source-commit 9f911c16 \
  --gemma-engine-dir .local/artifacts/v0110-port/gemma-4-e2b-it-awq/engine \
  --cosmos-engine-dir .local/artifacts/v0110-port/cosmos-reason2-2b/engine \
  --compress-closed-logs --result-root $root > $root.log 2>&1
echo "EXIT=$?" > $root/DONE
```
(`.local/results/v0110-port-fix-full24-3x-20261001/run.sh`)

**MMLU gate (single cell, `short` workload replaced by the MMLU trace, Gemma only, 1 repeat):**
```bash
cd /home/sslab/TensorRT-Edge-LLM
root=.local/results/v0110-port-fix-mmlu-20261001
common=(--models gemma --workloads short --variants independent --transition-predictors on --cuda-graphs on
  --serving-overlap-probes on --telemetry-level dispatch --build-root .local/baselines/v0110-port-9f911c16-20261001/bin
  --binary-source-commit 9f911c16 --gemma-engine-dir .local/artifacts/v0110-port/gemma-4-e2b-it-awq/engine
  --compress-closed-logs --trace-file short=.local/artifacts/datasets/mmlu/mmlu-zero-shot-trace.json)
python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py "${common[@]}" --repeats 1 --result-root $root/serving > $root.serving.log 2>&1
echo "EXIT=$?" > $root/DONE
```
(`.local/results/v0110-port-fix-mmlu-20261001/run.sh`)

**Single cell** (any one model/workload at `--repeats 1`): drop `--full12`, pass `--workloads <name>` and
`--models <model>` explicitly, as in the MMLU example above.

## 10. Known limitations and caveats

- **Single GPU, single card.** All measurements, including the frozen vLLM and upstream baselines, run on one
  10 GiB RTX 3080 (SM86); no multi-GPU or other SM-architecture results are part of this campaign
  (`notes/371...md`).
- **Fixed-length (`ignore_eos`) outputs, not production decoding.** Every request runs to its trace's
  `max_generate_length` regardless of EOS; this isolates throughput/latency from output-length variance but is not
  representative of production traffic where generation stops at EOS (`run_lifetime_encoded_admission.py`,
  `TRT_EDGELLM_IGNORE_EOS=1`; §6).
- **Day-to-day/run-to-run variance exists but is not separately quantified here.** The campaign relies on 3 repeats
  aggregated by median/mean; note 369 reports throughput "within noise" tolerances of roughly ±3% against a prior
  binary, but no formal variance/confidence-interval methodology was found in the runner.
- **Frozen vLLM baseline date.** The Gemma vLLM server log is dated 2026-09-12 (`server.log`); the baseline is not
  re-run per campaign, only re-aggregated from retained raw data (`rederive_frozen_vllm.py`), so later vLLM
  releases, driver updates, or thermal/clock differences between the vLLM capture date and later TensorRT
  Edge-LLM campaigns are not controlled for.
- **Bit-exact output equality is not a valid correctness bar for the Gemma INT4 engine** (note 368): batch-shape-
  dependent FP16 accumulation order in the INT4 GEMV/GEMM plugin, and vLLM's own non-determinism, mean serving-vs-
  batch-1 token agreement is an expected-to-be-partial signal, not a pass/fail gate; MMLU accuracy equivalence is
  used instead for output-quality.
- **Trace origin predates the retained artifacts.** The lineage from the v0.10.1 Cosmos traces to the Gemma
  capability-scaled set is recorded (section 5), but the synthetic generator run that first produced the Cosmos
  traces is not retained; the traces are fixed, hash-identified inputs rather than regenerated per campaign.
- The exact vision-engine build command for the port is not recorded in the port manifests (they reference the
  production `.local/current` vision engines). The v0.10.1 tip comparison uses `.local/results/tip-full24-3x-20260929`
  (note 369).

## Results (v0.11.0 port)

Final tree `f0849333`, full24 ×3 and MMLU on 2026-10-03 (notes 374 and 375).

| Comparison | Gemma | Cosmos | All 24 |
|---|---:|---:|---:|
| vs frozen vLLM (geomean of cell medians) | +40.4% | +16.6% | +28.0% |
| vs v0.10.1 tip `tip-full24-3x-20260929` | +2.5% | −1.1% | +0.7% |
| Runs below vLLM | 0/36 | 3/36 (all Cosmos balanced) | 3/72 |

- MMLU zero-shot through serving: 51.27%, equal to the batch-1 reference; same prediction on 14029/14031.
- Cosmos balanced is the one cell below vLLM on `f0849333` (4251.9 vs 4315.8 tok/s, −1.5%; note 375). After the second
  review fixes, a same-day interleaved A/B puts `d075df4d` at 4448.8 tok/s (+3.1% vs vLLM; note 376).
- Per-cell table: note 375. Result directories: `.local/results/v0110-port-final-full24-3x-20261003`,
  `.local/results/v0110-port-final-mmlu-20261003`.
