# 379 — Colab A100: capacity selection, the 64-row decode cap, and how to reproduce notes 377-379 (2026-10-05)

Follows notes 377 (Gemma 4 12B) and 378 (Cosmos-Reason2-2B, Gemma 4 E2B). Stopped by request during the E2B D256
step; the Cosmos sweep was not run.

## Conclusions
1. **Batch sizes tuned on one GPU silently capped another.** Dynamic decode chose decode batches from a built-in
   cost table measured on an RTX 3080 (P8/D64) that stops at 64 rows; with more rows ready it split decode into
   <=64-row dispatches, and admission rejected rows above the table. On A100 E2B at D128 decode stayed at 64 rows
   (3969 tok/s decode-heavy); after `f1d1bf8` it runs 128 rows (5911 tok/s, +49%). Any capacity comparison made
   with engines above D64 before `f1d1bf8` understates phase serving.
2. **A100 needs batches well above 64 for these 2B models, on both systems.** Throughput still rises at D192 for
   phase serving and vLLM (table below); D256 was not measured, so the knee is at or above 192.
3. **With batch and load scaled together, the systems are level overall but split by traffic type.** Geomean of
   balanced, decode-heavy and mixed, TRT/vLLM: 0.91 at D64, 0.94 at D128, 1.01 at D192. vLLM leads text-only work
   (balanced 0.55-0.64, decode-heavy 0.68-0.79), phase serving leads the vision mix (1.9-2.0x) with lower p95 TTFT
   and TPOT at every D.
4. **Memory: vLLM takes 92% of the GPU up front; phase serving uses what the plan needs.** E2B at D256 capacity
   peaked at 21.9-22.1 GB (phase) against 37.9-38.2 GB (vLLM). The probe-based planner predicted 21.1 GB for the
   phase engine (about 4% low).
5. **The remaining gap on text is prefill, not decode.** Note 378: small packed prefills cost about 17 ms of GPU each
   and many dispatch at batch 1; a decode-only step at 120-128 rows takes 11.4 ms (`f1d1bf8` build). That is the next optimization target, ahead of
   larger batches.
6. **Measurement hygiene.** A shared server per system (`--reuse-server`) cuts a campaign from about 70 min to
   about 6 min but carries phase scheduler state across workloads (note 378: one request starved 1 s in `short`).
   The gateway's default listen backlog (5) reset connections at 128 clients (fixed in `fe38dbe`).

## E2B capacity sweep (FP16, engine b256/P16/chunk 256, load scaled with D, 1 run per cell)
| D | System | balanced | decode-heavy | mixed | Geomean | Worst TPOT p95 (ms) | Worst TTFT p95 (ms) | GPU peak (MiB) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | phase | 1900 | 3781 | 1174 | 2036 | 57.2 | 1340 | 21908 |
| 64 | vLLM | 3452 | 5527 | 589 | 2239 | 137.2 | 5084 | 37944 |
| 128 | phase | 2662 | 5873 | 1432 | 2818 | 82.1 | 2638 | 22018 |
| 128 | vLLM | 4358 | 8301 | 745 | 2998 | 152.1 | 7808 | 37984 |
| 192 | phase | 2973 | 7345 | 1685 | 3326 | 102.8 | 3583 | 22120 |
| 192 | vLLM | 4658 | 9274 | 825 | 3291 | 254.4 | 11191 | 38204 |

Throughput is generated tok/s. Load per D: client in-flight = D, vLLM `--max-num-seqs D`, balanced and decode-heavy
4*D requests, mixed D/32 times the 64-request base mix. All cells completed every request at the fixed output
length. The worst TPOT/TTFT columns are dominated by the mixed workload. D64 phase numbers come from this b256/P16
engine and 256-request traces, so they differ from note 378's D64 engine (P8) and 288-request traces.

## Capacity selection method (reusable on a new GPU)
1. **Probe and plan** (`benchmarks/phase_serving/plan_phase_capacity.py`): run one serving probe on the target GPU
   and record its peak memory; the planner derives KV bytes per 128-token page from the exported `llm/config.json`
   (owning layers via `kv_sharing_donors`, bounded SWA pages per slot) and returns the maximum pool and decode-batch
   bounds for worst-case, p95 and mean request lengths. A100 results: E2B is compute-bound (D256 with a full KV pool
   fits in about 21 GB); Cosmos is KV-bound (about 1690 pages, D about 130 at p95 and 240 at mean request length).
2. **Build once at the upper bound, sweep at runtime** (`benchmarks/phase_serving/run_capacity_sweep.py`): it scales
   load with each D, runs both systems on the same traces, and recommends the smallest D within 95% of peak throughput
   that meets a TPOT p95 SLO.
3. **Do not reuse hard-coded cost tables across GPUs.** The decode table in `examples/llm/phaseSchedulerOptions.inc`
   is a 3080/Cosmos profile; after `f1d1bf8` it no longer caps larger engines, but its latency values still steer
   formation below 64 rows. A per-GPU measured profile (`TRT_EDGELLM_ENABLE_MEASURED_DECODE_BATCHING` with a
   calibration trace that covers the decode rows) is the follow-up.

## Reproduction (Colab A100-SXM4-40GB, driver 580.82.07, CUDA 13.0)
Everything under `.local/` is untracked; recreate it as follows.

### Environment
```bash
cd /content/TensorRT-Edge-LLM && git submodule update --init --recursive
# TensorRT 11.0.0.114 from the preconfigured CUDA apt repo, extracted instead of installed
mkdir -p .local/cache/tools/debs && cd .local/cache/tools/debs
V=11.0.0.114-1+cuda13.2
apt-get download libnvinfer11=$V libnvinfer-dev=$V libnvinfer-headers-dev=$V libnvinfer-plugin11=$V \
  libnvinfer-plugin-dev=$V libnvinfer-headers-plugin-dev=$V libnvonnxparsers11=$V libnvonnxparsers-dev=$V
D=../tensorrt-11.0.0.114-cuda13.2; mkdir -p $D; for f in *.deb; do dpkg -x $f $D; done
ln -sfn tensorrt-11.0.0.114-cuda13.2 ../tensorrt && cd /content/TensorRT-Edge-LLM
```
`.local/env.sh` (HF token in `.local/secrets/hf_token`, mode 600):
```bash
export REPO=/content/TensorRT-Edge-LLM
export TRT_PACKAGE_DIR=$REPO/.local/cache/tools/tensorrt/usr
export LD_LIBRARY_PATH=$TRT_PACKAGE_DIR/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
export LLM_SDK_DIR=$REPO
export EXPORT_VENV=$REPO/.local/cache/tools/export-venv
export BUILD_DIR=$REPO/.local/builds/colab-a100-sm80
export HF_HOME=$REPO/.local/cache/hf
export HF_TOKEN=$(cat $REPO/.local/secrets/hf_token)
export EDGELLM_PLUGIN_LIB=$BUILD_DIR/libNvInfer_edgellm_plugin.so
export EDGELLM_PLUGIN_PATH=$BUILD_DIR/libNvInfer_edgellm_plugin.so.1.0
```
Build and Python environments:
```bash
source .local/env.sh
cmake -S $REPO -B $BUILD_DIR -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR -DCUDA_CTK_VERSION=13.0 \
  -DCUDA_DIR=/usr/local/cuda-13.0 -DCMAKE_CUDA_ARCHITECTURES=80 -DCUTE_DSL_ARTIFACT_TAG=sm_80 \
  -DCUDA_DRIVER_LIB=/usr/lib64-nvidia/libcuda.so -DBUILD_UNIT_TESTS=ON -DCMAKE_BUILD_TYPE=Release
make -C $BUILD_DIR -j12
uv venv --python 3.12 $EXPORT_VENV
uv pip install --python $EXPORT_VENV/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.nvidia.com \
  -e ".[tools]" "tensorrt-cu13==11.0.0.114" requests pytest
uv venv --python 3.12 .local/cache/tools/vllm-0.31.0-venv
uv pip install --python .local/cache/tools/vllm-0.31.0-venv/bin/python "vllm==0.31.0" --torch-backend=auto
```

### Models, exports and engines (FP16)
```bash
for m in nvidia/Cosmos-Reason2-2B google/gemma-4-E2B-it google/gemma-4-12B-it; do
  $EXPORT_VENV/bin/hf download $m --local-dir .local/artifacts/models/${m#*/}; done
A=.local/artifacts/colab-a100; X=$EXPORT_VENV/bin/tensorrt-edgellm-export; B=$BUILD_DIR/examples
# Gemma 4 E2B: packed prefill, chunk cap 256; capacity engine at the D256 upper bound
$X .local/artifacts/models/gemma-4-E2B-it $A/gemma-4-e2b-it/onnx-fp16-packed-p256 --skip-audio --packed-prefill \
  --packed-prefill-max-chunk-tokens 256
$B/llm/llm_build --onnxDir $A/gemma-4-e2b-it/onnx-fp16-packed-p256/llm --engineDir $A/gemma-4-e2b-it/engine-b256-p16-c256-kv2048 \
  --maxInputLen 1024 --maxKVCacheCapacity 2048 --maxBatchSize 256 --maxPrefillBatchSize 16 --maxDecodeBatchSize 256 \
  --maxPrefillChunkTokens 256
$B/multimodal/visual_build --onnxDir $A/gemma-4-e2b-it/onnx-fp16-packed-p256/visual --engineDir $A/gemma-4-e2b-it/vision-e4 \
  --maxImageTokens 1120
# Cosmos-Reason2-2B (needs f857bca): chunk cap 1024, uniform 512 text/vision chunk, undercommitted pool
$X .local/artifacts/models/Cosmos-Reason2-2B $A/cosmos-reason2-2b/onnx-fp16-packed-p1024 --packed-prefill \
  --packed-prefill-max-chunk-tokens 1024
$B/llm/llm_build --onnxDir $A/cosmos-reason2-2b/onnx-fp16-packed-p1024/llm \
  --engineDir $A/cosmos-reason2-2b/engine-p8-d64-b80-c512-kv8192-pool1536 --maxInputLen 1024 --maxKVCacheCapacity 8192 \
  --maxBatchSize 80 --maxPrefillBatchSize 8 --maxDecodeBatchSize 64 --maxPrefillChunkTokens 512 \
  --maxKVPoolPages 1536 --allowKVPoolUndercommit
$B/multimodal/visual_build --onnxDir $A/cosmos-reason2-2b/onnx-fp16-packed-p1024/visual --engineDir $A/cosmos-reason2-2b/vision-e4 \
  --maxImageTokens 22528 --maxImageTokensPerImage 2816
# Gemma 4 12B: non-packed (vision-block attention), see note 377 for the full contract
```
The note 378 E2B campaign engine is the same export built with `--maxBatchSize 64 --maxPrefillBatchSize 8
--maxDecodeBatchSize 64`; the note 377 12B engine uses `--maxInputLen 1600 --maxKVCacheCapacity 2048 --maxBatchSize 24
--maxPrefillBatchSize 4 --maxDecodeBatchSize 24 --maxKVPoolPages 224 --allowKVPoolUndercommit`.

### Quality gate, campaigns and the sweep
```bash
P=benchmarks/phase_serving; PY=$EXPORT_VENV/bin/python
$PY $P/compare_hf_greedy.py reference --hf-dir .local/artifacts/models/<model> --dtype bfloat16 --output hf-bf16.json
#   start a phase server (run_serving_comparison.server_command) or vLLM, then:
$PY $P/compare_hf_greedy.py serve --model <served-name> --concurrency 9 --output trt.json
$PY $P/compare_hf_greedy.py score --hf-dir .local/artifacts/models/<model> --reference hf-bf16.json trt.json
# 12-workload campaigns (notes 377/378); traces from build_serving_workloads.py --bulk-requests 288
$PY $P/run_serving_comparison.py --config <campaign-config.json> --repeats 1 [--reuse-server] --output-dir <dir>
# Capacity plan and sweep (this note)
$PY $P/plan_phase_capacity.py --model-config $A/gemma-4-e2b-it/onnx-fp16-packed-p256/llm/config.json \
  --max-kv-capacity 2048 --max-batch 256 --probe-pool-pages 1024 --probe-max-batch 64 --probe-peak-mib 15621 \
  --gpu-total-mib 40960 --mean-request-tokens 676 --p95-request-tokens 1496
$PY $P/run_capacity_sweep.py --config .local/results/a100-gemma4-e2b-fp16/sweep-config.json \
  --tokenizer .local/artifacts/models/gemma-4-E2B-it --decode-batches 64 128 192 256 \
  --output-dir .local/results/a100-gemma4-e2b-fp16/capacity-sweep
```
Campaign configs are JSON with `model, hf_dir, engine_dir, vision_dir, build_dir, inputs, calibration_trace,
warmup_requests, in_flight, prefill_batch, decode_batch, stable_slots, vision_batch, initial_capacity,
encoder_input_tokens, prefill_chunk, prefill_batch_tokens, vllm_python, vllm_args`. The E2B sweep base used
`prefill_batch 16, prefill_batch_tokens 4096, encoder_input_tokens 1120, initial_capacity 4` and vLLM
`--dtype bfloat16 --max-model-len 2048 --max-num-batched-tokens 4096 --enable-chunked-prefill
--no-enable-prefix-caching --async-scheduling --limit-mm-per-prompt '{"image": 2, "audio": 0, "video": 0}'
--mm-processor-cache-gb 0 --seed 0 --gpu-memory-utilization 0.92` (`--max-num-seqs` set per D).

## Code changes behind notes 377-379 (branch `codex/v0110-phase-forward-port`)
| Commit | Change |
|---|---|
| `3484c71` | Stream syncs in `PhaseKVActiveViewTest` (flaky on A100) |
| `eaac6bd` | Phase prefill fills `vision_block_ids`; mixed-length prefill batching for entry-padded engines; prefill IO sized for the vision profile |
| `f857bca` | Loader keeps a checkpoint `lm_head` despite `tie_word_embeddings` (Cosmos matched HF only after this) |
| `f1d1bf8` | Static decode profile no longer caps engines above its 64 rows |
| `055aa1f`, `c265b22`, `fe38dbe` | Container-free gateway, client, workload builder, campaign runner, shared-server mode, capacity planner and sweep |

## Open items
- Prefill efficiency on text (17 ms per small packed prefill, batch-1 formation) — largest remaining gap to vLLM.
- Knee above D192 for E2B; Cosmos sweep (KV-bound, about D130-240) not run.
- Per-GPU measured decode profile instead of the 3080 table; scheduler starvation after a workload change.
- Gemma 4 12B: packed/chunked prefill with vision-block attention; exporter applies the block mask to global layers
  while HF keeps them causal (unverified A/B).
- Qwen3.6+ (hybrid GDN) needs the v0.10 Thor hybrid phase support ported and W4A16 quantization (no FP8 on SM80).

## Retained paths
- `.local/results/a100-gemma4-e2b-fp16/capacity-sweep` (validation; D64-D192, `STOPPED.txt`, `sweep-config.json`)
- `.local/results/a100-*-fp16/{full12-*,quality}` (notes 377/378)
