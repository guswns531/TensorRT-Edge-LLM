#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
result_root=${RESULT_ROOT:-$repo_root/.local/results/gemma4-v3-activity-diagnostic-20260913}
trace_root=${TRACE_ROOT:-$repo_root/.local/results/gemma4-e2b-awq-full12-20260911/inputs}
calibration=${CALIBRATION_TRACE:-$repo_root/.local/results/gemma4-packed-prefill-g4-20260912/generic-p8-d24-e4.json}
replay_tools=${REPLAY_TOOLS:-$repo_root/.local/results/v0101-forward-port/replay-tools}
build_root=${BUILD_ROOT:-$repo_root/.local/builds/v0101-release}
model_root=${MODEL_ROOT:-$repo_root/.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq}
result_root=$(realpath -m "$result_root")
trace_root=$(realpath "$trace_root")
calibration=$(realpath "$calibration")
replay_tools=$(realpath "$replay_tools")
build_root=$(realpath "$build_root")
model_root=$(realpath "$model_root")
engine_root=${ENGINE_ROOT:-$model_root/engine-packed-p8-d24-kv2048-p96}
allocatable_kv_pages=${ALLOCATABLE_KV_PAGES:-0}
encoded_capacity=${ENCODED_CAPACITY:-4}
encoded_admission_mode=${ENCODED_ADMISSION_MODE:-}
encoded_admission_capacity=${ENCODED_ADMISSION_CAPACITY:-12}
prefill_chunk=${PREFILL_CHUNK:-128}
prefill_batch_tokens=${MAX_PREFILL_BATCH_TOKENS:-$((8 * prefill_chunk))}
decode_active_prefill_chunk=${DECODE_ACTIVE_PREFILL_CHUNK:-0}
large_prefill_chunk_queue_threshold=${LARGE_PREFILL_CHUNK_QUEUE_THRESHOLD:-0}
kv_environment=(-e "TRT_EDGELLM_ALLOCATABLE_KV_PAGES=$allocatable_kv_pages")
if [[ "${KV_BUDGET_AT_MEASUREMENT:-0}" == 1 ]]; then
    kv_environment=(-e "TRT_EDGELLM_MEASUREMENT_KV_PAGES=$allocatable_kv_pages")
fi
formation_environment=()
if [[ "${DISABLE_WAVEFRONT_PREFILL:-0}" == 1 ]]; then
    formation_environment=(-e TRT_EDGELLM_DISABLE_WAVEFRONT_PREFILL=1)
fi
if [[ "${ENABLE_PREFILL_COHORT_REFILL:-0}" == 1 ]]; then
    formation_environment+=(-e TRT_EDGELLM_ENABLE_PREFILL_COHORT_REFILL=1)
fi
graph_environment=()
if [[ "${ENABLE_CUDA_GRAPHS:-0}" == 1 ]]; then
    graph_environment=(-e TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=1)
fi
vision_environment=()
if [[ "${ENABLE_VISION:-1}" == 1 ]]; then
    vision_environment=(-e TRT_EDGELLM_VISION_ENGINE_DIR=/opt/vision)
    if [[ "${TIERED_VISION_CONTEXT_MEMORY:-0}" == 1 ]]; then
        vision_environment+=(-e TRT_EDGELLM_TIERED_VISION_CONTEXT_MEMORY=1)
    fi
fi
encoded_admission_environment=()
if [[ -n "$encoded_admission_mode" ]]; then
    encoded_admission_environment=(-e "TRT_EDGELLM_MEASUREMENT_ENCODED_ADMISSION=$encoded_admission_mode"
        -e "TRT_EDGELLM_MEASUREMENT_ENCODED_CAPACITY=$encoded_admission_capacity")
fi
hf_root=$repo_root/.local/artifacts/models/gemma-4-e2b-it-awq/hf
image=nvcr.io/nvidia/tensorrt@sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3
cases=${CASES:-"balanced long-prefill bimodal mixed vision-heavy multi-image"}
repeats=${REPEATS:-1}
phase_policy=${PHASE_POLICY:-service-scaled-transition}
profiler_mount=()
profiler_command=()
if [[ "${NSYS_CAPTURE:-0}" == 1 ]]; then
    profiler_mount=(-v /opt/nvidia/nsight-systems/2026.1.3:/opt/nsys:ro)
    profiler_command=(/opt/nsys/target-linux-x64/nsys profile --sample none --cpuctxsw none
        --trace cuda,nvtx --output "/opt/results/run-{run}/nsys")
fi

for workload in $cases; do
    cell=$result_root/$workload
    mkdir -p "$cell"
    if [[ -f "$cell/aggregate.json" ]]; then
        continue
    fi
    PHASE_TRACE_CLIENT_IMPL=$replay_tools/run_vllm_trace_bench.py \
        python3 "$replay_tools/run_phase_http_trace_bench.py" \
        --gateway-script "$replay_tools/run_phase_openai_gateway.py" \
        --client-script "$replay_tools/run_vllm_trace_bench.py" \
        --trace "$trace_root/$workload.json" --output-dir "$cell" \
        --model google/gemma-4-e2b-it --repeats "$repeats" --max-workers 24 --max-in-flight 24 \
        --ready-timeout 180 --request-timeout 600 --policy-warmup-mode generic \
        --generic-warmup-trace "$calibration" --warmup-requests 49 \
        --phase-calibration-round-requests 49 --phase-calibration-min-requests 49 --ignore-eos -- \
        docker run --rm --gpus all --network none --cap-drop ALL \
        --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,size=268435456 \
        -i --user "$(id -u):$(id -g)" \
        -v "$build_root:/opt/edgellm:ro" \
        -v "$engine_root:/opt/model:ro" \
        -v "$model_root/visual-e4-soft280/visual:/opt/vision:ro" \
        -v "$hf_root:/opt/hf:ro" \
        -v "$repo_root/examples/multimodal/pics:/workspace/examples/multimodal/pics:ro" \
        -v "$cell:/opt/results:rw" \
        "${profiler_mount[@]}" \
        -e TRT_PACKAGE_DIR=/usr/local/tensorrt \
        -e EDGELLM_PLUGIN_PATH=/opt/edgellm/libNvInfer_edgellm_plugin.so.1.0 \
        -e LD_LIBRARY_PATH=/opt/edgellm:/usr/local/cuda/lib64:/usr/local/tensorrt/lib \
        -e CUDA_CACHE_PATH=/tmp/cuda-cache \
        -e TRT_EDGELLM_PHASE_IPC=1 -e TRT_EDGELLM_SEMANTIC_ONLY=1 -e TRT_EDGELLM_IGNORE_EOS=1 \
        -e TRT_EDGELLM_MAX_STABLE_SLOTS=24 -e TRT_EDGELLM_MAX_INFLIGHT=24 \
        "${kv_environment[@]}" "${formation_environment[@]}" "${graph_environment[@]}" \
        -e TRT_EDGELLM_MAX_PREFILL_BATCH=8 -e TRT_EDGELLM_MAX_DECODE_BATCH=24 \
        -e "TRT_EDGELLM_FIXED_PREFILL_CHUNK=$prefill_chunk" \
        -e "TRT_EDGELLM_MAX_PREFILL_BATCH_TOKENS=$prefill_batch_tokens" \
        -e "TRT_EDGELLM_DECODE_ACTIVE_PREFILL_CHUNK=$decode_active_prefill_chunk" \
        -e "TRT_EDGELLM_LARGE_PREFILL_CHUNK_QUEUE_THRESHOLD=$large_prefill_chunk_queue_threshold" \
        -e TRT_EDGELLM_ENABLE_DYNAMIC_DECODE=1 -e TRT_EDGELLM_ENABLE_DECODE_COHORT=1 \
        -e TRT_EDGELLM_ENABLE_BATCHED_VISION_PREFILL=1 -e TRT_EDGELLM_RELEASE_VISION_PREFILL_STORAGE=1 \
        -e "TRT_EDGELLM_MAX_ENCODED_VISION=$encoded_capacity" -e TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE=4 \
        "${encoded_admission_environment[@]}" \
        -e TRT_EDGELLM_VISION_ENCODER_MAX_INPUT_TOKENS=1120 -e TRT_EDGELLM_VISION_ENCODER_MAX_MEDIA=4 \
        -e TRT_EDGELLM_VISION_ENCODER_BATCH_WAIT_US=25000 -e TRT_EDGELLM_VISION_PREFILL_BATCH_SIZE=4 \
        -e TRT_EDGELLM_PREFILL_TTFT_HARD_GUARD=1 -e TRT_EDGELLM_VISION_IDLE_SLABS=0 \
        -e TRT_EDGELLM_GLOBAL_SCHEDULER=active -e TRT_EDGELLM_SYNCHRONIZE_DECODE_SAMPLING=1 \
        -e TRT_EDGELLM_GLOBAL_SAFE_PROBE_INTERVAL=0 -e TRT_EDGELLM_GLOBAL_FORMATION_REALIZED_DISPATCHES=4 \
        -e TRT_EDGELLM_IPC_ASYNC_REQUEST_ADAPTER=1 -e TRT_EDGELLM_IPC_REQUEST_ADAPTER_WORKERS=4 \
        -e TRT_EDGELLM_VISION_ASYNC_PREPARATION=1 -e TRT_EDGELLM_VISION_PREPARATION_PD_DISPATCH=1 \
        -e TRT_EDGELLM_VISION_ENCODER_ARBITER=1 -e TRT_EDGELLM_VISION_PREFIX_PREFILL=1 \
        -e TRT_EDGELLM_POLICY_WARMUP_MODE=generic -e "TRT_EDGELLM_PHASE_POLICY=$phase_policy" \
        "${vision_environment[@]}" \
        -e "TRT_EDGELLM_PHASE_ACTIVITY_PREFIX=/opt/results/run-{run}/activity" \
        -e TRT_EDGELLM_EMIT_PHASE_METRICS=1 -e TRT_EDGELLM_PHASE_TELEMETRY_LEVEL=full \
        "$image" "${profiler_command[@]}" \
        /opt/edgellm/examples/llm/llm_phase_context_smoke /opt/model /opt/hf \
        > "$cell/driver.log" 2>&1
done
