#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

preset="${1:-thor-throughput}"
image="${PHASE_IMAGE:-tensorrt-edge-llm:qwen38-phase-next}"
data_dir="${QWEN38_DATA_DIR:-/home/sslab/Desktop/TensorRT-Edge-LLM/data/qwen38}"
port="${PORT:-8001}"
checkpoint_dir="/data/nvfp4"
decode_engine_dir=""
optimized_core_hybrid=0

case "${preset}" in
    thor-throughput | thor-latency)
        engine_dir="/data/engines/phase-b32-p4-d32-kv4096"
        max_stable_slots=32
        serving_preset="${preset}"
        ;;
    thor-throughput-d64)
        engine_dir="/data/engines/phase-b64-p4-d64-kv4096"
        max_stable_slots=64
        serving_preset="thor-throughput"
        ;;
    thor-throughput-core-hybrid)
        engine_dir="/data/engines/phase-b64-p8-d64-fp8kv-input256"
        decode_engine_dir="/data/engines/phase-b64-p8-d64-fp8kv-gdn-int4"
        checkpoint_dir="/data/nvfp4-fp8kv-test"
        max_stable_slots=64
        serving_preset="thor-throughput"
        optimized_core_hybrid=1
        ;;
    *)
        echo "Usage: $0 [thor-throughput|thor-latency|thor-throughput-d64|thor-throughput-core-hybrid]" >&2
        exit 2
        ;;
esac

docker_args=(
    --rm
    --runtime nvidia
    --network host
    --shm-size 8g
)
if [[ "${IGNORE_EOS:-0}" == "1" ]]; then
    docker_args+=(-e TRT_EDGELLM_IGNORE_EOS=1)
fi
if [[ "${PHASE_TIMING_METRICS:-0}" == "1" ]]; then
    docker_args+=(-e TRT_EDGELLM_PHASE_TIMING_METRICS=1)
fi
if [[ "${GDN_SMALL_DECODE:-0}" == "1" ]]; then
    docker_args+=(-e TRT_EDGELLM_GDN_SMALL_DECODE=1)
fi
if [[ "${CAPTURE_PHASE_GRAPHS:-0}" == "1" ]]; then
    docker_args+=(-e TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=1)
fi
if [[ "${CAPTURE_PRODUCTION_GRAPHS:-0}" == "1" ]]; then
    docker_args+=(-e TRT_EDGELLM_CAPTURE_PRODUCTION_GRAPHS=1)
fi
if [[ "${ENABLE_IPC_SHAPE_WARMUP:-0}" != "1" ]]; then
    docker_args+=(-e TRT_EDGELLM_DISABLE_IPC_SHAPE_WARMUP=1)
fi
if [[ -n "${IPC_WARMUP_DECODE_BATCHES:-}" ]]; then
    docker_args+=(-e "TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES=${IPC_WARMUP_DECODE_BATCHES}")
fi
if [[ -n "${PREFILL_QUEUE_TARGET_US:-}" ]]; then
    docker_args+=(-e "TRT_EDGELLM_PREFILL_QUEUE_TARGET_US=${PREFILL_QUEUE_TARGET_US}")
fi
if [[ -n "${MAX_PREFILL_BATCH:-}" ]]; then
    docker_args+=(-e "TRT_EDGELLM_MAX_PREFILL_BATCH=${MAX_PREFILL_BATCH}")
fi
if [[ -n "${DECODE_QUEUE_TARGET_US:-}" ]]; then
    docker_args+=(-e "TRT_EDGELLM_DECODE_QUEUE_TARGET_US=${DECODE_QUEUE_TARGET_US}")
fi
if [[ -n "${decode_engine_dir}" ]]; then
    docker_args+=(-e "TRT_EDGELLM_DECODE_ENGINE_DIR=${decode_engine_dir}")
fi
if [[ "${optimized_core_hybrid}" == "1" ]]; then
    docker_args+=(-e TRT_EDGELLM_GDN_SMALL_DECODE=1)
    docker_args+=(-e TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=1)
    docker_args+=(-e TRT_EDGELLM_CAPTURE_PRODUCTION_GRAPHS=1)
    docker_args+=(-e TRT_EDGELLM_PREFILL_QUEUE_TARGET_US=10000)
fi

exec docker run "${docker_args[@]}" \
    -v "${data_dir}:/data" \
    -w /opt/TensorRT-Edge-LLM-v010 \
    -e TRT_EDGELLM_SEMANTIC_ONLY=1 \
    -e TRT_EDGELLM_PHASE_IPC=1 \
    -e TRT_EDGELLM_MAX_STABLE_SLOTS="${max_stable_slots}" \
    -e TRT_EDGELLM_MAX_INFLIGHT="${max_stable_slots}" \
    -e TRT_EDGELLM_SERVING_PRESET="${serving_preset}" \
    --entrypoint python3 \
    "${image}" \
    scripts/phase_openai_gateway.py \
    --host 127.0.0.1 --port "${port}" --model qwen38 -- \
    build/examples/llm/llm_phase_context_smoke \
    "${engine_dir}" "${checkpoint_dir}"
