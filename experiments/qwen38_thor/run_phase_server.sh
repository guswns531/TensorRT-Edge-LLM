#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

preset="${1:-thor-throughput}"
image="${PHASE_IMAGE:-tensorrt-edge-llm:qwen38-phase-next}"
data_dir="${QWEN38_DATA_DIR:-/home/sslab/Desktop/TensorRT-Edge-LLM/data/qwen38}"
port="${PORT:-8001}"

case "${preset}" in
    thor-throughput | thor-latency)
        engine_dir="/data/engines/phase-b32-p4-d32-kv4096"
        ;;
    *)
        echo "Usage: $0 [thor-throughput|thor-latency]" >&2
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

exec docker run "${docker_args[@]}" \
    -v "${data_dir}:/data" \
    -w /opt/TensorRT-Edge-LLM-v010 \
    -e TRT_EDGELLM_SEMANTIC_ONLY=1 \
    -e TRT_EDGELLM_PHASE_IPC=1 \
    -e TRT_EDGELLM_MAX_STABLE_SLOTS=32 \
    -e TRT_EDGELLM_MAX_INFLIGHT=32 \
    -e TRT_EDGELLM_SERVING_PRESET="${preset}" \
    -e TRT_EDGELLM_DISABLE_IPC_SHAPE_WARMUP=1 \
    --entrypoint python3 \
    "${image}" \
    scripts/phase_openai_gateway.py \
    --host 127.0.0.1 --port "${port}" --model qwen38 -- \
    build/examples/llm/llm_phase_context_smoke \
    "${engine_dir}" /data/nvfp4
