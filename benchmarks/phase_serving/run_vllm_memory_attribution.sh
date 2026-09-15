#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
mode=${1:?usage: run_vllm_memory_attribution.sh eager|graph}
if [[ "$mode" != eager && "$mode" != graph ]]; then
    echo "mode must be eager or graph" >&2
    exit 2
fi

result_root=${RESULT_ROOT:-$repo_root/.local/results/memory-attribution-20260915/vllm-$mode}
trace_root=${TRACE_ROOT:-$repo_root/.local/results/gemma4-e2b-awq-full12-20260911/inputs}
client=${CLIENT:-$repo_root/.local/results/v0101-forward-port/replay-tools/run_vllm_trace_bench.py}
model=/workspace/.local/artifacts/models/gemma-4-e2b-it-awq-vllm028-compat/hf
image=vllm/vllm-openai:v0.28.0
container=gemma4-vllm-memory-$mode
mkdir -p "$result_root"

cleanup() {
    docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

graph_args=(--enforce-eager)
if [[ "$mode" == graph ]]; then
    graph_args=(-cc.backend=eager --cudagraph-capture-sizes 1 2 4 8 16 24)
fi

docker run --rm -d --name "$container" --gpus all --ipc=host -p 8011:8000 \
    -v "$repo_root:/workspace:ro" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$image" "$model" \
    --served-model-name gemma4-e2b-awq \
    --dtype half \
    --trust-remote-code \
    --max-model-len 2048 \
    --max-num-seqs 24 \
    --max-num-batched-tokens 4096 \
    --kv-cache-memory-bytes "$((480 * 1024 * 1024))" \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --async-scheduling \
    --limit-mm-per-prompt '{"image":8,"audio":0}' \
    --allowed-local-media-path /workspace \
    --generation-config vllm \
    "${graph_args[@]}" >"$result_root/container-id.txt"

ready=0
for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8011/health >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -qx true; then
        break
    fi
    sleep 1
done
docker logs "$container" >"$result_root/server.log" 2>&1 || true
if ((ready == 0)); then
    echo "vLLM server failed to become ready" >&2
    exit 3
fi

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >"$result_root/ready-memory-mib.txt"
for workload in balanced multi-image; do
    python3 "$client" \
        --endpoint http://127.0.0.1:8011 \
        --model gemma4-e2b-awq \
        --trace "$trace_root/$workload.json" \
        --output-dir "$result_root/$workload" \
        --repeats 1 \
        --warmup-requests 0 \
        --max-workers 1 \
        --max-in-flight 1 \
        --request-limit 1 \
        --timeout 600 \
        --ignore-eos \
        --sample-gpu-memory
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        >"$result_root/after-$workload-memory-mib.txt"
done
docker logs "$container" >"$result_root/server.log" 2>&1 || true
