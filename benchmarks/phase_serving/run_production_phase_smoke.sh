#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model=${1:?Usage: run_production_phase_smoke.sh cosmos|gemma text|mixed [graphs=1]}
workload=${2:?Select text or mixed}
graphs=${3:-1}
build_root=${BUILD_ROOT:-$repo_root/.local/builds/v0101-release}
result_root=${RESULT_ROOT:-$repo_root/.local/results/runtime-contract-revalidation-20260926/production}
image=nvcr.io/nvidia/tensorrt@sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3

case "$model" in
    cosmos)
        engine=$repo_root/.local/artifacts/v0101-forward-port/text/cosmos-reason2-2b/engine-p8-d64-kv256-p128-vp1024-atomic
        vision=$repo_root/.local/artifacts/v0101-forward-port/cosmos-reason2-2b/vision-exact-gelu/engine/visual
        checkpoint=$repo_root/.local/artifacts/models/cosmos-reason2-2b/hf
        decode_graphs=64
        ;;
    gemma)
        engine=$repo_root/.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-packed-p8-d24-kv2048-p192
        vision=$repo_root/.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/visual-e4-soft280/visual
        checkpoint=$repo_root/.local/artifacts/models/gemma-4-e2b-it-awq/hf
        decode_graphs=24
        ;;
    *) exit 2 ;;
esac
engine=${ENGINE_DIR:-$engine}
vision=${VISION_DIR:-$vision}
case "$workload" in text|mixed) ;; *) exit 2 ;; esac
case "$graphs" in 0|1) ;; *) exit 2 ;; esac
test -f "$result_root/$workload.json"
test -x "$build_root/examples/llm/llm_inference"
name=$model-$workload-graphs$graphs
test ! -e "$result_root/$name.json"

docker run --rm --gpus all --network none --cap-drop ALL \
    --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,size=268435456 \
    --user "$(id -u):$(id -g)" \
    -v "$build_root:/opt/edgellm:ro" -v "$engine:/opt/model:ro" \
    -v "$vision:/opt/vision:ro" -v "$checkpoint:/opt/hf:ro" \
    -v "$repo_root/examples/multimodal/pics:/opt/images:ro" \
    -v "$result_root:/opt/results:rw" \
    -e TRT_PACKAGE_DIR=/usr/local/tensorrt \
    -e EDGELLM_PLUGIN_PATH=/opt/edgellm/libNvInfer_edgellm_plugin.so.1.0 \
    -e LD_LIBRARY_PATH=/opt/edgellm:/usr/local/cuda/lib64:/usr/local/tensorrt/lib \
    -e CUDA_CACHE_PATH=/tmp/cuda-cache \
    -e "TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=$graphs" \
    -e TRT_EDGELLM_ONLINE_GRAPH_CAPTURE=0 \
    -e "TRT_EDGELLM_MAX_DECODE_GRAPHS=$decode_graphs" \
    -e TRT_EDGELLM_MAX_PREFILL_GRAPHS=4 \
    -e TRT_EDGELLM_FIXED_PREFILL_CHUNK=128 \
    -e "TRT_EDGELLM_PHASE_WORKSPACE_MODE=${WORKSPACE_MODE:-shared_ep}" \
    -e "TRT_EDGELLM_PHASE_ACTIVITY_PREFIX=/opt/results/$name-activity" \
    "$image" /opt/edgellm/examples/llm/llm_inference \
    --phaseServing --phasePolicy service-scaled-transition \
    --engineDir /opt/model --multimodalEngineDir /opt/vision --checkpointDir /opt/hf \
    --inputFile "/opt/results/$workload.json" --outputFile "/opt/results/$name.json" \
    --dumpOutput 2>&1 | tee "$result_root/$name.log"
