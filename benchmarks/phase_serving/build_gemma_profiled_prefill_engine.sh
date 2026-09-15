#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
model_root=$repo_root/.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq
onnx_root=${ONNX_ROOT:-$model_root/onnx-int4-awq-packed-p512/llm}
engine_root=${ENGINE_ROOT:-$model_root/engine-profiled-p8x512-p8x128-d24-kv2048-p96}
build_root=${BUILD_ROOT:-$repo_root/.local/builds/v0101-release}
result_root=${RESULT_ROOT:-$repo_root/.local/results/gemma4-profiled-prefill-rebuild-20260915}
image=nvcr.io/nvidia/tensorrt@sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3

if [[ -f "$engine_root/llm.engine" ]]; then
    exit 0
fi

mkdir -p "$engine_root" "$result_root"
for sidecar in embedding.safetensors ple_embedding.safetensors external_int4_ffn_weights.safetensors; do
    if [[ ! -e "$engine_root/$sidecar" ]]; then
        ln "$onnx_root/$sidecar" "$engine_root/$sidecar"
    fi
done

docker run --rm --gpus all --network none --cap-drop ALL \
    --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,size=1073741824 \
    --user "$(id -u):$(id -g)" \
    -v "$build_root:/opt/edgellm:ro" -v "$onnx_root:/opt/onnx:ro" \
    -v "$engine_root:/opt/engine:rw" \
    -e TRT_PACKAGE_DIR=/usr/local/tensorrt \
    -e EDGELLM_PLUGIN_PATH=/opt/edgellm/libNvInfer_edgellm_plugin.so.1.0 \
    -e LD_LIBRARY_PATH=/opt/edgellm:/usr/local/cuda/lib64:/usr/local/tensorrt/lib \
    -e CUDA_CACHE_PATH=/tmp/cuda-cache \
    "$image" /opt/edgellm/examples/llm/llm_build \
    --onnxDir /opt/onnx --engineDir /opt/engine --maxInputLen 1024 \
    --maxBatchSize 24 --maxPrefillBatchSize 8 --maxDecodeBatchSize 24 \
    --maxKVCacheCapacity 2048 --maxKVPoolPages 96 --allowKVPoolUndercommit \
    --maxPrefillChunkTokens 512 --maxVisionPrefillChunkTokens 128 \
    --maxVisionPrefillBatchSize 8 > "$result_root/build.log" 2>&1
