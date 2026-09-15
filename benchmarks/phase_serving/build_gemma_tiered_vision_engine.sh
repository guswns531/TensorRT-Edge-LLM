#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
model_root=${MODEL_ROOT:-$repo_root/.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq}
onnx_root=${ONNX_ROOT:-$model_root/onnx-int4-awq-p128/visual}
engine_root=${ENGINE_ROOT:-$model_root/visual-tiered-e1-e4-soft280}
build_root=${BUILD_ROOT:-$repo_root/.local/builds/v0101-release}
result_root=${RESULT_ROOT:-$repo_root/.local/results/gemma4-tiered-vision-engine-20260915}
small_profile_max_image_tokens=${SMALL_PROFILE_MAX_IMAGE_TOKENS:-280}
image=nvcr.io/nvidia/tensorrt@sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3

if [[ -f "$engine_root/visual/visual.engine" ]]; then
    exit 0
fi

mkdir -p "$engine_root" "$result_root"
docker run --rm --gpus all --network none --cap-drop ALL \
    --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,size=1073741824 \
    --user "$(id -u):$(id -g)" \
    -v "$build_root:/opt/edgellm:ro" -v "$onnx_root:/opt/onnx:ro" \
    -v "$engine_root:/opt/engine:rw" \
    -e TRT_PACKAGE_DIR=/usr/local/tensorrt \
    -e EDGELLM_PLUGIN_PATH=/opt/edgellm/libNvInfer_edgellm_plugin.so.1.0 \
    -e LD_LIBRARY_PATH=/opt/edgellm:/usr/local/cuda/lib64:/usr/local/tensorrt/lib \
    -e CUDA_CACHE_PATH=/tmp/cuda-cache \
    "$image" /opt/edgellm/examples/multimodal/visual_build \
    --onnxDir /opt/onnx --engineDir /opt/engine \
    --minImageTokens 4 --maxImageTokens 1120 --maxImageTokensPerImage 280 \
    --smallProfileMaxImageTokens "$small_profile_max_image_tokens" > "$result_root/build.log" 2>&1
