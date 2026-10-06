#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
model_root=${MODEL_ROOT:-$repo_root/.local/artifacts/models/Cosmos-Reason2-2B}
build_root=${BUILD_ROOT:-$repo_root/.local/builds/colab-a100-sm80}
result_root=${RESULT_ROOT:?Set RESULT_ROOT to a new campaign directory}
onnx_root=${ONNX_ROOT:?Set ONNX_ROOT to a new export directory}
engine_root=${ENGINE_ROOT:?Set ENGINE_ROOT to a new engine directory}
small_tokens=${SMALL_PROFILE_MAX_IMAGE_TOKENS:-2816}
max_tokens=${MAX_IMAGE_TOKENS:-22528}
per_image_tokens=${MAX_IMAGE_TOKENS_PER_IMAGE:-2816}

: "${TRT_PACKAGE_DIR:?Set TRT_PACKAGE_DIR}"
: "${EXPORT_VENV:?Set EXPORT_VENV}"
export LD_LIBRARY_PATH="$build_root:$TRT_PACKAGE_DIR/lib:${LD_LIBRARY_PATH:-}"
export EDGELLM_PLUGIN_PATH="$build_root/libNvInfer_edgellm_plugin.so.1.0"

if [[ -e "$engine_root/visual/visual.engine" || -e "$engine_root/visual.engine" ]]; then
    echo "Refusing to overwrite a retained engine: $engine_root" >&2
    exit 1
fi
mkdir -p "$result_root" "$engine_root"
if [[ ! -f "$onnx_root/visual/model.onnx" ]]; then
    "$EXPORT_VENV/bin/python" -m tensorrt_edgellm.scripts.export \
        "$model_root" "$onnx_root" --components visual --dtype float16 \
        > "$result_root/export.log" 2>&1
fi
"$build_root/examples/multimodal/visual_build" \
    --onnxDir "$onnx_root/visual" --engineDir "$engine_root" \
    --minImageTokens 4 --maxImageTokens "$max_tokens" \
    --maxImageTokensPerImage "$per_image_tokens" \
    --smallProfileMaxImageTokens "$small_tokens" \
    > "$result_root/build.log" 2>&1
