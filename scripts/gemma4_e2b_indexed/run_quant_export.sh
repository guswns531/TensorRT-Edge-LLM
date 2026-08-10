#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_dir=${REPO_DIR:-/workspace/TensorRT-Edge-LLM}
work_dir=${WORK_DIR:-${repo_dir}/.local/gemma4-e2b}
model_id=${MODEL_ID:-google/gemma-4-E2B-it}
model_revision=${MODEL_REVISION:?MODEL_REVISION must be an immutable Hugging Face commit SHA}
mode=${MODE:-legacy}

python3 -m pip install -e "${repo_dir}[tools]"
hf download "${model_id}" --revision "${model_revision}" --local-dir "${work_dir}/hf"

tensorrt-edgellm-quantize llm \
    --model_dir "${work_dir}/hf" \
    --output_dir "${work_dir}/quant-int4-awq" \
    --quantization int4_awq \
    --dtype fp16 \
    --device cpu \
    --text_dataset wikitext \
    --num_samples 128 \
    --seed 0

indexed_args=()
if [[ "${mode}" == "indexed" ]]; then
    indexed_args+=(--indexed-kv-cache)
fi

tensorrt-edgellm-export \
    "${work_dir}/quant-int4-awq" \
    "${work_dir}/onnx-${mode}" \
    --skip-visual \
    --skip-audio \
    --skip-code2wav \
    --skip-action \
    --externalize-weights int4_ffn \
    "${indexed_args[@]}"
