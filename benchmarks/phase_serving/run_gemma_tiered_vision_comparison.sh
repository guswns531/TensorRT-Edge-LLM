#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
result_root=${RESULT_ROOT:-$repo_root/.local/results/gemma4-tiered-vision-comparison-20260915}
model_root=${MODEL_ROOT:-$repo_root/.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq}
baseline_vision_root=${BASELINE_VISION_ROOT:-$model_root/visual-e4-soft280/visual}
tiered_engine_root=${TIERED_ENGINE_ROOT:-$model_root/visual-tiered-e3-e4-soft280}
tiered_vision_root=$tiered_engine_root/visual
cases=${CASES:-"mixed vision-heavy multi-image"}
repeats=${REPEATS:-3}

SMALL_PROFILE_MAX_IMAGE_TOKENS=${SMALL_PROFILE_MAX_IMAGE_TOKENS:-840} \
    ENGINE_ROOT="$tiered_engine_root" \
    RESULT_ROOT="$result_root/tiered-engine-build" \
    "$repo_root/benchmarks/phase_serving/build_gemma_tiered_vision_engine.sh"

RESULT_ROOT="$result_root/independent" \
    VISION_ENGINE_ROOT="$baseline_vision_root" \
    CASES="$cases" REPEATS="$repeats" ENABLE_CUDA_GRAPHS=1 \
    ENCODED_ADMISSION_MODE=lifetime ENCODED_ADMISSION_CAPACITY=12 \
    "$repo_root/benchmarks/phase_serving/run_gemma_v3_activity_diagnostic.sh"

RESULT_ROOT="$result_root/tiered-e3-e4" \
    VISION_ENGINE_ROOT="$tiered_vision_root" TIERED_VISION_CONTEXT_MEMORY=1 \
    CASES="$cases" REPEATS="$repeats" ENABLE_CUDA_GRAPHS=1 \
    ENCODED_ADMISSION_MODE=lifetime ENCODED_ADMISSION_CAPACITY=12 \
    "$repo_root/benchmarks/phase_serving/run_gemma_v3_activity_diagnostic.sh"

python3 "$repo_root/benchmarks/phase_serving/compare_http_policy_variants.py" \
    --variant "independent=$result_root/independent" \
    --variant "tiered-e3-e4=$result_root/tiered-e3-e4" \
    --allow-token-trace-mismatch \
    --output-json "$result_root/comparison.json" \
    --output-csv "$result_root/comparison.csv"
