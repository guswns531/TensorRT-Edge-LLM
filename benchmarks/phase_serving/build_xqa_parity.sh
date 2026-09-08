#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 OLD_SOURCE OLD_BUILD NEW_SOURCE NEW_BUILD OUTPUT_DIR" >&2
    exit 2
fi
: "${TRT_PACKAGE_DIR:?Set TRT_PACKAGE_DIR}"
old_source=$1
old_build=$2
new_source=$3
new_build=$4
probe_output=$5
cuda_root=${CUDA_ROOT:-/usr/local/cuda}
mkdir -p "$probe_output"
probe_source="$new_source/benchmarks/phase_serving/xqaParityProbe.cpp"

# The cache parameter ABI differs; each executable must use its matching headers and archive.
nvcc -std=c++17 -I"$old_source/cpp" -I"$TRT_PACKAGE_DIR/include" "$probe_source" \
    "$old_build/cpp/libedgellmCore.a" -L"$cuda_root/lib64/stubs" -lcuda -ldl \
    -o "$probe_output/prebuilt"
nvcc -std=c++17 -DEDGELLM_PROBE_JIT -I"$new_source/cpp" -I"$TRT_PACKAGE_DIR/include" "$probe_source" \
    "$new_build/cpp/libedgellmCore.a" "$new_build/cpp/libedgellmPluginJit.a" \
    -L"$cuda_root/lib64/stubs" -lcuda -lnvrtc -ldl -o "$probe_output/jit"
