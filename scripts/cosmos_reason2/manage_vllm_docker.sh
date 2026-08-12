#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

readonly IMAGE="vllm/vllm-openai@sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2"
readonly CONTAINER_NAME="edgellm-vllm-cosmos"
readonly MODEL_NAME="nvidia/Cosmos-Reason2-2B"
readonly MATCHED_KV_BYTES="3758096384"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_dir="${repo_dir}/.local/cosmos-reason2-2b/hf"
result_dir="${repo_dir}/.local/vllm-cosmos-reason2-2b"
action="${1:-status}"

create_server()
{
    local mode="$1"
    if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        echo "Container ${CONTAINER_NAME} already exists; use start/stop or remove it explicitly." >&2
        exit 1
    fi
    if [[ ! -f "${model_dir}/model.safetensors" ]]; then
        echo "Missing local checkpoint: ${model_dir}/model.safetensors" >&2
        exit 1
    fi
    mkdir -p "${result_dir}/cache"
    local -a mode_args
    case "${mode}" in
    production)
        mode_args=(--gpu-memory-utilization 0.90)
        ;;
    memory-matched)
        mode_args=(--kv-cache-memory "${MATCHED_KV_BYTES}")
        ;;
    eager)
        mode_args=(--gpu-memory-utilization 0.90 --enforce-eager)
        ;;
    *)
        echo "Unknown mode: ${mode}" >&2
        exit 1
        ;;
    esac
    docker run --detach \
        --name "${CONTAINER_NAME}" \
        --gpus all \
        --ipc=host \
        --publish 8000:8000 \
        --volume "${model_dir}:/model:ro" \
        --volume "${result_dir}:/results" \
        --volume "${result_dir}/cache:/root/.cache/vllm" \
        "${IMAGE}" \
        /model \
        --served-model-name "${MODEL_NAME}" \
        --language-model-only \
        --dtype float16 \
        --kv-cache-dtype float16 \
        --max-model-len 2048 \
        --max-num-seqs 80 \
        --max-num-batched-tokens 8192 \
        --enable-chunked-prefill \
        --no-enable-prefix-caching \
        --generation-config vllm \
        "${mode_args[@]}"
}

case "${action}" in
create-production)
    create_server production
    ;;
create-memory-matched)
    create_server memory-matched
    ;;
create-eager)
    create_server eager
    ;;
start)
    docker start "${CONTAINER_NAME}"
    ;;
stop)
    docker stop "${CONTAINER_NAME}"
    ;;
status)
    docker ps --all --filter "name=^/${CONTAINER_NAME}$"
    ;;
logs)
    docker logs --follow "${CONTAINER_NAME}"
    ;;
*)
    echo "Usage: $0 {create-production|create-memory-matched|create-eager|start|stop|status|logs}" >&2
    exit 1
    ;;
esac
