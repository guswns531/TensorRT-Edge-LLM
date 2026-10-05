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
"""Size the KV page pool and decode batch for a phase-serving engine on a given GPU.

Step 1 of capacity selection (step 2 is the measured sweep in run_capacity_sweep.py).
Everything except the KV pool (weights, embedding/PLE tables, per-phase activation
workspaces, CUDA context) is taken from one measured probe server, so the plan transfers
to a new GPU by re-running a single probe there instead of modelling every allocation:

    budget_mib     = gpu_total_mib * utilization - headroom_mib
    fixed_mib      = probe_peak_mib - probe_kv_mib(probe_pool_pages, probe_max_batch)
    max_pool_pages = (budget_mib - fixed_mib - swa_mib(max_batch)) / full_page_mib

KV bytes follow KVCacheManager: one {2, pages, 128, heads, head_dim} FP16 tensor per
owning layer (kv_sharing_donors == -1); reduced-capacity (SWA) layers use the bounded
pool only when it is smaller than the full pool. Decode-batch bounds then follow from the
pool: worst case (every slot at max KV capacity) and typical (mean request length).
"""

import argparse
import json
import math
import pathlib

TOKENS_PER_PAGE = 128
SWA_BOUNDARY_HEADROOM_PAGES = 1
SWA_REPLACEMENT_PAGES = 1
MIB = 1024.0 * 1024.0


def kv_layers(config):
    """Return [(heads, head_dim, reduced_capacity_or_None)] for layers that own a KV pool."""
    layers = config.get("kv_layer_configs") or []
    count = config.get("num_attention_layers", config["num_hidden_layers"])
    if not layers:
        head_dim = config.get(
            "head_dim"
        ) or config["hidden_size"] // config["num_attention_heads"]
        layers = [{
            "num_kv_heads": config["num_key_value_heads"],
            "head_dim": head_dim
        }] * count
    donors = config.get("kv_sharing_donors") or [-1] * len(layers)
    owned = []
    for layer, donor in zip(layers, donors):
        if donor != -1:
            continue
        capacity = layer.get("kv_cache_capacity", 0)
        owned.append((layer["num_kv_heads"], layer["head_dim"], capacity
                      or None))
    return owned


def page_mib(layers, reduced):
    """MiB of one page across the owning layers that are (reduced=True) or are not SWA-bounded."""
    bytes_per_page = sum(2 * TOKENS_PER_PAGE * heads * dim * 2
                         for heads, dim, cap in layers
                         if (cap is not None) == reduced)
    return bytes_per_page / MIB


def swa_pages_per_slot(window):
    retained = math.ceil(
        window / TOKENS_PER_PAGE) + SWA_BOUNDARY_HEADROOM_PAGES
    return 2 * retained + SWA_REPLACEMENT_PAGES


def kv_mib(layers, pool_pages, max_batch, max_kv_capacity):
    """KV allocation for a pool, choosing bounded SWA exactly when it is smaller (DeploymentConfig)."""
    full = page_mib(layers, False) * pool_pages
    windows = {
        cap
        for _, _, cap in layers if cap is not None and cap < max_kv_capacity
    }
    swa_page = page_mib(layers, True)
    if not windows or swa_page == 0:
        return full + swa_page * pool_pages
    swa_pages = max_batch * swa_pages_per_slot(max(windows))
    return full + swa_page * min(pool_pages, swa_pages)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config",
                        type=pathlib.Path,
                        required=True,
                        help="Exported llm/config.json")
    parser.add_argument("--max-kv-capacity", type=int, required=True)
    parser.add_argument("--max-batch",
                        type=int,
                        required=True,
                        help="Target engine maxBatchSize (stable slots)")
    parser.add_argument("--probe-pool-pages", type=int, required=True)
    parser.add_argument("--probe-max-batch", type=int, required=True)
    parser.add_argument("--probe-peak-mib",
                        type=float,
                        required=True,
                        help="Measured serving GPU peak")
    parser.add_argument("--gpu-total-mib", type=float, required=True)
    parser.add_argument("--utilization", type=float, default=0.92)
    parser.add_argument("--headroom-mib", type=float, default=1024.0)
    parser.add_argument("--mean-request-tokens", type=float, required=True)
    parser.add_argument("--p95-request-tokens", type=float, required=True)
    args = parser.parse_args()

    config = json.loads(args.model_config.read_text())
    layers = kv_layers(config)
    probe_kv = kv_mib(layers, args.probe_pool_pages, args.probe_max_batch,
                      args.max_kv_capacity)
    fixed = args.probe_peak_mib - probe_kv
    budget = args.gpu_total_mib * args.utilization - args.headroom_mib
    full_page = page_mib(layers, False)
    swa_page = page_mib(layers, True)
    windows = {
        cap
        for _, _, cap in layers
        if cap is not None and cap < args.max_kv_capacity
    }
    swa_fixed = swa_page * args.max_batch * swa_pages_per_slot(
        max(windows)) if windows and swa_page else 0.0
    # Without bounded SWA every page also carries the SWA layers.
    per_page = full_page if swa_fixed else full_page + swa_page
    max_pool = int((budget - fixed - swa_fixed) // per_page)
    full_commit = args.max_batch * math.ceil(
        args.max_kv_capacity / TOKENS_PER_PAGE)
    pool = min(max_pool, full_commit)
    plan = {
        "owning_layers":
        len(layers),
        "page_mib_full_layers":
        round(full_page, 3),
        "page_mib_swa_layers":
        round(swa_page, 3),
        "fixed_mib_from_probe":
        round(fixed),
        "budget_mib":
        round(budget),
        "max_pool_pages":
        max_pool,
        "full_commit_pages":
        full_commit,
        "pool_pages":
        pool,
        "undercommit":
        pool < full_commit,
        "predicted_peak_mib":
        round(fixed +
              kv_mib(layers, pool, args.max_batch, args.max_kv_capacity)),
        "decode_batch_worst_case":
        pool // math.ceil(args.max_kv_capacity / TOKENS_PER_PAGE),
        "decode_batch_p95":
        pool // math.ceil(args.p95_request_tokens / TOKENS_PER_PAGE),
        "decode_batch_mean":
        pool // math.ceil(args.mean_request_tokens / TOKENS_PER_PAGE),
    }
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
