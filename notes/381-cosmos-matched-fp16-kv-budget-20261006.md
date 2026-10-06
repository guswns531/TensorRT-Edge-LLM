<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cosmos: matched FP16 and KV memory budget

This follow-up supersedes the primary comparison in note 380. FP16 versus BF16 was a precision
mismatch, although neither model was quantized. Prefix caching and CPU multimodal processor caching
being disabled did not disable vLLM's native cross-request image encoder cache. Cache-enabled BF16
results in note 380 remain a separate native-serving reference, not the headline for this contract.

## Contract

- Cosmos-Reason2-2B checkpoint revision `9ce19a195e423419c349abfc86fd07178b230561` on one A100-SXM4-40GB.
- Both systems use unquantized FP16 weights and FP16 KV; maximum sequence length is 8192.
- Physical KV allocation is 21 GiB (22548578304 bytes) in both systems: TRT has 1536 pages of
  128 tokens, and vLLM reports 196608 tokens. `--kv-cache-memory-bytes` overrides vLLM's
  utilization-based KV sizing, so U0.95 is not the effective KV budget in this comparison.
- Prefix and CPU multimodal processor caches are disabled in vLLM. Each image occurrence has
  a unique UUID, including a distinct calibration namespace. vLLM's native input processor uses
  this UUID as its cache identifier, preventing cross-request image embedding reuse while preserving
  image bytes. Temporary storage needed within a request remains enabled in both implementations.
- TRT retains its validated E8/P32/D256 contract, 2.5 GiB encoded buffer ceiling and 24 GiB
  managed-byte limit. The managed limit includes KV; it excludes weights and workspaces and is
  not a total GPU-memory cap. vLLM retains native encoder scheduling/storage; its temporary
  encoder buffer limit is not asserted to equal TRT's encoded ceiling.
- Same four traces as note 380, with only image UUID metadata added: 1024 balanced and decode-heavy
  requests each, 512 mixed and vision-heavy requests each; external client concurrency 512.
- Generic calibration: 1119 requests, followed by no workload-specific warmup in either system.
  Each vLLM pass starts a fresh server so UUIDs cannot be reused across passes. TRT has no
  cross-request image reuse and repeats on one server. Fixed output lengths ignore EOS.
- Measured source/binary commit: `dd773f05a7010634fa49e18543531bf9eff40c35`, clean source.
  Frozen runtime: `.local/artifacts/runtimes/cosmos-a100-dd773f05a701-20261006`.
  Engine: `.local/artifacts/colab-a100/cosmos-reason2-2b/engine-p32-d256-b256-c512-kv8192-pool1536`.
  Vision engine: `.local/artifacts/colab-a100/cosmos-reason2-2b/vision-e4/visual`.
  vLLM 0.31.0 has its own isolated virtual environment.

The vLLM encoder-reuse claim follows installed `entrypoints/chat_utils.py` (image UUID parsing),
`multimodal/processing/inputs.py:get_mm_hashes` (UUID overrides), and
`v1/core/encoder_cache_manager.py:check_and_update_cache` (identifier-keyed reuse).

## Search and confirmation

The previous TRT batch search is retained because its precision and cache contract are unchanged;
TRT is remeasured with the new UUID traces and warmup protocol. vLLM is screened again under the
new precision and KV budget. The objective is geometric mean generated tokens/s across the four
workloads, not minimum latency or a per-workload independently chosen setting. A best tested
candidate is not proof of a global optimum or a statistically unique optimum.

| vLLM max sequences | Batched token budget | balanced | decode-heavy | mixed | vision-heavy | Geometric mean |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 4096 | 4923 | 9857 | 874 | 593 | 2239 |
| 512 | 16384 | 5373 | 10549 | 884 | 601 | 2343 |
| 512 | 8192 | 5170 | 10392 | 890 | 603 | 2317 |

S512/T16384 leads this single-run screen by 1.1% over S512/T8192. That small difference does not
establish a uniquely optimal setting. Three-pass confirmation uses S512/T16384. All table rates below
are median generated tokens/s; each system completed 9216 measured requests, zero failures, and all
requested output lengths. The vLLM first screen pass is also its first confirmation pass.

| Workload | TRT FP16 | vLLM FP16 | TRT/vLLM | TRT peak MiB | vLLM peak MiB |
|---|---:|---:|---:|---:|---:|
| balanced | 5272 | 5364 | 0.983x | 37146 | 31784 |
| decode-heavy | 10597 | 10544 | 1.005x | 36930 | 31784 |
| mixed | 962 | 885 | 1.087x | 39820 | 31784 |
| vision-heavy | 664 | 601 | 1.104x | 39820 | 31784 |
| Geometric mean | 2445 | 2342 | 1.044x | — | — |

Under this matched contract, text throughput is similar and TRT has a modest image-workload
advantage. The cache-enabled 2.23x vLLM result in note 380 belongs to a different contract and must not
be used as the primary result here. Matching KV does not match total VRAM usage: vLLM uses less
measured VRAM. TRT temporary encoder/DeepStack storage and engine workspaces remain distinct.

This throughput confirmation does not close the existing output-quality gate in note 380, and no
default serving promotion is performed. The new FP16 vLLM output hashes are retained, but no new
nine-prompt semantic/exact-match claim is made for this configuration.

## Reproduction and retained evidence

Artifact root: `.local/results/a100-cosmos-reason2-2b-fp16/capacity-resume/matched-fp16-kv21/`.
`manifest.json` records the contract; `summary.json` records individual rates, medians, p95 latency,
peak memory and completeness. Per-cell commands, input hashes, request CSVs and server logs are
retained under the screen and confirmation directories. Source models, engines and runtime stay
protected behind the existing Cosmos lineage; no artifact cleanup is performed.

```bash
source .local/env.sh
source .local/current/cosmos-reason2-2b/serving.env
R=.local/results/a100-cosmos-reason2-2b-fp16/capacity-resume/matched-fp16-kv21
.local/cache/tools/export-venv/bin/python benchmarks/phase_serving/run_serving_comparison.py \
  --config "$R/s512-t16384.json" --systems vllm \
  --workloads balanced decode-heavy mixed vision-heavy --reuse-server \
  --workload-warmup-requests 0 --repeats 1 --output-dir "$R/reproduce-vllm-pass1"
# Repeat with a fresh output directory/server for each vLLM pass.
.local/cache/tools/export-venv/bin/python benchmarks/phase_serving/run_serving_comparison.py \
  --config "$R/trt.json" --systems trt \
  --workloads balanced decode-heavy mixed vision-heavy --reuse-server \
  --workload-warmup-requests 0 --repeats 3 --output-dir "$R/reproduce-trt"
```

UUID traces are exact copies of `../optimal-inputs/{workload}.json` with an added `uuid` field on
each image content part: `matched-fp16-{workload}-{request_index}-{message_index}-{part_index}`.
Calibration uses `workload=calibration`, separate from every measured trace. All image URLs, text,
arrival times and generation lengths are preserved. The copied inputs and their hashes are retained.
