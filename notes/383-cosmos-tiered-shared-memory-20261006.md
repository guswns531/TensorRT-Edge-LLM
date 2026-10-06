<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cosmos shared-memory follow-up on A100

The user requested shared execution memory after note 382 showed a modest throughput advantage and
higher memory than vLLM. This campaign implements Qwen3-VL tiered visual profiles, repairs shared
workspace ownership retirement, and tests vision engine capacity against the actual serving limit.
The active serving lineage is not promoted automatically; the earlier semantic quality gate stays open.

## Contract and implementation

FP16 Cosmos-Reason2-2B, A100-SXM4-40GB, the existing P32/D256 LLM engine and physical KV 21 GiB.
Same UUID workloads, fixed output lengths, calibration, HTTP concurrency 512 and E8/P32/D256.
The encoded byte ceiling starts at 2.5 GiB and managed memory at 24 GiB. Any changed encoded
ceiling is an explicit ablation. The vLLM result from note 381 remains reusable while its model,
precision, KV, requests, output lengths and unique-image contract stay unchanged.

Artifacts: `.local/results/a100-cosmos-reason2-2b-fp16/capacity-resume/shared-memory-followup/`.
The manifest records compiled source parent plus dirty patch hash, frozen binary, engines, configs,
request traces and final publication commit. Initial single-pass screens are diagnostics; confirmed
comparisons use three passes. Accepted cells require zero failures and full requested token outputs.

Qwen3-VL now accepts the builder's optional smaller profile. Its runner sizes buffers across all
profiles and selects the workspace-backed profile for each prepared input. The existing tiered arena
keeps P and the small E profile in disjoint slices; the large E profile overlaps the P slice and
requires exclusive ownership. No overlapping engines may use the same bytes concurrently.

The first tiered calibration stalled with CPU 101%, GPU 0%, and 35468 MiB allocated. Inspection
found that direct CUDA-event payload handoff retired E without releasing its exclusive P/D dispatch
guards. Full E/P sharing uses another retirement path and had completed previously; tiered E/P
and E/D use direct handoff. The fix releases guards only after the encoder event reports completion,
including both immediate and later event retirement. Failed pre-fix logs are retained and excluded.

## Capacity units

The original vision builder uses 22528 **merged** tokens total and 2816 merged tokens per image.
Qwen's merge factor is four, so the engine supports 90112 input patches total and 11264 per image.
The serving scheduler's encoder input limit is 22528 **patches**. The compact engine therefore uses
5632 merged tokens total, keeping 2816 per image and the same 22528-patch serving limit.
No image resolution, messages or E scheduler token limit is reduced by this change. Engine capacity
outside the retained serving contract is smaller; this is not a claim about arbitrary workloads.

The first full-size tiered engine has a 2816-merged-token small profile (11264 patches) and a
22528-merged-token large profile. Its small/large context memory is 605558784/4296586240 bytes;
P requires 1275074048 bytes. Its shared arena is 4296586240 bytes. Compared with its own independent
workspaces, the direct arena saving is 1275074048 bytes (1.188 GiB). Total peak can differ because
payload lifetimes, active cohorts and allocation sizes also change.

The explicit single-storage recheck gives 5464/10687/444/304 tok/s, peak 35968 MiB. Previous runs
already had one retained batch by default, so this is a control repeat, not a new memory optimization.

## Validation and results

Visual-only FP16 export completed, followed by full/compact engine builds and real serving inference.
The existing visual-profile configuration tests pass3/3; workspace safety, incremental-action,
CUDA-event activity and memory/storage policy tests pass28/28. Accepted measured runs complete
47616 requests with zero errors and full requested outputs. Another24 sequential/concurrent shape
and image probes complete64 tokens each, including a large single image (2770 prompt tokens),
a pair of the same large images (5522 prompt tokens), and transitions back to small images.
Woman/dog, panda and ER-diagram cues remain present. These spot checks do not close the earlier
full-model semantic/reference gate. All GPU processes stop; final NVML usage is0MiB/0%.

| Confirmed setup | Balanced tok/s | Decode-heavy tok/s | Mixed tok/s | Vision tok/s | Geo tok/s | Peak MiB | Repeats |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prior matched TRT, original engine, independent, encoded2.5GiB | 5272 | 10597 | 962 | 664 | 2445 | 39820 | 3 |
| Prior matched vLLM FP16/KV21, unique images | 5364 | 10544 | 885 | 601 | 2342 | 31784 | 3 |
| Compact, independent, encoded512MiB | 5391 | 10570 | 923 | 615 | 2385 | 33856 | 3 |
| Compact, tiered E/P, encoded512MiB | 5238 | 10685 | 885 | 604 | 2338 | 33308 | 3 |
| Compact, full E/P shared, two storage batches, encoded512MiB | 5307 | 10592 | 873 | 604 | 2333 | 32628 | 3 |

Each current confirmed row uses the same frozen corrected binary, compact engine, requests,
calibration and limits except the explicitly named workspace/storage mode. Historical rows from
note381 use their separately recorded engine/runtime identities. Workload order is balanced,
decode-heavy, mixed, vision-heavy; a process is reused across three repeats and all workloads.
Rates are medians; geomean is over the four workload medians. Peak is the maximum sampled total
GPU memory over the cells, including calibration in the first cell. These are fixed-output serving
benchmarks, not arbitrary high-resolution production-memory worst cases or proof of a global optimum.

The full shared/two-batch option saves7192MiB (7.023GiB,18.1%) versus prior matched TRT, with
geomean4.56% lower. It is0.40% below frozen matched vLLM throughput and uses844MiB (0.824GiB)
more total GPU memory. Relative to compact independent at the same512MiB budget it saves1228MiB
(1.199GiB) with2.16% lower geomean. Relative to compact tiered it saves680MiB (0.664GiB), while
geomean differs only0.22%; do not assign statistical significance to that tiny throughput difference.
This provides a concrete memory option; it does not establish a throughput/memory advantage over vLLM.

## Separate capacity, workspace and storage effects

| Single-pass screen | Balanced | Decode-heavy | Mixed | Vision | Geo | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| Full-size two-profile engine, independent before ownership fix | 5281 | 10574 | 981 | 681 | 2471 | 39440 |
| Full-size two-profile engine, tiered after fix | 5379 | 10629 | 949 | 658 | 2444 | 38260 |
| Compact engine, independent, encoded2.5GiB | 5483 | 10349 | 967 | 671 | 2463 | 36068 |
| Compact engine, tiered, encoded2.5GiB | 5495 | 10747 | 929 | 646 | 2440 | 35524 |
| Compact engine, tiered, encoded512MiB | 5285 | 10623 | 875 | 604 | 2334 | 33304 |
| Compact engine, full E/P sharing, one storage batch, encoded512MiB | — | — | 455 | 306 | — | 32210 |

Rates are tok/s. The full-size independent diagnostic used the pre-fix binary and is contextual,
not a strict workspace-only ablation. The same corrected binary is used for the other completed
screens and confirmations. Their engine changes are explicit capacity/profile comparisons.

The compact large-profile workspace is1164978176 bytes (1.085GiB), versus4659871744 bytes
(4.340GiB) in the original single-profile engine. Its small workspace is605558784 bytes (0.564GiB).
The tiered arena is1880632832 bytes (1.751GiB), with direct savings559419392 bytes (0.521GiB)
against its own independent allocation. Full E/P sharing uses the P-sized1275074048-byte arena
(1.188GiB), saving the compact vision workspace1.085GiB against independent allocation.
Capacity adjustment is the largest workspace reduction; this is not weight/model compression.

A single retained storage batch additionally requires no downstream vision payload before preparing
another E batch. `visionPayloadBytes()` includes decode-owned M-RoPE data, so this can wait through
D completion even when E workspace ownership has ended. Two retained batches retain bounded
encoder-output slabs while permitting preparation behind downstream decode. Sharing still serializes
E/P engine workspace use; it does not remove safety fences. Its confirmed mixed/vision throughput
873/604 tok/s is approximately1.92x/1.97x the single-batch screen455/306. The extra storage policy
is an explicit parameter change, not evidence that one shared workspace alone causes a55% penalty.

The compact E/D regression completes512 mixed requests at974 tok/s, peak35940MiB, after the same
1119-request calibration. The old E/D stall is repaired in the benchmark path. Direct E/D arena
saving is only19928576 bytes (19.005MiB); this is not selected for memory. The library runtime's
separate shared-E/D D32 capacity policy is not benchmarked by this IPC smoke path.

## Reproduction and disposition

`build_cosmos_tiered_vision_engine.sh` is checked in. Source `.local/env.sh`, set new ONNX/engine/result
paths, and call it with `MAX_IMAGE_TOKENS=5632`, `SMALL_PROFILE_MAX_IMAGE_TOKENS=2816`, and
`MAX_IMAGE_TOKENS_PER_IMAGE=2816`. The retained new export lives under
`.local/artifacts/colab-a100/cosmos-reason2-2b/onnx-fp16-tiered-validation/`; the compact engine is
`.local/artifacts/colab-a100/cosmos-reason2-2b/vision-tiered-compact5632-small2816/visual/`.
The script refuses to overwrite an engine and can reuse the same visual export for a second build.

For confirmed comparisons run `run_serving_comparison.py --config CONFIG --systems trt`
with `--workloads balanced decode-heavy mixed vision-heavy --reuse-server`
`--workload-warmup-requests 0 --repeats 3 --output-dir NEW_OUTPUT`.
Configs are `confirm-compact-independent-encoded512.json`,
`confirm-compact-tiered-encoded512.json`, and `confirm-compact-shared-two-encoded512.json`
under the campaign root. Exact server environments/client commands and trace hashes are retained.
Initial TRT-only screening configs inherit unused vLLM fields; they are not the native FP16 comparison
contract. Confirmed configs contain the matched FP16/KV21 native arguments, while the comparison uses
the unchanged frozen result from note381. No native rerun is needed for implementation-only TRT changes.

Selected memory option:
`TRT_EDGELLM_PHASE_WORKSPACE_MODE=shared_ep`,
`TRT_EDGELLM_SHARED_EP_SINGLE_STORAGE=0`,
`TRT_EDGELLM_MAX_ENCODED_VISION_BYTES=536870912`.
Use the compact visual engine and the frozen corrected binary recorded in the config. The `0` storage
flag means a maximum of two retained batches, not unbounded storage or disabled ownership safety.
Keep the earlier active lineage/default pointer unchanged because targeted regression checks do not
waive the outstanding semantic gate. All source changes, script and notes are committed/published;
models, engines and result data stay in the local artifact store.
