# vLLM 0.28 model compatibility and full-12 comparison

Date: 2026-09-11

## Outcome

The comparison was updated along both axes that matter: a newer vLLM release was tested with the new Gemma 4
checkpoint, and a fresh 12-workload performance campaign was run with the retained model that both runtimes can
execute under one contract.

vLLM 0.28.0 fixes the Gemma 4 architecture failure seen in vLLM 0.27.1. It resolves the heterogeneous d256/d512
attention layers and selects Triton attention on the RTX 3080. The retained `Chunity/gemma-4-E2B-it-AWQ-4bit`
checkpoint still cannot be used for a fair performance comparison: its AutoRound mixed-precision projection layout
is incompatible with vLLM's fused QKV loader. This is a different, later failure than the old global-`head_dim`
failure.

The fair performance control therefore remains `nvidia/Cosmos-Reason2-2B` in FP16. A fresh vLLM 0.28.0 server ran
all 12 retained request traces three times each with the same model, FP16 KV cache, 2,048-token context, 3.5 GiB KV
budget, 80 sequence capacity, 8,192-token scheduler budget, chunked prefill, disabled prefix cache, and identical
image preprocessing. Current V3 beats vLLM 0.28 on token throughput and mean E2E latency in all 12 workloads. It
beats vLLM on E2E p95 in 11 of 12, while four TTFT-sensitive VLM traces still expose a first-token trade-off.

## vLLM version and model compatibility

| Runtime/checkpoint | Result | Failure boundary |
|---|---|---|
| vLLM 0.27.1 + Chunity Gemma AWQ | Fail | Architecture conversion accesses ambiguous global `head_dim` |
| vLLM 0.28.0 + Chunity Gemma AWQ | Fail | Architecture succeeds; fused QKV requires every shard to use one precision |
| TensorRT-Edge-LLM v0.10.1 current + Chunity Gemma AWQ | Pass | Separate projection loading accepts the checkpoint; export → build → inference passes |
| vLLM 0.28.0 + Cosmos FP16 | Pass | Full 12-workload HTTP campaign completes |

The vLLM 0.28 model loader reports both Gemma attention dimensions and explicitly forces `TRITON_ATTN` because
SM86 does not provide the heterogeneous-head FlashAttention-4 path. During weight construction it then rejects
layer 15 `qkv_proj`: only some fused shards are quantized. The Chunity checkpoint skips quantization selectively
inside attention projections, whereas vLLM's fused QKV representation requires a common precision. Changing vLLM
alone is therefore insufficient for this particular checkpoint.

A future same-Gemma performance comparison needs one common checkpoint whose Q, K, and V projections use a uniform
quantization contract, followed by the complete TensorRT-Edge-LLM export → build → inference validation. Comparing
different Gemma quantizers would otherwise mix runtime performance with weight format and calibration differences.

## Fair Cosmos contract

| Item | vLLM 0.28 | Current V3 |
|---|---|---|
| Model | `nvidia/Cosmos-Reason2-2B` | Same |
| Weights / KV | FP16 / FP16 | Same |
| Maximum context | 2,048 tokens | 2,048 tokens |
| KV capacity | 3.5 GiB, 32,768 tokens | retained equal-capacity contract |
| Active sequence limit | 80 | 80 |
| Prefill scheduling | chunked, 8,192-token budget | packed 128-token chunks, P8 |
| Decode limit | scheduler up to 80 rows | D64 |
| Prefix cache | disabled | disabled |
| Images | at most 2, fixed 516,096 pixels | same request inputs |
| Repeats | 3 per workload after 64-request warmup | retained 3-repeat campaign |

Both comparisons use the same HTTP request traces and ignore EOS so generated work is deterministic. They do not
claim identical kernels or batching mechanisms; those differences are the systems being compared.

## Current V3 versus vLLM 0.28

Positive deltas mean Current V3 is better. For latency, positive means lower latency.

| Workload | vLLM 0.28 tok/s | V3 tok/s | Δ tok/s | Δ TTFT mean | Δ TTFT p95 | Δ TPOT mean | Δ TPOT p95 | Δ E2E mean | Δ E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 2,007.8 | 2,366.4 | +17.9% | +46.1% | +26.9% | +5.6% | +21.6% | +21.1% | +17.0% |
| balanced | 4,328.6 | 4,405.3 | +1.8% | +42.4% | +46.1% | -2.7% | -3.9% | +2.1% | +1.0% |
| decode-heavy | 4,954.1 | 5,154.3 | +4.0% | +47.3% | +51.7% | +1.7% | +1.6% | +3.8% | +3.4% |
| long-prefill | 1,123.6 | 1,344.8 | +19.7% | -0.9% | +15.2% | +31.5% | +28.7% | +17.8% | +20.3% |
| bimodal | 1,842.6 | 1,984.7 | +7.7% | -19.1% | -47.2% | +26.2% | +25.2% | +11.6% | +2.8% |
| text-heavy | 1,648.7 | 1,813.5 | +10.0% | +17.7% | +16.9% | +7.8% | +9.7% | +11.3% | +8.9% |
| mixed | 920.0 | 963.5 | +4.7% | -1.3% | -0.4% | +43.2% | +56.5% | +27.7% | +6.0% |
| vision-heavy | 578.8 | 630.8 | +9.0% | +12.1% | +3.6% | +52.4% | +68.0% | +35.5% | +9.3% |
| poisson | 1,796.5 | 1,902.1 | +5.9% | +38.0% | +12.1% | +6.8% | +11.2% | +8.8% | +7.9% |
| wave/drain | 95.8 | 97.5 | +1.8% | -21.3% | -26.7% | +31.3% | +31.2% | +10.4% | -14.2% |
| multi-image | 245.8 | 260.0 | +5.8% | -22.8% | -0.7% | +18.7% | +28.4% | +8.1% | +5.3% |
| late-vision | 2,363.3 | 2,452.1 | +3.8% | +16.1% | +25.8% | +2.9% | +2.6% | +4.4% | +3.7% |

Summary:

- Token throughput: Current wins 12/12; median improvement is 5.83%.
- TTFT mean/p95: Current wins 7/12 and 8/12; median improvements are 14.09% and 13.64%.
- TPOT mean/p95: Current wins 11/12 for both; median improvements are 13.25% and 23.40%.
- E2E mean/p95: Current wins 12/12 and 11/12; median improvements are 10.85% and 5.63%.

The remaining trade-off is not hidden by throughput. Bimodal, mixed, wave/drain, and multi-image retain worse mean
TTFT, and wave/drain also has worse E2E p95. Current prioritizes downstream decode continuity and completion enough
to win TPOT and mean E2E, but can defer the vision first-token path. This is the next profile-free policy target.

## vLLM 0.27.1 versus 0.28.0

Most workload throughput changes are within about 2%, except text-heavy (+27.6%) and late-vision (+9.2%) for 0.28.
These two larger changes must not be presented as a pure vLLM version effect: the retained 0.27.1 campaign used the
default PyTorch allocator, while the stable 0.28 campaign uses expandable segments. The old image was removed to
make disk space for 0.28 and was not re-run under the new allocator contract.

The architecture conclusion is clean even though the performance attribution is not: 0.28 fixes the heterogeneous
Gemma model definition, while the selected third-party AWQ weight layout remains unsupported.

## Memory stability finding

With a fixed 3.5 GiB KV cache and the default allocator, the first long-lived vLLM 0.28 server completed the early
workloads and then failed in the Qwen3-VL vision MLP while trying to allocate a contiguous 228 MiB activation. Only
111 MiB was free; 546 MiB was reserved but unallocated. This is direct evidence of both high real pressure and
allocator fragmentation.

Restarting with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` preserved every serving parameter and completed
all 36 measured runs in one long-lived server. After the campaign, vLLM occupied 9,576 MiB and the GPU had 290 MiB
free. The canonical Cosmos V3 runtime allocated 9,399 MiB after text-engine initialization, and its retained mixed-VLM
campaign had a 9,629 MiB median peak, leaving about 611 MiB against the nominal 10,240 MiB device capacity. These
values come from different memory instrumentation and should be treated as an approximate residency comparison rather
than an exact allocator A/B.

This result does not justify reducing vLLM's KV pool to make it pass. Instead, the equal-capacity result is retained
with the allocator setting disclosed. It also reinforces the current architecture's benefit: stable ownership and
preallocated phase-local workspaces avoid relying on a general-purpose caching allocator for the same activation
lifetime pattern.

## Artifacts

| Artifact | Path |
|---|---|
| Gemma compatibility manifest | `.local/results/gemma4-e2b-awq-vllm028-20260911/manifest.json` |
| vLLM 0.28 full-12 manifest | `.local/results/v0101-forward-port/vllm-028-cosmos-fresh-20260911/manifest.json` |
| Compact V3 comparison | `.local/results/v0101-forward-port/vllm-028-cosmos-fresh-20260911/vllm-028-v3-summary.csv` |
| Per-workload vLLM aggregates | `.local/results/v0101-forward-port/vllm-028-cosmos-fresh-20260911/*-expandable/aggregate.json` |
| Frozen vLLM 0.27.1 comparison | `.local/results/v0101-forward-port/v3-service-scale-20260910/vllm-fresh-equal-summary.csv` |

## Next steps

1. Keep Cosmos as the cross-runtime performance control until a uniform-QKV Gemma checkpoint passes both complete
   pipelines. Do not compare different Gemma quantizers as a runtime result.
2. Select one uniform W4A16 Gemma candidate, audit every fused QKV group before downloading/building, and retain
   only one checkpoint because current disk headroom cannot safely hold two complete Gemma engine lineages.
3. Address Current's bimodal and vision-wave TTFT regressions using observable ready-state and service-scale signals,
   without workload names or fixed SLO constants.
4. Add the vLLM allocator contract to future VLM manifests. A default-allocator OOM is not comparable with a
   preallocated TensorRT runtime unless both usable KV capacity and allocator policy are disclosed.
5. Repeat selected short, balanced, text-heavy, vision-heavy, and wave/drain points five times before promoting the
   numbers to citable status.
