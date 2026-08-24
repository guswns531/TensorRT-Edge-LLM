# Thor Qwen3.8 quality gate and optimized vLLM MTP c64

## Result

The Phase hybrid remains faster than stock vLLM vanilla on every core workload, but it does not yet beat the best
measured vLLM MTP-1 configuration on all workloads. MTP-1 wins short burst by 5.32% and wave burst by 0.56%; Phase wins
decode-heavy by 0.25%. The wave and decode-heavy differences are small enough that they should be treated as ties until
an interleaved A/B campaign reproduces them.

All HTTP numbers below are medians of three warmed streaming runs. Every run completed 128/128 requests under the locked
Thor clock contract: GPU GPC 1.575 GHz and EMC 4.266 GHz.

| workload | Phase hybrid | vLLM vanilla | vLLM MTP-1 | Phase vs MTP-1 |
| --- | ---: | ---: | ---: | ---: |
| short burst c64 | 286.78 tok/s | 235.27 tok/s | 302.88 tok/s | -5.32% |
| wave burst c64 | 275.04 tok/s | 256.56 tok/s | 276.60 tok/s | -0.56% |
| decode heavy c64 | 341.33 tok/s | 333.19 tok/s | 340.48 tok/s | +0.25% |

Machine-readable summaries:

- Phase: `data/qwen38/results/next/phase-hybrid-int4-fp8kv-graphs-input256-final/summary.json`
- vLLM vanilla: `data/qwen38/results/next/vllm-vanilla-c64/summary.json`
- vLLM MTP-1: `data/qwen38/results/next/vllm-mtp1-c64-16g-final/summary.json`

## Latency

| workload | Phase TTFT / TPOT / E2E | MTP-1 TTFT / TPOT / E2E |
| --- | --- | --- |
| short | 1,707 / 199 / 14,250 ms | 1,655 / 217 / 9,761 ms |
| wave | 1,839 / 203 / 22,279 ms | 2,898 / 236 / 16,094 ms |
| decode heavy | 1,663 / 181 / 47,949 ms | 2,104 / 210 / 33,765 ms |

Phase keeps lower TPOT in all three cases and lower TTFT in wave and decode-heavy. MTP-1's higher aggregate service rate
reduces median E2E because the second half of the 128-request campaign clears sooner. MTP also has a pronounced admission
tail: during saturated intervals it runs about 40-44 requests while 20 remain queued, producing p95 TTFT in the 10-33 s
range. Median-only throughput must not be interpreted as a universal latency win.

## MTP server contract

The successful MTP server used the official `nvcr.io/nvidia/vllm:26.06-py3` image and the same local ModelOpt NVFP4
checkpoint and FP8 KV scale 1.0 as the vanilla comparison. Prefix caching was disabled. The MTP-specific throughput setup
was text-only mode, async scheduling, a 16,384-token scheduler budget, performance mode, native one-token MTP, and a
manually sized 16 GiB KV cache:

```text
docker run --rm --runtime=nvidia --network host --ipc host \
  -v data/qwen38/nvfp4:/model:ro \
  -v data/qwen38/vllm-cache:/root/.cache/vllm \
  --entrypoint vllm nvcr.io/nvidia/vllm:26.06-py3 \
  serve /model --host 127.0.0.1 --port 8009 --served-model-name qwen38 \
  --language-model-only --max-model-len 4096 --max-num-seqs 64 \
  --max-num-batched-tokens 16384 --kv-cache-memory-bytes 16G \
  --gpu-memory-utilization 0.40 --kv-cache-dtype fp8 \
  --gdn-prefill-backend cutedsl --performance-mode throughput \
  --no-enable-prefix-caching --async-scheduling \
  --reasoning-parser qwen3 --generation-config vllm \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

The Thor image rejected the requested CuteDSL GDN prefill kernel and explicitly fell back to Triton/FLA. The option is
kept in the command to document the attempted best setting, but it provided no hidden acceleration.

The first one-variable-only MTP launch retained vanilla's automatic `--gpu-memory-utilization=0.40`. It failed in vLLM's
Jetson UMA profiler because free memory increased from 61.14 to 66.12 GiB during initialization. Manual KV sizing bypasses
that unstable inference. A 4 GiB screen was also rejected: it produced only 35,043 cache tokens and limited the 1,584-token
hybrid pages to about 22 concurrent requests. The final 16 GiB cache provides 141,539 tokens, enough for 89 minimum pages
and therefore the complete c64 workload.

One complete three-workload warmup was excluded from the measured repetitions. Cumulative warmup plus measurement counters
reported 88,806 accepted tokens from 123,319 drafts, a 72.01% acceptance rate. MTP-1 is used because the checkpoint has one
native MTP layer; the earlier MTP-3 screen repeated it and reached only 46.9% aggregate acceptance.

The MTP campaign averaged 39.4 W GPU power and 94.3 W board input, versus 46.3 W and 100.3 W for Phase. These are whole
campaign averages rather than per-workload energy measurements.

## Projection quality gate

`experiments/qwen38_thor/quality_prompts.json` expands the old one-prompt smoke into eight deterministic prompts covering
exact text, arithmetic, factual recall, Korean-to-English translation, Python generation, a word problem, summarization,
and logic. Both engines use the same uncalibrated FP8-KV checkpoint; the only intended difference is the GDN input
projection:

- reference: FP16 projection, `phase-b64-p8-d64-fp8kv-input256`
- candidate: group-128 weight-only INT4-RTN projection, `phase-b64-p8-d64-fp8kv-gdn-int4`

The outputs were exactly identical on 5/8 prompts. All eight answers were semantically task-correct for both engines. With
strict output-format grading, the FP16 reference passed 6/8 and INT4 passed 8/8: the FP16 reference added a Markdown fence
to the code answer and emitted placeholder text plus a duplicated answer on the train problem. The three INT4 divergences
were a different correct `clamp` implementation, removal of the FP16 placeholder artifact, and an equivalent logical
explanation. No visible regression was found in this gate.

Artifacts are `data/qwen38/results/quality-fp16-proj-fp8kv.json` and
`data/qwen38/results/quality-int4-rtn-proj-fp8kv.json`. This is a deterministic smoke/regression gate, not an accuracy
benchmark. A calibrated FP8-KV checkpoint and a real downstream suite remain mandatory before making a general quality
claim.

## Next performance target

The optimized competitor changes the target from vanilla decode kernels to speculative service rate. Closing the measured
gap requires either native Phase MTP-1 support or at least a 5.6% short-burst service-rate improvement without losing the
current TPOT and TTFT advantages. The next experiment should therefore implement a one-layer Phase draft/verify path and
gate it on acceptance-adjusted tokens per verify step. Scheduler-only work is unlikely to close the short-burst gap; wave
and decode-heavy are already within 0.6%.
