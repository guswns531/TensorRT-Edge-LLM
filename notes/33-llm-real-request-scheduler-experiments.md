# LLM Real-Request Scheduler Experiments

## Goal

This experiment removes the VLM encoder from the serving path and measures a
text-only online workload over independent prefill and decode TensorRT execution
contexts. Both contexts share one CUDA primary context, use separate nonblocking
streams, and share the stable indexed-linear KV cache.

The benchmark now accepts real chat JSON without a visual engine. Every request
is formatted with the model chat template, tokenized, admitted through the
production async API, sampled greedily, and decoded until its own output budget.
It records:

- request timestamps: scheduled arrival, submit, admission, first token, terminal;
- request metrics: prompt/output tokens, TTFT, TPOT, E2E, finish reason;
- scheduler dispatches: phase kind, P/D batch sizes, chunk tokens, queue wait,
  phase GPU time, makespan, and overlap ratio;
- CUDA-event kernel groups: prepare, engine, cache commit, and sample for prefill,
  plus prepare, engine, and sample for decode.

## Workload

The checked-in fixture contains 12 operational chat requests with 23–247 prompt
tokens and per-request output budgets from 8 to 32 tokens. Arrival offsets are
materialized from a seeded Poisson process. The main comparison uses 30 offered
requests/s, P2/D2 stable-slot partitioning, independent TensorRT contexts, and
three arrival seeds.

This is a real execution workload, but it is a small curated calibration trace,
not yet a public ShareGPT/Azure production trace. It is intended to expose
runtime behavior before adding a large trace adapter.

## Three-seed result

The table reports the median of the three run-level values.

| Policy | Achieved req/s | p95 TTFT ms | p95 TPOT ms | Mean P batch | Mean D batch | Median chunk |
|---|---:|---:|---:|---:|---:|---:|
| Whole prompt | 11.54 | 440.5 | 18.23 | 1.00 | 1.92 | 105 |
| Fixed 64 | 11.32 | 527.0 | 20.71 | 1.00 | 1.94 | 64 |
| Fixed 128 | 12.32 | 433.2 | 19.73 | 1.00 | 1.96 | 93.5 |
| Fixed 256 | 11.24 | 444.6 | 18.87 | 1.00 | 1.84 | 105 |
| Current adaptive, max 256 | 9.26 | 739.5 | 18.21 | 1.11 | 1.66 | 32 |

Fixed 128 is the best initial operating point for this trace: compared with
whole prompt, achieved throughput is about 6.8% higher and p95 TTFT about 1.7%
lower, although p95 TPOT is about 8.2% higher. The current adaptive policy is
not ready: it loses about 19.8% throughput and raises p95 TTFT about 67.9%
relative to whole prompt.

The fixed-128 policy was also run with a shared TensorRT execution context that
serializes prefill and decode. Across the same three seeds its medians were
10.96 requests/s, 510.58 ms p95 TTFT, and 22.51 ms p95 TPOT. Independent
contexts therefore improve throughput by about 12.4%, p95 TTFT by about 15.2%,
and p95 TPOT by about 12.3% on this workload.

## Kernel-group result

Across all three seeds, the median prefill engine time is 17.1–17.8 ms for every
policy, while prefill prepare, cache commit, and sampling are about 0.025–0.029,
0.003–0.004, and 0.021–0.023 ms. Decode engine time is about 6.38–6.44 ms, while
decode prepare and sample are each about 0.02 ms.

The important finding is that this workload is dominated by TensorRT engine
launch/service time, not host packing, embedding preparation, KV commit, or
sampling. The current adaptive policy interprets weak overlap as a reason to
halve the chunk repeatedly to 32 tokens. On this model, a 32-token prefill still
costs almost as much as a 100–128-token prefill. It therefore changes 36
whole-prompt prefill executions into 81 executions across the three runs without
obtaining enough extra prefill batching.

Prefill dynamic batching is also weak in this trace. Exact chunk-length
bucketing and Poisson arrivals leave the mean prefill batch near one. Decode
batching works: the mean decode batch is generally 1.8–2.0. Scheduler work
should therefore optimize prefill launch count and decode interference before
increasing the nominal prefill batch limit.

## Scheduler direction

The next adaptive version should replace the single prefill-ms-per-token EWMA
with a small online cost table keyed by initial/continuation state, batch bucket,
and chunk bucket.

Candidate chunk sizes should be discrete, initially whole remaining, 256, 128,
and 64. A candidate needs to pay an explicit launch-cost penalty, and 128 should
be the provisional lower bound on this RTX 3080/Gemma 4 engine until measurements
show that a smaller chunk materially protects a decode deadline.

The dispatch objective should combine:

1. predicted prefill completion and TTFT pressure;
2. predicted decode completion and TPOT pressure;
3. incremental launch cost from splitting the prompt;
4. measured decode inflation under overlap, not overlap ratio alone;
5. starvation aging and stable-slot admission pressure.

Overlap should be rejected when predicted decode inflation crosses its remaining
TPOT slack, even if both streams are runnable. A short configurable batch-hold
window can then be tested for uniform continuation chunks; it should never
consume the oldest request's remaining TTFT slack.

## Reproduction

Use scripts/gemma4_e2b_indexed/run_phase_real_request_suite.py with the indexed
engine and tests/test_cases/gemma4_llm_scheduler_real_requests.json. Each
scenario directory contains the materialized trace, request metrics, dispatch
metrics, raw kernel-group samples, and the runtime log.
