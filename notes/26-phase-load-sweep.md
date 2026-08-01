# Phase continuous-load sweep 결과

## 목적

단일 burst 결과만으로 scheduler 동작을 판단하지 않고 다음 축을 독립적으로 바꿨다.

- offered load: 10, 20, 25, 30, 35, 40, 1000 req/s
- prompt: 고정 128, 고정 512, 128~512 혼합
- output: 8 고정, 4~16 혼합, 24~32 long decode
- prefill: whole 512와 chunk 128
- batch limit: 2/2, prefill 1/decode 3, prefill 3/decode 1
- overlap threshold: total prefill tokens 128과 256

모든 workload는 RTX 3080, Gemma 4 E2B INT4 indexed engine, 같은 CUDA primary context, independent TensorRT
prefill/decode context, physical KV slot 4개로 실행했다. 각 process에서 phase별 1회 warmup 후 continuous load를
측정했다.

## 재현 도구

`scripts/gemma4_e2b_indexed/run_phase_load_suite.py`가 workload 실행, request/dispatch CSV 보존과 공통 summary
생성을 담당한다.

```bash
python3 scripts/gemma4_e2b_indexed/run_phase_load_suite.py \
  --bench build/examples/llm/llm_phase_bench \
  --engine-dir /workspace/artifacts/gemma4-e2b/engine-indexed \
  --output-dir /workspace/artifacts/gemma4-e2b/perf/phase/load-suite-v1 \
  --warmup 1 --iterations 1
```

`--scenario`를 반복해서 일부 workload만 선택할 수 있고, `--skip-existing`은 완료된 CSV를 재사용한다.

## Steady-load saturation curve

| Offered load | Achieved | Pending 진입 | TTFT p95 | E2E p95 | Overlap dispatch |
|---:|---:|---:|---:|---:|---:|
| 10 req/s | 10.21 req/s | 0 | 18.95 ms | 67.81 ms | 0 |
| 20 req/s | 19.45 req/s | 0 | 27.27 ms | 90.99 ms | 15 |
| 25 req/s | 24.21 req/s | 0 | 26.91 ms | 104.44 ms | 23 |
| 30 req/s | 25.63 req/s | 17 | 102.61 ms | 216.72 ms | 23 |
| 35 req/s | 25.64 req/s | 20 | 208.92 ms | 314.07 ms | 23 |
| 40 req/s | 25.79 req/s | 20 | 284.32 ms | 384.14 ms | 23 |
| 1000 req/s burst | 27.79 req/s | 20 | 771.96 ms | 832.27 ms | 1 |

이 configuration의 안정 영역은 약 25 req/s까지다. 30 req/s부터 achieved throughput은 25~26 req/s에 머무는데
queue latency만 빠르게 증가한다. 25/30 req/s 반복 결과도 각각 24.13/25.41 req/s와 TTFT p95
27.50/105.04 ms로 같은 knee를 재현했다.

1000 req/s burst는 batching으로 drain throughput이 27.79 req/s까지 올라가지만 TTFT가 크게 나빠지고 overlap은
1회로 줄었다. 따라서 burst throughput만 보고 scheduler가 좋아졌다고 판단하면 안 된다.

## Workload 종류별 결과

| Workload | Achieved | Generated token/s | TTFT p95 | E2E p95 | Overlap |
|---|---:|---:|---:|---:|---:|
| Long output 24~32 | 9.46 req/s | 250.74 | 1239.77 ms | 1540.72 ms | 4 |
| Whole prefill 512 | 13.21 req/s | 105.72 | 699.60 ms | 753.57 ms | 0 |
| Chunked 128, threshold 128 | 11.65 req/s | 93.23 | 871.04 ms | 924.26 ms | 0 |
| Chunked 128, threshold 256 | 12.63 req/s | 101.02 | 768.76 ms | 844.76 ms | 16 |
| Mixed prompt/output | 13.51 req/s | 125.81 | 885.26 ms | 1051.03 ms | 27 |
| Prefill BS1 / Decode BS3 | 10.04 req/s | 80.35 | 1022.66 ms | 1085.83 ms | 33 |
| Prefill BS3 / Decode BS1 | 5.29 req/s | 138.81 | 1562.85 ms | 2151.54 ms | 3 |

long-output workload는 request/s는 낮지만 decode batching으로 generated token/s는 가장 높다. request throughput과
token throughput을 반드시 함께 봐야 한다.

decode BS1은 가장 큰 병목이었다. 반대로 prefill BS1은 overlap 횟수는 33회로 많아졌지만 prefill batch 효율을
잃어 전체 throughput이 감소했다. overlap 횟수 자체도 최적화 목표가 될 수 없다.

## Chunked prefill 해석

기본 scheduler의 `maxOverlapPrefillTokens=128`은 요청당 token이 아니라 선택 batch의 총 token 수를 비교한다.
따라서 chunk 128 × BS2 = 256은 overlap 대상이 아니어서 실제 overlap이 0이었다.

benchmark에 `--maxOverlapPrefillTokens`를 노출하고 threshold를 256으로 바꾸자:

- overlap: 0 → 16
- throughput: 11.65 → 12.63 req/s, 약 8.4% 개선
- TTFT p95: 871.04 → 768.76 ms, 약 11.7% 개선
- E2E p95: 924.26 → 844.76 ms, 약 8.6% 개선

반복 측정도 12.66 req/s, TTFT p95 766.68 ms, overlap 16으로 재현됐다. 그래도 whole prefill의 13.19~13.21
req/s보다 낮다. 현재 workload에서는 chunk dispatch 횟수 증가 비용을 overlap 이득이 완전히 상쇄하지 못했다.

## 결론과 다음 측정

- 128-token/8-output/BS2+2 기준 sustainable load는 약 25 req/s다.
- 포화 이후에는 throughput보다 TTFT와 pending queue가 먼저 악화된다.
- chunk size와 overlap threshold는 batch size를 포함한 total token 관점에서 함께 설정해야 한다.
- 다음 scheduler는 workload별 고정 threshold 대신 measured GPU cost, queue deadline과 TTFT/TPOT SLO를 사용해야 한다.
- 다음 load suite에는 fixed interval 외 Poisson arrival, periodic burst, cancellation과 priority/tenant 혼합을 추가한다.

원본 결과:

- `/tmp/gemma4-e2b/perf/phase/load-suite-v1/summary.csv`
- `/tmp/gemma4-e2b/perf/phase/load-suite-repeat2/summary.csv`
- `/tmp/gemma4-e2b/perf/phase/load-suite-repeat2-tuned/summary.csv`
