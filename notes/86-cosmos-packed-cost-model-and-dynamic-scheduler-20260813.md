# Cosmos packed-prefill cost model과 dynamic scheduler 연결

## 결론

Cosmos-Reason2-2B FP16 packed-prefill P8/D32 engine을 independent TensorRT contexts에서
288-request balanced real-arrival trace로 측정했다. P=1/2/4/8과 D=1/2/4/8/16/32의 24개 조합이 모두
성공했다. 원시 CUDA-event dispatch로 packed 전용 schema v5 cost model을 생성했고, runtime과 matrix runner가
engine layout과 cost-model layout이 다르면 실행 전에 거부하도록 연결했다.

packed cost model은 prefill 81점, decode 11점, direct overlap 77점을 포함한다. 새 P8/D32 실행은
2,928.3 generated token/s로 이전 3-run median 2,934.5 token/s와 0.21% 차이여서 측정 재현성도 확인했다.

## Layout 계약

cost-model root에 `prefill_layout: dense|packed`를 기록한다. 새 detailed model은 schema 5, 단순 phase model은
schema 2다. 필드가 없는 기존 model은 호환성을 위해 dense로 해석한다. packed engine과 dense cost model 또는
dense engine과 packed cost model 조합은 Python harness와 C++ benchmark 양쪽에서 거부한다.

이는 packed prefill이 dense padded prefill보다 같은 logical batch의 token carrier와 kernel cost가 다르기 때문이다.
잘못된 표를 허용하면 scheduler가 prefill 비용과 decode slowdown을 체계적으로 과대 또는 과소평가한다.

## 24-case real-request sweep

- engine: packed P8/D32, maxBatchSize 32
- cache: indexed-paged FP16 KV, 128 tokens/page, 32 stable slots, 256 page bundles
- contexts: independent prefill/decode TensorRT execution contexts
- prefill: fixed chunk 128, token budget 1,024, ragged batching
- workload: 288 requests, output token 합 24,960, source arrival offset 보존, EOS 무시
- CUDA graph: off

24/24 case의 return code는 0이었다. 처리량 최고점은 P8/D32의 2,928.3 token/s였고 TTFT p95
6,964.6ms, TPOT p95 10.716ms, E2E p95 7,791.9ms였다. cost model에는 실제 관측된 ragged final cohort 때문에
prefill batch 3도 포함됐다. direct overlap coverage는 모든 nominal 조합의 완전한 Cartesian product가 아니라
실제로 동시에 발생한 shape만 기록한다.

## Dynamic scheduler A/B

같은 trace와 engine에서 packed cost model을 `throughput-balanced` profile에 연결해 3회 실행하고, 이전 fixed
P8/D32 3회와 median을 비교했다.

| metric | fixed P8/D32 | packed dynamic | 변화 |
| --- | ---: | ---: | ---: |
| generated token/s | 2,934.48 | 2,937.02 | +0.087% |
| TTFT p95 | 6,948.94ms | 6,943.63ms | -0.076% |
| TPOT p95 | 10.702ms | 10.688ms | -0.129% |
| E2E p95 | 7,774.91ms | 7,768.17ms | -0.087% |

세 dynamic run 모두 실제 최대 P8/D32를 형성했고 cost coverage miss와 latency-safe fallback은 0이었다. 따라서
연결은 정상이나 개선폭은 노이즈 수준이다. 이 overload workload에서는 fixed P8/D32가 이미 throughput 최적점이어서
dynamic policy가 작은 batch를 선택할 이유가 없기 때문이다. dynamic batching의 가치는 다음 단계에서 arrival-rate가
시간에 따라 변하는 mixed-load trace와 TTFT/TPOT SLO 전환으로 검증해야 한다.

원시 artifact는 `.local/cosmos-reason2-2b/packed-direct-cost-v5-20260813/`에 있다.

## 다음 단계

1. low/medium/high arrival-rate 구간을 한 trace에 섞어 fixed P8/D32와 dynamic policy를 비교한다.
2. direct overlap coverage가 없는 P4/D2, P8/D2/4/16/32 shape를 late-prefill probe로 보충한다.
3. packed engine의 D64/80-slot build가 10GB 메모리에서 graph headroom을 유지하는지 확인한 뒤 현재 최고 dense
   P8/D64 경로와 비교한다.
4. workload별 SLO를 만족하면서 처리량을 최대화하도록 cost table 기반 batch candidate score를 보정한다.
