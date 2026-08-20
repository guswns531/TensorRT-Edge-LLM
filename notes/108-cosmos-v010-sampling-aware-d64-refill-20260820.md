# Cosmos v0.10 sampling-aware D64 refill

## 결론

비대칭 P8/D64 engine에서 반복되던 `D64 -> D16` decode round의 원인은 admission이나 TensorRT profile이 아니라
비동기 sampling completion 순서였다.

```text
D64 TensorRT 완료
  -> D64 sampling ticket은 아직 CUDA event 대기
  -> scheduler에는 기존 tail D16만 존재
  -> D16을 즉시 dispatch
  -> 다음 poll에서 D64 token이 decode queue로 복귀
  -> D64/D16 반복
```

새 opt-in `decodeRefillBatchSize=64`는 prefill queue가 비어 있고, 현재 decode tail과 완료 대기 중인 decode sampling
row를 합치면 D64가 될 때만 tail dispatch를 보류한다. sampling event가 처리되면 D64를 즉시 재구성한다. GPU 작업을
인위적으로 sleep시키는 정책이 아니며 이미 완료 중인 비동기 sampling dependency만 기다린다.

Balanced fixed-output 3회 중앙값은 3,889.9 token/s다. 이전 P8/D64 최종값 3,693.3보다 5.3% 높고, 최초 B32
P8/D32 1,921.4보다 102.5% 높다.

## Balanced 3회 결과

조건은 288 requests, prompt 25,872 tokens, output 24,960 tokens, P8/D64, fixed chunk128, stable slots80,
in-flight80, graph-off, refill target64다.

| metric | latency profile (in-flight64) | throughput profile (in-flight80/refill64) | 변화 |
| --- | ---: | ---: | ---: |
| generated token/s | 3,693.3 | **3,889.9** | **+5.3%** |
| TTFT median | 2,336.5 ms | **2,035.8 ms** | **-12.9%** |
| TTFT p95 | 5,066.5 ms | **4,808.4 ms** | **-5.1%** |
| TPOT median | **15.29 ms** | 18.23 ms | +19.2% |
| TPOT p95 | **16.64 ms** | 20.04 ms | +20.4% |
| E2E median | **3,617.3 ms** | 3,729.2 ms | +3.1% |
| E2E p95 | 6,114.1 ms | **5,868.7 ms** | **-4.0%** |
| peak memory | 8,087 MiB | 8,087 MiB | 동일 |

세 run 처리량은 3,901.0 / 3,876.4 / 3,889.9 token/s다. output은 세 번 모두 24,960/24,960이며 peak
memory도 동일하다. refill은 throughput과 TTFT/p95 E2E를 개선하지만 active request가 64에서 80으로 늘어 median
TPOT과 median E2E를 희생한다.

따라서 하나의 무조건적인 기본값이 아니라 두 운영 profile이 적절하다.

```text
latency profile:    in-flight64, refill off  -> 3,693 tok/s, TPOT 15.29ms
throughput profile: in-flight80, refill64    -> 3,890 tok/s, TPOT 18.23ms
```

## Dispatch 변화

Balanced 첫 run에서 refill 적용 후:

- dispatch 558회
- D64 348회
- D16 0회
- overlap 1.6%
- GPU makespan 합 약 5.38초

기존 in-flight80은 D64와 D16이 매 token round 교대로 실행됐다. refill은 D16을 단순히 drop하지 않는다. D64
sampling ticket이 정말 64-row batch를 복구할 수 있을 때만 기다리며, output 종료로 복구할 row가 부족하면 작은
tail batch를 정상 실행한다.

## 다른 workload

| workload | 이전 v0.10 | refill P8/D64 | 변화 | vLLM fixed-output | vLLM 격차 |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 859.7 | 1,698.5 | +97.6% | 1,998.7 | -15.0% |
| balanced | 1,921.4 | 3,889.9 | +102.5% | 4,234.4 | -8.1% |
| decode-heavy | 2,554.5 | 4,476.6 | +75.2% | 4,797.9 | -6.7% |

Decode-heavy TPOT median/p95는 14.25/14.83ms로 vLLM 14.86/15.40ms보다 낮다. short는 전체 요청이 48개라
refill64가 dispatch를 보류하지 않으며 작은 workload 회귀가 없다.

## 구현 안전 경계

- 기본값 0: 기존 동작 유지
- refill target은 server max in-flight 이하만 허용
- prefill queue가 있으면 refill 대기로 prefill을 막지 않음
- decode queue가 비거나 이미 target 이상이면 대기하지 않음
- queued decode + pending decode sampling rows가 target 미만이면 tail을 즉시 실행
- pending sampling event가 output 종료를 반환하면 다음 poll에서 조건이 풀려 tail 실행
- decision helper는 GPU가 필요 없는 단위 테스트로 경계 검증
- 실제 trace에서 D16 제거와 output count를 통합 검증

## 새 engine cost table

P8/D64 engine SHA-256 `b9f87df7...09d4c5cf`에 대해 balanced/decode-heavy/forced-D64 decode-only
CUDA-event를 합쳤다. D64는 2,351 samples, median 8.641ms, p95 9.764ms다. 전체 관측 table은
`notes/results/cosmos-v010-p8d64-observed-cost-20260820.json`에 저장했다.

## 다음 단계

1. throughput/latency profile을 request SLO 또는 queue pressure로 online 전환한다.
2. refill 대기 횟수와 누적 wait time을 HTTP metrics에 노출한다.
3. D64 active cohort 옆 16 staging slot의 prefill-ready 개수를 snapshot에 추가한다.
4. graph shape를 측정 전에 priming한 뒤 refill on/off를 다시 비교한다.
5. long-prefill/bimodal에서 page pressure가 target64를 만들 수 없을 때 D32/D48 refill target을 동적으로 선택한다.

## Artifact

- results: `.local/cosmos-reason2-2b/rolling-refill-results-20260820/`
- balanced primary: `final-refill64-inflight80-balanced/`
- cost table: `notes/results/cosmos-v010-p8d64-observed-cost-20260820.json`
