# Cosmos v0.10 adaptive admission profile

## 결론

수동으로 선택하던 두 P8/D64 운영점을 pending backlog 기반 hysteresis로 자동 전환했다.

```text
latency mode
  admission limit 64
  sampling-aware refill off

pending request 발생
  -> throughput mode
  -> admission limit 80
  -> D64 sampling refill on

pending == 0 && active <= 64
  -> latency mode 복귀
```

Balanced fixed-output 3회 중앙값은 3,905.7 token/s다. 수동 throughput profile 3,889.9보다 0.4% 높아
측정 노이즈 범위에서 동등하며, 매 run 모드 전환은 정확히 두 번이었다. short 48-request trace는 pending backlog가
생기지 않아 latency mode를 유지하고, decode-heavy는 throughput mode로 승격한 뒤 tail에서 복귀했다.

## Balanced 3회

조건은 288 requests, prompt 25,872, output 24,960, P8/D64, stable80, latency admission64, throughput
admission80, refill target64, fixed chunk128, graph-off다.

| metric | 수동 throughput | adaptive | 변화 |
| --- | ---: | ---: | ---: |
| generated token/s | 3,889.9 | **3,905.7** | +0.4% |
| TTFT median | 2,035.8 ms | **2,026.5 ms** | -0.5% |
| TTFT p95 | 4,808.4 ms | 4,812.7 ms | +0.1% |
| TPOT median | 18.229 ms | **18.195 ms** | -0.2% |
| TPOT p95 | 20.043 ms | **19.997 ms** | -0.2% |
| E2E median | **3,729.2 ms** | 3,738.9 ms | +0.3% |
| E2E p95 | 5,868.7 ms | **5,855.8 ms** | -0.2% |
| peak memory | 8,087 MiB | 8,087 MiB | 동일 |

세 run 처리량은 3,918.3 / 3,878.7 / 3,905.7 token/s다. 모든 run에서 generated output은 정확히
24,960이며 모드별 dispatch 분포는 다음과 같다.

| run | throughput-mode dispatch | latency-mode dispatch | transition | refill wait polls |
| --- | ---: | ---: | ---: | ---: |
| 1 | 455 | 98 | 2 | 332 |
| 2 | 464 | 98 | 2 | 331 |
| 3 | 461 | 98 | 2 | 335 |

## Workload 선택 결과

| workload | adaptive token/s | vLLM fixed-output | 격차 | mode 동작 |
| --- | ---: | ---: | ---: | --- |
| short | 1,718.9 | 1,998.7 | -14.0% | 48 requests, latency 유지 |
| balanced | 3,905.7 | 4,234.4 | -7.8% | 두 번 전환 |
| decode-heavy | 4,477.4 | 4,797.9 | -6.7% | throughput 1,132 dispatch 후 latency 337 dispatch |

Decode-heavy TPOT median/p95는 14.29/14.77ms로 vLLM 14.86/15.40ms보다 낮다. adaptive 전환은 saturated
처리량을 훼손하지 않으며 workload 규모를 미리 지정할 필요를 없앤다.

## 구현 경계

- 기본값은 off이므로 기존 admission/refill 동작 유지
- latency limit은 1 이상이고 max in-flight 이하
- backlog enter threshold는 1 이상
- latency→throughput: pending queue가 threshold 이상일 때만
- throughput→latency: pending이 0이고 active가 latency limit 이하일 때만
- active가 80인 상태에서 조기 축소하지 않으므로 이미 admit한 요청을 취소하거나 eviction하지 않음
- latency mode에서는 decode refill target을 0으로 취급
- throughput mode에서만 configured D64 refill 사용
- mode, transition count, refill wait count를 각 `PHASE_METRIC`에 기록
- 순수 hysteresis decision과 refill decision을 GPU 없는 단위 테스트로 검증

## 해석

이 controller는 request별 SLO를 직접 최적화하는 최종 scheduler는 아니다. queue backlog를 production profile 선택의
첫 신호로 연결한 단계다. 큰 burst는 throughput mode가 최적이고, 64개 미만의 짧은 trace는 latency mode가 자연스럽다.

다음 개선은 backlog 개수만 보지 않고 다음 값을 함께 사용하는 것이다.

1. oldest TTFT slack
2. recent TPOT pressure
3. prefill-ready staging row 수
4. page-pool pressure
5. D64 graph coverage 여부

## Artifact

- balanced: `.local/cosmos-reason2-2b/adaptive-admission-results-20260820/final-balanced/`
- short/decode-heavy: `.local/cosmos-reason2-2b/adaptive-admission-results-20260820/`
- P8/D64 cost: `notes/results/cosmos-v010-p8d64-observed-cost-20260820.json`

