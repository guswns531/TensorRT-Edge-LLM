# Cosmos direct-cost v5 coverage 0과 workload regression gate

## 결론

production balanced trace에 남아 있던 direct-overlap coverage miss 116건을 targeted probe 하나로 제거했다. miss는
모두 continuation `past=128`, actual chunk=108, P2/P4, D37~64, maximum decode context=242~298에 집중되어
있었다. upper bucket으로 표현하면 두 shape뿐이다.

| P | chunk bucket | past bucket | D bucket | context bucket | initial | misses |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 4 | 128 | 128 | 64 | 512 | false | 80 |
| 2 | 128 | 128 | 64 | 512 | false | 36 |

64개 long-decode request를 먼저 실행하고 2초 뒤 prompt 234-token probe 16개를 투입했다. 그 결과 P2는 8개,
P4는 4개의 continuation sample을 얻었고, cost model v5에 필요한 direct points를 추가했다. v5 production 실행은
세 반복 모두 coverage miss 0, TPOT guard defer 7로 동일했다.

## balanced production 결과

Cosmos-Reason2-2B FP16 indexed-paged, independent TensorRT contexts, P4/D64, 288 requests, arrival 1000 req/s,
chunk 128, ragged padded-token budget 256의 3회 median이다.

| metric | static cap 128 | static cap 256 | cost-aware v5 | v5 vs 128 | v5 vs 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| generated token/s | 3207.0 | 3941.2 | 4100.5 | +27.86% | +4.04% |
| TTFT p95 | 6281.9 ms | 4668.2 ms | 4403.7 ms | -29.90% | -5.67% |
| TPOT p95 | 13.210 ms | 15.643 ms | 19.061 ms | +44.30% | +21.85% |
| E2E p95 | 7054.4 ms | 5579.5 ms | 5397.3 ms | -23.49% | -3.27% |

처리량 반복값은 4100.2/4106.3/4100.5 token/s였다. 출력 token/text는 static128 기준과 모두 일치했다.
TPOT p95는 50ms SLO 안이지만 정적 정책보다 높기 때문에 cost-aware를 latency-safe 기본값으로 사용하지 않는다.

## workload regression gate

balanced에서 얻은 정책이 작은 부하나 decode-heavy에서 불필요한 회귀를 만들지 확인했다. 모든 값은 3회
median이며 static128 ragged와 cost-aware v5를 같은 engine, CUDA graph budget, P4/D64에서 비교했다.

### 48-request short

| metric | static128 | cost-aware v5 | change |
| --- | ---: | ---: | ---: |
| generated token/s | 1595.6 | 1593.0 | -0.16% |
| TTFT p95 | 389.874 ms | 391.070 ms | +0.31% |
| TPOT p95 | 16.953 ms | 17.001 ms | +0.28% |
| E2E p95 | 608.635 ms | 609.708 ms | +0.18% |

cap 초과 후보가 없어 `overlap_evaluated_by_cost=0`, coverage miss=0, defer=0이었다. 최대 회귀는 0.32%로 3%
gate를 통과했다.

### 96-request decode-heavy

output length가 96/144/192/288/384인 실제 request mix다.

| metric | static128 | cost-aware v5 | change |
| --- | ---: | ---: | ---: |
| generated token/s | 3875.1 | 3867.3 | -0.20% |
| TTFT p95 | 1548.754 ms | 1555.403 ms | +0.43% |
| TPOT p95 | 13.373 ms | 13.364 ms | -0.06% |
| E2E p95 | 4357.609 ms | 4373.980 ms | +0.38% |

각 반복에서 cost evaluation은 1회, coverage miss=0, defer=0이었다. 최대 회귀는 0.43%로 3% gate를 통과했고
96개 출력은 세 반복 모두 static128과 일치했다.

## coverage 도구

`scripts/cosmos_reason2/summarize_overlap_cost_coverage.py`는 하나 이상의 `requests-dispatch.csv` 또는 결과
디렉터리를 읽어 raw lookup을 production upper bucket으로 집계한다. 출력 CSV는 바로 다음 controlled probe의
P/chunk/past/D/context 목표로 사용할 수 있다.

이전 v4 production에 실행하면 위의 두 probe shape와 miss 116건을 출력한다. v5 balanced, short, decode-heavy
검증 전체 2206 dispatch에 실행하면 `coverage_misses=0 recommended_probes=0`을 출력한다.

## 권장 preset과 다음 단계

- `latency-safe`: static overlap cap 128. TPOT tail을 우선하는 기본값이다.
- `throughput-balanced`: ragged token budget 256 + cost-aware admission + schema-v4/v5 direct table. 처리량과
  TTFT/E2E를 우선하되 TPOT target을 만족할 때 사용한다.
- coverage miss는 항상 decode-only로 유지하며 측정되지 않은 shape로 외삽하지 않는다.

다음 구현 단계는 이 두 정책을 명시적인 production scheduler preset으로 묶고, runtime에서 최근 TPOT p95가
목표에 가까워지면 cost-aware에서 latency-safe로 전환하는 hysteresis controller를 추가하는 것이다.

artifact는 `.local/cosmos-reason2-2b/ragged-direct-cost-v5-20260813/`에 있다.
