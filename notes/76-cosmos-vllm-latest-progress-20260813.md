# Cosmos 최신 scheduler와 vLLM 비교 및 발전 추이

## 결론

Cosmos-Reason2-2B FP16 text serving에서 최신 `throughput-balanced` 경로는 balanced workload의 실제 생성
token 처리량을 vLLM 대비 `-0.50%`까지 좁혔다. decode-heavy에서는 Current가 실제 생성 token/s `+4.75%`,
TPOT p95 `-18.85%`, E2E p95 `-3.09%`로 앞선다. 반면 short workload는 여전히 token/s `-23.57%`, TTFT p95
`+57.46%`, E2E p95 `+30.56%`로 vLLM보다 느리다.

초기 independent D32 balanced 결과는 vLLM보다 처리량이 약 41% 낮았다. indexed-paged KV, prefill token budget,
ragged batching, direct-cost overlap admission을 순서대로 추가하면서 Current 처리량은 2426.1에서 4114.0 token/s로
69.58% 증가했다. 같은 기간 TTFT p95는 8350.2ms에서 4339.4ms로 48.03%, E2E p95는 9229.0ms에서
5367.7ms로 41.84% 감소했다.

아직 production serving 전체에서 vLLM을 대체했다고 말할 수는 없다. short prompt admission과 batch formation,
메모리 footprint, 동일 transport 비교가 남아 있다.

## 공정 비교 조건

| 항목 | Current | vLLM |
| --- | --- | --- |
| model | `nvidia/Cosmos-Reason2-2B` revision `9ce19a1...` | 동일 local checkpoint |
| weights / KV | FP16 / FP16 | FP16 / FP16 |
| KV capacity | 32,768 tokens, 3,584MiB raw | 32,768 tokens, 3,584MiB raw |
| active sequence cap | 80 stable slots | `max-num-seqs=80` |
| prefix cache | 없음 | off |
| prefill | fixed chunk 128, dense ragged, P4 | chunked prefill, token budget 8192 |
| decode | independent TensorRT context, D64 | continuous batching |
| graph | phase별 CUDA graph cache, global reserve 0 | compile + CUDA graph |
| input | C++ runtime 직접 trace injection | localhost HTTP streaming |

각 workload는 Current와 vLLM이 동일한 materialized trace SHA를 사용한다. prompt token 합, arrival offset,
requested output limit, greedy 설정도 같다. 각 수치는 3회 실행의 median이다. vLLM은 64-request concurrent warmup
후 측정했다. Current는 프로세스마다 기본 graph를 준비하지만 trace 중 보지 못한 shape의 on-demand capture도
측정 시간에 포함한다.

이 비교에는 서로 반대 방향의 잔여 편향이 있다.

- Current의 direct C++ injection은 vLLM의 HTTP/JSON/streaming보다 유리하다.
- vLLM은 persistent server warmup 뒤 측정하지만 Current harness는 매 run 새 프로세스를 시작하므로 short trace의
  graph capture에는 Current가 불리하다.
- 두 backend의 수치와 EOS 판정이 달라 실제 생성 token 수가 다를 수 있다. 그래서 실제 token/s와 함께 request/s,
  duration, token 수를 확인해야 한다.

## 최신 세 workload 결과

### 3회 median

| workload | metric | Current | vLLM | Current 변화 |
| --- | --- | ---: | ---: | ---: |
| short 48 | generated token/s | 1545.0 | 2021.4 | -23.57% |
|  | TTFT p95 | 407.2ms | 258.6ms | +57.46% |
|  | TPOT p95 | 16.963ms | 25.493ms | **-33.46%** |
|  | E2E p95 | 644.3ms | 493.5ms | +30.56% |
| balanced 288 | generated token/s | 4114.0 | 4134.6 | -0.50% |
|  | TTFT p95 | 4339.4ms | 4118.8ms | +5.36% |
|  | TPOT p95 | 18.841ms | 18.579ms | +1.41% |
|  | E2E p95 | 5367.7ms | 5162.7ms | +3.97% |
| decode-heavy 288 | generated token/s | **4646.5** | 4435.8 | **+4.75%** |
|  | TTFT p95 | 8965.9ms | 8587.4ms | +4.41% |
|  | TPOT p95 | **13.466ms** | 16.595ms | **-18.85%** |
|  | E2E p95 | **11259.2ms** | 11617.6ms | **-3.09%** |

short는 두 backend 모두 정확히 1040 tokens를 생성했으므로 처리량 차이가 그대로 duration 차이다. balanced는
Current/vLLM이 24,384/23,712 tokens를 생성했다. 실제 token/s 차이는 -0.50%지만 request completion duration은
5927.0/5735.0ms로 Current가 3.35% 길다. 동일 requested-token 예산으로 정규화한 처리량 차이도 -3.24%다.

decode-heavy는 Current가 매번 57,984 tokens를 생성했고 vLLM은 median run에서 56,962 tokens를 생성했다. Current의
실제 token/s는 +4.75%, request/s는 +2.95%다. Current가 더 많은 token을 생성했는데도 전체 요청 완료시간은
2.86% 짧았다.

### scheduler 동작

- short: cap 초과 cost evaluation 0, coverage miss 0, fallback 0이다. 현재 격차는 overlap 판단이 아니라 작은
  workload에서의 prefill formation과 graph warmup/capture 비용이다.
- balanced: 반복당 cost evaluation 48, guard defer 7, coverage miss 0, fallback 0이다. 50ms TPOT target 안에서
  throughput path를 유지한다.
- decode-heavy: 반복당 cost evaluation 21, coverage miss 1이고 491 dispatch가 latency-safe fallback 상태였다.
  decode pressure가 높을 때 큰 prefill overlap을 막은 결과 TPOT과 E2E가 vLLM보다 낮다.

## 단계별 발전 정도

아래는 동일한 12-prompt/288-request/output4x workload family다. 마지막 두 단계는 Poisson seed만 다르며 prompt와
output-length 분포는 같다.

| 단계 | 핵심 변화 | token/s | 직전 대비 | TTFT p95 | TPOT p95 | E2E p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| early independent | indexed-linear, P8/D32 | 2426.1 | - | 8350.2ms | 12.700ms | 9229.0ms |
| indexed-paged static | page pool과 larger admission | 3107.7 | +28.10% | 6339.8ms | 11.360ms | 7114.3ms |
| token budget 256 | compatible prefill 묶기 | 3648.2 | +17.39% | 5162.6ms | 16.710ms | 6093.3ms |
| ragged prefill | 다른 final chunk right-padding | 3941.2 | +8.03% | 4658.8ms | 15.633ms | 5577.7ms |
| direct-cost v5 | 안전한 cap 초과 overlap | 4100.5 | +4.04% | 4403.7ms | 19.061ms | 5397.3ms |
| throughput-balanced | rolling-p95 hysteresis preset | 4114.0 | +0.33% | 4339.4ms | 18.841ms | 5367.7ms |

초기 대비 최종 변화는 token/s `+69.58%`, TTFT p95 `-48.03%`, E2E p95 `-41.84%`다. 대신 aggressive prefill
overlap으로 TPOT p95는 12.700ms에서 18.841ms로 48.35% 증가했다. 최신 hysteresis는 이 trade-off를 없애는 것이
아니라 target에 가까워질 때 static-128로 되돌아가도록 제한한다.

초기 Current/vLLM 처리량은 2426.1/4119.5 token/s로 Current가 41.11% 낮았다. 최신 동일 workload family는
4114.0/4134.6 token/s로 0.50% 낮다. 약 40.6 percentage point의 처리량 격차를 회수했다.

## 메모리 비교

두 runtime 모두 raw FP16 KV를 정확히 3,584MiB로 맞췄다. 동일 balanced 실행 중 `nvidia-smi` process memory를
샘플링한 결과다.

| 항목 | Current max-throughput | Current reserve-256 | vLLM memory-matched |
| --- | ---: | ---: | ---: |
| raw KV | 3584MiB | 3584MiB | 3584MiB |
| observed process memory | **9728MiB peak** | 9575MiB after run | **8180MiB peak** |
| vLLM 대비 추가 사용 | +1548MiB | +1395MiB | - |
| balanced token/s | 4114.0 | 4100.5 | 4134.6 |

Current의 두 independent TensorRT context workspace는 prefill 약 1088MiB, decode 약 513MiB다. max-throughput
실행은 prefill/decode graph를 4/30개 캐시해 graph charge 약 140MiB를 더 사용했다. reserve-256은 graph를 2/2개로
제한해 약 153MiB를 절약하지만 10GB GPU에서 free memory는 약 299MiB라 512MiB headroom gate를 여전히 못 넘는다.

vLLM보다 Current가 많이 쓰는 약 1.5GiB는 KV allocator 차이보다 독립 TensorRT context workspace와 phase별 I/O,
CUDA graph cache가 주원인이다. independent context overlap이 decode-heavy 성능을 만드는 동시에 메모리 비용도
만드는 구조적 trade-off다.

## 다음 우선순위

1. short trace 전에 representative P/D graph shape를 warmup해 cold capture를 측정 구간 밖으로 옮기고 A/B한다.
2. prefill P4의 dense ragged `[B,Smax]`를 true packed/varlen token layout으로 바꾸거나 더 큰 token-budget profile을
   별도 engine/context로 제공한다.
3. prefill/decode workspace를 phase profile별로 더 작게 빌드해 최소 512MiB headroom을 확보한다.
4. Current에도 vLLM과 동일한 HTTP streaming frontend를 연결해 transport가 완전히 같은 production E2E를 측정한다.
5. short/balanced/decode-heavy별 preset routing을 둔다. short는 prefill-first batch formation, balanced는 현재
   throughput-balanced, decode-heavy는 현재 hysteresis fallback이 가장 적합하다.

artifact는 다음에 있다.

- Current: `.local/cosmos-reason2-2b/vllm-latest-comparison-20260813/`
- Current balanced: `.local/cosmos-reason2-2b/throughput-balanced-hysteresis-20260813/`
- vLLM: `.local/vllm-cosmos-reason2-2b/latest-comparison-20260813/`
- vLLM balanced: `.local/vllm-cosmos-reason2-2b/throughput-balanced-trace-20260813/`
