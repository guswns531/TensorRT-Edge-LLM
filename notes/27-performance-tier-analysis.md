# Legacy부터 continuous scheduler까지 단계별 성능 분석

## 결론

현재 성능 효과는 두 종류로 구분해야 한다.

1. actual text BS4 request 100 batch에서 indexed-only의 총 GPU time은 legacy보다 `1.14%` 늘었다. indexed KV는
   정상 prefill/decode를 빠르게 만드는 기능이 아니라 eviction의 KV compaction을 제거하는 기반이다.
2. 실제 request lifecycle과 arrival queue를 포함한 6개 workload에서 independent TensorRT context는 shared
   serialized context보다 throughput을 `1.20~24.45%` 높이고, 5개 부하의 TTFT p95를 `6.71~85.85%` 낮췄다.
   비포화 20 rps에서는 TTFT가 7.23% 늘었지만 E2E p95와 TPOT p95는 각각 33.74%, 42.91% 줄었다.

즉 indexed KV가 안전한 ownership과 compaction-free eviction을 만들고, 그 위의 phase scheduler가 indexed
lookup 비용보다 큰 overlap 이득을 만드는 구조다.

## 비교 tier

| Tier | 구성 | 효과 | 비교 방법 |
|---|---|---|---|
| L0 완전 legacy | active row = physical row, 기존 실행 | 기준점 | L1과 actual text BS4 100 batch 비교 |
| L1 indexed only | stable `kv_slot_ids`, 기존 실행 순서 | eviction KV copy 제거 | L0와 actual text request 직접 비교 |
| L2 indexed + shared TRT | queue/stream 2개, TRT context 1개, event 직렬화 | 안전 fallback | L3와 동일 continuous trace 비교 |
| L3 indexed + independent TRT | CUDA context 1개, TRT context/workspace/I/O/stream 2개 | kernel overlap | L2와 동일 continuous trace 비교 |
| L4 production adapter | L3 + request context pack/scatter | 실제 batch 연결 | adapter on/off 비교 |
| L5 continuous scheduler | L4 + admission/batching/backpressure | 부하 제어 | 현재 legacy 동등 경로 없음 |

주 비교는 actual text request를 처리하는 `llm_inference`와 actual request context, admission, 실제 greedy
sampling, 반복 decode, terminal release를 수행하는 continuous `llm_phase_bench`다. fixed-shape phase 측정은
kernel overlap의 보조자료로만 사용한다.

## L0 대 L1: actual text request의 indexed-only 비용

실제 chat-template text 요청 4개에 서로 다른 stop 조건을 주어 active batch가 4→3→2→1로 줄어드는 입력을
100 batch 반복했다. 각 engine은 warmup 20 batch 뒤 prefill 100회와 generation/eviction 500회를 profiler로
집계했다. 두 출력 파일의 SHA256은
`500efce553448fcbec39d622720df374dc6f88c332e689070785395e82eb8578`로 같다.

| Stage | Legacy | Indexed only | 변화 |
|---|---:|---:|---:|
| Prefill median / p95 | 17.6551 / 17.8228 ms | 17.6067 / 17.8129 ms | -0.27% / -0.06% |
| Generation-step median / p95 | 5.9976 / 6.8363 ms | 6.1302 / 6.9324 ms | +2.21% / +1.41% |
| 전체 GPU time | 4,844.62 ms | 4,899.61 ms | +1.14% |
| Peak VRAM | 8,480 MiB | 8,486 MiB | +6 MiB |

이 workload의 prompt는 batch 전체 76 token이고 생성은 batch당 15 token이라 eviction 때 이동하는 KV가 작다.
따라서 compaction 제거 이득보다 indexed lookup 비용이 더 크게 보인다. 긴 survivor KV의 eviction은 별도 direct
benchmark가 필요하다. engine 크기는 121,104 byte, 약 `0.0121%` 증가했다.

## L1/L2 대 L3: 동일 continuous request trace

두 topology에 seed 0의 같은 request ID, arrival offset, prompt length, output budget을 replay했다. shared mode도
indexed KV, admission과 queue batching을 사용하지만 TensorRT context 하나를 event로 직렬화한다. independent
mode는 같은 CUDA primary context에서 prefill/decode TensorRT context와 stream을 분리한다.

| Workload | Throughput | TTFT p95 | E2E p95 | TPOT p95 | Pending |
|---|---:|---:|---:|---:|---:|
| steady 20 rps | 19.23→19.46, +1.20% | 25.03→26.84, +7.23% | 135.52→89.79, -33.74% | 15.86→9.06, -42.91% | 0→0 |
| steady 25 rps | 20.05→24.12, +20.29% | 194.44→27.52, -85.85% | 340.92→104.49, -69.35% | 25.36→11.03, -56.52% | 19→0 |
| burst 1000 rps | 25.74→27.74, +7.79% | 831.47→774.28, -6.88% | 901.30→833.82, -7.49% | 13.15→9.85, -25.06% | 20→20 |
| long output 24~32 | 8.88→9.47, +6.56% | 1328.91→1239.74, -6.71% | 1650.98→1540.08, -6.72% | 17.42→16.13, -7.44% | 12→12 |
| chunked 512/128 | 10.74→12.61, +17.39% | 922.43→766.55, -16.90% | 1011.63→846.15, -16.36% | 32.44→24.89, -23.29% | 8→8 |
| mixed lengths | 10.86→13.52, +24.45% | 1136.68→884.89, -22.15% | 1339.80→1050.35, -21.60% | 38.98→25.54, -34.47% | 12→12 |

steady 25 rps의 큰 tail 개선은 독립 mode가 포화 knee를 20 rps 부근에서 24 rps 이상으로 옮긴 결과라서 단순
kernel speedup으로 해석하면 안 된다. 반대로 비포화 20 rps에서는 prefill TTFT가 1.81 ms 나빠졌지만 이후 decode
service가 빨라 E2E와 TPOT은 크게 개선됐다. scheduler는 TTFT만이 아니라 TTFT/TPOT/E2E SLO를 함께 봐야 한다.

이 결과는 한 번 생성한 deterministic trace의 topology별 1회 replay다. batch histogram과 overlap dispatch 수는
두 mode에서 같아서 workload 선택 차이는 없지만, 최종 gate에는 process-level 3회 반복을 추가해야 한다. 전체
원자료 요약은 `gemma4-e2b-real-request-context-comparison.csv`에 있다.

## Fixed-shape kernel 보조 측정

기존 6개 workload의 100-sample 결과에서 shared TRT context는 event 직렬화 때문에 median
`+0.29~+1.38%`, p95 `+0.17~+1.35%`의 scheduler 비용만 있었다. independent TRT context는 모든 workload의
makespan을 줄였다.

| Active rows | Prompt / past KV | Median 감소 | p95 감소 |
|---:|---:|---:|---:|
| 1+1 | 128 / 128 | 12.53% | 12.28% |
| 1+1 | 512 / 512 | 6.90% | 6.88% |
| 1+1 | 1024 / 1536 | 4.14% | 4.05% |
| 2+2 | 128 / 128 | 10.48% | 9.87% |
| 2+2 | 512 / 512 | 4.22% | 4.24% |
| 2+2 | 1024 / 1536 | 2.32% | 2.31% |

2026-08-01 현재 코드와 production adapter로 BS1 prompt 128 + KV 128을 warmup 20회, 100회 재측정했다.

| Mode | Sequential median / p95 | Scheduled median / p95 | 결과 |
|---|---:|---:|---:|
| Shared TRT | 32.7844 / 33.1172 ms | 32.5553 / 32.9168 ms | median -0.70%, 잡음 수준 |
| Independent TRT | 24.7706 / 24.8443 ms | 20.9707 / 21.1141 ms | median -15.34%, 1.1812x |

재측정은 기존 결론과 방향이 같지만 한 scenario의 한 process 실행이라 기존 matrix를 대체하지 않는다. shared와
independent의 sequential 절대값 차이는 context/profile topology가 다르므로 각 row 내부만 직접 비교해야 한다.
짧은 workload는 두 phase가 겹칠 면적이 커 이득이 크고, 장문 prefill은 한 phase가 makespan 대부분을 차지해
상대 개선 상한이 작다. 동시 실행 중 개별 phase는 contention으로 느려질 수 있으므로 makespan, TTFT, TPOT을
함께 최적화해야 한다.

## L4와 L5

BS2 prompt 512 + decode KV 512에서 production adapter 비용은 sequential median/p95 `+0.15/+0.17%`, overlap
`+0.04/+0.21%`였고, host pack/scatter median은 약 `53/12 us`였다. adapter 포함 overlap speedup은 1.0529x다.

continuous workload의 prompt 128, output 8, BS2+2 구성은 약 25 req/s까지 안정적이었다. 30 req/s부터 처리량은
25~26 req/s에 머물고 queue latency가 증가했다. chunked prefill 128에서 overlap threshold를 batch total
128→256 token으로 맞추면 throughput은 11.65→12.63 req/s, `8.4%` 증가했고 TTFT p95는
871.04→768.76 ms, `11.7%` 감소했다. 이는 scheduler policy끼리의 직접 비교이지 legacy 대비 수치는 아니다.

## KV eviction과 메모리 효율

35개 attention layer는 FP16 KV, KV head 1개이며 d256 28개와 d512 7개다. 한 physical slot의 token 하나는
모든 layer에 걸쳐 43,008 byte, 정확히 42 KiB다. legacy compaction은 capacity 전체가 아니라 survivor의 실제
`seqLen`만 복사한다.

| Survivor length | 한 slot 이동 | 두 slot 이동 | Indexed |
|---:|---:|---:|---:|
| 128 | 5.25 MiB | 10.50 MiB | 0 MiB KV D2D |
| 512 | 21.00 MiB | 42.00 MiB | 0 MiB KV D2D |
| 1536 | 63.00 MiB | 126.00 MiB | 0 MiB KV D2D |
| 2048 | 84.00 MiB | 168.00 MiB | 0 MiB KV D2D |

indexed branch는 slot ID와 length metadata만 갱신하고 `compactKVCacheBatched()`를 호출하지 않는다. 위 byte는
제거된 이론적 copy volume이다. 아직 Nsight Systems로 legacy eviction의 실제 DRAM byte와 latency를 직접 측정하지
않았으므로 실측값으로 표현하면 안 된다.

indexed-linear는 allocation을 줄이지 않는다. 네 slot을 미리 고정 할당해 외부 파편화와 재할당은 없지만,
짧은 request가 2048-token capacity를 다 쓰지 않는 내부 낭비는 남는다. independent context의 관측 VRAM은
8,626 MiB로 shared 7,898 MiB보다 약 728 MiB 컸고 1,240 MiB headroom을 남겼다. 현재 선택은 KV capacity
효율보다 stable ownership과 overlap을 우선한다.

## 다음 검증

1. legacy/indexed eviction 4→2를 length 128/512/1536에서 CUDA event와 Nsight Systems로 직접 비교한다.
2. 동일 arrival trace A/B harness를 만들고 legacy의 drain-and-rebatch 기능 차이를 함께 보고한다.
3. production adapter phase matrix를 3회 반복하며 GPU clock/temperature와 VRAM peak를 저장한다.
4. throughput, TTFT/TPOT/E2E p95, pending depth, batch histogram을 공동 gate로 사용한다.

원자료는 `gemma4-e2b-real-request-context-comparison.csv`, `gemma4-e2b-perf-gate-confirmed.csv`,
`gemma4-e2b-phase-context-comparison.csv`, `/tmp/gemma4-e2b/perf/real-request-*-repeat100-profile.json`,
`/tmp/gemma4-e2b/perf/phase/real-request-tier-{shared,independent}/summary.csv`에 있다.
