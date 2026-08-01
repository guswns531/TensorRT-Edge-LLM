# Legacy부터 continuous scheduler까지 단계별 성능 분석

## 결론

현재 성능 효과는 두 종류로 구분해야 한다.

1. indexed-linear KV는 정상 prefill/decode를 빠르게 만들지 않는다. stable physical slot으로 eviction 때 KV
   compaction을 제거하며, 정상 kernel 비용은 median `+0.14~+2.57%`다.
2. 실제 makespan 개선은 같은 CUDA primary context 안의 독립 TensorRT context와 stream으로 prefill/decode를
   겹칠 때 나온다. 기존 6개 workload에서 median latency가 `2.32~12.53%` 감소했다.

즉 indexed KV가 안전한 ownership과 compaction-free eviction을 만들고, 그 위의 phase scheduler가 indexed
lookup 비용보다 큰 overlap 이득을 만드는 구조다.

## 비교 tier

| Tier | 구성 | 효과 | 비교 방법 |
|---|---|---|---|
| L0 완전 legacy | active row = physical row, 기존 실행 | 기준점 | L1과 같은 `llm_bench` shape |
| L1 indexed only | stable `kv_slot_ids`, 기존 실행 순서 | eviction KV copy 제거 | L0와 직접 비교 |
| L2 indexed + shared TRT | queue/stream 2개, TRT context 1개, event 직렬화 | 안전 fallback | 자체 sequential과 비교 |
| L3 indexed + independent TRT | CUDA context 1개, TRT context/workspace/I/O/stream 2개 | kernel overlap | 자체 sequential과 비교 |
| L4 production adapter | L3 + request context pack/scatter | 실제 batch 연결 | adapter on/off 비교 |
| L5 continuous scheduler | L4 + admission/batching/backpressure | 부하 제어 | 현재 legacy 동등 경로 없음 |

L0/L1의 `llm_bench`는 한 phase의 CUDA-event 시간이고, L2~L4의 `llm_phase_bench`는 prefill과 decode를 함께
넣은 makespan이다. 두 벤치의 absolute millisecond를 직접 빼면 안 된다.

## L0 대 L1: indexed KV만 켠 비용

RTX 3080, CUDA graph 비활성, warmup 20회, 측정 100회, 전체 3회 결과다. greedy 출력 SHA256은 두 engine이
`1918f649c96695ea807985d3e7a98c4257d3d429f3b9557277f4957472dfcb2a`로 같다.

| Phase | Shape | Median 변화 | p95 변화 |
|---|---|---:|---:|
| Prefill | BS1/BS4 × input 128/512/1024 | `+0.14~+0.90%` | `+0.06~+0.48%` |
| Decode | BS1/BS4 × past KV 128/512/1536 | `+0.80~+2.57%` | `-1.25~+2.97%` |

모두 3% gate를 통과했다. engine 크기는 1,001,923,868 byte에서 1,002,044,972 byte로 121,104 byte,
약 `0.0121%` 증가했다. BS1 component median 합은 다음과 같다. request E2E latency가 아니라 overlap 전 component
cost를 보기 위한 값이다.

| Prompt / past KV | Legacy | Indexed | 변화 |
|---|---:|---:|---:|
| 128 / 128 | 25.1235 ms | 25.4594 ms | +1.337% |
| 512 / 512 | 55.9956 ms | 56.2683 ms | +0.487% |
| 1024 / 1536 | 112.5921 ms | 112.9126 ms | +0.285% |

## L2 대 L3: context 분리와 overlap

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

## L0부터 L3까지 정규화 추정

L0→L1 component ratio와 기존 L1-sequential→L3-overlap ratio만 곱했다. 동일 request E2E 직접 측정이 아니라
단계별 효과를 보여 주는 `normalized estimate`다.

| BS1 workload | Indexed 비용 | Overlap 효과 | Legacy 대비 추정 순효과 |
|---|---:|---:|---:|
| prompt 128 / KV 128 | +1.337% | -12.53% | -11.36%, 약 1.128x |
| prompt 512 / KV 512 | +0.487% | -6.90% | -6.45%, 약 1.069x |
| prompt 1024 / KV 1536 | +0.285% | -4.14% | -3.87%, 약 1.040x |

보수적인 기존 matrix를 써도 indexed 비용을 상쇄하고 이득이 남는다. 다만 production 결론에는 같은 arrival trace의
end-to-end A/B가 필요하다. legacy runtime은 stable ownership 없이 continuous admission을 안전하게 수행할 수 없어
L5와 완전히 같은 실험은 현재 불가능하다.

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

원자료는 `gemma4-e2b-perf-gate-confirmed.csv`, `gemma4-e2b-phase-context-comparison.csv`,
`/tmp/gemma4-e2b/perf/phase/tier-current-*.csv`, `/tmp/gemma4-e2b/perf/phase/load-suite-v1/summary.csv`에 있다.
