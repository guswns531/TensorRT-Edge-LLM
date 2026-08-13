# Cosmos short TTFT 분해와 P8 검증

## 결론

request timestamp, prefill adapter packing 경계, scheduler dispatch, 기존 CUDA-event kernel group을 request ID로
연결해 TTFT를 겹치지 않는 구간으로 분해했다. P4/512 short trace의 병목은 host packing이나 sampling이 아니라
prefill scheduler 대기다.

| P4/512 TTFT 구성 | median | p95 | p95 상위 3개 요청 평균 |
| --- | ---: | ---: | ---: |
| 전체 TTFT | 193.50ms | 289.26ms | 299.31ms |
| arrival → submit | 0.064ms | 1.523ms | 0.082ms |
| admission | 0.000ms | 0.000ms | 0.000ms |
| **prefill scheduler 대기** | **171.80ms** | **257.05ms** | **254.89ms** |
| host adapter pack | 0.032ms | 0.059ms | 0.059ms |
| prefill stream service wall | 21.86ms | 50.45ms | 44.24ms |
| first-token 완료 전달 | 0.002ms | 0.003ms | 0.003ms |

stream service 안에서는 `prefill_engine`이 p95 50.21ms로 거의 전부다. `prepare` p95는 0.176ms,
cache commit은 0.007ms, sampling은 0.033ms다. 따라서 embedding/packing/sampling micro-optimization으로는 남은
vLLM TTFT 차이를 닫을 수 없다.

## 계측 방식

```text
scheduled arrival
    | arrival_queue_us
submit
    | admission_us
stable slot admission
    | scheduler_wait_us (각 chunk의 queue residence 합)
dispatch selected
    | host_pack_us
serving adapter packed
    | phase_service_wall_us
    |   +-- prefill_prepare_gpu_us
    |   +-- prefill_engine_gpu_us
    |   +-- prefill_cache_commit_gpu_us
    |   +-- prefill_sample_gpu_us
prefill completion callback
    | first_token_delivery_us
first token visible
```

`phase_service_wall_us`는 host clock의 enqueue-to-completion wall time이고 네 CUDA-event group의 합은 실제 stream
service다. 두 값의 차이는 stream/completion residual로 따로 보존한다. 각 request에서 모든 구간의 합이 TTFT와
정확히 같지 않으면 analyzer가 실패한다.

한 prefill dispatch에 여러 request가 들어가므로 해당 CUDA-event 시간은 각 request가 경험한 latency로 연결한다.
이를 request별 GPU work의 합으로 해석하면 안 된다. 목적은 TTFT critical path 분해다.

## tail 원인

P4의 p95 상위 요청은 ID 11/23/35이며 모두 131-token prompt다. 첫 128-token chunk 뒤 3-token continuation이
남는다. throughput-oriented bucket score가 더 많은 useful token을 가진 initial bucket들을 먼저 처리하면서 이 final
continuation batch가 마지막 prefill dispatch까지 기다렸다.

wavefront cohort 4와 8도 한 번씩 실험했지만 token/s가 각각 1857.1/1854.3으로 낮아지고 TTFT p95가
334.09/335.09ms로 악화됐다. cohort를 고정하면 완료 가능한 짧은 요청들이 반대로 기다리는 문제가 생긴다. 따라서
단순 continuation 우선이나 고정 cohort는 채택하지 않는다.

## P8/1024 공정 비교

같은 engine, indexed-paged KV, D64, CUDA graph byte budget, independent contexts에서 P8 전용 shape profile을 만들고
독립 프로세스로 3회 측정했다.

| 조건 | token/s | TTFT median | TTFT p95 | TPOT p95 | E2E p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P4 / token budget 512 | 1971.3 | 193.50ms | 289.26ms | 15.129ms | 500.43ms |
| **P8 / token budget 1024** | **1995.6** | **180.58ms** | **282.91ms** | **14.959ms** | **493.99ms** |
| vLLM warm server | 2021.4 | 147.87ms | 258.58ms | 25.493ms | 493.46ms |

P8은 P4 대비 token/s `+1.24%`, TTFT p95 `-2.19%`, TPOT p95 `-1.12%`, E2E p95 `-1.29%`다.
vLLM 대비 token/s는 `-1.28%`, E2E p95는 `+0.11%`이고, TTFT p95는 아직 `+9.41%`다.

P8의 scheduler wait p95는 247.94ms로 P4보다 9.11ms 줄었지만 prefill stream service p95는 65.56ms로
15.11ms 늘었다. request별 상관관계 때문에 최종 TTFT는 개선됐지만 P를 계속 키우는 것만으로 큰 이득을 기대하기
어렵다.

P8은 decode graph 23개까지 priming해 종료 시 9752.9MiB를 사용하고 121.4MiB가 남았다. P4의 221.4MiB보다
약 100MiB 더 사용한다. 현재 메모리 우선순위에서는 허용 가능하지만 production 기본값으로 고정하기에는 여유가 작다.

## 구현 위치

| 역할 | 위치 |
| --- | --- |
| request별 prefill dispatch/pack/completion timeline | `examples/llm/llm_phase_bench.cpp` |
| Cosmos runner의 timeline artifact 연결 | `scripts/cosmos_reason2/run_real_request_kv_matrix.py` |
| request/timeline/kernel CSV join과 TTFT partition | `scripts/cosmos_reason2/analyze_ttft_breakdown.py` |
| partition closure 단위 테스트 | `tests/python-unittests/test_ttft_breakdown.py` |

48개 요청의 output token 수와 text는 계측 전 P4 reference와 모두 일치했다. 새 timeline 계측을 켠 P4 throughput은
직전 1974.0 token/s 대비 1971.3 token/s로 `-0.14%`이며 측정 변동 범위다.

## 다음 단계

1. short 최적 후보는 P8/1024로 유지하되 balanced/decode-heavy의 P4/256 회귀 gate를 바꾸지 않는다.
2. dense ragged `[B,Smax]` 대신 true packed/varlen prefill binding을 구현해 P8 stream service 증가를 줄인다.
3. continuation과 initial bucket을 완료 이득, useful tokens, predicted GPU cost로 함께 점수화하되 고정 wavefront는 쓰지
   않는다.
4. 대표 graph shape 선택도 빈도뿐 아니라 `예상 capture 절감시간 / graph bytes`를 포함한다.

artifact:

- `.local/cosmos-reason2-2b/ttft-breakdown-20260813/`
- `.local/cosmos-reason2-2b/ttft-policy-search-20260813/`
- `.local/vllm-cosmos-reason2-2b/latest-comparison-20260813/short/`
