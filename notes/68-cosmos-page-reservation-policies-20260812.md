SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Cosmos paged-KV reservation 정책 구현과 실험

## 결론

기본값과 범용 권장값은 계속 `full`이다. `bounded-overcommit`은 동시 입장을 조금 늘려 bimodal 처리량과
E2E를 개선할 수 있지만, growth lease가 부족하면 token 사이 대기가 크게 늘어난다. 따라서 현재 결과만으로
production default를 바꾸지 않고 opt-in 실험 정책으로 둔다.

## 구현된 정책

128-token page-bundle에서 요청의 full reservation을 `F`, base reservation을 `B`, tail을 `F - B`라고 한다.

- `full`: `B = F`. 기존 동작과 동일하다.
- `headroom`: prompt와 설정한 output headroom까지만 `B`로 보장한다.
- `bounded-overcommit`: full reservation에서 요청당 최대 N개 bundle을 할인하되 prompt 자체의 page보다 작아지지
  않는다.

overcommit 정책의 admission 조건은 다음과 같다.

```text
sum(all active base reservations)
+ sum(largest L active tails)
<= total page bundles
```

`L`은 동시에 base를 넘어 성장할 수 있는 요청 수다. 입장 후 선택된 요청은 terminal까지 sticky growth lease를
유지한다. lease가 없는 요청은 base 경계에 도달하면 decode queue에 그대로 남고, 다른 요청이 끝나 lease와 physical
pages를 반환하면 다시 runnable해진다.

이 조건은 현재 growth owner의 tail 합이 항상 largest-L tails 합보다 작거나 같음을 이용한다. 따라서 prefill/decode
overlap 중에도 이미 허가한 요청이 page-pool exhaustion으로 실패하지 않는다.

```text
request admission
      |
      +-- base + top-L tail guarantee fits --> stable slot + logical reservation
      |                                          |
      |                                          +--> lazy physical allocation
      |                                          |
      |                 no growth lease ---------+--> wait at base page boundary
      |                 sticky growth lease -----+--> may grow to full reservation
      |
      +-- guarantee does not fit -------------> bounded pending admission queue

terminal/cancel --> physical pages release --> base/tail guarantee release --> next sticky lease
```

## 코드 위치

| 기능 | 위치 |
| --- | --- |
| production reservation config | `cpp/runtime/scheduling/phaseAsyncServer.h` |
| base/full 계산, admission guarantee, sticky lease | `cpp/runtime/scheduling/phaseContextServingFacade.{h,cpp}` |
| blocked work를 보존하는 eligibility gate | `cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}` |
| real-trace CLI | `examples/llm/llm_phase_bench.cpp` |
| matrix runner와 status metadata | `scripts/cosmos_reason2/run_real_request_kv_matrix.py` |

CLI는 `--pageReservationMode`, `--pageReservationHeadroomTokens`,
`--pageReservationOvercommitBundles`, `--pageReservationGrowthRequests`를 제공한다. 아무 옵션도 주지 않으면 기존
`full` 동작이다.

## 성능 결과

공통 조건은 Cosmos-Reason2-2B FP16 indexed-paged, RTX 3080 10GB, independent prefill/decode TensorRT contexts,
P4/D64, fixed chunk 128, prefill token budget 256, pool 256 bundles, 64 slots, 288 requests다.

### Long-prefill

| policy | token/s | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | initial pending | observed max D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 912.23 | 24,269.09 | 18.66 | 25,875.16 | 253 | 35 |
| headroom64, L8 | 899.46 | 24,771.58 | 21.01 | 26,257.54 | 252 | 36 |
| bounded1, L8 | 791.17 | 29,266.79 | 74.60 | 29,970.21 | 248 | 37 |

이 workload는 output이 비교적 짧다. reservation을 줄여도 추가 입장은 1~5개뿐이고 page 경계 대기가 생겨 이득이
없다. `full`이 명확히 가장 좋다.

### Bimodal mixed

| policy | token/s | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | initial pending | observed max D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 1,493.55 | 21,354.44 | 17.84 | 24,329.80 | 233 | 54 |
| bounded1, L16 | 1,570.27 | 20,858.24 | 51.08 | 22,913.62 | 229 | 56 |
| bounded1, L32 | 1,477.86 | 21,737.85 | 18.55 | 24,386.91 | 231 | 56 |

L16은 full 대비 throughput +5.14%, TTFT p95 -2.32%, E2E p95 -5.82%지만 TPOT p95가 186% 악화됐다. L32는
TPOT을 full의 +4.0%까지 회복하지만 throughput -1.05%, TTFT +1.80%, E2E +0.23%로 전체 이득이 사라진다.

## 메모리 관점

이 정책은 engine weights, TensorRT context workspace, CUDA graph cache, KV page-pool의 물리적 크기를 줄이지 않는다.
줄이는 것은 admission 시점의 `reserved-but-unused` 논리적 fragmentation이다. 그 결과 더 많은 stable slots가 실제
physical pages를 사용할 수 있어 peak allocation은 오히려 높아질 수 있다. bimodal full의 admission 관측 peak는
223 bundles, bounded L16은 239 bundles였다. 모든 완료 run은 최종 `allocated=0`, `available=256`으로 page와 slot을
누수 없이 반환했다.

## 판정과 다음 단계

- `full`을 production default로 유지한다.
- `headroom`과 `bounded-overcommit`은 workload-specific opt-in으로 유지한다.
- throughput 우선이라도 TPOT SLO가 있는 서비스에는 L16 결과를 사용하지 않는다.
- 다음 최적화는 고정 L이 아니라 decode cost table과 TPOT pressure를 이용해 growth lease 수를 동적으로 조절하는
  것이다. 단, lease를 이미 사용해 base를 넘은 요청의 소유권은 terminal까지 취소하면 안 된다.

## 검증

- TensorRT 26.06 / CUDA 13.3 build 성공
- 관련 C++ test 27/27 성공
- long-prefill 및 bimodal 각 정책 288-request GPU run 완료
- 모든 완료 run에서 page-pool 최종 allocated bundle 0
- Python syntax compile 성공
