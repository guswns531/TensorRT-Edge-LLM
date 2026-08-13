# Cosmos production phase-shape CUDA graph priming

## 결론

exact request replay 없이 production serving이 실제로 사용하는 prefill/decode adapter와 TensorRT context buffer에
CUDA graph를 미리 capture하는 경로를 구현했다. short 48-request Cosmos trace의 3회 median은 다음과 같다.

| 조건 | generated token/s | TTFT p95 | TPOT p95 | E2E p95 |
| --- | ---: | ---: | ---: | ---: |
| 기존 Current, budget 256, cold | 1542.0 | 408.18ms | 17.017ms | 645.51ms |
| budget 512, cold | 1862.2 | 295.69ms | 16.078ms | 531.34ms |
| budget 512, exact-trace replay | 1958.0 | 290.21ms | 15.440ms | 503.98ms |
| **budget 512, phase-shape profile** | **1974.0** | **288.68ms** | **15.091ms** | **499.70ms** |
| vLLM warm server | 2021.4 | 258.58ms | 25.493ms | 493.46ms |

phase-shape profile은 기존 budget-256 Current보다 throughput `+28.02%`, TTFT p95 `-29.28%`, E2E p95
`-22.59%`다. budget-512 cold와 비교해도 throughput `+6.01%`, E2E p95 `-5.95%`다. vLLM 대비 throughput
격차는 `-2.35%`, E2E p95는 `+1.26%`까지 줄었고 TPOT p95는 Current가 `40.80%` 낮다. exact replay보다
약 `0.82%` 빠른 차이는 3회 변동 범위로 보며 shape priming이 replay upper bound를 재현했다고 해석한다.

## 동작 방식

```text
이전 dispatch-cost-table.csv
        |
        v
phase shape histogram builder
        |
        +-- prefill: batch, padded chunk, initial/continuation, max past KV
        +-- decode:  batch, max context
        |
        v
versioned JSON profile (request text/arrival/output 없음)
        |
        v
idle PhaseContextServingFacade
        |
        +-- synthetic stable slots + temporary KV pages
        +-- production-owned prefill/decode adapters와 streams
        +-- production independent TensorRT execution contexts
        |
        v
각 shape를 2회 enqueue -> auto capture/replay
        |
        +-- synthetic page lease 전부 반환
        +-- global slot lengths 0으로 reset
        +-- scheduler/dispatch/kernel telemetry reset
        |
        v
실제 online request admission 시작
```

중요한 차이는 임시 warmup engine이나 별도 buffer를 쓰지 않는다는 점이다. CUDA graph key에는 binding 주소가
포함되므로 실제 serving adapter와 다른 buffer로 capture하면 production enqueue가 cache hit하지 않는다.
`PhaseContextServingFacade::primeCudaGraphShapes()`는 facade가 idle일 때만 동작하고, 실제 serving-owned adapter,
stream, stable-slot mapping을 사용한다.

continuation prefill과 decode의 nonzero KV length는 단순 host length 덮어쓰기로 만들지 않는다. indexed-paged의
page-table 및 system-prompt reuse invariant를 지키기 위해 `PhaseBatchState::prepare()/commit()`으로 synthetic KV
ownership을 정상 경로로 만든다. 각 반복 후 page lease를 반환하고 마지막에 모든 slot length를 0으로 되돌린다.

## profile 생성과 실행

```bash
python3 scripts/cosmos_reason2/build_phase_graph_warmup_profile.py \
  --dispatch-csv previous/dispatch-cost-table.csv \
  --output phase-graph-profile.json \
  --max-prefill-shapes 3 \
  --max-decode-shapes 32 \
  --repetitions 2
```

Cosmos runner에는 다음 두 옵션을 추가했다.

- `--cuda-graph-warmup-profile phase-graph-profile.json`: exact trace가 아닌 phase binding shape를 priming한다.
- `--workload-preset short|balanced|decode-heavy`: 명시적인 `--prefill-token-budget`이 없을 때 short는 512,
  balanced/decode-heavy는 256을 선택한다. 사용자가 budget을 지정하면 preset보다 우선한다.

short profile은 prefill 상위 3개와 decode 19개, 총 22개 shape를 사용했다. 측정 종료 시 prefill graph 4개,
decode graph 20개가 cache에 있었고 측정 구간의 `post_enqueue_captures=0`이었다. 즉 실제 요청 처리 중 graph capture가
발생하지 않았다. 최종 paged pool도 `allocated=0, available=256`이었다.

## 메모리와 정확성

- 실행 전 CUDA 사용량: 9548.9MiB
- 실행 후 CUDA 사용량: 9652.9MiB
- graph cache 포함 증가량: 약 104MiB
- 남은 여유: 221.4MiB

10GB RTX 3080에서는 메모리 여유가 작다. 이번 단계는 사용자의 “메모리는 괜찮다”는 우선순위에 따라 latency와
throughput을 택했다. 더 많은 shape를 무조건 priming해서는 안 되며 prefill/decode graph byte budget이 계속 상한이다.

profile r1 결과를 budget-512 cold r1과 request별로 비교해 48개 요청 모두 생성 token 수와 text가 일치했다.
GPU 단위 테스트는 initial/continuation prefill과 decode priming 후 synthetic page bundle이 모두 반환되고 네 slot의
global length가 전부 0인지 검사한다.

## 코드 위치

| 역할 | 위치 |
| --- | --- |
| shape 계약과 production priming API | `cpp/runtime/scheduling/phaseContextServingFacade.{h,cpp}` |
| profile JSON load와 측정 전 priming | `examples/llm/llm_phase_bench.cpp` |
| telemetry에서 profile 생성 | `scripts/cosmos_reason2/build_phase_graph_warmup_profile.py` |
| Cosmos preset/profile runner 연결 | `scripts/cosmos_reason2/run_real_request_kv_matrix.py` |
| KV/page cleanup GPU test | `unittests/phaseDispatchWorkerTest.cpp` |
| profile/preset Python tests | `tests/python-unittests/test_phase_graph_warmup_profile.py`, `test_cosmos_workload_preset.py` |

## 다음 단계

1. short TTFT p95의 vLLM 대비 11.64% 격차를 prefill prepare, TensorRT enqueue, sampling의 CUDA-event group으로
   다시 분해한다.
2. 대표 shape 선택을 단순 빈도에서 `빈도 x capture 절감시간 / graph bytes` 점수로 바꾼다.
3. profile을 일정 주기로 갱신하되 graph cache 교체는 admission을 멈추지 않는 epoch boundary에서 수행한다.
4. dense ragged prefill을 true packed/varlen 입력으로 바꿔 P4의 padding과 graph shape cardinality를 줄인다.

artifact:

- `.local/cosmos-reason2-2b/phase-shape-warmup-20260813/`
- `.local/cosmos-reason2-2b/trace-graph-warmup-20260813/short-shape-profile.json`
- `.local/vllm-cosmos-reason2-2b/latest-comparison-20260813/short/`
