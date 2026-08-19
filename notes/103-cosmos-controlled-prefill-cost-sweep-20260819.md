# Cosmos controlled prefill shape cost sweep

## 결론

동일 tied embedding/LM-head Cosmos-Reason2-2B max-128 engine에서 `P × chunk × D × past-KV` controlled
microbenchmark를 구축하고 실행했다. 요청한 256개 조합 중 indexed-paged KV 256 bundles와 80 slots에서 실제로
소유권을 만들 수 있는 172개를 자동 선별했다.

- initial prefill: 88 cases
- continuation prefill: 84 cases
- 각 case: warmup 5, sequential/independent-overlap 각각 20 samples
- 결과: 172/172 성공
- controlled model: prefill 156, direct overlap 172, decode 11 points
- controlled + real-request hybrid model: prefill 193, direct overlap 251, decode 16 points
- hybrid replay와 실제 5-workload에서 cost coverage miss 0

비교 가능한 86개 `(P,D,past,phase)` pair에서 queue-scaled decode penalty를 포함한 effective token throughput은
C128이 C64보다 86/86 모두 높았다. 현재 engine에서 C64는 일반 최적 shape가 아니라 C128이 TPOT/debt feasibility를
넘을 때만 남겨 둘 비상 후보다.

## continuation past-KV를 어떻게 만들었나

paged KV는 nonzero length를 `resetForNewSequences()`로 직접 주입하지 못한다. 그렇게 하면 page ownership 없이
system-prompt cache를 재사용하려는 잘못된 상태가 되므로 runtime이 거부한다.

controlled fixed benchmark는 다음 순서를 사용한다.

```text
page allocator reset
       |
       v
PhaseBatchState.prepare(synthetic work items)
       |  ensureCapacity(slot, past KV)
       |  physical page-table upload
       v
PhaseBatchState.commit(past KV)
       |  global slot length = past KV
       v
timed continuation prefill / decode enqueue
```

즉 page bundle과 stable slot length는 production과 같은 allocator/commit API를 거친다. 다만 prefix KV 값 자체는
semantic prompt로 계산하지 않고 기존 device buffer 내용을 사용한다. attention kernel의 주소, page-table traversal,
연산량과 실행시간을 측정하기 위한 synthetic cost benchmark이며 정확성/출력 benchmark가 아니다. 실제 semantic
정확성은 real-request trace가 담당한다.

새 `--prefillPastKVLen`은 fixed-shape microbenchmark에서만 사용할 수 있다. continuous-load 또는 real trace와 함께
지정하면 실행 전에 거부한다.

## Page-pool feasibility

runner는 case 실행 전에 다음 bundle 수를 계산한다.

```text
prefill pages = P * ceil((prefill past + chunk) / 128)
decode pages  = D * ceil((decode past + 1) / 128)

prefill pages + decode pages <= 256
P + D <= 80
```

따라서 uniform D64×KV512나 D32×KV1024처럼 256 bundles를 넘는 shape는 실행하지 않는다. real trace에서는 같은
max context라도 row 길이가 ragged하므로 total page 수가 작아 실행 가능한 경우가 있다. controlled-only model이
long/bimodal replay에서 83.3%/79.6% coverage였던 이유다.

## 실행 범위

| 축 | 값 |
| --- | --- |
| Prefill phase | initial, continuation |
| Prefill batch | 1, 2, 4, 8 |
| Chunk | 64, 128 |
| Decode batch | 8, 16, 32, 64 |
| Past KV | 128, 512, 1024, 1536 |
| Requested / feasible | 256 / 172 |
| Warmup / samples | 5 / 20 per mode |
| Mode | sequential solo, independent overlap |

fixed-path raw kernel metadata도 함께 수정했다.

- `prefill_tokens`: per-row가 아니라 `P×chunk`
- `decode_context_tokens`: per-row가 아니라 `D×past`
- initial/continuation rows와 prefill past-KV 기록
- 측정 전 준비 호출의 zero-metadata kernel row 제거

## Controlled cost 결과

controlled dispatch는 sequential sample 하나를 prefill-only와 decode-only로 분리하고, concurrent sample은 direct
overlap row로 기록한다. 기존 `build_prefill_wavefront_cost_model.py` schema와 호환된다.

| Model | Prefill | Direct overlap | Decode | Real replay coverage |
| --- | ---: | ---: | ---: | ---: |
| Controlled only | 156 | 172 | 11 | short/balanced/decode 100%, long 83.3%, bimodal 79.6% |
| Hybrid controlled+real | 193 | 251 | 16 | 5 workload 모두 100% |

hybrid model은 controlled uniform point를 기반으로 하고 page-feasible real ragged dispatch를 fallback coverage로 함께
집계한다. 불가능한 uniform shape를 외삽하지 않는다.

### C64 대 C128

동일 P/D/past/phase에 두 chunk가 모두 있는 direct overlap pair는 86개다. 점수는 joint scheduler와 같이 계산했다.

```text
score = useful tokens /
        (prefill p95 + D/64 * decode slowdown p95)
```

결과는 C64 0승, C128 86승이다. C64에 가장 유리한 P8/D8/initial/KV128도 C128 대비 score가 약 8% 낮았다.
past가 길거나 D가 커질수록 대체로 C128 우세가 더 커졌다.

## Hybrid model 실제 5-workload smoke

기존 fresh fixed run과 hybrid joint 한 run을 비교했다. 출력은 모든 요청에서 동일했고 coverage miss와 C64 선택은
모두 0이었다.

| Workload | Throughput | TTFT med / p95 | TPOT med / p95 | E2E p95 |
| --- | ---: | ---: | ---: | ---: |
| short | -0.47% | -0.14% / +0.81% | +1.31% / +1.02% | +0.48% |
| balanced | **+0.40%** | -0.51% / -0.44% | -0.48% / -0.43% | -0.41% |
| decode-heavy | **+0.21%** | -0.29% / -0.22% | -0.23% / -0.17% | -0.22% |
| long-prefill | -0.11% | -0.10% / **-1.21%** | -0.81% / -0.99% | -0.39% |
| bimodal | -0.24% | +0.00% / **-1.40%** | +0.59% / -0.81% | +0.18% |

새 표는 기존 real-only 표보다 C64를 보수적으로 제거하면서도 3% performance gate를 유지했다.

## Mixed-load 3-run median

low 24 requests → 1,000 req/s burst 192 requests → 6.5초 gap → recovery 72 requests trace를 사용했다. fixed와
hybrid joint 모두 P8/D64, ragged packed prefill, output length 고정 조건이다.

| Overall metric | Fixed | Hybrid joint | 변화 |
| --- | ---: | ---: | ---: |
| Generated token/s | 1839.916 | 1839.991 | +0.004% |
| TTFT median / p95 | 232.79 / 2238.27ms | 236.33 / 2261.86ms | +1.52% / +1.05% |
| TPOT median / p95 | 12.749 / 17.197ms | 12.811 / 17.440ms | +0.48% / +1.41% |
| E2E median / p95 | 1649.80 / 3305.62ms | 1657.69 / 3318.64ms | +0.48% / +0.39% |

구간별 특징은 다음과 같다.

- low: TTFT median -2.89%, TPOT p95 -1.18%, E2E p95 -1.12%
- burst: TTFT p95 +0.94%, TPOT p95 +1.17%, E2E p95 +0.54%
- recovered: TTFT p95 -4.64%, TPOT p95 +0.19%, E2E p95 +0.38%

세 run에서 coverage miss 0, C64 선택 0, drain-mode dispatch 9였다. 부하 전환에도 3% gate 안이지만 fixed 대비
명확한 overall speedup은 없다. 따라서 hybrid joint도 experimental opt-in을 유지한다.

## 구현 위치

- `examples/llm/llm_phase_bench.cpp`
  - `--prefillPastKVLen`
  - synthetic paged ownership/length setup
  - continuation profile selection
  - fixed kernel metadata 보정
- `scripts/cosmos_reason2/run_controlled_prefill_shape_cost_suite.py`
  - engine/page feasibility filtering
  - resumable case execution
  - sequential/overlap normalization과 case summary
- 기존 `scripts/cosmos_reason2/build_prefill_wavefront_cost_model.py`
  - controlled dispatch와 real dispatch를 같은 scheduler schema로 집계

## Artifact와 다음 단계

- root: `.local/cosmos-reason2-2b/controlled-prefill-cost-20260819/`
- full sweep: `full-w5-i20/`
- controlled model: `full-w5-i20/cost-model.{json,csv}`
- hybrid model: `full-w5-i20/cost-model-hybrid.{json,csv}`
- mixed load: `mixed-fixed-r1..r3`, `mixed-joint-r1..r3`

다음 단계는 C64 threshold를 더 튜닝하는 것이 아니다. C128이 명확히 우세하므로 다음 우선순위는 phase CUDA graph
hit/miss 비용을 cost schema에 분리하는 것이다. graph capture가 없는 shape, warm graph hit, cold capture를 각각
측정해 score가 graph 상태를 고려하도록 해야 한다.
