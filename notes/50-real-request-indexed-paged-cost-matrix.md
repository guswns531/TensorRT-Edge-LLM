# Real-request indexed-linear/indexed-paged cost matrix

## 실행 목적

이 단계의 비교 대상은 같은 Cosmos text-only real-request trace를 다음 조합에 넣는 것이다.

| 축 | 값 |
| --- | --- |
| prefill cap | 1, 2, 4, 8 |
| decode cap | 1, 2, 4, 8, 16 |
| TensorRT context | `shared` serialized, `independent` concurrent |
| cache | `indexed-linear`, `indexed-paged` |
| prefill | fixed 128-token chunk |
| 측정 | request TTFT/TPOT/E2E, dispatch CUDA event, kernel-group CUDA event |

`shared`와 `independent`는 각각 별도 실행한다. 따라서 overlap 이득은 같은 trace와 같은 batch cap에서
`independent`와 `shared`의 makespan/TTFT/TPOT를 비교해 계산한다. CUDA primary context는 두 경우 모두
하나이며, `independent`만 prefill/decode TensorRT execution context와 I/O/workspace를 분리한다.

## 재현 명령

현재 checkout에서 사용할 real trace와 엔진은 다음과 같다.

```bash
python3 scripts/cosmos_reason2/run_real_request_kv_matrix.py \
  --bench build/examples/llm/llm_phase_bench \
  --source-trace notes/results/gemma4-llm-real-request-poisson-seed1-20260804/fixed128_p2d2_30rps/trace.json \
  --output-dir .local/cosmos-reason2-2b/real-request-kv-matrix \
  --engine indexed-linear=.local/cosmos-reason2-2b/engine-fp16-indexed-asym-p8-d16-i1024-kv2048 \
  --engine indexed-paged=.local/cosmos-reason2-2b/engine-fp16-paged-p8-d16-b80 \
  --prefill-batches 1 2 4 8 \
  --decode-batches 1 2 4 8 16 \
  --context-modes shared independent \
  --repeat-count 4 \
  --slot-count 16 \
  --past-kv-len 0 \
  --page-bundles 80 \
  --tokens-per-page 128 \
  --continue-on-error
```

`--repeat-count 4`는 기존 12-request trace를 48개 request로 확장해 decode BS8/16이 실제로
형성될 수 있게 한다. `--continue-on-error`는 GPU가 없는 호스트에서도 전체 case의 실행 상태를 남기기 위한 옵션이다.
`--past-kv-len 0`은 paged KV v1이 지원하지 않는 system-prompt cache reuse를 피하기 위한 real-trace
warmup 설정이다.
정상 GPU 실행에서는 이 옵션을 빼고 첫 실패에서 중단하는 것이 좋다. 모든 case는 동일한
`materialized-trace.json`을 사용하며, 엔진별 디렉토리 아래에 다음 파일을 만든다.

- `requests.csv`: request-level TTFT/TPOT/E2E와 실제 admission 상태
- `requests-dispatch.csv`: 선택된 prefill/decode batch, CUDA-event phase cost, overlap ratio와 실제 page-pool snapshot
- `kernel-groups.csv`: prefill/decode prepare, engine, cache commit, sample group cost
- `run.log`: 엔진별 실행 로그

상위 결과는 다음 파일이다.

- `request-summary.csv`
- `dispatch-cost-table.csv`
- `kernel-cost-table.csv`
- `page-pressure-model.csv`
- `status.csv`

## page-pool pressure와 admission

`page-pressure-model.csv`는 paged pool에 대해 보수적인 admission 정책을 계산한다. 각 request가
admitted 될 때 `ceil((prompt_tokens + max_output_tokens) / 128)` bundle을 미리 보유한다고 가정하고,
기존 request가 완료될 때까지 다음 request를 기다린다. 기록 항목은 다음과 같다.

- `modelled_peak_pressure`: `peak allocated bundles / page-bundles`
- `modelled_page_block_events`: pool 또는 stable slot이 부족해 completion까지 기다린 횟수
- `modelled_max_pending`: 모델상 대기 중인 active request 수

동시에 `requests.csv`에는 runtime callback이 관측한 `admission_available_slots`,
`admission_pending_queue_depth`, `admission_page_pool_*`가 기록된다. 즉 모델 pressure와 실제
stable-slot admission backpressure를 같은 request row에서 대조할 수 있다.

이는 현재 runtime이 이미 page pool에 대해 완전한 admission blocking을 수행한다는 뜻이 아니다.
현재 production facade의 실제 backpressure는 stable-slot pending queue이며, paged allocator의
transactional exhaustion은 아직 dispatch-level retry 정책으로 연결되지 않았다. 따라서 이 CSV를
통해 “이 workload와 page budget이면 필요한 pressure”와 “실제 `admission_status`에서 관측된 slot
backpressure”를 분리한다. 다음 구현 단계에서는 이 모델을 `availableBundles`/필요 bundle 수 preflight와
연결해 pool 부족을 pending admission 또는 작은 decode batch로 바꿔야 한다.

## 실제 GPU 실행 결과 (2026-08-11)

NVIDIA runtime이 연결된 TensorRT 26.06/CUDA 13.3 컨테이너에서 새로 빌드한
`llm_phase_bench`와 plugin으로 80개 case(2 cache × 2 context mode × 4 prefill cap ×
5 decode cap)를 모두 실행했다. `status.csv`의 80행이 모두 `return_code=0`이며 OOM 없이
trace가 종료됐다. 결과 디렉토리는 다음과 같다.

`.local/cosmos-reason2-2b/real-request-kv-matrix-20260811-rerun/`

아래 수치는 20개 batch 조합에 대한 `request-summary.csv`의 **scenario median-of-medians**다.
즉 각 조합의 request median/p95를 먼저 계산한 뒤 20개 조합의 중앙값을 취했다.

| KV | context | TTFT med/p95 (ms) | TPOT med/p95 (ms) | E2E med/p95 (ms) | generated tok/s med |
| --- | --- | ---: | ---: | ---: | ---: |
| indexed-linear | shared(serialized) | 81.352 / 411.876 | 43.579 / 50.011 | 922.710 / 1475.094 | 385.08 |
| indexed-paged | shared(serialized) | 82.492 / 411.809 | 43.693 / 50.032 | 925.920 / 1478.058 | 384.32 |
| indexed-linear | independent(concurrent) | 24.661 / 81.514 | 25.315 / 38.191 | 559.204 / 940.210 | 444.87 |
| indexed-paged | independent(concurrent) | 24.286 / 88.860 | 25.629 / 38.527 | 565.456 / 949.272 | 442.53 |

동일 context mode에서 indexed-paged/indexed-linear의 평균 회귀는 TTFT median
`+1.36% (shared)`, `+0.10% (independent)`, TPOT median `+0.43%/+0.54%`, E2E median
`+1.12%/+0.72%`였다. Kernel-group 기준으로는 prefill engine median이
`14.829→14.822 ms (shared)`, `17.464→17.429 ms (independent)`, decode engine median이
`6.307→6.348 ms`, `6.310→6.341 ms`로 모두 1% 이내였다. Request-level TTFT/E2E p95의
일부 조합은 trace queue noise로 3%를 넘었으므로, “3% gate”는 scheduler lookup에
kernel-group median/p95를 우선 사용하고 request-tail은 별도 admission 지표로 보존한다.

### Sequential 대 overlap

`shared`에서는 prefill+decode dispatch의 overlap ratio가 항상 0이었다. `independent`에서는
두 phase가 동시에 포함된 dispatch에서 overlap ratio 중앙값이 indexed-linear `0.334`,
indexed-paged `0.331` (p95 `0.341`/`0.338`)이었다. independent로 바꾸면 shared 대비
scenario median-of-medians가 TTFT 약 `-39~−40%`, TPOT 약 `-25%`, E2E 약 `-25%` 개선됐다.
이는 CUDA primary context는 공유하되 TensorRT execution context/I/O/workspace를 분리한
현재 설계의 overlap 효과다.

### Kernel-group cost table

전체 원시 표는 `kernel-cost-table.csv`, dispatch 표는 `dispatch-cost-table.csv`에 있다.
주요 group의 20개 case median-of-medians는 다음과 같다.

| context | group | indexed-linear med/p95 (ms) | indexed-paged med/p95 (ms) |
| --- | --- | ---: | ---: |
| shared | prefill_engine | 14.829 / 16.295 | 14.822 / 16.160 |
| shared | decode_engine | 6.307 / 9.796 | 6.348 / 9.838 |
| independent | prefill_engine | 17.464 / 19.301 | 17.429 / 19.217 |
| independent | decode_engine | 6.310 / 8.895 | 6.341 / 8.786 |

`prefill_prepare`, `prefill_cache_commit`, `decode_prepare`, `decode_sample`도 모든 mode에서
sub-millisecond이며 두 cache 간 차이는 측정 노이즈 범위다.

## Independent context를 주 경로로 볼 때의 해석

실제 서비스 목표가 CUDA primary context 하나 + independent TensorRT execution context 두 개라면
아래 20개 independent case를 주 성능표로 사용하고 `shared`는 회귀 확인용으로만 유지한다.

| prefill cap | decode cap | linear TTFT med (ms) | paged TTFT med (ms) | linear TPOT med (ms) | paged TPOT med (ms) | paged tok/s / linear |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1210.5 | 1217.8 | 126.8 | 127.3 | 0.995 |
| 1 | 2 | 303.8 | 310.2 | 64.8 | 65.1 | 0.996 |
| 1 | 4 | 23.5 | 23.7 | 25.3 | 25.4 | 0.993 |
| 1 | 8 | 23.6 | 23.1 | 10.0 | 10.0 | 1.000 |
| 1 | 16 | 24.7 | 22.9 | 10.0 | 10.0 | 1.001 |

prefill cap 2/4/8의 값은 위 cap 1과 거의 같지만, 이는 prefill cap이 실제로 포화되지 않았기
때문이다. independent trace에서 관측된 prefill dispatch는 모든 cap에서 `BS=1` 56회였다.
decode cap 16도 이 trace에서는 실제 최대 decode batch 9까지만 형성됐다. 따라서 이 실험으로
`prefill BS=8` 또는 `decode BS=16`의 포화 비용을 결론 내릴 수는 없고, 더 높은 arrival rate와
더 많은 동시 request/output length를 넣은 saturation trace가 필요하다.

현재 trace에서 decode cap을 1→2→4→8로 키우는 효과는 크다(TPOT 약 127→65→25→10 ms).
8→16은 이미 약 10 ms에서 plateau이므로 scheduler의 기본 decode cap은 8, reserve cap은
16으로 두는 것이 합리적이다. prefill은 실제 BS1에서 약 23 ms 수준이며, prefill queue가
쌓일 때만 cap 2/4/8을 사용하도록 admission window를 별도로 조정해야 한다.

Independent 기준 indexed-paged의 평균 차이는 TTFT median `+0.10%`, TTFT p95 `+1.38%`,
TPOT median `+0.54%`, TPOT p95 `+0.28%`, E2E median `+0.72%`, E2E p95 `+0.29%`,
generated token/s `-0.27%`다. 따라서 independent 운영 기준에서는 paged KV의 실행 비용은
사실상 동일하고, 선택 이유는 성능 향상보다 stable slot eviction과 page-pool memory 효율이다.
TTFT p95의 최악 조합은 `p2_d4`에서 `+17.4%`였으나 median과 TPOT/E2E tail에는 재현되지 않아
queue arrival noise로 분리해 기록한다.

### Page-pool pressure와 admission backpressure

`page-pressure-model.csv`는 80-bundle pool, 128-token/page, 48-request trace에 대해
요청 전체(prompt+max output)를 보수적으로 미리 보유한다고 가정한 모델이다. 두 engine 모두
modelled peak pressure 중앙값 `0.2875~0.29375`, 최대 `0.30`, page-block event 중앙값 `15.5`,
최대 `31`이었다. 실제 stable-slot admission queue의 관측 pending은 중앙값 `5.5`, 최대
`25`였다. indexed-linear 행의 page pressure는 실제 linear allocator가 page를 쓰는 뜻이
아니라 동일 workload의 paged 요구량을 비교하기 위해 같이 계산한 가상값이다.

현재 callback의 `admission_page_pool_*`은 admission 시점 snapshot이고, page allocation은
이후 prefill/decode prepare에서 일어나므로 대부분 0으로 보인다. 다음 admission 정책 단계에서
`availableBundles()`를 reservation 전에 검사하고, 부족하면 pending admission 또는 작은
decode batch로 되돌리는 것이 필요하다.

### 실행 중 발견해 수정한 paged 경로 버그

초기 paged smoke는 decode warmup에서 illegal address가 났다. Compute Sanitizer가
`applyRopeWriteKV`가 page `-1`에 쓰는 것을 보고, 고정형 decode 경로가
`PhaseBatchState::prepare(..., decode=false)`를 호출해 첫 decode token page를 예약하지 않는
문제를 찾았다. `decode=true`로 수정한 뒤 paged smoke 6/6과 전체 matrix 40/40이 통과했다.
또한 paged plugin은 TensorRT binding shape를 max batch로 유지하면서 helper에는 active logical
batch view를 전달하도록 수정했다. `--past-kv-len 0`은 paged v1의 system-prompt cache reuse
제약을 피하기 위한 조건이다.

## 해석 규칙

1. `independent`의 latency가 `shared`보다 낮고 overlap ratio가 양수이면 같은 CUDA primary context 안의
   독립 TensorRT context overlap 이득으로 해석한다.
2. `prefill` batch가 커질수록 decode kernel cost와 TTFT p95가 증가하는지 확인한다. prefill은
   decode보다 작은 cap을 선택하는 근거가 된다.
3. paged의 kernel-group cost가 linear보다 3% 이내이고 VRAM pressure가 크게 낮으면, scheduler는
   stable ownership을 유지한 채 larger decode batch를 시도할 수 있다.
4. `modelled_peak_pressure`가 1에 가까운 case에서는 latency 숫자만 비교하지 않고, admission delay와
   page-block event를 함께 기록한다.
