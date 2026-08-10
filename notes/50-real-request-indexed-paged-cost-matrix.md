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
  --page-bundles 80 \
  --tokens-per-page 128 \
  --continue-on-error
```

`--repeat-count 4`는 기존 12-request trace를 48개 request로 확장해 decode BS8/16이 실제로
형성될 수 있게 한다. `--continue-on-error`는 GPU가 없는 호스트에서도 전체 case의 실행 상태를 남기기 위한 옵션이다.
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

## 현재 실행 상태

이 checkout의 현재 호스트에는 `/dev/nvidia*`가 없고 `nvidia-smi`가 NVIDIA driver 통신에 실패한다.
또한 host linker에는 TensorRT runtime `libnvinfer.so.11`이 없어 현재 build binary가 preflight 단계에서
실행되지 않는다.
따라서 새 40개 batch/context case × 2 cache engine의 CUDA median/p95는 아직 수집할 수 없다.
기존 GPU artifact로 확인된 기준은 다음과 같다.

- indexed-linear BS1: prefill 14.969 ms, decode median 6.2368 ms, peak VRAM 9,294 MiB
- indexed-paged BS1: prefill 14.907 ms, decode median 6.2555 ms, peak VRAM 6,828 MiB
- indexed-linear 대비 indexed-paged decode median 회귀: 0.30%
- teacher-forced logits: BS2 worst cosine 0.99992612, BS4 worst cosine 0.99994773

GPU가 다시 노출되면 위 명령이 요청한 전체 matrix와 pressure table을 생성한다. 결과 gate는
각 cache/context/batch case의 median과 p95를 보존하고, paged가 indexed-linear 대비 3%를 넘는
회귀를 보이면 해당 조합을 phase scheduler cost lookup에 넣지 않는 것이다.

## 해석 규칙

1. `independent`의 latency가 `shared`보다 낮고 overlap ratio가 양수이면 같은 CUDA primary context 안의
   독립 TensorRT context overlap 이득으로 해석한다.
2. `prefill` batch가 커질수록 decode kernel cost와 TTFT p95가 증가하는지 확인한다. prefill은
   decode보다 작은 cap을 선택하는 근거가 된다.
3. paged의 kernel-group cost가 linear보다 3% 이내이고 VRAM pressure가 크게 낮으면, scheduler는
   stable ownership을 유지한 채 larger decode batch를 시도할 수 있다.
4. `modelled_peak_pressure`가 1에 가까운 case에서는 latency 숫자만 비교하지 않고, admission delay와
   page-block event를 함께 기록한다.
