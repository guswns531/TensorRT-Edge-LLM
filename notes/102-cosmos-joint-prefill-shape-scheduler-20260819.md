# Cosmos joint prefill batch/chunk scheduler

## 결론

Cosmos-Reason2-2B의 independent prefill/decode TensorRT context 경로에 `(prefill batch, chunk)`를 함께 고르는
cost-aware scheduler를 opt-in으로 연결했다. fixed-128이나 단순 queue-pressure chunk 정책과 달리, 같은 queue
snapshot에서 profiled `64/128` chunk와 가능한 prefill batch를 모두 평가한다.

동일 tied embedding/LM-head max-128 engine, P8/D64, graph-off real-request 5-workload를 fresh process로 세 번씩
비교한 최종 권장 설정은 decode penalty weight 1.0, prefill drain backlog 1,024 tokens다.

- balanced generated token/s `+0.52%`, TTFT p95 `-1.20%`, TPOT p95 `-0.73%`
- long-prefill generated token/s `+0.43%`, TTFT p95 `-1.76%`, TPOT p95 `-1.54%`
- short/decode-heavy/bimodal 처리량은 `-0.34/-0.15/-0.14%`
- 전체 latency 최악 회귀는 bimodal TPOT median `+0.62%`, short E2E p95 `+0.37%`
- 모든 joint dispatch의 cost coverage miss는 0
- 각 fixed/joint 반복에서 output token, finish reason, greedy output text가 요청별로 같았다.

성능 개선은 작지만 이전 threshold-only dynamic chunk의 long/bimodal 회귀를 제거하고 3% gate 안에서 일부 workload의
TTFT/TPOT을 동시에 개선했다. 기본값은 여전히 fixed이며 이 정책은 engine-specific cost table이 있을 때만 켜는
experimental opt-in이다.

## 왜 batch와 chunk를 함께 골라야 하나

`P8×64`와 `P4×128`은 모두 512 prefill tokens지만 실행 특성이 다르다.

```text
same useful tokens
       |
       +-- P8 x C64  -> 많은 row, 짧은 interference window
       |
       +-- P4 x C128 -> 적은 row, 긴 interference window

decode batch, past KV, TensorRT tactic에 따라 어느 쪽이 좋은지가 달라진다.
```

이전 구현은 chunk를 queue pressure로 먼저 정한 뒤 그 chunk 안에서 batch만 골랐다. 따라서 cost table에 C64/C128이
모두 있어도 서로 직접 비교할 수 없었다. 새 경로는 productive bucket에서 두 축을 하나의 후보 공간으로 평가한다.

## 선택식

일반 상태의 후보 점수는 다음과 같다.

```text
decode queue pressure = min(decode queued / max decode batch, 1)

effective cost =
    profiled prefill p95
  + decode penalty weight
      * decode queue pressure
      * profiled decode slowdown p95
  + configured host/enqueue cost

score = useful prefill tokens / effective cost
```

decode queue가 D1 수준일 때 slowdown을 100% 벌점으로 주면 P8을 P2/P4로 과도하게 줄였다. queue occupancy를 곱해
빈 decode queue에서는 prefill drain을 유지하고 D64에 가까울 때만 slowdown을 강하게 반영한다.

prefill remaining tokens가 1,024 이상이면 drain mode로 전환한다. 이때 TPOT/debt feasibility를 통과한 후보 중
`P×chunk`가 가장 큰 shape를 먼저 고르고 같은 progress에서 점수를 비교한다. long/bimodal backlog에서 real-request
sample p95 노이즈가 P1/P2를 과도하게 선호해 queue drain을 늦추는 현상을 막는다.

## 안전 경계

joint search는 기존 bucket selector가 가장 큰 productive candidate를 고른 경우에만 실행한다.

- 117/109/33-token처럼 한 turn에 끝나는 completion bucket은 chunk를 다시 고르지 않는다.
- completion bucket의 batch도 기존 fixed-compatible 최대값을 유지한다.
- final tail은 `--adaptiveChunkSplitCompletion`을 명시하지 않는 한 분할하지 않는다.
- profiled shape가 없으면 기존 fixed 선택으로 돌아간다.
- TPOT hard guard를 함께 쓸 때 모든 후보가 unsafe하면 prefill을 보내지 않고 decode-only로 전환한다.
- 기능을 켜려면 dynamic prefill, bounded adaptive candidates, scheduler cost JSON이 모두 필요하다.

초기 구현은 completion P4×117보다 productive P2×128을 먼저 보내 short 처리량을 약 8% 낮췄다. 첫 수정 후에도
33-token completion bucket에서 legacy dynamic prefill이 P4를 P3으로 줄여 tail을 늘렸다. 위 두 경계를 추가한 뒤
fresh fixed 대비 short 회귀가 `-0.34%`로 줄었다.

## Cost table

보존된 동일 tied max-128 real-request CUDA-event dispatch와 activation stress trace로 packed cost table을 다시
생성했다.

| 항목 | coverage |
| --- | --- |
| Prefill batch | P1--P8 |
| Productive chunk | C64, C128 |
| Concurrent decode bucket | D0, D8, D16, D32, D64 |
| Prefill past-KV bucket | 0, 128, 512, 1024 |
| Points | prefill 67, direct overlap 83, decode 15 |

이 표는 모든 Cartesian product가 아니라 실제 request trajectory에서 관측된 conservative upper bucket이다. 최종
5-workload joint 실행에서는 coverage miss가 없었다. 다른 model, engine build, tied-weight layout, P/D cap 또는 CUDA
graph 설정에 그대로 재사용하면 안 된다.

## 3-run median

조건은 FP16 Cosmos-Reason2-2B, tied embedding/LM-head, indexed-paged FP16 KV 256 bundles, 80 slots, packed prefill,
P8/D64, max chunk 128, token budget 1,024, independent TensorRT contexts, graph off다. latency 변화의 음수는 개선이다.

| Workload | Fixed / joint token/s | Throughput | TTFT med / p95 | TPOT med / p95 | E2E med / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 1954.16 / 1947.50 | -0.340% | -0.162% / -0.186% | +0.355% / -0.000% | -0.051% / +0.371% |
| balanced | 3955.39 / 3976.01 | **+0.521%** | **-1.878% / -1.200%** | **-1.002% / -0.727%** | +0.034% / **-1.006%** |
| decode-heavy | 4481.45 / 4474.87 | -0.147% | +0.224% / +0.170% | +0.184% / +0.184% | +0.215% / +0.148% |
| long-prefill | 1254.24 / 1259.62 | **+0.429%** | **-0.638% / -1.764%** | **-1.245% / -1.535%** | **-0.627% / -0.940%** |
| bimodal | 1823.26 / 1820.70 | -0.140% | -0.141% / **-1.571%** | +0.619% / **-1.039%** | +0.306% / +0.031% |

세 joint run 전체의 productive C64 선택은 balanced 1회, decode-heavy 1회, 나머지 workload 0회였다. 현재 cost
curve에서 성과 대부분은 작은 chunk를 자주 쓰는 효과가 아니라 다음 세 가지에서 나온다.

1. queue가 건강한 productive bucket에서 cost-aware P를 고른다.
2. completion bucket은 fixed batch로 빠르게 끝낸다.
3. 큰 backlog는 P×chunk progress를 우선해 drain한다.

따라서 이 결과를 “C64가 C128보다 빠르다”는 증거로 해석하면 안 된다. 오히려 현재 engine에서는 C64가 필요한
구간이 매우 드물다는 근거다.

## 구현 위치

- `cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}`
  - bounded `(P, chunk)` enumeration
  - cost/decode-pressure score
  - backlog drain regime
  - completion bucket 보존과 fallback
- `cpp/runtime/scheduling/phaseDispatchWorker.cpp`
  - candidate count, score, drain-mode metric 전달
- `examples/llm/llm_phase_bench.cpp`
  - joint policy/penalty/drain CLI
  - dispatch CSV 계측
- `scripts/cosmos_reason2/run_real_request_kv_matrix.py`
  - real-request harness 연결
  - `prefill-shape-table.csv` 집계
- `scripts/cosmos_reason2/replay_prefill_shape_policy.py`
  - 저장된 dispatch snapshot에 cost policy를 재적용하는 local decision replay
- `unittests/phaseQueueSchedulerTest.cpp`
  - decode interference trade-off, throughput mode, drain mode, completion 보호와 config gate

## Replay의 의미와 한계

decision replay는 각 recorded dispatch에서 어떤 P/C를 골랐을지만 비교한다. 선택이 바뀐 뒤의 queue residence,
admission, page pressure와 KV frontier를 다시 시뮬레이션하지 않으므로 E2E latency 예측기가 아니다. replay는 나쁜
threshold를 GPU 실행 전에 제거하는 용도이고, 최종 판단은 반드시 real TensorRT trace로 한다.

## Artifact와 다음 단계

- root: `.local/cosmos-reason2-2b/joint-prefill-20260819/`
- cost model: `cost-existing.json`, `cost-existing.csv`
- replay: `replay-*-drain1k.{json,csv}`
- GPU A/B: `gpu-<workload>-fixed[-rN]`, `gpu-<workload>-joint[-rN]`
- long/bimodal final: `gpu-*-joint-drain1k[-rN]`

다음 단계는 graph-off real trace에서 추출한 표를 더 복잡하게 튜닝하는 것이 아니다. controlled microbenchmark로
P1/2/4/8 × C64/128 × D8/16/32/64 × past-KV 128/512/1024/1536을 채우고, CUDA graph hit/miss 비용을 별도 필드로
분리한다. 그 뒤 mixed-load low/burst/recovery trace에서 threshold 1,024가 부하 전환에도 안정적인지 검증한다.
