# Cosmos v0.10 host-gap 제거와 v0.9.1 fresh parity

## 결론

v0.10 phase IPC 경로의 고정 1ms poll sleep, sampling CUDA event 반복 생성, 공유 host token staging, FIFO ticket
head-of-line blocking을 제거했다. Balanced fixed-output 처리량은 adaptive admission 기준 3,905.7에서 4,346.8
token/s로 11.3% 증가했다. 새 engine-specific overlap cost를 연결한 최종값은 4,364.3 token/s다.

같은 checkpoint에서 v0.9.1 native ONNX와 P8/D64 engine을 다시 `export -> build -> inference`하고 동일 HTTP/SSE
client와 fixed-output trace로 세 번 측정했다. v0.9.1은 4,468.1 token/s다. 현재 v0.10의 처리량 격차는
`-2.32%`로 3% gate 안이다.

| backend | token/s | TTFT med/p95 | TPOT med/p95 | E2E med/p95 | peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0.9.1 native P8/D64 | **4,468.1** | **1,692/3,959 ms** | 16.14/18.46 ms | **3,242/5,021 ms** | 8,419 |
| v0.10 host-opt | 4,346.8 | 1,814/4,334 ms | 16.16/17.96 ms | 3,334/5,243 ms | **8,087** |
| v0.10 host-opt + cost-aware | 4,364.3 | 1,807/4,291 ms | **16.15/17.57 ms** | 3,293/5,224 ms | **8,087** |
| vLLM fixed-output | 4,234.4 | 1,859/4,389 ms | 17.06/17.89 ms | 3,423/5,377 ms | 7,959 |

v0.10 cost-aware는 vLLM보다 처리량 3.1% 높고 TPOT median/p95도 낮다. v0.9.1과 비교하면 peak memory를
332MiB 줄이고 TPOT p95를 4.8% 개선하지만, TTFT median/p95는 6.8/8.4% 느리다. 따라서 처리량 parity는
달성했지만 모든 latency dimension의 3% parity는 아직 아니다.

## Host gap 원인과 수정

수정 전 balanced representative run은 553 dispatch, CUDA makespan 약 5.37초, client wall 약 6.37초였다.
`wall - CUDA makespan`이 약 1.00초였다.

### 고정 poll sleep

IPC loop가 progress 여부와 무관하게 매 iteration 다음을 실행했다.

```text
sleep_for(1 ms)
```

이를 server/input/output progress가 없을 때만 `yield()`하는 방식으로 변경했다. CUDA event는 계속 non-blocking query하고
GPU stream을 host에서 synchronize하지 않는다.

### Sampling event와 host buffer

기존에는 매 sampling batch마다 CUDA event를 생성·파괴하고 prefill/decode별 하나의 pinned host tensor를 다음 ticket이
재사용했다. 새 slot pool은 다음을 보장한다.

- ticket별 독립 pinned host token buffer
- ticket별 event ownership
- 완료 후 slot/event 재사용
- server shutdown에서도 release callback으로 pool 반환
- ready ticket 전체를 한 poll에서 수집
- 앞 ticket이 아직 미완료여도 뒤의 ready ticket을 처리

Balanced에서는 event slot 2개만 생성해 run당 약 560회 재사용했다.

## Host gap 결과

| 항목 | 수정 전 | 수정 후 |
| --- | ---: | ---: |
| balanced host gap | 약 1,000 ms | 277~324 ms |
| balanced token/s | 3,905.7 | 4,346.8 |
| 변화 | - | **+11.3%** |
| sampling event slots | 매 batch 생성 | 2 |
| event reuse | 없음 | 약 560회/run |

세 host-opt run 처리량은 4,328.8 / 4,346.8 / 4,407.4 token/s 범위이며 output은 매번 정확히 24,960이다.

## 다른 workload

| workload | host 최적화 전 | host 최적화 후 | 변화 | vLLM | 현재/vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 1,718.9 | 1,894.8 | +10.2% | 1,998.7 | -5.2% |
| balanced | 3,905.7 | 4,364.3 | +11.7% | 4,234.4 | +3.1% |
| decode-heavy | 4,477.4 | 5,170.8 | +15.5% | 4,797.9 | +7.8% |

Decode-heavy는 dispatch가 많아 고정 poll 비용 제거 효과가 가장 크다. host-opt run은 1,466 dispatch, D64 1,020회,
CUDA makespan 13.82초, host gap 0.67초다. TPOT median/p95는 12.43/12.84ms다.

## 새 overlap cost와 A/B

Host 최적화 이후 비용은 이전 engine/runtime table과 분리했다. 새 generator가 `PHASE_METRIC`의 batch, chunk,
prefill past-KV, decode context, CUDA-event timing을 bucket으로 집계한다.

- decode points: 20
- prefill points: 31
- direct-overlap points: 23
- engine SHA-256: `b9f87df7...09d4c5cf`

무조건 overlap trace는 3,510.0 token/s로 느렸다. 새 cost model을 throughput-balanced scheduler에 연결하면
3회 중앙값 4,364.3 token/s로 기본 host-opt보다 0.4% 높고 TPOT p95는 2.2% 낮다. 개선 폭은 작지만 회귀가 없어
engine fingerprint가 일치할 때만 사용하는 opt-in 후보로 유지한다.

Cost artifact는 `notes/results/cosmos-v010-p8d64-hostopt-cost-20260820.json`이다.

## v0.9.1 fresh 계약

- source/runtime: root branch, Edge-LLM `0.9.1` 기반
- ONNX: root exporter가 Cosmos checkpoint에서 fresh 생성
- engine: root builder/plugin, P8/D64, slots80, page bundles256, tied FP16 head
- warmup past KV: 128; D64×KV512는 page256을 초과하므로 사용하지 않음
- trace SHA: `290d34061a...49d6538`
- fixed output: 288 requests, prompt 25,872, output 24,960
- graph: off
- HTTP/SSE client: v0.10/Current와 동일

v0.9.1 세 run은 4,468.1 / 4,446.1 / 4,468.4 token/s이고 token trace hash도 세 번 동일했다. v0.10 online
batch trajectory는 run마다 달라 token hash는 다르지만 모든 request와 output token count는 같다. cross-version exact
token identity는 이 성능 gate의 증거가 아니며 별도 controlled BS1 gate가 필요하다.

## 남은 병목

1. v0.10 TTFT p95가 v0.9.1보다 8.4% 느리다.
2. short는 vLLM보다 5.2%, 과거 v0.9.1 headline보다 여전히 크게 낮다.
3. host gap은 줄었지만 decode-heavy에서 약 0.67초가 남는다.
4. production event loop는 busy-yield 대신 CUDA event/IPC input을 함께 깨우는 wait primitive가 필요하다.
5. CUDA Graph는 runtime cold capture가 아니라 사전 shape priming 후 다시 평가해야 한다.

## Artifact

- v0.10 host-opt: `.local/cosmos-reason2-2b/host-gap-results-20260820/`
- v0.9.1 native ONNX/engine/results: `.local/cosmos-reason2-2b/v091-fresh-20260820/`
- overlap cost generator: `scripts/cosmos_reason2/build_phase_cost_model_from_gateway.py`
- overlap cost: `notes/results/cosmos-v010-p8d64-hostopt-cost-20260820.json`

