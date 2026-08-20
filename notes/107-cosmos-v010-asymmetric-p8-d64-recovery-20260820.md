# Cosmos v0.10 asymmetric P8/D64 성능 복구

## 결론

v0.9.1에서 사용하던 비대칭 TensorRT profile과 undercommitted indexed-paged KV pool을 v0.10 forward-port에
이식했다. `maxBatch=80`, prefill profile P8, decode profile D64, KV2048, 256 page bundles의 tied engine이
`export -> build -> inference`를 통과했다.

동일 balanced fixed-output HTTP trace의 3회 중앙값은 다음과 같다.

| 구성 | token/s | TTFT med/p95 | TPOT med/p95 | E2E med/p95 | ready/peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 이전 v0.10 B32 P8/D32 | 1,921.4 | 5,612/11,112 ms | 15.66/16.88 ms | 6,842/12,188 ms | 8,459/8,465 MiB |
| v0.10 asymmetric P8/D64 | **3,693.3** | **2,337/5,066 ms** | **15.29/16.64 ms** | **3,617/6,114 ms** | **8,081/8,087 MiB** |

변화는 처리량 `+92.2%`, TTFT median `-58.4%`, TPOT median `-2.4%`, E2E median `-47.1%`, peak memory
`-378MiB`다. 같은 fixed-output vLLM 4,234.4 token/s와의 격차도 `-54.6%`에서 `-12.8%`로 줄었다.

v0.9.1의 EOS-enabled 대표값 4,413.4 token/s보다는 16.3% 낮다. 이는 완전히 같은 측정 계약은 아니지만,
남은 차이의 위치는 dispatch 분석으로 좁혀졌다. v0.10 kernel 자체는 이미 controlled 비교에서 더 빨랐고, 현재
남은 차이는 phase overlap/admission 궤적과 graph lifecycle이다.

## 구현한 계약

### 비대칭 profile

builder에 다음 옵션을 추가했다.

```text
--maxBatchSize 80
--maxPrefillBatchSize 8
--maxDecodeBatchSize 64
--maxKVCacheCapacity 2048
--maxKVPoolPages 256
--allowKVPoolUndercommit
```

Prefill의 `context_lengths`, page table, RoPE, embedding, last-token, PLE/deepstack profile은 P8까지만 만들고,
decode binding은 D64까지 만든다. global max batch 80은 stable slot/page-table address space 상한으로 남는다.

### Undercommitted page pool

최악의 이론상 profile occupancy는 `80 * ceil(2048/128) = 1,280 pages`지만 실제 물리 pool은 256 pages다.
이 계약은 다음 안전장치가 있을 때만 활성화된다.

- builder에서 명시적 `--allowKVPoolUndercommit` 필요
- 초기 global page table은 identity mapping이 아니라 빈 row로 시작
- `StableKVPageManager`가 leased row에만 유효 page ID를 materialize
- public `LLMInferenceRuntime::handleRequest()`는 undercommitted engine을 거부
- allocator-backed independent phase server만 실행 허용
- phase-local P8/D64 page table은 global active-row 상한 이하에서 별도 크기를 사용

### Phase-sized I/O

global B80 크기로 두 phase의 tensor를 모두 할당하지 않는다.

```text
prefill I/O: [P8, max input 1024, hidden]
decode I/O:  [D64, 1, hidden]
```

Cosmos deepstack도 prefill `[8,1024,2048]`, decode `[64,1,2048]`로 분리된다. 이 변경과 작은 TensorRT tactic
덕분에 D64 engine이 B32 engine보다 오히려 378MiB 적은 peak memory를 사용했다.

## 단계별 복구 결과

모든 행은 288 requests, prompt 25,872, generated output 24,960, fixed chunk128, graph-off 단일 smoke다.

| 단계 | 실제 max D | token/s | 의미 |
| --- | ---: | ---: | --- |
| B32 기존 | 32 | 1,921 | forward-port 시작점 |
| D64 engine, cost table D32까지만 | 32 | 2,846 | engine만 키워도 P profile/I/O 절감 효과 |
| forced static D64, in-flight80 | 64 | 3,375 | D64 상한 활성화 |
| dynamic D64 cost 확장, in-flight80 | 64 | 3,262 | D64+D16 tail round가 남음 |
| dynamic D64, in-flight64 | 64 | **3,759** | D16 tail 제거, 단일-run 최고 |
| 위 구성 + cold CUDA Graph | 64 | 3,657 | capture 비용으로 -2.7% |

최종 3회 graph-off 결과는 3,749.6 / 3,693.3 / 3,637.8 token/s다. 세 run에서 requested/generated output은
모두 24,960으로 같았고 peak memory도 8,087MiB로 같았다.

## 왜 in-flight64가 in-flight80보다 빨랐나

in-flight80에서 decode queue가 80 rows가 되면 한 token round가 다음처럼 갈라졌다.

```text
D64 약 8.6ms + D16 약 6.6ms = 약 15.2ms/round
```

in-flight64에서는 대부분의 steady-state decode가 D64 한 번으로 끝난다. 최종 run의 decode GPU 합은 약 3.95초,
prefill GPU 합은 약 1.75초였다. D64 cohort가 충분히 길게 유지되어 TPOT median이 15.29ms까지 내려왔다.

stable slots 80은 계속 유용하지만 “leased request 80개를 매 token마다 모두 공정하게 round-robin”하는 것은 이
workload에서 비효율적이다. admission slot 수와 매 round active decode cohort 크기를 분리해야 한다.

## Overlap 실험

무조건 1,024-token overlap을 허용하면 overlap dispatch는 18.7%까지 늘었지만 decode batch가 35/45 또는
64/16으로 갈라져 3,260~3,393 token/s에 머물렀다. 보존된 v0.9.1 direct-overlap cost JSON을 v0.10 runtime에서
읽는 경로도 연결했지만, v0.9.1 engine에서 생성한 table은 새 engine의 admission 궤적을 복원하지 못해
3,361 token/s였다.

따라서 다음 cost table은 새 P8/D64 engine fingerprint로 다시 생성해야 한다. “overlap 비율을 높이는 것”이 목표가
아니라 D64 cohort를 유지하면서 GPU makespan에서 실제로 숨길 수 있는 prefill만 선택해야 한다.

## CUDA Graph

현재 coordinator는 first-seen shape를 runtime 중 capture한다. balanced 한 lifecycle에서는 cold capture 비용을
amortize하지 못해 graph-off 3,759 대비 graph-on 3,657 token/s로 2.7% 느렸다. v0.9.1의 성능 결과는 workload
shape를 미리 priming한 뒤 측정했으므로 아직 동일 조건이 아니다.

다음 graph gate는 측정 요청이 도착하기 전에 P/D frequency profile을 replay하고, capture 완료 후 scheduler history와
KV ownership을 초기화한 상태에서 시작해야 한다.

## Decode cohort 실험

`enableDecodeCohortBatching` opt-in도 추가했다. 선택된 decode request ID를 유지하고 완료된 row만 대기 요청으로
교체하는 기능이며 단위 테스트를 통과한다. 하지만 현재 async admission에서는 cohort가 충분히 prefilled되기 전
output-length wave가 끝나는 구간이 있어, aggressive overlap과 조합했을 때 3,306 token/s에 머물렀다. 따라서
기본값으로 채택하지 않고 다음 rolling-admission 설계의 기반으로만 유지한다.

## 검증

- asymmetric/undercommit/scheduler focused tests: 144/144 pass
- 추가 decode-cohort/page-table tests: 16/16 pass
- P8/D64 TensorRT engine build: pass
- tied-engine materialization SHA/CRC contract: pass
- independent phase semantic inference: pass
- fixed-output balanced trace: 288/288 requests, 24,960/24,960 tokens, 3회 pass
- GPU headroom: 약 1.78GiB

## 다음 단계

1. 새 engine 전용 P1/2/4/8 × D16/32/48/64 overlap cost table을 생성한다.
2. prefill admission과 decode cohort를 분리해 D64 active set 옆의 16 stable slots를 prefill staging으로 사용한다.
3. D64 cohort가 비면 즉시 prefilled request로 교체하고, 아직 prefill 중인 request 때문에 D16 tail round를 만들지 않는다.
4. graph frequency profile을 측정 전에 priming하고 graph-off와 3회 재비교한다.
5. 동일 fixed-output 조건의 v0.9.1 engine을 fresh 실행해 남은 16%를 정확히 재산정한다.

## Artifact

- engine: `.local/cosmos-reason2-2b/asymmetric-20260820/`
- results: `.local/cosmos-reason2-2b/asymmetric-results-20260820/`
- selected 3-run result: `final-dynamic-fixed128-p8d64-inflight64/`

