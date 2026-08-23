# Cosmos serving 개선 우선순위 1~4 구현과 단계별 성능

## 결론

요청한 네 단계를 모두 구현하고 각 단계에서 동일 HTTP/SSE trace를 재실행했다. 가장 명확한 이득은 short 응답
경로에서 나왔다. D64 cohort와 online cost는 decode-heavy latency를 소폭 개선했지만 처리량 변화는 측정 노이즈
범위였다. 추가 frequency graph profile은 회귀하여 채택하지 않았다.

최종 선택은 다음과 같다.

- native token/completion callback과 batched output pump: 채택
- D64 stable cohort + 나머지 16 stable slot의 P8 staging: 채택
- confidence 기반 online decode cost: throughput mode에서만 채택
- frequency/LRU graph cache 통계와 configurable warmup: 기능 채택
- 8-shape frequency warmup profile: 기각, 기존 5-shape profile 유지

## 단계별 결과

| 단계 | workload | token/s | TTFT med | TPOT med | E2E p95 | 판정 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 기존 선택값 | short, 3-run median | 2,362.2 | 93.6ms | 11.91ms | 434.7ms | 기준 |
| P1 output pump | short, 3-run median | 2,406.1 | 91.0ms | 11.49ms | 425.9ms | 채택 |
| P2 cohort | balanced smoke | 4,574.1 | 1,637ms | 15.93ms | 4,961ms | 소폭 개선 |
| P2 cohort | decode-heavy smoke | 5,287.7 | 4,807ms | 12.17ms | 13,267ms | 소폭 개선 |
| P3 online cost | balanced, 3-run median | 4,495.6 | 1,637ms | 16.04ms | 5,056ms | 대조군과 동일 |
| P3 online cost | decode-heavy smoke | 5,292.6 | 4,780ms | 12.21ms | 13,250ms | 유지 |
| P4 8-shape graph | short, 3-run median | 2,360.0 | 76.7ms | 12.21ms | 435.1ms | 기각 |
| 최종 latency gate | short, 3-run median | **2,410.4** | 96.6ms | **11.26ms** | **425.5ms** | 선택 |

최종 short는 기존 선택값보다 처리량 2.0%, TPOT median 5.5%, E2E p95 2.1% 개선됐다. TTFT median은 run 간
변동으로 3.2% 높지만 p95는 176.5 대비 181.1ms로 비슷한 범위다.

Balanced/decode-heavy의 최종 3-run 중앙 처리량은 4,486.6/5,268.9 token/s로 기존 4,531.2/5,275.5 대비
-1.0%/-0.1%다. 둘 다 3% gate 안이며 최종 추가 smoke는 4,607.4/5,257.4 token/s였다. 처리량을 추가로 올렸다고
단정하지 않고 latency와 적응 기반을 확보한 결과로 해석한다.

## P1: native completion과 batched output pump

`IndependentPhaseAsyncServer::setEventCallbacks()`를 추가했다. Sampling ticket을 수집한 뒤 다음 GPU dispatch를 먼저
enqueue하고, 그 후 token/completion을 native callback으로 이동한다. Callback이 없으면 기존
`tryPopToken()/tryPopCompletion()` queue 동작을 유지한다.

IPC backend의 output writer도 record마다 mutex unlock, `std::endl`, flush를 반복하지 않는다. 한 poll에서 만들어진
token/completion/metric record를 한 번의 queue insertion과 output batch로 합친다. Host 계측은 ingress, server poll,
serialization, output batch/record/byte 수를 별도로 기록한다.

## P2: D64 cohort와 16-slot P8 staging

비대칭 engine에서 다음 조건이면 decode cohort를 기본 활성화한다.

```text
stable slots 80 > max decode rows 64 > max prefill rows 8
```

D64 cohort ID는 request가 끝날 때만 제거된다. Decode cohort 밖에서 prefill을 마친 최대 16개 request는 stable KV
ownership을 유지한 채 staging된다. Cohort row가 완료되면 staging request가 다음 decode batch에 들어가며 KV
payload copy나 stable slot renumbering은 없다. `decode_cohort_size`가 phase metric에 추가됐다.

## P3: confidence 기반 online decode cost

Offline JSON cost는 prior로 유지한다. Decode-only CUDA-event sample을 다음 key로 모은다.

```text
(decode batch size, ceil(max context / 512))
```

- window: 최근 32개
- 최소 confidence: 8개
- statistic: p95
- static prior 보정 한도: +/-25%
- overlap sample은 decode-only cost를 오염시키지 않도록 제외
- latency mode에서는 학습과 적용을 정지
- backlog throughput mode에서만 활성화

Latency mode 차단 전 short의 한 run이 크게 흔들렸다. Mode gate 후 short 중앙 처리량이 2,410 token/s로
회복됐다. Online bucket/sample 수는 metric으로 노출된다.

## P4: frequency/budget graph cache

`EngineExecutor` graph cache가 다음 통계를 제공한다.

- execute calls
- graph hits/misses
- captures
- launch failures
- entries
- frequency/LRU evictions

Cache를 줄일 때 hit count가 낮고 오래 사용하지 않은 graph부터 제거한다. Startup decode shape는
`TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES`로 지정할 수 있으며 profile 밖 값은 거부한다.

과거 실측 빈도에서 D64와 short tail shape를 골라 다음 profile을 실험했다.

```text
D5,D12,D18,D28,D40,D44,D48,D64
```

Ready memory는 약 8,088에서 8,110MiB로 22MiB 증가했고 short 처리량은 2,406에서 2,360 token/s로 감소했다.
따라서 cache 기능은 유지하지만 production 기본값은 기존 `D8,D16,D32,D48,D64`다.

## 검증

- 실제 Cosmos P8/D64 TensorRT engine build/runtime 연결
- short 48 requests, balanced/decode-heavy 288 requests
- 모든 run의 requested/generated token 수 일치
- scheduler/server/graph 집중 unit tests: 78/78 pass
- 새 online confidence/cohort/warmup validation tests pass
- rejected graph profile도 OOM 없이 완료, ready GPU headroom 약 1.71GiB

## Artifact

- 전체 결과: `.local/cosmos-reason2-2b/priority1-4-20260823/`
- 선택 engine: `.local/cosmos-reason2-2b/asymmetric-20260820/engine-b80-p8-d64-kv2048-tied/`
- 구현 worktree: `.local/upstream-v010/`
