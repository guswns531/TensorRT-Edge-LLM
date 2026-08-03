# SLO 기반 customizable scheduler

## 요청별 policy 입력

`PhaseSchedulingHints`가 request admission부터 terminal까지 유지된다.

- `priority`: tenant priority class. 기본 범위는 0–3이다.
- `ttftTargetUs`: prefill queue에서 사용할 요청별 TTFT budget. 0이면 scheduler 기본값을 상속한다.
- `tpotTargetUs`: decode queue에서 사용할 요청별 TPOT budget. 0이면 scheduler 기본값을 상속한다.

stable KV slot이 없어 pending admission queue에 머문 요청도 이 값을 잃지 않는다. Encoder가 먼저 실행되는 VLM
요청은 encoder reservation snapshot에 힌트를 저장하고, CUDA event handoff 뒤 생성되는 prefill work item에 다시
연결한다.

## 기본 정책

각 phase queue snapshot은 oldest wait뿐 아니라 요청별 target으로 정규화한 최대 SLO pressure와 최고 priority를
제공한다. 두 queue 중 하나가 target을 넘으면 기본 metrics policy는 다음 점수를 비교한다.

```text
score = max_slo_pressure + priority_pressure_weight * priority / max_priority
```

priority 항은 기본 0.25로 제한되어, 낮은 priority의 심하게 overdue된 요청을 무한히 굶기지 않는다. Deadline을
넘지 않은 상태에서는 기존 EWMA GPU cost와 overlap efficiency 결정을 유지한다. `metricsPolicy` callback을 설정하면
queue depth, candidate token cost, wait, SLO pressure, priority, EWMA와 마지막 CUDA-event metric을 받아 이 기본
결정을 완전히 교체할 수 있다.

`enablePriorityBatching=true`이면 phase를 고른 뒤 같은 phase queue 안에서도 effective priority가 높은 요청부터
batch를 만든다. effective priority는 `priority + wait_us / priorityAgingUs`이며 기본 aging window는 1초다. 따라서
높은 priority 요청이 먼저 실행되지만 오래 기다린 낮은 priority 요청은 매초 한 class씩 승급되어 starvation을
피한다. Prefill은 priority가 가장 높은 요청의 chunk/profile bucket을 먼저 고른 뒤 같은 bucket에서 batch를 채운다.

`ttftTargetUs`는 scheduler에 admission된 뒤 prefill queue residence budget이다. 실제 arrival부터 admission까지의
pending 시간까지 포함한 end-to-end TTFT SLO를 적용하려면 admission 시 남은 budget을 계산해 요청별 힌트로 넣는다.
`tpotTargetUs`도 token 간 전체 지연의 근사치로 decode queue residence를 사용하며, host/network 시간을 포함한 최종
SLO 판정은 request CSV에서 별도로 수행한다.

## 실행 방법

`llm_phase_bench` continuous-load 모드에 다음 옵션을 추가했다.

```bash
./build/examples/llm/llm_phase_bench \
  --engineDir /tmp/gemma4-e2b/engine-indexed \
  --prefillBatch 2 --decodeBatch 2 --prefillChunkSize 128 \
  --adaptiveScheduler --adaptiveChunking \
  --loadRequests 100 --arrivalRate 20 \
  --loadPromptMin 128 --loadPromptMax 1024 \
  --loadOutputMin 8 --loadOutputMax 64 \
  --ttftTargetMs 500 --tpotTargetMs 50 --loadPriorityClasses 4 \
  --loadCsv requests.csv --outputCsv dispatch.csv
```

priority class는 고정 seed schedule의 request 순서에 0부터 round-robin으로 부여된다. request CSV에도 priority가
기록되므로 class별 TTFT/TPOT violation을 재계산할 수 있다.

## 검증

- custom policy가 phase별 최대 SLO pressure와 priority를 받는지 확인했다.
- 두 phase가 모두 overdue일 때 bounded priority가 decision에 반영되는지 확인했다.
- pending admission을 거친 요청의 TTFT/TPOT/priority가 보존되는지 확인했다.
- 관련 scheduler/lifecycle/facade GPU unit test 21개와 `llm_phase_bench` 빌드가 통과했다.
