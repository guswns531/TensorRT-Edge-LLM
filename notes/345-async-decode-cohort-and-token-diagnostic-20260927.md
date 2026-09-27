<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 345. Async decode equal-work transition and token diagnostic

## Question and scope

344의 service 경로 회귀를 수동 비용표 복구 없이 분석한다.
이번에는 serving policy를 수정하거나 full12를 다시 돌리지 않는다.

1. 기존 로그의 Gemma3개 divergent requests는 서로 다른 prompt인가?
2. 같은 logical prefix/cohort를 async server로 실행하면 D8 vs D4+D4의 다음 D8 비용은 어떤가?
3. P를 BS1로 고정했을 때 D1/D4/D8에서도 output 차이가 발생하는가?

## Retained evidence

344의 balanced에서 mismatch requests는4/28/52, 첫 divergence 위치는 zero-based66/3/13이다.
이 셋은 동일한 incident-report prompt다. 같은 prompt의 전체 반복 ID는4/16/28/40/52다.

- Static 안에서도 unique output2개: request28이 request4와 token3부터 다르다.
- Service 안에서는 unique output3개다.
- 따라서 새 host-service model만이 최초로 만든 현상이라고 주장할 수 없다.
  그러나 기존에도 있었다는 이유로 correctness 검사를 생략하거나 단순 FP16 오차로 확정하지 않는다.

## Implementation

`examples/llm/phaseAsyncDecodeTrial.inc`는 `TRT_EDGELLM_ASYNC_DECODE_TRIAL` opt-in 진단이다.
`llm_phase_context_smoke.cpp`에서 semantic server 초기화 직후 실행하고 종료한다.
Production scheduler/기본값/engine/KV 구현은 변경하지 않는다.

- 실제 tokenizer/chat template로 retained request4를 처리한다.
- 모든 row는 동일 prompt다. 각 요청을 하나씩 prefill해서 P batch shape를 BS1로 고정한다.
- D를 막은 상태로 전체 ready cohort를 구성한 뒤 explicit global action API로 지정 membership을 실행한다.
- 기존 coordinator, KV view, async sampling ticket, commit 경로를 사용한다.
- Graph1/split/dense를 미리 확보하고 timed decode는 graph replay를 요구한다.
- 각 dispatch 후 `pollCompletions()`만 진행하여 policy가 추가 action을 선택하지 못하게 한다.
- 첫 tick과 공통 successor의 총 host 완료시간, GPU event time, prepare→poll-drained service,
  직전 token commit→다음 prepare 간격, 실제 row 순서, output tokens를 보존한다.
- 종료마다 request 수/토큰 수/slot 반환을 확인한다.

첫 implementation은 actual rows와 candidate rows의 순서까지 동일해야 한다고 검사하여 실패했다.
기존 `PhaseDispatchWorker::preservePhaseBatchRowAffinity`는 selected membership을 유지하면서
직전 D의 row 위치를 보존한다. 예를 들어 D4+D4 다음 D8은 후반4rows가 앞쪽으로 이동할 수 있다.
수정 진단은 정확한 membership을 검사하고 actual row order를 별도로 저장한다.
이는 production row-affinity 제거 또는 exact action-order parity 보장을 구현한 것이 아니다.

## Experiment contract

| Cell | Rows | 첫 tick | 후속 tick | 관측 |
|---|---:|---|---|---|
| Gemma two_tick | 8 | D8 / D4+D4 / D1×8 | D8 / D8 / D1×8 | 각5회, 앞서 각1회 warmup |
| Gemma quality | 8 | D8 / D4+D4 / D1×8 | 같은 partition127 decode turns | 각2회 + 각1회 warmup, output128 |
| Cosmos two_tick | 64 | D64 / D32+D32 / D1×64 | D64 / D64 / D1×64 | 각5회, 앞서 각1회 warmup |

Variant 순서는 block마다 회전한다. 재시작 독립5회가 아니라 in-process 반복이다.
First tick은 prefill에서 나온 첫 output token 다음의 decode다.
Prefill은 timed decode horizon에서 제외한다. Vision engine도 로드하지 않는다.
동일 prompt/first-token prefix와 logical row work를 다시 구성하는 실험이지
동일 physical page bytes를 clone한 same-snapshot counterfactual은 아니다. 실제 slots는 raw에 저장한다.
기존 row-affinity가 작동하므로 batch-size-only와 row-order-only 효과를 완전히 분리하지 않는다.
Sampling은 실제 async 경로지만 진단 driver는 완료를 기다린 뒤 다음 action을 제출한다.
따라서 일반 concurrent HTTP serving의 host gap이나 TTFT/TPOT/E2E로 해석하지 않는다.
또한 commit→prepare에는 진단 기록/명시적 candidate 작성 비용도 포함된다.

## Reproduction and provenance

```bash
python3 benchmarks/phase_serving/run_async_decode_trials.py \
  --build-root .local/baselines/async-decode-trial-v2-20260927/bin \
  --result-root .local/results/async-decode-trial-v2-20260927
```

Base source `370a67f` + retained patch/new diagnostic sources.
Binary SHA256 `352b9bb3256ebdb93338d27f7c782af202fa979ffd32f69f67a737ea180a7ff3`.
초기 실패는 `.local/results/async-decode-trial-20260927`에 보존한다.
각 command/engine hash/request/config/repeat/summary 위치는 manifest에 저장한다.
이 통제 fixture에 대응하는 vLLM 결과가 없으므로 비교는 N/A다.
344의 balanced frozen comparison을 가져와 이 microbenchmark의 승리로 표시하지 않는다.

## Results

수정 진단3/3 cells 완료. Prompt length는 Gemma108, Cosmos102 tokens다.
아래 값은5개 in-process observation의 mean / nearest-rank p95다.
표본5개에서 p95는 최대값이므로 안정적인 tail 추정/신뢰구간을 뜻하지 않는다.

| 모델/실행 | 첫 tick 완료, ms | 공통 successor 포함 두 tick, ms | 첫 tick GPU 합, mean ms |
|---|---:|---:|---:|
| Gemma D8 | 13.000 / 13.066 | 25.902 / 26.002 | 12.629 |
| Gemma D4+D4 | 13.076 / 13.102 | 25.983 / 26.011 | 12.626 |
| Cosmos D64 | 8.376 / 8.391 | 16.718 / 16.740 | 8.049 |
| Cosmos D32+D32 | 14.379 / 14.611 | 22.723 / 22.942 | 13.785 |

- Gemma split: 첫 tick +0.59%, 두 tick +0.31%, GPU 합 -0.026%.
  동일 real prompt에서 GPU 이득은 사실상 없고 완료시간도 개선되지 않았다.
- Cosmos split: 첫 tick +71.68%, 두 tick +35.92%, GPU 합 +71.26%.
  이 ready64 통제점에서는 dense가 명확히 유리했다.
- 첫 tick service 합(prepare→poll-drained)은 Gemma12.969→13.027ms,
  Cosmos8.291→14.236ms다.
- 다음 turn의 commit→prepare mean은 Gemma dense25.5µs/split20.1µs,
  Cosmos81.2µs/67.9µs다. 관측된 비용으로 fixed multi-ms launch gap 가설을 지지하지 않는다.
  단 실제 HTTP concurrent loop의 gap을 측정한 값은 아니다.
- 이 결과만으로344의 약4% serving 회귀 전체를 설명하지 않는다.
  Prefix/context/state와 harness가 다른343의 숫자와도 직접 혼합 평균하지 않는다.

### Output and row evidence

Gemma quality: 동일108token prompt, P BS1, 고정128token 출력.
각 variant는8rows×2반복=16개 sequence를 비교했다. 매번 slots0..7을 사용한 것이 raw에서 확인됐다.
각 variant 내부16개 sequence는 모두 동일했다.

| D 실행 | Variant 내부 unique outputs | D1 reference와 exact 일치 | 첫 차이 |
|---|---:|---:|---|
| D1×8 | 1 | 16/16 | 없음 |
| D4+D4 | 1 | 16/16 | 없음 |
| D8 | 1 | 0/16 | output index13, 즉14번째 토큰 |

첫13tokens는 동일하므로 동일 autoregressive input prefix에서 D8과 D1/D4가 갈라진다.
이는 **D batch 실행 경로만 달라도 output divergence가 재현됨**을 보여준다.
Prefill batch 차이나 RLS policy를 원인으로 반드시 가정할 필요가 없다.
그러나 logit error/top-2 margin을 아직 측정하지 않았으므로 정상 FP16/tactic 차이인지,
shape-specific attention/KV 오류인지는 미확정이다. D1을 semantic ground truth라고도 주장하지 않는다.

Retained HTTP와의 exact mapping:

| 경로 | Requests | 통제 output과 일치 |
|---|---|---|
| Static | 4/16/40/52 | D8 |
| Static | 28 | D8/D1 어느 것과도 다름 |
| Service | 16/28/40 | D8 |
| Service | 52 | D1/D4 |
| Service | 4 | D8/D1 어느 것과도 다름 |

따라서 service52의 차이는 이 통제점과 정합적이지만 나머지 다른 sequence까지 원인을 해결한 것은 아니다.
Two-tick fixtures는 모두3token 출력이 같았다: Gemma variant당40/40 sequences,
Cosmos320/320 sequences. 짧은 horizon parity를128token quality gate로 대체하면 안 된다.

Row-affinity 기록:

- Dense D8 뒤 D8: `[0,1,2,3,4,5,6,7]`.
- D4+D4 뒤 D8: `[4,5,6,7,0,1,2,3]`.
- Cosmos도32..63이 먼저 오는 같은 패턴이다.

이는 ownership/membership 손실이 아니라 runtime의 기존 row reuse mechanism이다.
`globalCandidateParity`는 scheduler materialization 시점의 검증이며 worker의 row-affinity 적용 후
순서까지 같음을 뜻하지 않는다. Dispatch hash나 numerical determinism 분석에서는 actual rows를 써야 한다.

## Architectural assessment and next step

새 host-service 모델은 GPU-only보다 많은 비용을 측정하지만 아직 **개별 dispatch 값**이다.
`PhaseQueueScheduler::selectDecodeBatchSize()`의 service DP는 ready N rows를 partition했을 때 비용을 계산한 뒤
첫 batch 크기만 반환한다. 해당 N rows가 한 token씩 전진할 때까지 residual partition을 lease하지 않는다.
실제 sampling 완료/새로운 P completion 이후에는 다른 ready frontier로 다시 결정할 수 있다.
따라서 DP가 평가한 equal-work partition과 실행 trajectory가 항상 같다고 볼 수 없다.
이번 forced experiment는 그 partition을 끝까지 실행했을 때의 비용을 보여주며,
production에서 그 불일치가 회귀의 주원인인지까지 검증하지는 않았다.

다음 작업은 두 가지로 좁힌다.

1. **Quality:** output index13 직전 동일 prefix의 D1/D4/D8 logits/top-2 gap을 비교한다.
   Graph/eager를 한 점에서만 나눠 수치 경계와 shape-specific 잘못된 실행을 구분한다.
2. **Selection realization:** model이 선택한 partition의 ready request IDs와 token-progress frontier를 기록한다.
   다음 dispatch가 같은 frontier의 residual인지, 이미 진행한 row의 다음 token인지 구분해
   predicted equal-work cost와 actual completion을 비교한다. 불일치가 확인되면 작은 cohort lease를 검토한다.

지금 즉시 static D table 복구, D8 금지, Gemma 전용 D4 rule, 임의 launch penalty는 추가하지 않는다.
Automatic model의 calibration/선택을 유지하되 평가 단위와 실제 실행 단위를 맞추는 방향이다.
이번에는 serving 성능 개선이나 full12/vLLM 승리를 새로 주장하지 않는다.

## Validation

- C++ diagnostic target build 성공, 실제 async trials3개 완료.
- Python report tests 신규5개 + 기존 equal-work4개 통과.
- 누락 episode/dispatch/token, eager fallback, invalid timing, membership 손실을 검사한다.
  행 순서만 바뀌면 허용하되 반드시 raw에 저장한다.
- Production default, `.local/current` 포인터, engine/model 모두 변경 없음.
