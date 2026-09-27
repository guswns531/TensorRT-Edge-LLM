<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 347. Gemma 단일 decode 턴 분기와 tail partition 통제 실험

## 목적

[346](346-decode-logit-and-partition-realization-20260927.md)의 두 미해결 질문을 좁힌다.

1. D8과 D4/D1이 출력 index 13에서 갈리는 현상이 이전 12턴의 서로 다른 batch history 없이도
   해당 한 턴의 batch shape만으로 재현되는가?
2. HTTP service DP가 tail에서 선택한 D1×3 계열 split이 동일한 작업량의 D3 dense보다
   실제 host 완료시간과 GPU 비용에서 빠른가?

Production scheduler, engine, KV allocator, `.local/current/*`는 변경하지 않았다.

## 구현과 실험 계약

`phaseAsyncDecodeTrial.inc`에 opt-in `branch_turn`을 추가했다. 이 설정이 없으면 기존
trial의 D4 early-split/D1 whole-run 방식은 유지된다. 설정하면 모든 variant가 branch 전후에
동일한 dense batch를 실행하고, 지정한 한 턴에서만 dense/split/singleton으로 갈린다.
`graph_batches`는 미리 capture할 batch 크기를 제한한다. Python runner는 각 턴의 실제
dispatch membership, graph/eager 실행, logits와 branch→successor 완료시간을 검사한다.

| Cell | 동일한 이전 decode | 단일 branch | 동일한 다음 decode | 출력 | 반복 |
|---|---|---|---|---:|---:|
| Logits | D8 × 12턴 | D8 / D4+D4 / D1×8 | D8 | 15 tokens | 각 5회 |
| Tail drain | D3 × 137턴 | D3 / D2+D1 / D1×3 | D3 | 140 tokens | 각 5회 |

두 cell 모두 Gemma의 동일한 incident-report prompt를 사용하고 각 row를 BS1로 prefill한다.
Prompt는 108 tokens다. Tail branch 시점의 예정 context는 약 246 tokens다. 모든 episode에서
stable slots는 logits cell `[0..7]`, tail cell `[0..2]`로 같았다. Branch 전 batch shape,
graph/eager mode, row order 및 생성 token prefix도 variant 사이에 동일했다.

Logits cell은 D1/D4/D8 graph를 사용한다. Tail cell은 D1/D2만 capture해 D3는 eager,
D1/D2는 graph로 실행한다. 이는 HTTP trace에서 D3 graph hit가 없고 D1 graph가 사용된
조건에 더 가깝다. 각 결과의 실제 graph flag는 raw dispatch마다 검증했다.

실행 명령:

```bash
python3 benchmarks/phase_serving/run_async_decode_trials.py \
  --branch-only \
  --build-root .local/builds/v0101-validation \
  --result-root .local/results/decode-single-turn-20260927
```

결과 manifest에는 소스 커밋 `eebbfa7` + `source.patch`, binary SHA256
`b83fee93506d1001ae188064deb0da42e4b09055f367fba42e5b9a60bffa3b29`,
plugin/engine/trace hashes, Docker image digest, 각 command와 raw/summary 위치가 있다.
GPU는 RTX 3080 10 GiB, driver 610.57.04, TensorRT Docker 26.06이다.

## 결과 1: 단일 턴만 바꿔도 output divergence

각 variant의 8 rows × 5회 = 40 sequences는 내부에서 모두 동일했다. 첫 13개 output
tokens와 branch 이전의 모든 D8 dispatch가 variant 사이에 일치했다.

| Branch | index 13 top-1 / top-2 | top-2 margin | D1 reference exact |
|---|---|---:|---:|
| D8 | 9249: 7.822895 / 236786: 7.815614 | 0.007281 | 0/40 |
| D4+D4 | 236786: 7.953797 / 9249: 7.844735 | 0.109063 | 40/40 |
| D1×8 | D4+D4와 동일 | 0.109063 | 40/40 |

모든 반복에서 variant별 top-8 logits signature가 동일했다. 따라서 앞선 턴의 batch
history가 달라야만 divergence가 발생한다는 가설은 배제된다. 하지만 episode마다 KV를
별도 생성했으므로 **같은 물리 KV bytes를 clone해 한 커널만 다르게 실행한 증명은 아니다**.
또한 D1을 정답으로 규정하거나 FP16 경계인지 shape-specific kernel/KV 오류인지 확정하지
않는다. D8에서 top 두 후보 간 차이는 작지만 후보 236786의 D4 대비 logit 차이는 약
0.138184로 더 크다.

Logits cell의 branch host time은 row 0 logits D2H inspection이 split/singleton의
다음 dispatch 사이에 끼므로 공정한 성능 비교로 사용하지 않는다. 이 cell은 numerical
diagnostic이다.

## 결과 2: 약 246-token context에서 D3가 split보다 빠름

Tail cell은 logits D2H inspection이 없다. 각 variant 3 rows × 5회 = 15 sequences의
전체 140-token output이 정확히 일치했다. Branch 직전 137턴은 모두 동일한 D3 eager
경로다. 시간은 5개 in-process 반복의 mean이며, 독립 process 반복이나 HTTP E2E가 아니다.

| 단일 branch | branch host 완료, ms | branch + 공통 D3 successor, ms | branch GPU 합, ms |
|---|---:|---:|---:|
| D3 eager | 7.002 | 14.036 | 6.820 |
| D2+D1 graph | 12.483 | 19.586 | 12.109 |
| D1×3 graph | 18.565 | 25.653 | 17.999 |

D1×3는 D3보다 branch 완료가 `+165.1%`, 공통 successor까지 `+82.8%` 느리다.
D3 branch의 5회 범위는 6.964–7.045 ms, D1×3은 18.527–18.616 ms다.
그래프를 사용할 수 있는 D1에도 불구하고 3회 dispatch 비용이 압도적이다.

346의 실제 HTTP balanced 측정에서는 service가 static보다 D dispatch를 296→322회
늘렸고, tail의 비자명한 service DP partition은 모두 D1 반복이었다. 41개 split
snapshot은 서로 독립하지 않으며 실제 request들의 context 길이는 여기의 동일 길이
3 rows와 다르다. 따라서 이 microbenchmark만으로 전체 throughput -3.42%의 인과효과를
정량화하지 않는다. 다만 같은 context bucket의 D3 dense 대 D1×3를 직접 재보면
"D1 split이 더 싸다"는 선택을 재검토할 강한 근거가 된다.

`selectDecodeBatchSize()`는 dense D3의 service와 GPU estimate가 없으면 D3를 그대로
실행한다. HTTP에서 `[1,1,1]`을 선택했다면 해당 bucket의 dense estimate가 존재하고,
선택에 쓰인 보수적 비용은 D1×3 합보다 컸다는 것이 코드의 DP 규칙에서 따라온다.
하지만 그 값과 표본 구성을 직접 기록하지 않았으므로 p95 outlier, bucket 혼합, sampling
host delay 중 무엇이 주원인인지는 아직 미확정이다.

## 검증과 다음 좁은 단계

- 기존 Python trial 테스트 6개 + branch/graph coverage 신규 1개 통과.
- TensorRT 26.06 Docker에서 `llm_phase_context_smoke` 재빌드 및 GPU 2/2 cells 완료.
- 변경 후 기존 `--inspection-only` graph/eager 2 cells도 다시 실행했다. 이전 결과와
  variant별 top-8 logits 및 전체 reference token sequence가 정확히 일치했다.
- Production serving 결과나 vLLM 대응 microbenchmark는 이번 변경에서 생성하지 않았다.
  기본 runtime과 `.local/current/*` 포인터는 그대로다.

다음은 service DP의 decision 시점에 dense D3와 split D1의 **실제 사용된 estimate,
sample count, graph mode, bucket**을 함께 기록하고, 동일 tail frontier에서 후보를
강제 실행해 ranking error를 확인하는 것이다. 그 전에 model-specific D3 규칙이나
static D 표를 되살리지 않는다. Output-quality gate는 별도로 계속 보류한다.
