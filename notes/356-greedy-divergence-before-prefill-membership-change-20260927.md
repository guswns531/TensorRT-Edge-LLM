<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 356. P reference ablation의 greedy 분기는 P membership보다 먼저 발생

## 이유와 진단 계약

[355](355-trusted-prefill-service-reference-ablation-20260927.md)의 두 paired
balanced run에서 baseline과 opt-in의 request별 greedy output은 각각 61/64만
exact였다. 두 run 모두 동일 prompt인 requests 1/13이 같은 19번째 생성 token
(0-based index 18)에서 `506` 대 `1156`으로 갈렸다. 각 run 내부에서는 두 요청의
48-token 출력이 서로 exact였다. 이 분기는 EOS 이후가 아니다.

요청 1의 logit만 캡처하는 opt-in runner 옵션을 추가했다. 처음에는 매 token을
캡처했으나, 이 GPU→CPU 복사가 decode formation을 바꿔 두 경로 모두 token
`506`/동일 행 순서가 되었다. 이 결과를 원래 분기의 logit 증거로 사용하지 않는다.
이후 `--diagnostic-logit-request-id 1 --diagnostic-logit-step 18`을 추가해 해당
한 step에서만 logits를 저장했다. 캡처 이전에는 진단 D2H가 없다. Baseline과
trusted-cover 경로의 단일-step 실행은 동일 binary SHA256
`c3ab0084419876dd5a260203ca669a7b8e78dcdd3ecd91d0c803ca8770059067`,
동일 engine/trace/generic startup/ordered backend ingress를 사용한다. 양쪽 모두
64/64 요청을 완료하고 backend submit 순서 0→63을 만족했다. 단일-step 캡처도
버퍼 할당과 분기 시점 복사를 동반하므로 production performance 수치로 쓰지 않는다.

Retained evidence:

- `.local/results/prefill-trusted-20260927-{baseline,filtered}/`:
  캡처 없는 paired 2회씩의 CSV/dispatch/P formation.
- `.local/results/prefill-logit-20260927-{baseline,filtered}/`:
  매-step 캡처가 trajectory를 바꾸는 음성 대조.
- `.local/results/prefill-logit-step18-20260927-{baseline,filtered}/`:
  한-step logit binary/metadata, 요청 CSV, scheduler log와 manifest.
- `.local/results/prefill-logit-step18-20260927-filtered/output-cohort-request1.json`:
  새 `analyze_output_cohort_divergence.py`의 결합 결과.

## 관측된 분기

단일-step 캡처에서 원래의 `506 ↔ 1156` 분기가 재현됐다. Metadata의
`selected_token`은 CSV token과 동일하고 FP32 dump의 argmax와도 동일하다.

| 경로 | 선택 token | logit(506) | logit(1156) | top-2 margin |
|---|---:|---:|---:|---:|
| Baseline | 506 | 16.627171 | 16.540367 | 0.086803 |
| Trusted P cover | 1156 | 16.648808 | 16.702795 | 0.053988 |

전체 262144-vocab vector의 양 경로 평균 절대 차이는 0.03525, RMS는 0.04459였다.
따라서 이번 불일치는 HTTP 응답 조립이나 argmax 이후 token 기록 문제가 아니다.
**sampling으로 들어온 logits 자체가 달랐다.** 수치 차이의 근원이 어떤 kernel,
row layout, 이전 overlap 또는 KV state인지까지는 아직 분리하지 못했다.

## Formation과 시간 순서

두 단일-step run의 첫 P membership 차이는 26번째 P dispatch(0-based index 25)다.
그 전까지 initial P cohorts는 같은 요청 집합을 처리했다. 이 P 차이의 dispatch
시각은 요청 1의 분기 token을 만드는 18번째 decode 실행보다 양쪽 모두 약
1.75초 뒤다. 따라서 **이 paired run의 request 1 분기를 초기 P5/P6 membership
차이로 설명할 수 없다.** 실제로 초기 P5는 동일했다.

반면 요청 1이 포함된 D cohort는 첫 decode부터 달랐다. 분기 token을 만드는
decode execution(0-based index 17)에서는 양 경로 모두 동일한 24개 request를
실행했지만 행 순서가 달랐다. Baseline은 `[0,1,2,3,12,13,14,15,...]`,
trusted-cover 경로는 `[0,1,2,12,13,14,3,15,...]`로 시작했다. Logit metadata의
batch member 순서도 이 metric과 일치한다. 따라서 row order/초기 D formation
차이가 수치 분기와 **동반**한다. 이 실험만으로 row order만을 원인으로 확정할 수는
없다. 실행 전 history, graph/tactic, context 상태도 함께 변했을 수 있다.

## 판단과 다음 검증

출력 동일성 미통과는 P covering 신뢰도 설정의 처리량 개선 여부와 별개다.
단일-step logit은 실제 수치 분기를 확인했지만 corruption이나 구현 오류의 위치를
특정하지 않는다. 기본 정책을 바꾸지 않고, 다음에는 같은 decode member set과
KV ownership을 고정한 상태에서 행 순서만 바꾸는 controlled replay가 가능한지
먼저 확인한다. 그 전에는 `506`과 `1156` 중 하나를 정답으로 선언하거나 모든
cross-policy greedy 차이를 버그/허용 가능한 FP16 오차 중 하나로 단정하지 않는다.
