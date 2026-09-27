<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 352. 같은 HTTP send도 같은 server ingress/admission은 아니다

## 질문과 방법

[351](351-prefill-arrival-feedback-20260927.md)에서 client 상한64를 사용하면
static/service의 HTTP send 차이 p95가 0.04~0.08 ms로 줄었다. 그런데 P batch는
여전히 달랐다. 이제 그 차이가 server submit→admission→첫 P 실행 중 어느
경계에서 시작하는지 확인했다.

`analyze_prefill_arrival_coupling.py`가 `PHASE_TIMELINE`의 `server_submit`,
`server_admit`, `prefill_start`, `first_token`을 request ID로 연결한다. Measurement
epoch의 dispatch timestamp를 기준으로 timeline을 분리하고, 각 stage의 순서와
stage 사이 평균/p95, paired relative-time 차이를 보고한다. 모든 64개 요청에서
필수 stage가 있어야 분석을 완료한다. Synthetic submit-order inversion과 누락
stage를 검사하는 단위 테스트를 추가했다. Serving runtime과 workload는 수정하지
않았고 351의 기존 GPU HTTP 결과를 재분석했다.

## Client64, 같은 입력에 대한 3개 paired run

Client의 scheduled JSON 요청 및 actual HTTP send는 거의 같아도 서버 submission은
thread/gateway/IPC 진행에 따라 바뀌었다. 표의 p95 차이는 각 run의 첫 server
submit을 0으로 맞춘 뒤 동일 request의 상대 시각을 비교한 값이다.

| Run | Client send 차이 p95 | Server submit 상대시각 차이 p95 | 첫 submit 순서 차이 | Admit 상대시각 차이 p95 | 첫 P-start 순서 차이 |
|---|---:|---:|---|---:|---|
| 1 | 0.071 ms | 3.61 ms | 25↔26 | 222.52 ms | 25↔26 |
| 2 | 0.076 ms | 8.16 ms | 25↔26 | 216.96 ms | 3↔12 |
| 3 | 0.040 ms | 6.78 ms | 35↔36 | 197.91 ms | 3↔12 |

첫 run에서 client는 양쪽 모두 25를 26보다 먼저 전송했다. 서버에서는 static이
25→26으로 submit했지만 service는 26→25였다. 첫 submit 시각은 각 run의 첫
server submit에서 잰 상대 ms로 static 25/26=`30.96/31.15`, service
25/26=`31.56/31.33`이다. Active capacity가 24이므로 먼저 submit된 쪽이
다음 admit 기회를 얻었다. Admit 시각은 static 25/26=`607.26/832.45`, service
25/26=`841.51/607.90` ms였다. 첫 P 실행도 이 admit 순서를 따랐다.
즉 첫 run의 P cohort 차이는 **client send 후 backend submission 순서가 뒤집힌
것만으로도 설명 가능**하며, 같은 P-ready snapshot에서 candidate generator가
잘못 선택했다고 볼 수 없다.

반대로 둘째·셋째 run에서는 P-start 순서가 request 3/12 부근에서 먼저 달라졌다.
이 시점은 각각 server submit 순서의 첫 inversion(25/26, 35/36)보다 앞선다.
첫 P batch를 비교하면 static은 `[1,2,3,12,13,14,15]`, service는
`[1,2,12,13,14]`였다. 양쪽 모두 serial P action이었고 공통 request들의
ready token count는 같았다. 그러나 static ready set에는 run에 따라 15/16까지,
service ready set에는 14/15까지 있었으며 request age와 cost posterior도 같지
않았다. 따라서 이 조기 차이는 **server-internal timing/formation**이 남는다는
증거지만 같은 완전한 snapshot의 policy-only 선택 실패로 확정할 수 없다.

## 기다림의 위치와 증폭

Client64 첫 run의 request당 mean/p95 단계 시간(ms):

| 단계 | Static | Service |
|---|---:|---:|
| Server submit→admit | 1133.57 / 2743.13 | 1112.99 / 2674.99 |
| Admit→첫 P start | 22.00 / 100.83 | 25.60 / 101.34 |
| 첫 P start→첫 token | 47.41 / 104.11 | 50.67 / 105.24 |

Client64는 64개 HTTP 요청을 약 40 ms 안에 보내지만 GPU active slot은 24개다.
따라서 client wait가 거의 0이 된 대신 평균 약 1.1초가 server admission queue에서
소요된다. Client 상한24에서는 server submit→admit이 거의 즉시지만 그만큼
client slot에서 기다린다. 수 µs~수 ms의 submit 순서/시각 차이가 약 200 ms의
admission 차이로 커질 수 있다. 이것은 요청별 latency와 다음 P formation을
동시에 바꾼다.

## 해석과 다음 단계

상한64는 HTTP send 일정은 맞추지만 **backend ingress 순서까지 고정하지 않는다**.
반대로 backend 순서 차이만이 전부도 아니다. 둘째·셋째 run에는 그보다 먼저 P
선택 차이가 있다. Policy-only causality를 주장하려면 다음 중 하나가 필요하다.

1. JSON request를 정해진 `(arrival_offset_us, request_index)` 순서로 backend에
   submit하고 응답은 계속 비동기로 받는 opt-in deterministic ingress harness.
2. 동일한 P-ready snapshot/ownership/cost state를 복제해 action만 바꾸는 replay.

기존 closed-loop24와 client64 HTTP 결과는 각각 유효한 E2E workload다. 다만
"같은 JSON trace"만으로 policy mechanism parity나 P batch 회귀의 인과효과를
단정하지 않는다. 이 분석은 runtime policy, engine, `.local/current`를 변경하지
않았고 새 GPU/vLLM benchmark도 실행하지 않았다. Exact output-quality gate는
여전히 별개로 미통과다.

재분석 JSON은 351의 closed-loop/service 및 client64/service 결과 디렉토리의
`prefill-server-transition-pair-repeat-*.json`에 보존한다. 원본 campaign manifest가
바이너리/engine/trace/hash와 command를 기록한다.
