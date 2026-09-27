<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 351. Prefill batch formation과 client arrival feedback

## 질문

[350](350-decode-partition-rolling-frontier-20260927.md)에서 Gemma balanced의
measured-service는 static보다 같은 P row work를 5~7회 더 많은 dispatch로 처리했다.
이것이 동일한 ready state에서 P batch를 잘못 만드는 scheduler 문제인지,
실제 HTTP arrival 자체가 달라져 생긴 결과인지 분리했다.

기존 harness는 64개 JSON request를 같은 `scheduled_arrival_us`로 읽지만
`--max-workers 24 --max-in-flight 24`로 client의 동시 요청 수를 제한한다.
따라서 client는 앞선 요청이 완료될 때까지 뒤 요청의 HTTP send를 보류한다.
같은 scheduled trace도 policy별 완료시간에 따라 **실제 서버 ingress가 달라지는
closed-loop workload**다. 이것은 유효한 실사용 계약이지만 고정 ingress의 policy-only
비교와는 다르다.

## 구현·측정 계약

`run_lifetime_encoded_admission.py`에 `--client-max-in-flight`를 추가했다. 기본값 0은
기존 모델 설정(현재 Gemma 24)을 그대로 사용한다. 진단에서만 64를 전달해
client worker/in-flight 상한을 함께 올린다. Backend의 max in-flight, stable slots,
engine, KV, 정책, calibration, trace는 변경하지 않았다. 서버는 초과 요청을 기존
pending queue에서 관리한다. `analyze_prefill_arrival_coupling.py`는 실제 send 차이,
scheduled-arrival latency, P row work/배치 및 최초 P 차이 시점의 ready state를
같은 request ID로 연결한다. Python 계약 테스트 48개와 신규 분석기 테스트 2개가
통과했다.

Primary 비교의 실행 파일 SHA256은 모든 셀에서 동일한
`1372aa36d7e427ca614651846b37699f290a69780960a950a0e7a6eae40d4afb`다.
GPU는 RTX 3080 10 GiB, driver 610.57.04, TensorRT 26.06이다. Gemma AWQ engine,
balanced 64 requests/6035 prompt tokens/5440 generated tokens도 동일하다.

- Closed-loop24: static/service 각 2개 독립 process.
- Client64: static/service 각 3개 독립 process. 실제 send는 scheduled arrival에서
  p95 약 0.1 ms 이내라 이 trace에 대해서는 고정 ingress에 가깝다.
- 각 셀은 실제 HTTP/SSE request를 사용하고 full telemetry를 저장했다.
- Command, source/runner/binary/engine identity, trace hash와 결과는
  `.local/results/prefill-arrival-closed24-current-20260927-{static,service}/` 및
  `.local/results/prefill-arrival-openloop[-repeat]-20260927-{static,service}/`
  아래 manifest에 있다.

Client64는 서버의 동시 active slot을 64로 확장한 것이 아니다. Server cap은 기존 24다.
또한 closed-loop24와 client64는 offered-load 계약이 다르므로 한쪽의 절대
TTFT/tokens/s를 다른 쪽보다 우월하다는 증거로 직접 사용하지 않는다.

## P formation과 처리량

아래는 각 계약 내부에서의 run 평균이다. 2~3회로 통계적 유의성을 주장하지 않는다.

| Client 계약 | Static tokens/s | Service tokens/s | Service 변화 | Static P dispatch | Service P dispatch | Paired actual send 차이 p95 |
|---|---:|---:|---:|---:|---:|---:|
| Closed-loop24 | 1232.17 | 1200.17 | -2.60% | 39.0 | 44.0 | 17.78–61.47 ms |
| Client64 | 1219.49 | 1221.37 | +0.15% | 44.7 | 38.3 | 0.04–0.08 ms |

모든 run에서 P row appearance는 정확히 84회, useful prompt tokens는 6035다.
Client64의 service는 세 run에서 P 38/39/38회, static은 47/44/43회다.
Closed-loop24는 service P 46/42회, static 39/39회다. 따라서 350에서 관측한
`service P fragmentation`은 **고정된 입력에서 일관된 메커니즘 결함이 아니다**.
Client feedback이 정책별 ingress/cohort 형성을 크게 바꾼다.

Closed-loop24 두 run 모두 첫 P membership 차이는 11번째 P dispatch에서 발생했다.
Static은 `[28,29]`, service는 `[28]`을 실행했다. 당시 scheduler decision의
P-ready set도 static에는 `[28,29]`, service에는 `[28]`이었다. 즉 같은 ready
snapshot에서 P candidate generator가 2→1로 갈라진 것이 아니다. Request 29의
실제 HTTP send는 static보다 service에서 각각 20.40/28.53 ms 늦었다.
이는 제한된 client slot이 앞선 완료에 따라 다음 요청 전송을 늦춘 직접 증거다.

Client64에서는 정책 간 같은 request의 실제 send 차이 p95가 세 paired run에서
0.071/0.076/0.040 ms였다. 그런데 P batch는 여전히 달랐다. 도착을 맞춰도
admission·phase progress·ready 시각은 정책에 따라 움직인다. Service가 이
계약에서는 오히려 더 큰 P batch를 형성했다. 따라서 "모든 차이는 client 때문"도
아니며, `ingress feedback`과 `server-internal transition`을 분리해야 한다.

## Latency 해석

아래는 `scheduled_arrival_us`부터 계산한 TTFT/E2E의 run 평균 mean/p95(ms)다.
HTTP send 기준과 달리 client에서 기다린 시간까지 포함한다.

| 계약/경로 | Arrival TTFT | Arrival E2E | Client wait |
|---|---:|---:|---:|
| Closed-loop24 static | 1215.47 / 2766.99 | 2516.04 / 4066.67 | 1132.74 / 2722.78 |
| Closed-loop24 service | 1225.50 / 2817.90 | 2547.54 / 4132.06 | 1143.78 / 2752.52 |
| Client64 static | 1211.49 / 2792.11 | 2539.17 / 4098.15 | 0.07 / 0.12 |
| Client64 service | 1203.86 / 2743.44 | 2508.95 / 4047.46 | 0.07 / 0.11 |

Closed-loop에서 send 기준 TTFT mean은 약 80 ms, client64에서는 약 1200 ms로
급증한다. 이는 처리 능력이 갑자기 15배 나빠진 것이 아니라 대기 장소가 client
slot에서 server pending queue로 이동했기 때문이다. Scheduled-arrival 기준으로 보면
두 계약 모두 약 1.2초 TTFT/2.5초 E2E다. Client64 service의 도착 기준 TTFT/E2E는
static보다 약간 낮지만, 3회에서 나온 방향성일 뿐 최종 우위 주장은 아니다.

## 한계와 다음 결정

Static/service의 exact greedy output은 closed-loop24에서 59~62/64 requests,
client64에서 60~62/64 requests만 일치했다. Output-quality gate는 계속 실패한다.
이 새 arrival 계약에 맞춘 vLLM 비교도 실행하지 않았으므로 외부 우위 주장은 없다.

이번 결과로 P batch를 강제로 키우거나 decode DP frontier lease를 추가하지 않는다.
다음 연구용 비교는 입력 계약을 명시해 **closed-loop24와 client64를 별도 workload**로
평가하고, policy-only 진단에는 client64 또는 미리 기록된 실제 ingress replay를
사용하는 것이다. 서버 내부 원인 분석은 동일한 HTTP send 시각에서 P-ready 및
admission 전이를 추적한다. Default serving/client 계약은 이번 구현에서 변경하지 않았다.
