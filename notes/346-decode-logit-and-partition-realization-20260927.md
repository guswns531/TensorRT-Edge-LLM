<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 346. Gemma decode logits와 service partition 실현성 진단

## 이어받은 문제

[345](345-async-decode-cohort-and-token-diagnostic-20260927.md)에서 Gemma의 같은 prompt를 BS1로 prefill한 뒤
D1/D4와 D8의 greedy 출력이 14번째 출력 토큰(index 13)에서 갈라졌다. Gemma balanced HTTP에서 새
host-service 정책의 처리량은 같은 binary의 static 경로보다 4.11% 낮았지만, 그 원인은 아직 분리되지 않았다.

이번 변경은 두 질문을 직접 측정하도록 준비한다.

1. 출력이 갈라질 때 D1/D4/D8의 top logits가 근접한 경계인가? Graph replay와 eager에서 차이가 같은가?
2. service DP가 평가한 decode partition은 sampling/requeue 후 실제 동일 ready frontier에 실행되는가?

## 구현

`phaseAsyncDecodeTrial.inc`에 지정 decode turn의 row 0 FP32 logits top 8 기록을 추가했다. P는 계속 BS1이고
8개 stable slot과 실제 async sampling ticket을 사용한다. Pinned host buffer로 decode stream에서만 D2H 복사하며,
CPU가 읽어야 할 때만 해당 stream을 동기화한다. `graph_replay` 입력으로 graph/eager를 각각 별도 process에서
실행한다. `run_async_decode_trials.py --inspection-only`는 같은 request와 15 출력 토큰으로 두 cell을 실행하고,
top-2 간격, committed argmax, 실행 모드를 검증한다. 이 계측은 diagnostic trial에만 적용한다.

`PhaseQueueScheduler`의 opt-in `captureDecodePartitionTrace`는 measured host-service DP의 예상 batch
partition, 당시 ready request IDs와 KV 길이, 전체 partition의 예상 host-service/GPU 비용을 보존한다.
`PhaseDispatchWorker`가 이를 실제 dispatch metrics와 함께 넘긴다. HTTP harness의
`--decode-partition-diagnostic --telemetry-level full`이 진단을 켠다. 기본 serving 설정에는 이 수집이 없다.

`analyze_decode_partition_realization.py`는 full gateway log의 dispatch membership과 decode token commit
timeline을 결합한다. 각 예상 split에 대해 첫 batch가 계획대로 나갔는지, 남은 frontier를 한 token씩 모두
전진시키기 전에 이미 실행한 row가 다시 선택됐는지, 다른 row가 끼었는지, 전체 commit horizon이 얼마인지
계산한다. 예상 service 합과 실제 commit horizon은 같은 시간 경계가 아니다. 실제 horizon에는 requeue,
다른 phase, 새 요청과 host gap도 포함된다. 따라서 그 차이만으로 GPU 비용 모델의 오차라고 해석하지 않는다.

## 현재 검증 결과

- TensorRT 26.06 컨테이너에서 `llm_phase_context_smoke`와 `unitTestRuntime` 빌드 성공.
- service DP의 `[4]` / `[2,2]` 및 frontier/비용 기록을 포함한 C++ targeted 3개 통과.
- Python: serving runner 계약 46개, async logits 요약 6개, partition trajectory 3개 통과.
- pre-commit 통과. 기본 production scheduler 선택 로직과 `.local/current/*` 포인터는 변경하지 않았다.

최초 작성 시점에는 GPU 실행을 못했다. 호스트의 실행 커널은 `6.8.0-142-generic`인데 NVIDIA 610 모듈은
`6.8.0-139-generic`용만 설치돼 있었다. `lsmod`에 NVIDIA 모듈이 없고 `/dev/nvidia*`가 없으며,
`nvidia-smi`와 `docker --gpus all`이 driver-not-loaded 오류를 냈다. APT 시뮬레이션에서 현재 커널용
패키지는 이용 가능하지만 이를 설치하면 NVIDIA userspace 16개 패키지 업그레이드도 동반한다.
이전 턴에는 호스트 드라이버를 변경하지 않았다. 아래 재실행 결과는 사용자 측 설치 이후 별도로 얻었다.

## GPU 복구 후 재실행 결과

실행 커널 `6.8.0-142-generic`용 `nvidia-driver-610-open 610.57.04`가 설치되고 NVIDIA 모듈이
로드된 뒤, 호스트 권한의 `nvidia-smi`와 TensorRT 26.06 Docker 내부 `nvidia-smi`에서
RTX 3080 10 GiB를 확인했다. 일반 sandbox의 `/dev/nvidia*` 부재는 호스트 상태가 아니므로
GPU 실험은 Docker 접근 권한을 사용했다. 소스 커밋은 `9fdd9cc`이며 engine, plugin, binary,
trace SHA와 명령은 각 결과 manifest에 있다.

### Gemma D1/D4/D8 logits

`.local/results/decode-logits-20260927/{manifest,summary}.json`에 graph/eager 각각 2회,
variant당 8 rows × 2회 = 16 sequences를 보존했다. Prompt prefill은 모두 BS1이다.
검사 지점은 output index 13을 생성하는 decode turn 12다.

| 실행 | D8 top-1 / top-2, logit | D4+D4 top-1 / top-2, logit | D1×8 top-1 / top-2, logit |
|---|---|---|---|
| graph | 9249: 7.822895 / 236786: 7.815614 | 236786: 7.953797 / 9249: 7.844735 | D4+D4와 동일 |
| eager | graph D8과 동일 | graph D4+D4와 동일 | graph D1×8과 동일 |

D8의 top-2 margin은 `0.007281`, D4/D1은 `0.109063`이다. D8은 D1 reference와
0/16 sequences 일치하고 D4/D1은 16/16 일치했다. Graph와 eager의 기록된 top-8 logits는
각 variant에서 정확히 같았다. 따라서 graph replay만의 문제라는 가설은 이 실험에서 지지되지 않는다.
하지만 D8과 D4/D1은 갈림 직전까지 다른 batch shape로 여러 decode turn을 실행했으므로,
동일한 물리 KV 상태에서 마지막 한 turn의 batch shape만 바꾼 반사실 실험은 아니다.
이 결과만으로 FP16 수치 경계와 shape-specific KV/attention 오류를 구분하거나 D1을 정답으로
승인할 수 없다. 두 후보 token 중 236786의 D8 logit은 D4/D1보다 약 `0.138184` 낮다.

### HTTP partition 실현성과 동일 binary 대조

`.local/results/decode-partition-20260927/`의 service와
`.local/results/decode-partition-shadow-20260927/`의 static-shadow를 같은 binary/engine/trace,
full telemetry, 동일 driver로 각각 한 번 실행했다. 두 경로 모두 64 requests, 5,440 output
tokens를 완료했다. 단일 실행이므로 통계적 우열 또는 promotion 근거가 아니다.

| Gemma balanced | Static-shadow | Measured host-service | Service 변화 |
|---|---:|---:|---:|
| Generated tokens/s | 1221.92 | 1180.15 | -3.42% |
| TTFT mean / p95, ms | 85.28 / 211.08 | 78.53 / 207.46 | -7.92% / -1.72% |
| TPOT mean / p95, ms | 15.70 / 17.35 | 15.74 / 17.08 | +0.29% / -1.53% |
| E2E mean / p95, ms | 1392.21 / 2178.59 | 1392.16 / 2156.48 | 약 0% / -1.02% |
| D dispatches / dispatch interval sum | 296 / 3769.14 ms | 322 / 3890.74 ms | +8.78% / +3.23% |
| P dispatches / dispatch interval sum | 42 / 831.77 ms | 45 / 886.02 ms | +7.14% / +6.52% |

Service의 D1 dispatch는 61회로 static의 20회보다 많다. D24는 양쪽 모두 148회다.
Service 측정 구간의 비자명한 예측 partition은 41개 snapshot으로,
`[1,1,1]` 34개, `[1,1]` 4개, `[1,1,1,1]` 3개였다. 이 snapshot들은 같은 tail
trajectory에서 연달아 나오므로 서로 독립한 41개 사례가 아니다.
`analyze_decode_partition_realization.py` 결과에서는 first batch와 전체 frontier가
각각 41/41 일치했고, drain 전 같은 row의 중복 재선택 및 외부 row 삽입은 0건이다.
Predicted service 평균 19.65 ms와 observed commit horizon 평균 18.67 ms는 시간 경계가
달라 직접적인 예측 오차로 취급하지 않는다. 이 trace의 관측된 작은 split에 대해서는
"DP가 선택한 partition이 비동기 경로에서 실현되지 않는다"는 가설이 지지되지 않았다.

Static/service의 exact output은 62/64 requests, 5349/5440 token positions만 일치했다.
Output-quality gate는 여전히 미통과다. 현재 throughput 하락과 추가 D1 실행은 함께
관측됐으나, 같은 snapshot을 두 정책으로 강제 replay한 결과가 아니므로 split 하나가
전체 -3.42%를 야기했다고 단정하지 않는다. Frozen vLLM 결과는 GPU driver가 달라진
이번 진단의 직접 비교치로 재사용하지 않았다.

다음 최소 실험은 (1) 동일한 D8 prefix/KV를 만든 뒤 마지막 한 decode turn에서만 D8과
D4를 갈라 logits를 재측정하고, (2) 실제 tail snapshot에서 D3 dense와 D1×3의
service·request completion을 반복 비교하는 것이다. 그 결과 전에는 D8 금지나
Gemma 전용 D3 규칙을 추가하지 않는다.

## GPU 복구 후 작은 재현 순서

1. `nvidia-smi`와 GPU 컨테이너 실행을 확인한다. 빌드 binary와 engine/hash를 새 manifest에 기록한다.
2. Gemma logits 두 cell을 실행한다.

   ```bash
   python3 benchmarks/phase_serving/run_async_decode_trials.py \
     --inspection-only \
     --build-root .local/builds/v0101-validation \
     --result-root .local/results/decode-logits-20260927
   ```

3. Gemma balanced 한 cell만 full telemetry로 실행한다. CLI는 기존 runner의 host-service variant를 사용한다.

   ```bash
   python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py \
     --models gemma --workloads balanced \
     --variants independent-autotune-service --repeats 1 \
     --startup-budget-ms 120000 \
     --telemetry-level full --decode-partition-diagnostic \
     --build-root .local/builds/v0101-validation \
     --result-root .local/results/decode-partition-20260927
   ```

4. 해당 cell의 `gateway.log.gz`에 다음 분석을 적용한다.

   ```bash
   python3 benchmarks/phase_serving/analyze_decode_partition_realization.py \
     <gateway.log.gz> --measurement-epoch 1 \
     --output .local/results/decode-partition-20260927/realization.json
   ```

5. top-2 margin과 양쪽 토큰의 logit 차이를 검토한다. Partition은 first-batch mismatch,
   duplicate-before-drain, outside-frontier, full realization 비율과 D/P dispatch 변화를 함께 본다.
   이후 필요한 수정만 적용하고 동일 trace로 재측정한다. vLLM은 workload/runtime 계약이 바뀌지 않았으므로
   이 진단 단계에서는 [344](344-measured-startup-decode-service-20260927.md)의 frozen 결과를 참고한다.

Graph/eager에서 같은 작은 logit 경계가 보이더라도 누적 KV 상태가 batch shape마다 달랐다는 점이 남는다.
shape-specific 오류 가능성을 배제하려면 동일 KV bytes에서 한 decode step만 갈라지는 추가 실험이 필요하다.
현재는 D1을 정답으로 간주하거나 출력 차이를 정상적인 FP16 오차로 확정하지 않는다.
