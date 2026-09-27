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

GPU 실행은 아직 못했다. 호스트의 실행 커널은 `6.8.0-142-generic`인데 NVIDIA 610 모듈은
`6.8.0-139-generic`용만 설치돼 있다. `lsmod`에 NVIDIA 모듈이 없고 `/dev/nvidia*`가 없으며,
`nvidia-smi`와 `docker --gpus all`이 driver-not-loaded 오류를 냈다. APT 시뮬레이션에서 현재 커널용
패키지는 이용 가능하지만 이를 설치하면 NVIDIA userspace 16개 패키지 업그레이드도 동반한다.
이 턴에는 호스트 드라이버를 변경하지 않았다. 따라서 logit 차이의 크기, graph/eager 차이,
실제 HTTP partition 불일치 빈도와 성능 효과는 **미측정**이다.

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
