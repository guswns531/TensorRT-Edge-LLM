# 실험·검증 체크리스트

## 1. 환경 기록

- GPU model, compute capability, SM 수, TPC 수
- Jetson/DRIVE power mode와 clock lock 상태
- CUDA runtime/driver version
- TensorRT version
- engine build flags와 auxiliary stream 수
- model/precision/KV dtype
- max batch, input, sequence, KV capacity
- CUDA graph on/off
- partition backend와 partition granularity

## 2. baseline

- 기존 single-stream output token parity
- prefill GPU time
- decode iteration GPU time와 TPOT
- encoder GPU time
- TTFT
- throughput/goodput
- peak device memory
- `nsys` timeline에서 engine/aux stream 확인

## 3. correctness gate

각 구현 phase마다 다음을 통과해야 다음 단계로 간다.

- 동일 seed/greedy에서 output token 일치
- KV length가 token length와 일치
- request slot 간 KV contamination 없음
- encoder output lifetime이 prefill 완료까지 유지
- batch 종료/취소 시 event와 slot 회수 누락 없음
- error path에서 stream/context/event/workspace leak 없음
- Compute Sanitizer racecheck/memcheck 가능한 최소 model 통과
- 반복 실행과 다양한 arrival ordering에서 deadlock 없음

## 4. concurrency 확인

단순히 두 stream을 만들었다고 overlap이 생긴 것은 아니다.

- Nsight Systems에서 서로 다른 stream의 kernel overlap 확인
- event 기반 phase duration과 Nsight 구간 비교
- overlap ratio 기록

```text
overlapRatio =
  (encoderGpu + prefillGpu + decodeGpu - wallGpuWindow)
  / min(sumOfConcurrentPhaseGpuTimes, wallGpuWindow)
```

정확한 metric 정의는 실험 도구에서 고정하고 모든 비교에서 동일하게 사용한다.

## 5. interference matrix

다음 조합을 각각 측정한다.

| Pair | sweep |
|---|---|
| encoder + decode | image/audio size × decode batch/context |
| prefill + decode | prompt tokens × decode batch/context |
| encoder + prefill | modality size × prompt tokens |
| encoder + prefill + decode | 2-phase 검증 뒤 제한적으로 |

각 조합에서 isolated 대비 slowdown, TTFT, TPOT, throughput, memory bandwidth, SM utilization을 기록한다.

## 6. partition backend 검증

### Noop

- 여러 stream/context만으로 실제 concurrent kernel 실행 확인
- stream priority 효과 확인

### CUDA Green Context

- device resource query 결과 기록
- minimum partition size와 alignment 준수
- 각 stream이 받은 resource query/검증
- TensorRT deserialize/context create/enqueue 호환성
- CUDA graph와 event interop
- disjoint partition에서도 forward progress가 보장되지 않는 경우 탐지

### libsmctrl

- upstream validator test
- CUDA/driver/GPU 조합별 mask offset 확인
- TPC range와 실제 실행 SM 검증
- main stream과 TensorRT aux stream 모두 검증
- mask 변경 중 in-flight work 처리 규칙 확정
- 실패 시 즉시 Noop/serial fallback

## 7. scheduler acceptance criteria

- 기능 off일 때 baseline과 동작/성능 차이가 측정 오차 이내
- deadline guard가 설정된 workload에서 TPOT violation 증가 없음
- 동시 실행으로 throughput 또는 goodput 개선
- scheduler CPU overhead와 event polling overhead 별도 보고
- predictor 오차 P50/P95 보고
- workload shift 시 safe policy로 fallback
- starvation 방지: encoder/prefill 최대 queue age 상한

## 8. 권장 개발 순서

1. 작은 vanilla LLM + 고정 입력으로 phase event trace
2. VLM encoder와 decode 두 context/workspace 분리
3. 두 요청으로 encoder/decode overlap
4. LLM prefill/decode execution context 분리
5. KV physical slot 고정과 compaction 제한
6. prefill/decode overlap
7. fixed scheduler
8. fixed partition backend
9. profile table과 adaptive scheduler
10. engine segmentation feasibility study
11. speculative decoding/system prompt cache/CUDA graph 재도입

## 9. 모델 validation

모델-facing 변경의 최종 검증은 저장소 규칙대로 다음 순서를 지킨다.

```text
export -> build -> inference
```

export-only 성공은 runtime concurrency와 KV cache correctness의 근거가 아니다.
