# Cosmos 모든 workload 처리량 우위와 production shape priming

## 결론

Cosmos-Reason2-2B FP16, RTX 3080 10GB, fixed-output HTTP/SSE trace에서 Current v0.10이 short, balanced,
decode-heavy 세 workload 모두 fresh v0.9.1과 memory-matched vLLM 0.27.1보다 높은 generated token/s를 기록했다.

| workload | Current P8/D64 | fresh v0.9.1 | Current/v0.9.1 | vLLM 0.27.1 | Current/vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| short, 48 requests | **2,362.2** | 1,353.4 | **+74.5%** | 1,976.6 | **+19.5%** |
| balanced, 288 requests | **4,531.2** | 4,468.1 | **+1.4%** | 4,283.7 | **+5.8%** |
| decode-heavy, 288 requests | **5,275.5** | 4,911.5 | **+7.4%** | 4,831.0 | **+9.2%** |

Current 선택 구성은 P8/D64, stable indexed-paged KV, 256 page bundles, fixed-128 packed prefill, adaptive
admission, sampling-aware D64 refill, startup shape priming, CUDA Graph prepared-shape capture 후 runtime capture
동결이다. Ready 상태의 GPU 사용량은 8,088MiB이며 약 1.78GiB headroom을 남겼다.

## 해결한 병목

### 측정 요청 중 metric IPC

기존 production trace는 모든 phase dispatch마다 JSON metric을 만들고 stdout pipe로 flush했다. v0.9.1 내부
replay와 vLLM에는 없는 비용이었다. 이제 외부 metric retention과 serialization은 기본적으로 꺼져 있으며
`TRT_EDGELLM_EMIT_PHASE_METRICS=1`일 때만 활성화된다. 외부 보관을 꺼도 scheduler 내부 CUDA-event telemetry는
계속 갱신된다. 장시간 server에서 coordinator metric vector가 무한히 커지는 문제도 함께 제거했다.

### first-seen TensorRT/CUDA shape 비용

Client warmup 20 rows는 실제 short trace의 D48과 balanced/decode-heavy의 D64를 준비하지 못했다. 첫 D48는 약
24.5ms였지만 정상 상태는 약 7.7ms였다. IPC endpoint가 ready를 알리기 전에 decode profile에 맞춰 다음 bucket을
실제 greedy inference로 준비한다.

```text
D8 -> D16 -> D32 -> D48 -> D64
```

각 warmup request는 stable lease와 KV page를 정상적으로 획득·반납한다. prefix cache가 활성화된 경우 warmup record도
ready 전에 제거한다. P8/D64에서는 168개 짧은 warmup request가 약 0.45초 걸렸다.

### CUDA Graph lifecycle

이전 graph-on 경로는 실제 요청에서 처음 본 shape를 동기 capture해 첫 run을 악화시켰다. 이제 startup warmup이
준비한 shape만 capture한 뒤 capture를 동결한다. 준비하지 않은 shape는 graph를 새로 만들지 않고 `enqueueV3`로
실행한다. 이 방식은 short TPOT p95를 graph-off의 27.84ms에서 24.13ms로 낮췄다.

## latency 비교

| workload/backend | TTFT med/p95 | TPOT med/p95 | E2E med/p95 |
| --- | ---: | ---: | ---: |
| short Current | **93.6/176.5ms** | 11.91/24.13ms | **363.6/434.7ms** |
| short v0.9.1 | 276.6/589.1ms | **7.96/8.80ms** | 439.3/740.8ms |
| short vLLM | 238.3/257.4ms | **9.92**/24.44ms | 442.9/507.4ms |
| balanced Current | **1,609/4,099ms** | **16.04/17.57ms** | **3,158/5,009ms** |
| balanced v0.9.1 | 1,692/**3,959ms** | 16.14/18.46ms | 3,242/5,021ms |
| balanced vLLM | 1,805/4,352ms | 16.38/17.58ms | 3,350/5,309ms |
| decode-heavy Current | 4,820/**10,662ms** | **12.30/12.80ms** | **7,783/13,290ms** |
| decode-heavy v0.9.1 | **4,783**/11,135ms | 12.87/13.73ms | 8,523/13,984ms |
| decode-heavy vLLM | **4,435**/11,470ms | 14.74/15.20ms | 8,685/14,538ms |

“모든 workload에서 빠르다”는 primary throughput과 E2E/p95 기준이다. 모든 개별 latency 숫자를 이긴 것은 아니다.
short TPOT median은 v0.9.1/vLLM보다 느리고, decode-heavy TTFT median은 v0.9.1보다 0.8%, vLLM보다 8.7%
느리다. 반면 해당 workload의 TTFT p95, TPOT, E2E와 처리량은 Current가 더 좋다.

## P16 실험을 채택하지 않은 이유

남은 decode-heavy TTFT median을 줄이려고 P16/D64 엔진을 fresh build하고 semantic inference까지 통과시켰다.
그러나 TensorRT가 P8 엔진과 다른 tactic을 선택했다. 엔진/ready memory는 약 200MiB 줄었지만 balanced 처리량이
4,531에서 4,278 token/s로 떨어지고 TPOT도 16.04ms에서 17.33ms로 악화됐다. P16 엔진은 삭제했고 P8/D64를
유지했다.

## 비교 계약과 검증

- 동일 Cosmos checkpoint, FP16 backbone/KV, max sequence 2,048
- 동일 HTTP streaming client, arrival offsets, prompt/output lengths, EOS 무시
- Current와 v0.9.1: independent TensorRT prefill/decode contexts
- Current/v0.9.1: P8/D64, 256 page bundles
- vLLM: 3.5GiB KV cache, max 80 sequences, chunked prefill, prefix cache off
- 모든 run은 요청 수와 requested/generated output token 수가 정확히 일치
- vLLM endpoint는 token ID를 반환하지 않아 cross-backend token hash gate에는 사용하지 않음
- Current semantic greedy inference와 stable lease 전량 반환을 별도로 통과
- C++ 전체 suite: 988개 중 944 pass, 42 platform skip; 누락된 phase-limit fixture 두 건 수정 후 focused pass

## Artifact

- 결과: `.local/cosmos-reason2-2b/all-cases-win-20260823/`
- 선택 engine: `.local/cosmos-reason2-2b/asymmetric-20260820/engine-b80-p8-d64-kv2048-tied/`
- v0.10 source worktree: `.local/upstream-v010/`
- scheduler cost: `notes/results/cosmos-v010-p8d64-hostopt-cost-20260820.json`
