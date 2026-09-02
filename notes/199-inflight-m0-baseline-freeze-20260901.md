# In-flight-aware scheduler M0 baseline freeze

## 1. 결과

M0 baseline freeze를 완료했다. 이 단계에서는 production policy나 runtime behavior를 변경하지 않았다.

고정한 기준은 다음과 같다.

```text
production default
  transition-safe myopic selector

research opt-in
  contextual P+D/E+P/E+D
  + equal-work H2
  + first concrete sampling boundary
  + final-action strict dominance
```

M1 이후 변경은 이 기준과 same-runtime A/B로 비교한다.

## 2. 추가한 artifact

### Baseline manifest

```text
benchmarks/phase_serving/manifests/inflight_m0_baseline.json
```

다음을 고정한다.

- repository HEAD와 핵심 source file SHA256
- runtime executable과 TensorRT plugin SHA256
- text/vision engine와 engine config SHA256
- Cosmos Reason2-2B config/checkpoint SHA256
- 384-request long-lived trace SHA256
- final myopic/H2 commands와 aggregate result SHA256
- GPU, driver, CUDA, compiler, TensorRT container digest
- request, engine capacity, policy, output correctness contract

작업 트리는 intentionally dirty research snapshot이므로 git HEAD만 baseline identity로 사용하지 않는다.
Executable, source, engine, trace, result artifact hash를 함께 사용한다.

### Unified telemetry schema

```text
benchmarks/phase_serving/manifests/phase_event_schema_v1.json
```

M1이 출력할 append-only `PHASE_SCHEDULER_EVENT` record의 V1 contract를 고정한다. 기존 사용자 token/completion
record가 이미 `PHASE_EVENT` prefix를 사용하므로 scheduler telemetry는 별도 prefix를 사용한다.

- `decision`: snapshot, ready/in-flight state, bounded candidates, selected action
- `dispatch`: decision/plan/action correlation, direction, cohort, planned outstanding mask
- `completion`: GPU start/end, host visibility, observed outstanding mask, action fidelity

Clock domain도 분리한다.

- host timestamp: `CLOCK_MONOTONIC` nanoseconds
- GPU timestamp: per-run CUDA event origin으로부터의 elapsed microseconds

GPU 실행 중 정확한 progress를 관측한다고 가정하지 않는다. M1 snapshot은 host-observed dispatch age와 event readiness를 사용한다.

### Validator

```text
benchmarks/phase_serving/validate_m0_baseline.py
```

실행:

```bash
python3 benchmarks/phase_serving/validate_m0_baseline.py \
  --workspace-root /home/sslab/TensorRT-Edge-LLM
```

검증 결과:

```text
M0 baseline validation passed: 26 artifacts, 7.84 GiB hashed
```

## 3. Frozen environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 10GiB, SM86 |
| Driver | 610.43.02 |
| CUDA compiler | 13.3 |
| C++ compiler | GCC 13.3.0 |
| TensorRT image | `nvcr.io/nvidia/tensorrt:26.06-py3` |
| Image digest | `sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3` |
| TensorRT | 11.0.0.114 |
| Model | `nvidia/Cosmos-Reason2-2B` |
| Runtime binary SHA256 | `c6404b35efe1d4597c1e0b4b9211b6b3a39b78d17fc919c3a3a902e82a20b7d8` |
| Text engine SHA256 | `96393b233813fe0b956a55441d4ca0bdbe6a6de6ae6b2f50fd2db0f7bab298df` |
| Vision engine SHA256 | `ea090592762a96d2dc82f10c167b60f4f85fa6ccc4810cb362cef22556fcf99d` |
| Trace SHA256 | `94d8d5299063b6ed2ffa808c1cc37edc152d1d35735399de4a69eb40afc1f207` |

## 4. Frozen execution contract

| Setting | Value |
|---|---:|
| Requests/run | 384 |
| Prompt tokens/run | 133,134 |
| Requested output tokens/run | 17,568 |
| Repeats | 3 |
| Max active/stable slots | 80 |
| Prefill batch cap | 8 |
| Decode batch cap | 64 |
| Encoder batch cap | 8 |
| Fixed P chunk | 128 tokens |
| KV pool | 256 pages |
| KV capacity | 2,048 tokens |
| Request adapter workers | 8 |
| EOS | ignored |
| Output gate | exact greedy token trace + VLM semantic checks |

## 5. Frozen performance result

| Metric | Production myopic | Research H2 | H2 delta |
|---|---:|---:|---:|
| Generated tok/s | 584.561 | 586.179 | +0.28% |
| TTFT mean | 1,136.089 ms | 1,102.249 ms | -2.98% |
| TTFT p95 | 3,391.840 ms | 3,289.755 ms | -3.01% |
| TPOT mean | 25.253 ms | 25.287 ms | +0.13% |
| TPOT p95 | 39.923 ms | 39.441 ms | -1.21% |
| E2E mean | 2,277.855 ms | 2,250.373 ms | -1.21% |
| E2E p95 | 4,132.749 ms | 4,049.495 ms | -2.01% |
| Peak VRAM | 9,481 MiB | 9,473 MiB | parity |

양쪽의 세 run 모두 token trace SHA256가 다음 값으로 동일하다.

```text
5c0aa22ee6aa52638fa1ea09c5d4b7fc2ec225a7870d8870aaf1693bd5fa6e21
```

H2는 aggregate상 개선됐지만 production default로 승격하지 않는다.

```text
realized H2 episodes       10
known D budgets             9
local D budget violations   9
regret vs D-gap corr        0.388
```

이 mismatch가 M1--M5의 in-flight-aware incremental 구조가 해결해야 하는 기준 문제다.

## 6. M0에서 의도적으로 하지 않은 것

- 새 scheduler policy 적용
- 기존 `PHASE_METRIC`, `PHASE_TIMELINE`, `PHASE_FORMATION_EPISODE` 제거
- `PHASE_SCHEDULER_EVENT` runtime emission 구현
- 새 engine build 또는 inference 재실행
- frozen vLLM anchor 재실행
- external cost registry 또는 TTL 추가

새 inference를 실행하지 않은 이유는 M0가 새 성능 결과를 만드는 단계가 아니라 기존 성공 binary와 결과를
immutable artifact hash로 고정하는 단계이기 때문이다. GPU와 Docker metadata는 host에서 다시 조회해
manifest에 기록했다.

## 7. M1 입력

다음 단계는 `In-flight snapshot and unified event log`다.

구현 범위:

1. E/P/D context별 dispatch state와 host-observed age를 snapshot에 추가한다.
2. decision, dispatch, completion에 stable ID를 부여한다.
3. `PHASE_SCHEDULER_EVENT` V1 JSONL을 shadow telemetry로 출력한다.
4. 기존 policy decision은 변경하지 않는다.
5. planned/observed outstanding mask와 action fidelity를 검증한다.
6. 기존 189/189 unit gate와 exact long-lived smoke를 통과시킨다.

M1 결과는 M0 manifest와 binary hash가 달라지는 것이 정상이다. 비교의 기준 engine/model/trace/request contract는 동일하게 유지한다.
