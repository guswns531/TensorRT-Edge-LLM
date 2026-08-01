# Gemma 4 E2B phase execution 연결 및 성능 결과

## 완료한 연결

`llm_phase_bench`의 queue worker와 `PhaseRequestLifecycle`이 실제 indexed Gemma TensorRT callback을 호출한다.

- prefill/decode가 각각 별도 `PipelineIO`, `TensorMap`, CUDA stream을 사용한다.
- `PhaseBatchState`가 active row의 stable `kv_slot_ids`와 physical-slot KV length를 gather/commit한다.
- lifecycle smoke는 request submit, prefill, decode, terminal slot release를 실제 engine enqueue로 검증한다.
- `--sharedContext`는 한 `IExecutionContext`와 workspace만 사용한다.
- 기본 모드는 sibling context 두 개로 실제 kernel overlap의 상한을 측정한다.

```text
prefill queue -> batch + slot IDs -> prefill stream -> TensorRT profile 0
                                                 |
                                      prefill-done CUDA event
                                                 |
decode queue  -> batch + slot IDs -> decode stream  -> TensorRT profile 1
                                                 |
                                       decode-done CUDA event
                                                 |
                                   KV commit / requeue / slot release
```

shared-context 모드에서는 prefill event가 완료된 뒤에 decode의 host-side `prepare()`를 호출한다.
decode stream에 wait event만 넣고 즉시 profile을 변경하면 같은 context의 host state를 이전 enqueue와 동시에
바꿀 수 있기 때문이다. 따라서 stream과 queue는 분리되지만 kernel은 안전하게 직렬 실행된다.

## RTX 3080 측정 결과

환경은 RTX 3080 10GB, driver 610.43.02, CUDA 13.3, TensorRT 11.0.0.114,
Gemma 4 E2B INT4-AWQ backbone과 FP16 PLE/embedding/LM head/KV cache다.
각 scenario는 CUDA graph를 끄고 warmup 20회 뒤 CUDA event 100회를 측정했다.

| Mode | Worst median delta | Worst p95 delta | 결과 |
|---|---:|---:|---|
| 한 context, 두 stream, event 직렬화 | +1.38% | +1.35% | 3% gate 통과 |
| 두 context, 실제 overlap | -12.53% | -12.28% | 모든 scenario 개선 |

전체 12개 결과는 [phase context comparison CSV](gemma4-e2b-phase-context-comparison.csv)에 있다.
shared-context scheduler 비용은 scenario에 따라 median +0.29~+1.38%, p95 +0.17~+1.35%였다.
independent-context overlap의 makespan 개선은 장문에서 2.32~4.22%, 짧은 작업에서 10.48~12.53%였다.

동시 실행 중 phase별 단독 시간은 늘어난다. BS1/input128 예에서 decode는 5.74 ms에서 8.39 ms로
느려졌지만 prefill과 겹쳐 전체 makespan은 24.42 ms에서 21.36 ms로 줄었다. 따라서 scheduler objective는
개별 kernel latency가 아니라 TTFT/TPOT SLO와 전체 makespan을 함께 사용해야 한다.

## VRAM과 correctness

worst-case BS2 prefill 1024 + BS2 decode past 1536에서 `nvidia-smi`로 관측했다.

- shared context: 7,898 MiB, headroom 1,968 MiB
- independent contexts: 8,626 MiB, headroom 1,240 MiB
- 두 방식 모두 요구한 512 MiB headroom을 만족한다.
- 두 번째 context/workspace의 관측 비용은 약 728 MiB다.

실제 `llm_basic` greedy inference는 성공했고 결과 SHA256은
`1918f649c96695ea807985d3e7a98c4257d3d429f3b9557277f4957472dfcb2a`로 기존 결과와 같다.
raw CSV와 log는 `/tmp/gemma4-e2b/perf/phase`에 보존했다.

## 적용 범위

이번 단계에서 scheduler/lifecycle에서 실제 TensorRT phase enqueue까지의 benchmark execution adapter는 연결됐다.
일반 `handleRequest()`의 여러 독립 request context를 하나의 동적 phase batch로 pack하는 public serving adapter는
아직 별도 작업이다. 현재 성능 수치는 random embedding 기반 phase engine 성능이며 tokenizer, sampling, network
queue 시간을 포함한 end-to-end TTFT/TPOT는 아니다.
