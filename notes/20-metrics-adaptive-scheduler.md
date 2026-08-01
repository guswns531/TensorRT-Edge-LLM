# Metrics 기반 adaptive scheduler

## 기본 제공 정책

`PhaseQueueSchedulerConfig::enableMetricsPolicy=true`이면 queue-only threshold 대신 다음 순서로 결정한다.

```text
한 queue만 존재 -> 그 phase
두 queue 존재
  -> queue wait target 초과? 정규화 pressure가 큰 phase
  -> decode burst limit 초과? prefill
  -> metrics warmup 부족? 기존 token-threshold policy
  -> predicted prefill GPU time이 허용 범위이고 overlap 효율이 충분? overlap
  -> decode
```

예측값은 CUDA event sample의 EWMA다.

```text
prefill_cost = EWMA(prefill_gpu_ms / prefill_tokens)
decode_cost  = EWMA(decode_gpu_ms / sum(past_kv_length))
predicted_prefill_ms = prefill_cost * candidate_prefill_tokens
```

decode cost는 현재 기본 decision에 직접 사용하지 않지만 telemetry로 제공한다. 다음 튜닝에서 prefill/decode
비용 비율이나 TTFT/ITL objective에 사용할 수 있다.

## 기본값

- prefill queue wait target: 5000 us
- decode queue wait target: 2000 us
- predicted prefill overlap 상한: 30 ms
- 관측 overlap ratio 하한: 0.05
- metrics warmup: 2 dispatch
- EWMA alpha: 0.2

정책은 opt-in이다. 따라서 기존 engine/runtime의 queue decision은 바뀌지 않는다. `llm_phase_bench`에서는
`--adaptiveScheduler`로 serving facade smoke에 활성화한다.

## 커스터마이즈

`PhaseMetricsSchedulingPolicy`는 다음 두 immutable 입력을 받는다.

- `PhaseQueueSnapshot`: queue depth, candidate token cost, oldest wait, decode burst
- `PhaseSchedulerTelemetry`: EWMA costs, overlap EWMA/sample 수, 마지막 dispatch 전체 metrics

callback이 설정되면 built-in policy를 완전히 대체한다.

```cpp
config.metricsPolicy = [](PhaseQueueSnapshot const& q, PhaseSchedulerTelemetry const& t) {
    if (q.decodeOldestWaitUs > 1000.0)
    {
        return PhaseDispatchKind::kDecode;
    }
    if (t.prefillGpuMsPerToken * q.prefillCandidateTokens < 8.0F)
    {
        return PhaseDispatchKind::kOverlap;
    }
    return PhaseDispatchKind::kPrefill;
};
```

외부 policy object를 lambda capture하면 SLO, tenant priority, SM-mask backend 상태를 추가 상태로 사용할 수
있다. callback은 빈 queue를 선택하면 기존 scheduler validation에서 실패하므로 반드시 snapshot을 확인해야 한다.

## 실제 Gemma smoke

RTX 3080, Gemma 4 E2B INT4 indexed engine에서 `--adaptiveScheduler`를 켠 actual sampling/repeated decode
serving smoke가 통과했다. dispatch metrics는
`/tmp/gemma4-e2b/perf/phase/adaptive-scheduler-smoke-dispatch.csv`에 기록했다. 같은 실행의 timed synthetic
phase workload는 단일 sample 기준 sequential 138.7827 ms, concurrent 123.6152 ms, 1.1227x였다.

이 smoke는 처음에 모든 요청이 함께 prefill queue에 들어가고 final prefill 뒤 decode가 생기므로 serving loop
내에서 두 queue가 오래 공존하지 않는다. 따라서 adaptive decision의 성능 우위 증거가 아니라 engine 연결과
telemetry feedback 안정성 검증이다. 실제 정책 비교는 continuous admission load generator로 prefill/decode
queue를 동시에 채운 뒤 warmup을 제외한 median/p95와 TTFT/ITL을 측정해야 한다.
