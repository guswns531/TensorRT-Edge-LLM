# Cosmos ragged direct-cost admission과 controlled overlap probe

## 결론

ragged prefill의 `maxOverlapPrefillTokens`를 무조건 128 또는 256으로 고정하지 않고, 실제 CUDA-event
`Co(P,D,shape)` 비용표가 있는 큰 prefill만 overlap 후보로 허용하는 opt-in cost-aware admission을 구현했다.
기존 cap 안의 요청은 기존 동작을 그대로 유지하며, cap을 넘는 요청만 direct-cost TPOT guard로 평가한다. 비용표가
없는 shape는 decode-only로 되돌아간다.

Cosmos-Reason2-2B FP16 indexed-paged, independent TensorRT contexts, P4/D64, 288-request balanced trace의 3회
median은 다음과 같다.

| metric | static cap 128 | static cap 256 | cost-aware | cost-aware vs 128 | cost-aware vs 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| generated token/s | 3207.0 | 3941.2 | 4107.4 | +28.08% | +4.22% |
| TTFT p95 | 6281.9 ms | 4668.2 ms | 4325.1 ms | -31.15% | -7.35% |
| TPOT p95 | 13.210 ms | 15.643 ms | 17.893 ms | +35.45% | +14.38% |
| E2E p95 | 7054.4 ms | 5579.5 ms | 5344.3 ms | -24.24% | -4.22% |

세 정책 모두 같은 288개 request의 generated text가 정확히 일치했다. cost-aware의 TPOT p95는 두 정적 정책보다
높지만 설정한 50ms SLO 안이다. 따라서 현재 balanced workload의 처리량/TTFT/E2E 우선 권장은 cost-aware이고,
TPOT tail을 최우선으로 두면 cap128을 유지한다.

## admission 동작

```text
prefill + decode queues
        |
        v
candidate padded footprint = rows * padded chunk
        |
        +-- footprint <= static cap ------> legacy overlap
        |
        +-- footprint > static cap
                   |
                   v
        direct Co(P,D,chunk,past,max-context) lookup
             |                           |
             | covered + TPOT safe       | uncovered / unsafe
             v                           v
        overlap, selected P          decode-only, prefill stays queued
```

`--costAwareOverlapAdmission`은 기본값이 off다. 켜면 direct overlap cost, TPOT hard guard, coverage enforcement가
함께 적용된다. `prefill_cost_coverage_miss`, lookup P/chunk/past, planned D/max-context를 dispatch CSV에 남겨
“비용 때문에 거절”과 “coverage가 없어 거절”을 분리해 분석할 수 있다.

## 비용표를 공정하게 만드는 방법

첫 28-case 행렬은 P=1/2/4/8 × D=1/2/4/8/16/32/64를 모두 실행했고 direct overlap 145점을 얻었다. 하지만
일반 burst trace에서는 모든 prefill이 먼저 끝나므로 decode context가 자란 뒤의 P4/D64 overlap이 거의 생기지
않았다. 이 불완전한 비용표를 production에 적용했을 때 985회 prefill이 지연되고 처리량이 1978.9 token/s까지
떨어졌다. fail-closed 안전성은 맞았지만 비용표 수집 방법이 틀린 사례다.

이를 위해 두 도구를 추가했다.

- `build_overlap_probe_trace.py`: D64 장기 decode 요청을 먼저 넣고 1s/2s 뒤 prefill probe를 도착시킨다.
- `run_real_request_kv_matrix.py --preserve-arrival-offsets --ignore-eos`: trace의 명시적 도착 시간을 보존하고,
  controlled cost probe에서는 EOS 조기 종료 없이 지정 output 길이까지 decode한다.

또한 cost model과 runtime lookup을 같은 보수적 shape 계약으로 맞췄다.

- decode context: 합계/평균이 아니라 scheduler가 사용하는 planned maximum context를 bucket한다.
- chunk length: 32/64/96/128 upper bucket으로 만들고 실제 chunk 이상인 가장 작은 비용점을 선택한다.
- past KV, D, maximum decode context도 실제 값 이상의 upper bucket만 선택한다.
- coverage가 없으면 작은 값으로 외삽하지 않고 decode-only로 유지한다.

최종 비용표는 prefill 107점, decode 14점, overlap 124점이며 chunk={32,64,96,128},
P={1,2,4,8}, D={1,2,4,8,16,32,64}를 포함한다. production trace에는 여전히 반복당 116건의 coverage miss가
있지만, 이들은 안전하게 overlap되지 않으며 성능 결과의 반복성은 4108.2/4107.4/4099.3 token/s였다.

## 구현 위치와 검증

- `phaseQueueScheduler.{h,cpp}`: cap 초과 후보만 direct-cost로 평가하고 upper-bucket shape를 선택한다.
- `phaseDispatchWorker.cpp`, `llm_phase_bench.cpp`: cost evaluation, coverage miss와 lookup shape를 기록한다.
- `build_prefill_wavefront_cost_model.py`: maximum decode context와 chunk upper bucket 비용표를 생성한다.
- `run_real_request_kv_matrix.py`: cost-aware 실행, arrival 보존, EOS probe 옵션을 전달한다.
- `build_overlap_probe_trace.py`: late-prefill controlled trace를 생성한다.

스케줄러/dispatch worker C++ test 43개가 통과했고, P/D 28-case GPU 행렬 및 late-prefill 1s/2s probe가 모두
정상 종료했다. 전체 artifact는 `.local/cosmos-reason2-2b/ragged-direct-cost-v4-20260813/`에 있다.

## 다음 단계

1. coverage miss lookup histogram에서 빈 P3/chunk/context 조합만 targeted probe로 추가한다.
2. 48-request short와 decode-heavy trace에서도 cost-aware를 3회 반복해 static128보다 3% 이상 나빠지면 off한다.
3. TPOT p95 예산을 명시적으로 설정하고 처리량 최대와 latency-safe 두 production preset을 나눈다.
4. 비용표가 충분해지면 static cap은 안전한 fallback으로만 남기고 workload별 online 선택을 연결한다.
