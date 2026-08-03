# SLO scheduler 100-request 성능 결과

## 조건

- GPU/model: RTX 3080 10GB, Gemma 4 E2B INT4-AWQ indexed engine
- 실행 구조: 하나의 CUDA primary context, 독립 TensorRT prefill/decode context와 stream
- 동일 trace: seed 0, 100 requests, 6 req/s, prompt 128–1024, output 8–64
- batching: prefill BS2, decode BS2, 최대 prefill chunk 128
- SLO: TTFT 500ms, TPOT 50ms, priority class 0–3

모든 실행은 100/100 terminal completion과 stable slot 반환을 확인했다. Raw CSV는
`/tmp/gemma4-e2b/perf/slo-100`에 저장했다.

## 전체 결과

| 정책 | req/s | token/s | TTFT p50/p95 | TPOT p50/p95 | dispatch |
|---|---:|---:|---:|---:|---:|
| queue-default, fixed chunk | 4.707 | 167.177 | 2373 / 4282ms | 17.97 / 24.11ms | 1912 |
| metrics, adaptive chunk | 3.556 | 126.313 | 5540 / 10754ms | 11.41 / 20.62ms | 2643 |
| SLO priority, adaptive chunk | 3.460 | 122.897 | 6327 / 11680ms | 10.85 / 19.09ms | 2766 |
| SLO priority, fixed chunk | 4.472 | 158.839 | 2824 / 5403ms | 9.83 / 25.57ms | 2212 |

이 workload에서는 adaptive chunk가 prefill 실행 수를 늘려 throughput을 크게 낮췄다. SLO policy와 chunk policy는
독립 설정이어야 하며, 현재 권장 SLO 실험값은 fixed 128-token chunk다. Queue-default 대비 SLO fixed-chunk의
throughput 비용은 5.0%다.

## Priority별 tail 재분배

SLO fixed-chunk에서 TPOT p95는 다음과 같다.

| priority | queue-default | SLO priority | 변화 |
|---:|---:|---:|---:|
| 0 | 22.03ms | 31.23ms | +41.8% |
| 1 | 22.83ms | 16.54ms | -27.5% |
| 2 | 23.93ms | 15.55ms | -35.0% |
| 3 | 23.20ms | 11.35ms | -51.1% |

따라서 priority scheduling은 전체 지연을 무료로 줄이는 최적화가 아니라 제한된 GPU 용량을 tenant class 사이에
재분배하는 QoS 기능이다. 1초 aging으로 class 0도 starvation 없이 모두 완료됐다. TTFT는 prompt 길이와 slot
admission 대기의 영향이 커 class별 단조 감소를 보이지 않았다. 다음 튜닝은 prefill과 decode priority를 분리하고,
arrival부터 slot admission까지의 남은 TTFT budget을 힌트에 반영해야 한다.

## 재현 분석

```bash
python scripts/gemma4_e2b_indexed/summarize_slo_results.py \
  --run fixed=/tmp/gemma4-e2b/perf/slo-100/fixed/requests.csv \
  --run adaptive=/tmp/gemma4-e2b/perf/slo-100/adaptive/requests.csv \
  --run slo_adaptive=/tmp/gemma4-e2b/perf/slo-100/slo/requests.csv \
  --run slo_fixed=/tmp/gemma4-e2b/perf/slo-100/slo-fixed-chunk/requests.csv
```

summarizer는 request ID, arrival, prompt, output trace가 모든 run에서 같은지 먼저 검증한 뒤 전체 및 priority별
TTFT/TPOT와 dispatch 수를 출력한다.

