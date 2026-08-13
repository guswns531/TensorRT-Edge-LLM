# Cosmos variable-length packed prefill과 backlog-aware chunk 정책

## 결론

Cosmos-Reason2-2B FP16 packed-prefill 계약을 고정 128 token에서 export/build 시 지정 가능한 최대 logical row
길이로 확장했다. 256-token 계약의 P8/D64, 80-slot, 256-page-bundle engine을 RTX 3080 10GB에서 빌드했고,
실제 236-token packed row를 포함한 real-request 실행이 성공했다.

12개 요청에서 chunk 128/256과 P1/P8 네 실행의 greedy output text와 output token count는 모두 같았다. 다만
256 계약은 prefill workspace를 84MiB에서 128MiB로 늘리고 실행 전 GPU 사용량을 9086.9MiB에서
9804.9MiB로 높였다. 남은 free memory가 약 69MiB이므로 이 engine에서 큰 CUDA graph cache까지 유지하는 것은
10GB 장치의 production 기본값으로 안전하지 않다.

## 구현 계약

- export: `--packed-prefill-max-chunk-tokens`
- build: `--maxPrefillChunkTokens`; 0이면 export 최대값 상속
- engine config: `packed_prefill_max_chunk_tokens`, builder override `max_prefill_chunk_tokens`
- attention plugin serialization: `packed_prefill_max_chunk_tokens`
- runtime phase contract: engine에 기록된 최대값을 scheduler와 packed adapter가 공유
- legacy packed ONNX/engine: 필드가 없으면 128로 해석

builder는 packed carrier의 prefill profile 최대 token 수를
`maxPrefillBatchSize * maxPrefillChunkTokens`로 계산한다. plugin은 runtime total-token carrier 길이와 계약 최대 row
길이 중 작은 값을 FMHA 최대 logical Q 길이로 사용한다.

## 3회 median: fixed 128 대 fixed 256

조건은 P8/D64, independent TensorRT contexts, graph off, prefill token budget 1024, 동일 real-arrival trace다.

| workload | chunk | generated token/s | TTFT p95 | TPOT p95 | E2E p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| short 48 | 128 | 2411.5 | 200.88ms | 15.199ms | 410.00ms |
| short 48 | 256 | 2401.4 | 198.00ms | 15.800ms | 404.98ms |
| balanced 288 | 128 | 4438.7 | 3983.44ms | 18.681ms | 5053.39ms |
| balanced 288 | 256 | 4580.5 | 3812.03ms | 17.594ms | 4880.58ms |
| decode-heavy 288 | 128 | 5041.7 | 10804.42ms | 13.709ms | 13627.00ms |
| decode-heavy 288 | 256 | 5076.9 | 10641.04ms | 13.597ms | 13514.17ms |

256은 balanced에서 token/s +3.20%, TTFT p95 -4.30%, TPOT p95 -5.82%, E2E p95 -3.42%다.
decode-heavy 이득은 token/s +0.70%이고, short는 token/s -0.42%와 TPOT p95 +3.95% 대신 TTFT/E2E가 약간
좋다. 따라서 workload에 관계없는 하나의 큰 chunk가 항상 우월하지는 않다.

## Queue-drain / steady-state 정책

scheduler에 다음 opt-in knob를 추가했다.

- `decodeActivePrefillChunkTokens`: decode queue가 활성일 때의 steady-state row cap
- `largePrefillChunkQueueThreshold`: admitted prefill queue가 이 값 이상이면 최대 chunk로 burst를 drain
- 기존 `prefillCompletionBonusTokens`: 작은 final continuation row가 다른 productive bucket에 무한히 밀리지 않게
  bucket score에 completion credit 부여

benchmark CLI는 `--decodeActivePrefillChunkSize`와 `--largePrefillChunkQueueThreshold`, Cosmos matrix runner는
각각 `--decode-active-chunk-size`와 `--large-chunk-queue-threshold`로 연결한다. 모두 기본값 0이어서 기존 정책은
변하지 않는다.

처음 구현한 `decode empty: 256 / decode active: 128` 정책은 short를 보호했지만 balanced에서 최적 fixed 256보다
2.40% 느렸다. backlog threshold 8/16/32도 admitted queue가 page/slot admission에 의해 작게 유지되어 큰 row를
충분히 만들지 못했고 fixed 256을 넘지 못했다. 이 정책은 부하가 시간에 따라 크게 변하는 production trace를 위한
customization seam으로 유지하고, 현재 기본값으로 채택하지 않는다.

## 검증

- C++ packed contract/config/model phase tests 통과
- `PhaseQueueSchedulerTest.*`: 50/50 통과
- plugin/engine rebuild 성공
- P1/P8, chunk 128/256 실제 inference 성공
- 128/256에서 두 개의 row가 128보다 컸고 최대 row는 236
- 네 correctness 실행의 output mismatch 0/12
- short/balanced/decode-heavy fixed A/B: 18/18 성공
- queue policy와 threshold sweep: 모두 성공, 기본 정책 승격은 보류

원시 artifact:

- `.local/cosmos-reason2-2b/packed-max256-correctness-20260813/`
- `.local/cosmos-reason2-2b/packed-max256-chunk-ab-20260813/`
- `.local/cosmos-reason2-2b/packed-max256-drain-policy-20260813/`
- `.local/cosmos-reason2-2b/packed-max256-threshold-sweep-20260813/`

## vLLM 대비 다음 병목

packed prefill과 D64는 prefill-heavy 처리량 격차를 크게 줄였지만, 현재 최신 D64 trace는 EOS를 무시해 requested
output을 전부 생성한 반면 기존 vLLM production trace는 EOS 종료를 허용했다. balanced/decode-heavy의 raw
token/s를 그대로 최신 공정 비교라고 부를 수 없다. 다음 순서는 다음과 같다.

1. EOS 정책, 실제 generated token 수, arrival trace, transport warm 상태를 맞춘 vLLM/Current 재측정
2. past-KV 128/512/1024/1536에서 D16/24/32/48/64 decode-only와 overlap cost table 생성
3. CUDA graph를 유지할 수 있는 128-contract engine과 graph-off 256-contract engine의 workload router 비교
4. TensorRT phase I/O/deepstack buffer와 두 execution-context workspace 중복을 줄여 graph headroom 확보
5. short TTFT의 남은 scheduler wait를 continuous admission과 prefill token-budget formation으로 축소
