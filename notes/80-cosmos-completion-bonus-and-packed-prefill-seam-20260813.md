# Cosmos completion-aware prefill 실험과 packed prefill 경계

## 결론

131-token prompt의 마지막 3-token continuation을 먼저 끝내도록 completion-aware bucket score를 구현했지만,
현재의 padded prefill engine에서는 전체 short-trace TTFT p95가 악화됐다. 이 정책은 기본값 `0`인 실험용
knob로만 유지하고 workload preset에는 넣지 않는다.

정적 우선순위로 해결되지 않는 이유는 continuation을 앞당길 때 별도 TensorRT enqueue가 추가되기 때문이다.
진짜 다음 병목은 서로 다른 request의 token을 한 compact binding에 넣는 packed/varlen prefill이다.

## 구현

`PhaseQueueSchedulerConfig::prefillCompletionBonusTokens`는 prefill을 끝내는 continuation row에 virtual useful-token
credit을 준다. credit은 다음 두 조건으로 제한한다.

1. 남은 row가 configured chunk의 절반보다 작아야 한다.
2. 실제 credit은 `min(configured bonus, chunk size - row tokens)`이다.

따라서 chunk 128, bonus 128에서 3-token tail은 125-token credit을 받지만 108-token tail은 credit을 받지 않는다.
기본값 0에서는 기존 bucket ordering이 그대로다. CLI와 real-request runner에는 각각
`--prefillCompletionBonusTokens`, `--prefill-completion-bonus-tokens`를 추가했다.

## 실제 trace 결과

조건은 Cosmos Reason2-2B FP16 indexed-paged, independent TensorRT contexts, P8/D64, fixed chunk 128,
prefill token budget 1024, CUDA graph shape priming, 48-request short trace다.

| 정책 | token/s | TTFT median | TTFT p95 | TPOT p95 | E2E p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bonus 0, 3-run median | 1995.6 | 180.58ms | 282.91ms | 14.959ms | 493.99ms |
| 무제한 completion bonus pilot | 2006.6 | 200.52ms | 315.37ms | 16.754ms | 499.31ms |
| unused-chunk cap pilot | 2024.3 | 199.15ms | 313.28ms | 16.678ms | 494.80ms |
| half-chunk guard pilot | 2011.3 | 199.75ms | 314.42ms | 16.688ms | 498.11ms |

모든 pilot은 reference와 48/48 request의 output token 수, finish reason, output text가 일치했다. pilot은 각 1회라
throughput 차이는 채택 근거로 쓰지 않는다. 반면 세 변형 모두 TTFT p95가 약 30ms 나빠졌으므로 명확한 reject다.

무제한 정책은 108-token continuation을 B2로 너무 일찍 실행했다. guard를 추가하면 이 문제는 줄지만, 3-token
continuation B4 자체가 21~33-token initial request들을 한 dispatch 뒤로 미룬다. 기존 정책의 tail request
11/23/35는 빨라졌지만 새로운 tail은 short request 3/13/25로 이동했다. 이는 우선순위 문제가 아니라 별도
prefill launch 비용 문제다.

## true packed/varlen이 필요한 지점

현재 dense ragged는 host에서 `[B, Smax]`로 right padding한다. true packed는 `[T, hidden]`과 cumulative sequence
lengths를 사용해 initial/continuation의 useful token을 한 번의 실행으로 처리해야 한다. 현재 공통 FMHA runner에는
compact layout 지원이 있지만 engine 전체 계약은 아직 padded batch다.

필요한 변경은 다음 순서다.

1. Python export의 attention custom-op schema에 compact prefill mode와 cumulative Q lengths를 추가한다.
2. `cpp/plugins/attentionPlugin/`에서 batch를 Q의 첫 dimension으로 추론하지 않고 context-length tensor에서 얻는다.
3. RoPE와 KV write가 `[B,S]` physical offset 대신 compact token index와 row→stable KV slot mapping을 함께 사용하게
   한다.
4. chunked-prefix K/V gather/deinterleave도 compact row offsets를 사용한다.
5. `tensorrt_edgellm/models/default/modeling_default.py`의 batch-dimension `GatherND` last-token selection에 compact
   index 경로를 추가한다.
6. builder profile과 runtime binding registry에 total-token dynamic axis를 연결한다.
7. export → build → inference 뒤 dense reference와 output을 비교하고 P1/P2/P4/P8 kernel-group cost를 다시 만든다.

`cpp/kernels/contextAttentionKernels/contextFMHARunner.*`의 `isSPadded=false`는 FMHA 계산 기반이 이미 있다는 뜻이지,
현재 exported decoder가 곧바로 packed input을 받을 수 있다는 뜻은 아니다. RoPE/KV write, last-token gather,
TensorRT binding shape까지 함께 바뀌어야 안전하다.

## 검증

- scheduler 단위 테스트 46개 통과
- 작은 final continuation 선택과 거의 full continuation 비선택 경계 테스트 추가
- `llm_phase_bench` 재빌드 성공
- 실험 옵션 기본값 0, production preset 변경 없음

artifact:

- `.local/cosmos-reason2-2b/ttft-completion-bonus-20260813/`
- `.local/cosmos-reason2-2b/ttft-policy-search-20260813/`

