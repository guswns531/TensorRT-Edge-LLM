# 261. VLM calibration asset 누락과 warmup 응답 검증 교정

최종 108회 결과와 추가 engine-only 비교는 [262번 보고서](262-validated-v0101-full12-kv-and-engine-control-20260909.md)에 모두 정리했다. 이 문서의 단일 smoke 수치를 최종 정책 순위로 사용하지 않는다.

## 발견

Generic VLM trace는319 requests(239 text +80 vision)를 담고 있지만80 vision 모두
삭제된 `.local/upstream-v010/examples/multimodal/pics/woman_and_dog.jpeg`를 참조했다.
Old 로그의 encoderStarts=85와 현재5(측정 요청만) 차이를 조사하다 확인했다.
현재 trace 파일이 동일하다는 사실만으로 실제 calibration이 동일하다고 판단하면 안 된다.

Retained `run_vllm_trace_bench.py`는 measurement에서 HTTP/error를 검사하지만,
warmup의 `execute_requests()` 반환 rows는 검사하지 않고 제출 수만 completed_warmup에 더했다.
따라서 completed_warmup_requests=319와 calibration_converged=true는 vision 성공 증거가 아니다.
기존 로그에 요청별 warmup response를 남기지 않았으므로 과거 실패80건의 상세 error를 복원해
관측했다고 주장하지 않는다. 누락 asset과 encoder 실행수 차이는 직접 확인됐다.

## 수정

- 원본 trace나 과거 결과를 덮어쓰지 않는다.
- Replay CLI `--asset-remap OLD=NEW`로 명시한 prefix만 교정한다.
- Workload와 calibration의 모든 file URL을 실행 전에 검사하고 파일 SHA256을 기록한다.
- 현재 root의 woman_and_dog.jpeg는 이전 f4eb53e tree와 같은 Git blob이다:
  `9fdc040050624556464ffa5112dde397ccd792c6`.
- `guarded_trace_client.py`는 신뢰하는 retained client를 사용하되 warmup을 포함한 모든
  response의 HTTP200/error없음/요청수 일치를 검증한다. 실패하면 serving 측정으로 넘어가지 않는다.
- `warmup-validation.json`에 각 batch의 expected/responses/failed와 request classes를 남긴다.
- Legacy client 구현 경로는 command manifest의 environment에 명시하며 runtime/policy를 바꾸지 않는다.

## 사전 테스트

HTTP500, HTTP200+SSE error, missing response는 모두 거부했다. 정상 response는 허용한다.
Remap 없는 원본 전체12 manifest는 missing image로 사전 거부된다. 올바른 remap 후14개
입력 문서와5개 고유 asset 검증이 통과했다. GPU 실행 전에 완료한 검사다.

## 비교 해석

미최적화 suite는 build confound, 첫 Release VLM suite는 calibration asset confound가 있다.
각각의 raw 수치를 보존하되 공정한 old/current policy gain으로 인용하지 않는다.
Text-only Release 수치는 이미지 문제의 영향이 없지만 새 complete suite와 분리해 표시한다.
Engine/KV/RLS 구현은 이번 수정으로 변경하지 않는다. 측정 workload의 요청 내용도 바꾸지 않는다.
VLM calibration만 원래 의도한 동일 이미지의 성공 실행으로 복구한다.

수정 후 결과 위치: `.local/results/v0101-forward-port/validated-parity-20260909/`.
Full suite 완료 여부는 그 디렉토리의 completion.json으로 확인한다.

## 교정 후 대표 smoke

VLM warmup319/319 responses, failed0을 실제 확인했다. Encoder starts/completions도85로,
80 calibration+5 measurement가 실행됐다. 두 workload에서 V0/V1/V2 output hashes가 동일하다.

| Workload | V0 token/s | V1 token/s | V2 token/s |
|---|---:|---:|---:|
| Balanced | 4406.22 | 4420.34 | 4537.52 |
| Multi-image | 304.18 | 239.20 | 238.35 |

각1회다. 이미지 warmup을 복구하면 반드시 더 빨라진다고 가정하면 안 된다. Multi-image에서는
V0와 learned policy의 차이가 오히려 커졌다. 이는 policy/observation/engine interaction을
다시 검증해야 한다는 신호이며 같은 실제 학습 조건의 full12 결과가 우선이다.
입력 검증을 건너뛰어 더 잘 나온 수치를 production winner로 승격하지 않는다.

같은 smoke의 measurement-start/final graph counters를 빼면 P miss delta는 모두4,
D miss delta는 V0=34/V1=54/V2=57이다. 이 계약은 graph capture/replay가0이므로 eager dispatch의
proxy로 사용한다. Calibration까지 포함한 누계 counter를 직접 비교하지 않았다.
V1/V2에서 D cohort fragmentation이 증가한 후보 증거이며, 반복성 및 forced action의 인과 효과를
검증하기 전에는 모델 하나가 근본 원인이라고 단정하지 않는다.

## 반복 결과에 따른 추가 주의

후속 full12 V0의 Multi-image3회 중앙값도238.51 token/s로 내려갔다. P delta는4로 같았으나
D delta는42/57/57이었다. 따라서 smoke V0=304 vs V1=239만으로 학습 정책의 인과 손실을
주장하면 안 된다. 같은 V0에서도 cohort/arrival/preparation timing 변화가 큰 차이를 만든다.
Old reference가1회라는 한계를 포함해 반복 분산과 engine-only control을 우선 검토한다.

## Frozen vLLM과 KV 계약의 추가 확인

Equal-cap Balanced server.log/metrics에서 vLLM0.27.1, dtype=float16,
KV dtype=float16, kv_cache_memory_bytes=3758096384, block_size=16,
num_gpu_blocks=2048, 총32768tokens를 확인했다. 우리도KV3584MiB/32768tokens이나
page128이므로256pages다. 동일 bytes가 동일 fragmentation 특성을 뜻하지 않는다.
Cosmos의 aggregate page payload는 우리14MiB vs vLLM1.75MiB다.
이는 v0.10.0→v0.10.1 사이 새 차이가 아니라 두 serving stack의 기존 차이다.

또한 이번 current log는 page reservation mode=full, growth_owners=0이다.
Stable page lease는 admission/completion에 따라 동적으로 예약/반환되지만,
이번 실행을 headroom 모드의 decode-step별 pool growth 실험으로 설명하지 않는다.
실제 physical pool allocation은 시작 때 고정한다.
