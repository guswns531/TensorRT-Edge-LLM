# E/P/D/Copy CUDA activity-mask timeline

## 목표

GPU 하드웨어 utilization을 추측하는 대신, serving runtime이 실제 CUDA stream에 enqueue한 작업의 시작과
끝만으로 다음 네 activity bit를 만든다.

```text
E = 0001  encoder engine
P = 0010  prefill dispatch
D = 0100  decode dispatch
C = 1000  independently issued memory copy
```

동일한 CUDA event epoch에 대한 interval을 sweep하면 임의 시점의 mask가 만들어진다.

```text
time  ───────────────────────────────────────────>

E       [==========]              [=======]
P             [=============]
D       [===] [===] [===] [===] [===] [===]
C            [--]          [---]

mask    0101 1111 0110 ... 0000 ...
```

이 값은 SM occupancy나 kernel utilization이 아니다. `E/P/D/C 중 어느 stream service interval이 GPU
timeline에서 열려 있었는가`만 나타낸다. 따라서 scheduler가 계획한 overlap과 실제 outstanding execution을
직접 비교하는 action-fidelity 신호로 사용할 수 있다.

## 구현 구조

### 공통 recorder

`cpp/runtime/scheduling/phaseActivityTimeline.{h,cpp}`에 opt-in recorder를 추가했다.

1. 하나의 synchronized CUDA event를 공통 epoch로 만든다.
2. 각 작업의 stream에 timing start/end event를 기록한다.
3. 완료된 event만 비동기로 수집하고, 종료 시에는 모두 drain한다.
4. `[start, end)` boundary를 sweep하면서 네 bit의 active count를 관리한다.
5. 동일 종류의 interval이 겹쳐도 boolean이 아니라 count로 관리하므로 먼저 끝난 interval이 bit를 잘못
   내리지 않는다.

모든 stream은 같은 CUDA context에 속해야 하며, 하나의 interval은 시작한 stream에서 끝나야 한다. recorder는
mutex로 보호되어 encoder thread와 P/D worker가 동시에 interval을 등록할 수 있다.

### runtime 연결

```text
PhaseThreeCoordinator
├── PhaseVisionAdapter
│   ├── encoder stream: E interval
│   └── copy stream:    C interval (fallback output D2D가 있을 때만)
└── IndependentPhaseAsyncServer
    └── IndependentPhaseCoordinator
        └── PhaseDispatchWorker
            ├── prefill stream: P interval
            └── decode stream:  D interval
```

- `PhaseDispatchWorker`는 실제 prefill/decode enqueue callback을 각각 P/D interval로 감싼다.
- `PhaseVisionAdapter`는 `MultimodalRunner::infer()`를 E interval로 감싼다.
- vision runner가 external output binding을 지원하지 않으면 다음 handoff로 실제 독립 Copy stream을 쓴다.

```text
encoder stream  infer ── record encoderDone
                                  │
copy stream                 wait ─┴─ D2D output copy ── readyEvent
                                                        │
prefill consumer                         payload ready ─┘
```

- direct output binding을 지원하면 encoder가 retained slab에 바로 쓰므로 C interval과 copy 자체가 없다.
- P/D 내부에서 같은 P/D stream에 enqueue되는 작은 metadata copy는 별도 C로 중복 계상하지 않는다. 이를
  C로도 표시하면 하나의 물리 stream 작업을 P+C overlap으로 잘못 표현하게 된다.
- recorder의 public `begin/end`는 임의의 동일-context copy stream을 받을 수 있다. 향후 KV offload나 별도
  admission upload stream이 생기면 같은 C bit에 연결할 수 있다.

### 관측 window

interval CSV는 공통 epoch 상대 절대 시각을 보존한다. mask와 summary는 첫 GPU 작업 시작부터 마지막 GPU 작업
종료까지를 active span으로 삼고 첫 작업을 0ms로 rebase한다.

```text
process/engine setup idle     첫 GPU work       내부 gap       마지막 GPU work    teardown idle
-------------------------|================| 0000 |================|-----------------
                         <--------- summary / mask window -------->
```

따라서 엔진 load, vision context 생성, HTTP client 연결 대기를 GPU service idle로 과장하지 않는다. 반면 active
span 내부의 실제 dispatch gap은 `0000`으로 그대로 남는다.

## 출력

`llm_phase_context_smoke`에 다음 환경 변수를 주면 계측이 켜진다.

```bash
TRT_EDGELLM_PHASE_ACTIVITY_PREFIX=/path/to/run-mask
```

기본 serving에서는 recorder를 만들지 않으므로 event 추가 비용이 없다. opt-in 실행은 다음 세 파일을 만든다.

### `run-mask-intervals.csv`

원본 CUDA event interval이다.

```text
interval_index,interval_id,correlation_id,kind,bit,name,start_ms,end_ms,duration_ms
0,1,23000,encoder,0001,encoder_engine,...
1,2,1,prefill,0010,prefill_dispatch,...
```

`correlation_id`는 E/C에서는 대표 request ID, P/D에서는 dispatch index다.

### `run-mask-segments.csv`

요청한 mask timeline이다.

```text
segment_index,active_span_start_ms,active_span_end_ms,duration_ms,mask,binary_mask,encoder,prefill,decode,copy,active_streams
0,0.000000,13.973511,13.973511,2,0010,0,1,0,0,1
1,13.973511,14.033203,0.059692,0,0000,0,0,0,0,0
2,14.033203,30.298828,16.265625,2,0010,0,1,0,0,1
```

### `run-mask-summary.csv`

16개 mask 각각의 duration/ratio와 파생 지표를 기록한다.

```text
all_idle  = duration(mask == 0000)
epd_idle  = duration(mask == 0000 or mask == 1000)
any_epd   = window - epd_idle
epd_triple = duration(mask == 0111 or mask == 1111)
four_way  = duration(mask == 1111)
E/P/D/C duty = 해당 bit가 켜진 모든 mask duration의 합 / window
```

각 duty의 합은 overlap을 중복 포함하므로 100%를 넘을 수 있다. `any_epd`는 하나 이상의 E/P/D가 열린 wall-time
비율이라 100%를 넘지 않는다.

## 실제 검증

환경은 RTX 3080 10GB, TensorRT 11.0, CUDA 13.3, Cosmos Reason2-2B FP16 tied engine, P8/D64,
fixed prefill chunk 128이다.

### 24-request text HTTP trace

실제 OpenAI/SSE gateway와 production IPC adapter를 통해 24개 요청, prompt 2,156 tokens, output 520 tokens를
완료했다. token hash도 결정적으로 일치했다.

| metric | 결과 |
|---|---:|
| active-span window | 353.094 ms |
| any E/P/D | 98.97% |
| internal all-idle | 1.03% |
| P duty | 36.11% |
| D duty | 62.86% |
| E/C duty | 0% / 0% |
| generated throughput | 1,465.73 token/s |
| TTFT mean / p95 | 78.42 / 145.11 ms |
| TPOT mean / p95 | 10.77 / 20.08 ms |
| E2E mean / p95 | 276.42 / 348.33 ms |

이 trace에서는 실제 mask가 `0010`, `0100`, 내부 `0000`만 나왔다. independent P/D context가 존재해도 현재
scheduler가 이 짧은 trace에서 P+D를 동시에 outstanding으로 만들지 않았다는 뜻이다. recorder가 overlap을
놓친 것이 아니라 현재 execution의 action fidelity를 그대로 보여 준다.

### 실제 Cosmos VLM 단일 요청

`giant_panda.jpeg`를 visual engine에 입력한 실제 `E -> P -> D` 실행은 다음과 같았다.

| metric | 결과 |
|---|---:|
| active-span window | 86.482 ms |
| any E/P/D | 99.09% |
| internal all-idle | 0.91% |
| E duty | 25.04% |
| P duty | 23.66% |
| D duty | 50.39% |
| C duty | 0% |

semantic output은 `The panda bear, with its distinctive black...`으로 정상이다. 이 Cosmos visual runner는 retained
slab에 direct output을 쓰므로 C=0이 올바른 결과다. 불필요한 copy를 만들어 mask를 채우지는 않는다.

### correctness와 build

- `unitTest`, `llm_phase_context_smoke` TensorRT/CUDA build 통과
- `PhaseActivityTimelineTest.*`, `PhaseDispatchWorkerTest.*`, `PhaseKernelGroupRecorderTest.*`: 8/8 통과
- synthetic sweep에서 `0000`, `0101`, `0011`, `1011`, `1111`, `0111` 경계와 duty 합산 검증
- 실제 서로 다른 CUDA stream의 E/C event를 공통 epoch로 환산하는 GPU test 통과
- Copy-only `1000`은 `all_idle`이 아니지만 `epd_idle`인 경계 조건 검증
- startup/trailing idle 제거와 active-span 내부 `0000` 보존 검증

Copy stream handoff까지 포함한 전체 C++ suite는 1,147개 중 1,101 pass, 42 skip이었다. 최초 실행에서 네
수치 test가 실패했지만 `NormalizeImage.Accuracy`는 단독 재실행에서 통과했고,
`RopeWriteKvDecodeTreeAttention.Accuracy`도 이어진 3회 반복을 모두 통과했다. 남은 두 persistent failure는
변경 전부터 SM86에서 재현되는 Yarn/M-RoPE 약 `1.1e-3~1.2e-3` tolerance다. 이번 변경은 해당 kernel을
수정하지 않는다.

## 해석과 다음 사용처

이 mask를 scheduler의 online cost에 바로 넣기 전에 다음 순서가 안전하다.

1. 계획한 action의 phase set과 실제 non-zero mask set을 비교한다.
2. P+D를 선택했는데 `0110`이 전혀 생기지 않거나, serial E를 선택했는데 `0101`이 생기는 action-fidelity
   위반을 찾는다.
3. workload별로 `0000`, single-phase, two-way, three-way, four-way wall time을 집계한다.
4. overlap mask가 생긴 dispatch의 makespan과 isolated phase cost를 연결한다.
5. overlap 자체가 아니라 deadline-safe effective work compression이 양수일 때만 scheduler가 해당 action을
   선택하게 한다.

현재 구현은 정확히 “작업 interval이 열려 있는가”를 측정한다. 실제 SM이 얼마나 찼는지, memory-copy engine이
몇 % 사용됐는지까지 보려면 Nsight Systems/CUPTI hardware counter가 별도로 필요하며 이 mask와 혼동하지 않는다.
