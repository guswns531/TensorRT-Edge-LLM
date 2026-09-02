# R7 Prepared Directional Execution 구현과 최종 검증

날짜: 2026-09-02

브랜치: `codex/v010-phase-forward-port`

대상: `nvidia/Cosmos-Reason2-2B`, FP16, RTX 3080 10 GiB, TensorRT 11.0/CUDA 13.3

선행 문서: `notes/219-completion-residual-r0-r6-results-20260902.md`

후속 검증: `notes/221-process-local-calibration-and-saturation-r8-results-20260902.md`

## 1. Executive summary

R0--R6의 핵심 blocker는 scheduler가 선택한 residual offset과 GPU에서 실제로
materialize된 newcomer kernel start가 일치하지 않는다는 것이었다. R7에서는 정책 score를
추가로 조정하지 않고 실행 mechanism을 고쳤다.

```text
기존
decision -> incumbent enqueue -> host-side newcomer preparation -> late kernel start

R7 controlled path
decision
  -> exact P/D cohort materialization or asynchronous E preparation
  -> newcomer stream에 device wait 선설치
  -> incumbent stream의 start signal
  -> mapped semaphore release
  -> exact newcomer TensorRT enqueue/replay
```

결과는 다음과 같다.

1. E/P/D 양방향 6개와 target `0/0.25/0.5/0.75/0.9`, 총 30개 실제 HTTP cell을
   모두 실행했다.
2. 모든 cell의 planned action과 실제 outstanding phase가 일치했고 action fidelity는
   `30/30`이었다. Token hash도 전 cell에서 결정적이었다.
3. causal하게 도달 가능한 대표 target에서 P→D는 `0.5 -> 0.524`,
   `0.75 -> 0.771`, `0.9 -> 0.926`; E→D는 `0.5 -> 0.524`,
   `0.75 -> 0.761`; P→E는 `0.9 -> 0.909`로 재현됐다.
4. target 0/0.25가 일부 direction에서 늦는 것은 gate 오류가 아니라 newcomer의
   exact cohort/profile 준비가 완료되기 전에 incumbent가 이미 진행한 **causal floor**다.
5. P→D와 E→D newcomer decode는 graph replay가 실제 사용됐고 host execute submit은
   대체로 `70--106 us`였다. 반대로 D→P의 P submit은 약 `4.8--5.0 ms`여서
   early-offset 실현과 overlap 수익성을 제한했다.
6. production 기본 경로는 controlled injection과 분리했다. 일반 VLM request는 기존
   encoder-first action semantics를 보존하며, 실험 gate는 명시적으로 요청한 첫 pair에만
   one-shot으로 적용된다.
7. Release build의 최종 12-workload 조합에서 frozen vLLM 대비 token throughput은
   `12/12` 모두 높았다. Frozen Current 대비로는 대부분 ±3%이며 multi-image는
   `+14.34%`, vision-heavy는 `-3.54%`, bimodal은 `-3.08%`였다.

R7로 **decision realization fidelity**는 해결했다. 그러나 completion-aware policy
authority는 아직 승격하지 않는다. 각 controlled cell이 fresh process의 첫 관측이므로
process-local completion predictor가 warmup되기 전에 injection이 발생해
`completion_prediction_ready=false`였다. 외부 registry나 TTL을 추가하지 않는 설계에서
다음 문제는 mechanism이 아니라 동일 process 안의 safe calibration과 natural action
coverage다.

## 2. 최종 아키텍처

### 2.1 Production plane과 controlled research plane

```text
                         OpenAI-compatible HTTP request
                                      |
                                      v
                    Request DAG + stable KV/vision leases
                                      |
                                      v
                         Global E/P/D ready snapshot
                                      |
                    +-----------------+------------------+
                    |                                    |
                    v                                    v
          Production action selector             Controlled R7 injection
          feasibility -> SLO -> value             direction + target fraction
                    |                                    |
                    v                                    v
       ordinary encoder-first E action       exact incumbent/newcomer preparation
                    |                                    |
                    +-----------------+------------------+
                                      v
                    shared CUDA context, independent TRT contexts
                       E context       P context       D context
                       E stream        P stream        D stream
                                      |
                                      v
                           CUDA start/end event plane
                                      |
                 +--------------------+---------------------+
                 |                    |                     |
                 v                    v                     v
          activity mask E/P/D/C   exact cost update   research side channel
```

Controlled path만 다음 release dependency를 추가한다.

```text
newcomer stream:  wait(mapped gate == 1) -> TensorRT work already submitted
                                      ^
                                      |
gate stream:      wait(incumbent signal) -> optional delay -> host release
                                      ^
                                      |
incumbent stream: write(mapped incumbent flag = 1) immediately after start
```

CPU가 incumbent 실행 중에 바쁜 CUDA kernel을 띄우는 이전 방식과 달리, GPU SM을
소비하지 않는 stream memory operation으로 dependency를 표현한다. Gate는 같은 CUDA
context 안의 독립 TensorRT execution context를 연결할 뿐 context를 합치지 않는다.

### 2.2 Prepare와 execute의 분리

P/D는 scheduler가 exact rows를 materialize한 뒤 one-shot preamble을 phase stream에
설치한다. Preamble은 해당 dispatch의 start event 직전에 gate를 arm하고, callback은
오직 한 번 소비된다.

E는 HTTP/image preprocessing과 TensorRT profile 준비를 encoder submission 앞에서
수행한다.

```text
image request
   -> decode/resize/normalize
   -> QwenViT optimization profile selection
   -> bindings/workspace preparation
   -> directional gate arm
   -> encoder execute
```

이 분리 덕분에 측정 target은 scheduler decision timestamp가 아니라 실제 incumbent CUDA
start 기준이 된다.

### 2.3 Host realization timing

각 in-flight phase는 다음 host timestamp를 노출한다.

```text
prepare_start_host_ns
prepare_end_host_ns
execute_start_host_ns
execute_end_host_ns
graph_replay
```

분석기는 이를 다음처럼 사용한다.

```text
decision_to_enqueue = phase enqueue host time - global decision time
prepare_host        = prepare end - prepare start
execute_submit_host = execute end - execute start
actual_fraction     = (newcomer GPU start - incumbent GPU start)
                      / incumbent isolated reference
```

따라서 offset miss를 scheduler, input/profile preparation, TensorRT submission, GPU
dependency 중 어느 구간이 만들었는지 분리할 수 있다.

## 3. 코드 구조와 책임

| 책임 | 구현 위치 |
|---|---|
| mapped semaphore gate와 arm/signal | `cpp/kernels/phase/phaseCudaDirectionalGate.cu` |
| gate public API/ownership | `cpp/runtime/scheduling/phaseCudaDirectionalGate.h` |
| exact P/D dispatch 직전 one-shot preamble | `cpp/runtime/scheduling/phaseDispatchWorker.{h,cpp}` |
| P/D prepare/execute/graph timing | `cpp/runtime/scheduling/independentPhaseCoordinator.{h,cpp}` |
| async server stream/event/preamble forwarding | `cpp/runtime/scheduling/independentPhaseAsyncServer.{h,cpp}` |
| E/P/D controlled pair 조립과 production order | `cpp/runtime/scheduling/phaseThreeCoordinator.{h,cpp}` |
| multimodal generic prepare API | `cpp/multimodal/multimodalRunner.{h,cpp}` |
| QwenViT TensorRT profile preparation | `cpp/multimodal/qwenViTRunner.{h,cpp}` |
| vision adapter prepare call | `cpp/runtime/scheduling/phaseVisionAdapter.cpp` |
| unified host execution telemetry schema | `cpp/runtime/phase/mechanism/phaseUnifiedEvent.h` |
| 실제 HTTP 6-direction runner | `benchmarks/phase_serving/run_directional_injection_matrix.py` |
| causal reachability/fidelity analyzer | `benchmarks/phase_serving/analyze_directional_injection.py` |
| smoke CLI와 graph/client concurrency knobs | `examples/llm/llm_phase_context_smoke.cpp` |
| C++ mechanism tests | `unittests/phaseShellTest.cpp` |
| Python artifact tests | `tests/python-unittests/test_directional_injection.py` |

새 옵션은 연구 실행에만 opt-in이다. Default production에서 directional control은 disabled다.

## 4. 왜 R2 gate가 실패했고 R7이 동작하는가

R2는 incumbent start 뒤 delay event를 만들었지만 newcomer TensorRT enqueue가 늦었다.

```text
요청 target 50%
incumbent start ---- 50% gate release ---------------- end
                       |
CPU:              [binding/profile/TRT enqueue........]
                                                    |
GPU newcomer:                                        start (~92%)
```

즉 gate release는 빨랐지만 실행할 newcomer work가 GPU queue에 없었다. R7은 반대로
newcomer wait를 먼저 queue에 materialize하고 incumbent signal을 나중에 넣는다.

```text
CPU:  prepare newcomer -> queue wait -> submit incumbent
GPU:  newcomer wait --------------------+
      incumbent start -> signal --------+-> target delay -> newcomer start
```

여기서도 절대 0%가 항상 가능한 것은 아니다. Exact cohort를 아직 만들 수 없거나 encoder
input/profile 준비가 끝나지 않았다면 scheduler가 pair를 확정하는 시각 자체가 늦다. 분석기는
이를 `target_causally_reachable`과 `target_late_by_us`로 구분한다. 도달 불가능한 target을
gate fidelity 실패로 계산하지 않는다.

## 5. Six-direction 30-cell 결과

조건은 E1, P 최대 8, D 최대 32, target `0/0.25/0.5/0.75/0.9`다. `Compression`은
동일 pair의 isolated serial-equivalent horizon 대비 실제 action makespan 압축률이다.
양수는 이득, 음수는 interference가 더 큼을 뜻한다. `Reachable=N`인 cell은 target보다
앞에서 newcomer를 시작할 준비가 물리적으로 끝나지 않은 경우다.

|Direction|Target|Actual|Reachable|Overlap|Inc/New rows|Prepare us|Submit us|Graph|Compression|Fidelity|
|---|---:|---:|:---:|:---:|---:|---:|---:|:---:|---:|:---:|
|P→D|0.00|0.491|N|Y|8/1|208|70|Y|+3.05%|Y|
|P→D|0.25|0.507|N|Y|8/1|210|79|Y|+2.63%|Y|
|P→D|0.50|0.524|Y|Y|8/1|211|78|Y|+1.36%|Y|
|P→D|0.75|0.771|Y|Y|8/1|202|73|Y|-3.24%|Y|
|P→D|0.90|0.926|Y|Y|8/1|208|106|Y|-18.89%|Y|
|D→P|0.00|0.114|N|Y|1/8|379|4857|N|-0.34%|Y|
|D→P|0.25|0.296|N|Y|1/8|356|4848|N|-2.29%|Y|
|D→P|0.50|0.532|N|Y|1/8|184|4974|N|-10.06%|Y|
|D→P|0.75|0.789|N|Y|1/8|378|4860|N|-15.80%|Y|
|D→P|0.90|0.944|Y|Y|1/8|190|4825|N|-19.37%|Y|
|E→D|0.00|0.517|N|Y|1/1|224|72|Y|+15.51%|Y|
|E→D|0.25|0.530|N|Y|1/2|233|74|Y|+14.28%|Y|
|E→D|0.50|0.524|Y|Y|1/1|221|72|Y|+14.48%|Y|
|E→D|0.75|0.761|Y|Y|1/2|220|72|Y|+12.81%|Y|
|E→D|0.90|1.036|Y|N|1/1|259|77|Y|-0.10%|Y|
|D→E|0.00|0.000|Y|Y|1/32|1035|2831|N|+7.00%|Y|
|D→E|0.25|0.289|N|Y|32/1|15223|3717|N|+5.69%|Y|
|D→E|0.50|0.614|N|Y|32/1|15938|24353|N|-1.27%|Y|
|D→E|0.75|0.830|N|Y|32/1|15100|25571|N|-6.90%|Y|
|D→E|0.90|0.927|N|Y|32/1|2135|19900|N|-9.72%|Y|
|E→P|0.00|1.072|N|Y|1/2|333|10430|N|-10.30%|Y|
|E→P|0.25|0.862|N|Y|1/3|343|7868|N|+7.11%|Y|
|E→P|0.50|0.532|N|Y|1/8|235|4917|N|+2.80%|Y|
|E→P|0.75|0.833|N|Y|1/4|343|7620|N|+8.73%|Y|
|E→P|0.90|0.907|Y|Y|1/4|232|4906|N|+7.09%|Y|
|P→E|0.00|0.006|N|Y|1/1|1753|12450|N|+22.16%|Y|
|P→E|0.25|0.262|N|Y|3/1|3429|16331|N|+14.66%|Y|
|P→E|0.50|0.510|N|Y|5/1|1079|6356|N|+10.53%|Y|
|P→E|0.75|0.760|N|Y|4/1|9827|21661|N|-1.90%|Y|
|P→E|0.90|0.909|Y|Y|3/1|1087|7025|N|+3.22%|Y|

### 5.1 해석

- 같은 pair도 offset에 따라 수익성 부호가 바뀐다. P→D는 0.5에서 `+1.36%`지만
  0.9에서 `-18.89%`다.
- 방향도 중요하다. P→D 0.5는 소폭 이득이지만 D→P 0.5는 `-10.06%`다.
- small E→D의 middle offset은 `+12.81--14.48%`로 강한 기회다. 하지만 0.9는
  incumbent 종료 뒤 시작해 이득이 사라진다.
- E/P cell의 row 수가 target별로 같은 것은 아니다. Pair decision이 exact ready cohort를
  materialize하는 시점과 연결되므로 action-induced formation effect가 함께 관측된다.
  따라서 이 표는 pure kernel microbenchmark가 아니라 실제 request-level execution 결과다.
- `Fidelity=Y`는 requested direction의 action과 실제 phase order/outstanding set이
  일치했다는 뜻이지, causally impossible target까지 정확히 맞았다는 뜻은 아니다.

## 6. Release build와 성능 재검증

초기 성능 run은 CMake build type이 비어 있어 약 30% regression처럼 보였다. 이 run은
근거에서 제외했다. `CMAKE_BUILD_TYPE=Release`로 재구성하고 동일 binary lifecycle에서
다시 측정했다.

Production VLM path에서 일반 E action의 순서를 P/D-first로 바꿨던 중간 구현은
vision-heavy E formation을 악화시켰다. Controlled experiment는 prepared gate를 갖고 있지만
ordinary traffic까지 그 순서를 강제할 이유가 없으므로 일반 경로를 encoder-first로 복원했다.
그 결과 vision-heavy throughput regression은 `-6.52%`에서 `-3.54%`로 회복했고,
multi-image는 frozen Current 대비 `+14.34%`가 됐다.

### 6.1 최종 12-workload 조합

Text-only 5개는 Release 12x3 run을 사용한다. E path 복원은 text-only를 건드리지 않으므로
그 결과를 그대로 사용했다. VLM 7개는 복원 뒤 fresh-process 3회 median을 사용한다.
Latency는 작을수록 좋다.

|Workload|R7 tok/s|vs frozen Current|vs vLLM|TTFT mean/p95 ms|TPOT mean/p95 ms|E2E mean/p95 ms|Peak MiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|short|2448.21|-1.94%|+23.43%|97.34/186.88|13.09/24.99|335.04/414.80|9237|
|balanced|4410.74|+1.16%|+2.10%|70.59/172.05|12.46/14.33|1131.84/1780.08|9237|
|decode-heavy|5208.37|-0.84%|+7.29%|73.89/199.66|10.66/11.30|2824.98/4320.76|9237|
|long-prefill|1196.25|+3.19%|+6.72%|2092.71/2613.95|26.40/31.80|4341.71/6082.80|9237|
|bimodal|1882.35|-3.08%|+0.75%|1952.93/4130.07|18.17/28.20|4460.22/9328.68|9237|
|text-heavy|1986.62|+0.47%|+21.52%|370.38/1112.23|23.84/37.46|1596.88/1677.54|9389|
|mixed|1085.62|+0.54%|+17.81%|812.24/2291.15|34.54/42.55|2414.07/2606.26|9461|
|vision-heavy|680.34|-3.54%|+17.46%|1447.30/3302.14|32.73/54.26|2780.47/3556.37|9439|
|poisson|2009.35|-0.59%|+11.63%|225.09/764.78|20.31/36.44|1544.74/1982.71|9407|
|wave/drain|97.80|+0.02%|+2.04%|263.19/313.15|7.94/9.00|509.63/520.95|9573|
|multi-image|321.41|+14.34%|+31.44%|253.45/289.03|7.76/9.31|492.02/497.45|9563|
|late-vision|2557.50|+0.37%|+8.40%|122.45/428.59|9.21/9.27|1442.32/1803.75|9383|

### 6.2 vLLM latency delta

아래 양수는 R7 latency가 vLLM보다 그 비율만큼 짧다는 뜻이다. Frozen vLLM의 request CSV
각 run에서 mean/p95를 재구성한 뒤 valid run의 median을 사용했다.

|Workload|TTFT mean|TTFT p95|TPOT mean|TPOT p95|E2E mean|E2E p95|vLLM peak MiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|short|+44.35%|+29.24%|+2.04%|+5.48%|+21.48%|+17.73%|9035|
|balanced|+51.91%|+52.90%|+17.58%|+17.67%|+21.26%|+20.74%|9039|
|decode-heavy|+51.84%|+49.43%|+23.85%|+24.81%|+25.12%|+25.68%|9039|
|long-prefill|+28.92%|+39.51%|+18.60%|+15.63%|+23.78%|+22.62%|9323|
|bimodal|+22.16%|+3.78%|+23.23%|+35.58%|+20.67%|+10.39%|9327|
|text-heavy|+12.14%|+9.81%|+18.40%|+20.93%|+17.83%|+17.68%|9791|
|mixed|+7.13%|+9.87%|+26.46%|+49.55%|+19.76%|+17.02%|9799|
|vision-heavy|+15.40%|+11.10%|+48.62%|+54.84%|+32.50%|+15.93%|9763|
|poisson|+48.62%|+15.56%|+8.49%|+20.21%|+14.19%|+12.56%|9497|
|wave/drain|-4.13%|+25.92%|+36.14%|+47.88%|+20.10%|+19.93%|9619|
|multi-image|+2.45%|+32.61%|+37.53%|+45.55%|+23.63%|+23.95%|9035|
|late-vision|+20.18%|+32.18%|-24.00%|+6.59%|+8.48%|+7.71%|9071|

R7은 vLLM보다 throughput 12/12, E2E mean/p95 12/12가 좋다. 예외 latency 축은
wave/drain TTFT mean `4.13%` 악화와 late-vision TPOT mean `24.00%` 악화다.
둘 다 p95와 E2E는 R7이 더 좋으므로 분포/phase ordering trade-off로 남긴다.

## 7. E/P/D/Copy activity와 idle

각 workload의 3회 중 token throughput 중앙값 run을 선택했다. HTTP warmup에서 재사용된
request ID는 마지막 completed lifecycle만 선택해 제외했다. CUDA event interval을 4-bit
mask로 sweep했다.

```text
E = 0001, P = 0010, D = 0100, Copy = 1000
```

Duty 합이 100%를 넘을 수 있는 이유는 실제 overlap 때문이다. `Any overlap`은 E/P/D 중
두 개 이상이 동시에 실제 CUDA-event 작업 구간인 시간이다.

|Workload|Window ms|Idle|E duty|P duty|D duty|Copy duty|Any overlap|E+P|E+D|P+D|E+P+D|D gap p95 ms|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|short|422.1|1.46%|0.00%|47.12%|62.26%|0.00%|10.85%|0.00%|0.00%|10.85%|0.00%|14.316|
|balanced|5656.7|2.51%|0.00%|44.43%|77.68%|0.00%|24.62%|0.00%|0.00%|24.62%|0.00%|7.632|
|decode-heavy|14374.7|2.51%|0.00%|17.46%|89.97%|0.00%|9.94%|0.00%|0.00%|9.94%|0.00%|5.272|
|long-prefill|20861.7|1.28%|0.00%|89.56%|53.28%|0.00%|44.12%|0.00%|0.00%|44.12%|0.00%|49.205|
|bimodal|23457.9|1.35%|0.00%|52.48%|70.82%|0.00%|24.65%|0.00%|0.00%|24.65%|0.00%|18.562|
|text-heavy|1681.5|3.44%|24.56%|39.68%|43.62%|0.00%|11.30%|0.00%|0.00%|11.30%|0.00%|102.103|
|mixed|2672.1|3.33%|31.30%|40.15%|37.64%|0.00%|12.42%|0.82%|0.00%|11.60%|0.00%|100.008|
|vision-heavy|3596.8|4.38%|35.16%|43.00%|31.79%|0.00%|14.34%|5.64%|0.00%|8.70%|0.00%|111.735|
|poisson|2300.1|3.18%|18.22%|36.88%|52.87%|0.00%|11.15%|3.51%|0.00%|7.64%|0.00%|37.251|
|wave/drain|6515.9|70.50%|8.02%|8.36%|14.77%|0.00%|1.65%|0.00%|0.00%|1.65%|0.00%|17.056|
|multi-image|491.4|2.80%|26.97%|26.47%|51.25%|0.00%|7.49%|0.00%|0.00%|7.49%|0.00%|9.190|
|late-vision|1797.9|2.63%|9.64%|15.32%|75.83%|0.00%|3.43%|0.00%|0.00%|3.43%|0.00%|1.617|

### 7.1 Idle 해석

- wave/drain을 제외한 burst/continuous workload의 all-idle은 `1.28--4.38%`다.
  GPU에 단순히 넣을 ready work가 없어 생기는 큰 빈칸은 아니다.
- wave/drain `70.50%`는 의도된 request wave 사이 arrival gap이다. 이 구간은 scheduler가
  overlap으로 제거할 수 없다.
- 자연 traffic의 주요 overlap은 P+D다. long-prefill은 active window의 `44.12%`,
  balanced는 `24.62%`가 P+D다.
- 자연 E+D는 이 12개 대표 run에서 `0%`다. 이는 E+D가 무가치하다는 뜻이 아니다.
  Controlled E1→D에서는 `+12.81--15.51%` compression을 확인했다. 현재 production
  selector가 SLO/formation/cold uncertainty 때문에 이 기회를 보수적으로 사용한다는 뜻이다.
- Copy duty가 `0%`인 것은 현재 measured interval에서 별도 `encoder_output_copy` CUDA
  interval이 발생하지 않았기 때문이다. Copy stream이 삭제됐다는 의미는 아니다. 현재
  telemetry만으로 copy가 발행되지 않은 것과 다른 binding/interval에 포함된 것을 구분할 수
  없으므로 관측된 별도 Copy 작업만 정확히 0으로 보고한다.

따라서 overlap 연구의 목적은 idle 제거 하나가 아니다.

```text
낮은 load / arrival gap: overlap 불가능하거나 불필요
busy continuous load:     이미 busy한 E/P/D의 makespan 압축
large phase shape:        interference/formation loss가 커져 serial이 더 좋을 수 있음
```

## 8. Correctness와 validation

최종 검증은 다음과 같다.

- C++ targeted tests: `19/19` pass
  - directional gate/action semantics
  - one-shot P/D dispatch preamble
  - unified timing schema
  - canonical ordering 관련 기존 tests
- Python directional analyzer tests: `7/7` pass
- Release `llm_phase_context_smoke`와 `unitTest` build 성공
- 30 directional cell token hash deterministic
- 12-workload fresh runs token identity deterministic
- `git diff --check` clean

실험용 `.local` artifact는 commit하지 않는다. Source, tests, notes만 repository history에
남긴다.

## 9. 현재 한계

### 9.1 Process-local learner cold start

30-cell matrix는 cell마다 fresh server process를 띄운다. Controlled injection은 그 process의
첫 pair에서 발생해 online completion model의 minimum samples를 채우지 못한다.

```text
completion_prediction_ready = false (30/30)
```

이는 fidelity 실패가 아니다. 실제 completion observation은 기록되지만 그 action을 선택한
시점에는 posterior가 아직 없다. 외부 cost registry, persisted learning, TTL은 사용하지
않는다는 기존 결정을 유지한다.

### 9.2 Natural E overlap evidence density

Natural 12-workload에서 E+P는 일부 생기지만 E+D는 없다. Controlled E1+D의 잠재력은 크지만
이를 바로 static rule로 승격하면 workload/shape fine-tuning이 된다. 동일 process의 warmup
후 continuous feature와 uncertainty가 충분한 상태에서 action choice가 바뀌는지 검증해야 한다.

### 9.3 Vision-heavy variance

Vision-heavy throughput은 frozen Current 대비 `-3.54%`로 3% gate를 0.54pp 벗어났다.
그러나 E2E mean은 `8.30%` 짧고 E2E p95는 `2.69%` 이내였으며, 중간 P/D-first 구현의
`-6.52%`보다 회복했다. E batch formation과 process-level variance를 5회 이상 반복해
promotion 여부를 결정해야 한다.

### 9.4 Bimodal variance

Bimodal도 `-3.08%`로 gate 경계다. 다른 workload를 위한 새 rule을 추가하지 않고 반복
신뢰구간과 request ordering을 먼저 확인한다.

## 10. 다음 실행 계획

우선순위는 다음과 같다.

1. **R7 freeze와 5-repeat confirmation**
   - vision-heavy, bimodal, balanced, multi-image를 같은 Release binary로 5회 이상 반복한다.
   - token identity, E formation, TTFT/TPOT/E2E mean/p95를 함께 gate한다.
2. **In-process calibration trace**
   - fresh-process 첫 pair가 아니라 동일 server에서 isolated E/P/D와 safe pair observation을
     먼저 수집한다.
   - persisted registry 없이 process-local posterior만 준비한다.
3. **Shadow-only E+D contextual evaluation**
   - natural candidate에서 예상 mean/uncertainty, protected slack, successor formation을 기록한다.
   - static `if E1/D32` rule은 추가하지 않는다.
4. **Safe authority promotion gate**
   - posterior lower-confidence bound가 positive이고 TTFT/TPOT slack이 충분한 E+D만 허용한다.
   - 동일 snapshot의 serial action과 bounded replay/controlled reference로 regret를 검증한다.
5. **P/D prepared execution generalization**
   - graph-covered shape에서 prepare/submit gap이 실제 decode cycle을 줄이는지 saturation trace로
     측정한다.
   - completion visibility, sampling, state commit, candidate formation, TRT submit을 계속 분리한다.
6. **Copy-plane observability**
   - zero-copy와 explicit D2D/H2D copy를 telemetry에서 구분한다.
   - copy interval이 없을 때 `not-issued`와 `unobserved`를 구분하는 schema를 추가한다.
7. **Nsight selected points**
   - P→D 0.5/0.9, E→D 0.5/0.75, D→P 0.5를 반복해 SM active, DRAM, L2,
     concurrent kernel, host launch gap을 수집한다.
8. **Generality**
   - 두 번째 GPU/모델에서 동일 continuous feature model이 다른 payoff surface에 적응하는지
     검증한다.

Production 원칙은 바꾸지 않는다.

```text
deterministic mechanism decides legality
profile-free process-local controller estimates value
uncertain or SLO-unsafe overlap falls back to serial/current action
```

## 11. Artifact 위치

```text
.local/completion-r7-six-direction-e1-final-20260902/
  directional-injection-samples.csv
  directional-injection-artifact.json

.local/completion-r7-release-current-12x3-20260902/
  text-only 5 workloads, fresh process x3

.local/completion-r7-production-order-vlm-7x3-20260902/
  production order restoration 이후 VLM 7 workloads, fresh process x3

.local/completion-r7-activity-final-20260902/
  workload별 measured intervals/segments/activity-summary.json

.local/final-contextual-provenance-canonical-20260902/
  frozen Current reference

.local/profile-free-global-20260827/r4-vllm-12x3/
  frozen vLLM reference와 request CSV
```

Build type이 비어 있던 다음 artifact는 성능 근거로 사용하지 않는다.

```text
.local/completion-r7-current-12x1-20260902/
```

## 12. 최종 판단

R7은 R2의 실패 원인이 scheduler value가 아니라 **준비되지 않은 실행을 늦게 제출한
mechanism**이었다는 것을 확인하고 해결했다. 이제 계획한 direction과 실제 GPU outstanding
set이 일치하며, reachable target은 representative point에서 수 percentage point 이내로
재현된다.

동시에 결과는 항상 overlap해야 한다는 주장을 반박한다. P→D는 offset이 늦어질수록
`+1.36%`에서 `-18.89%`로 바뀌고, E→D는 middle offset에서 강한 이익이 있지만 종료 뒤
시작하면 이익이 사라진다. Natural workload도 GPU idle이 거의 없으므로 핵심 문제는
work-conserving 여부가 아니라 **어떤 concrete shape를 어느 residual point에 실행할지**다.

따라서 다음 연구 단계는 새 workload rule이 아니라 prepared execution 위에서 process-local
uncertainty가 충분할 때만 authority를 여는 것이다. Mechanism fidelity는 확보했고,
policy benefit과 generality는 다음 gate로 남는다.
