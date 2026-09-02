# M6 Six-Direction Repeated Calibration

## 1. Outcome

The Gate-C calibration matrix now contains ready-capable completion labels for all six ordered E/P/D directions. The
matrix uses no workload class or exact policy key: each process learns from continuous action features, one canonical
pair posterior, and one ordered-direction residual posterior.

The directional harness was corrected before this run. `--repeats N` previously produced N gateway logs but returned
only `run-001` to the analyzer. It now returns and validates all N logs, so repeat coverage and calibration statistics
cannot silently omit later runs.

## 2. Repeated cells

The final additions used target offsets 25%, 50%, and 75%, with two real-request repeats per cell:

```text
E -> P: 3 offsets x 2 repeats
E -> D: 3 offsets x 2 repeats
```

The same traces naturally generated reverse-direction labels when the actual outstanding order was P->E or D->E.
This is desirable: direction identity follows measured execution order rather than the requested experiment label.

Every repeat preserved exact token identity:

- E/P trace: 56/56 generated tokens, stable hash
  `1ce07cd3506a9139e5ec26798b256568b541d071b5227de6ec8cb9c9d577f1a5`;
- E/D trace: 6,176/6,176 generated tokens, stable hash
  `2c9e7624bed45a92567232b6802fd8da60e1da0595014a70cc6d2947042f28d5`.

## 3. Completion calibration

Ready coverage is measured from the pre-update posterior. It is therefore online calibration evidence rather than a
fit-after-label statistic.

| Direction | Obs | Ready obs | Ready incumbent coverage | Ready newcomer coverage | Predicted safe | False safe |
|---|---:|---:|---:|---:|---:|---:|
| P -> D | 457 | 419 | 96.4% | 97.9% | 0 | 0 |
| D -> P | 118 | 116 | 100.0% | 97.4% | 0 | 0 |
| E -> P | 40 | 18 | 100.0% | **83.3%** | 11 | 0 |
| P -> E | 34 | 32 | 100.0% | 100.0% | 14 | 0 |
| E -> D | 33 | 20 | 100.0% | **85.0%** | 9 | 0 |
| D -> E | 128 | 117 | **93.2%** | 100.0% | 111 | 0 |

The P/D rows come from the repeated balanced/injection Gate-C artifacts in note 210. E/P and E/D come from the new
multi-offset matrices.

The important result is not that every interval is calibrated. Three components remain under the nominal 95%
confidence target. The important result is that direction sparsity is no longer the blocker and the under-coverage is
now measurable without a per-workload threshold.

Pair-value posterior summaries:

| Pair | Observations | Ready observations | Ready interval coverage | Ready normalized MAE | Predicted safe | False safe |
|---|---:|---:|---:|---:|---:|---:|
| E+P | 88 | 40 | 95.0% | 0.134 | 0 | 0 |
| E+D | 161 | 113 | 100.0% | 0.038 | 90 | 0 |
| P+D | 599 | 535 | 97.8% | 0.136 | 43 | 0 |

Pair-value false-safe and component-completion false-safe are distinct. The first protects an overlap-benefit decision;
the second protects the per-phase deadline projection. Both currently report zero observed false-safe events, but the
E newcomer completion intervals still need widening before active promotion.

## 4. Actual overlap versus completion labels

The E/P and E/D cells did not pass the M2 material kernel-interval-overlap bucket test. Nevertheless, they produced
valid ordered completion observations because independent contexts were outstanding and completed across the same H1
boundary. These statements are intentionally separate:

```text
legal/outstanding E+P or E+D action       yes
component completion-vector label         yes
material concurrent kernel interval       no in these cells
evidence that overlap is profitable        no
```

The calibration data may teach the model to reject or serialize an action. It must not be presented as a throughput
gain from physical kernel overlap.

## 5. Scheduler overhead

| Matrix | Decision mean | Decision p95 | Decision p99 |
|---|---:|---:|---:|
| E/P | 24.45 us | 48.20 us | 58.05 us |
| E/D | 58.48 us | 96.54 us | 122.70 us |

The E/D trace creates thousands of decode decisions and remains below 100 us at p95. The previously measured balanced
trace was higher at 422.26 us p95 because full continuous-batching candidate formation, not the 16-dimensional RLS
matrix operation, dominates that host path.

## 6. Gate decision

| Requirement | Status |
|---|---|
| all six ordered directions observed | pass |
| at least four observations per direction | pass |
| exact output identity | pass |
| action fidelity | pass in prior unified validation; no new violations reported |
| same-snapshot top-1/regret | 10/10 and zero regret in balanced pilot |
| pair-value false-safe | 0 observed |
| completion false-safe | 0 observed |
| nominal completion coverage | **fail for three components** |
| broad repeated alternative coverage | incomplete |
| active-policy promotion | blocked |

## 7. Next implementation

The next change should estimate one global or pair-family conformal scale from standardized pre-update completion
residuals and apply it only to uncertainty:

```text
calibrated_sigma(pair/component)
    = global_or_pair_scale * online_RLS_sigma
```

It must not add direction-, shape-, trace-, or workload-specific thresholds. Evaluation will use held-out chronological
samples so the scale is never fitted and scored on the same label. After coverage approaches the nominal confidence
and false-safe remains low, repeat balanced, vision-heavy, and 39/48.8/97.5 saturation in shadow mode. Active M7 and
the 12-workload gate remain blocked until those checks pass.

Artifacts:

```text
.local/inflight-m6-gate-c-calibration-20260901/
  encoder-to-prefill/
    directional-injection-artifact.json
    contextual-summary.json
  encoder-to-decode/
    directional-injection-artifact.json
    contextual-summary.json
```

Update: the pair-family chronological conformal calibrator and six-direction held-out pilot are complete in
`notes/212-m7-pair-conformal-uncertainty-20260901.md`. The pilot repaired the measured under-coverage without adding a
direction or workload rule, but active promotion remains blocked on broader natural-trace shadow validation.
