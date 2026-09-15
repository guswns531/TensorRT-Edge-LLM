# Local artifact retention and cleanup

## Outcome

The workspace reached 100% filesystem usage because TensorRT engine variants accumulated faster than result logs. The cleanup therefore preserves models, ONNX, promoted engines, source worktrees, baselines, manifests, and aggregate results while removing reproducible superseded engines and redundant raw logs.

Before cleanup, `.local/` occupied about 98 GiB. The largest components were 51 GiB of v0.10.1 forward-port artifacts, including 40 GiB for Gemma 4, and 7.6 GiB of results. Repeated `gateway.log` files across all result roots occupied about 2.76 GiB.

## Retention classes

| Class | Retention |
|---|---|
| Source worktree | Always retain; source changes must also be committed and pushed |
| Model/checkpoint | Retain unless an explicit model retirement is approved |
| ONNX | Retain for promoted engines and expensive export lineages |
| Promoted engine | Retain and protect through `.local/current/` |
| Citable result | Retain manifest, aggregate data, inputs, and one raw trace when the conclusion requires it |
| Validation result | Retain manifest and aggregate data; retain raw failure evidence only |
| Diagnostic result | Retain the conclusion and summary; raw logs are disposable |
| Scratch result | Delete after its conclusion is incorporated into a numbered note |

Cleanup is reference- and state-driven rather than age-driven. A note reference preserves provenance, not necessarily the serialized TensorRT engine itself, provided the corresponding ONNX and builder contract remain available.

## Protected active artifacts

The active Gemma path is:

- LLM engine: `engine-packed-p8-d24-kv2048-p96`
- vision engine: `visual-e4-soft280`
- ONNX: `onnx-int4-awq-p128` and `onnx-int4-awq-packed-p128`

The P512/P128 profiled engine was later rebuilt for the diagnostic campaign in note 311. It is retained only while
that transition experiment remains active; it is not the promoted engine in `.local/current/gemma4/`.

The Cosmos model, ONNX, no-VP/VP engines, and vision engine referenced by `.local/current/` also remain protected. All clean and modified source worktrees and all upstream baselines remain protected.

## Cleanup campaign

The auditable allowlist, engine SHA256 values, and retained paths are recorded in `.local/results/cleanup-20260915/manifest.json`.

The first stage removed failed build intermediates and redundant raw logs. The second stage removed superseded asymmetric, profiled-prefill, wide-prefill, and alternate-capacity TensorRT engines while retaining their result summaries, manifests, notes, and source ONNX. Its logical directory total was 30.91 GiB. Shared hard links reduced the physical recovery: the filesystem moved from 100% full to 89% used with 25 GiB available.

Model-specific pointers now protect both `.local/current/cosmos/` and `.local/current/gemma4/`, and `.local/current/active` points to Gemma 4. The previous flat Cosmos pointers remain only for compatibility.

## Follow-up status

The runtime was rebuilt with the host UID/GID, 246 related unit tests passed, and the graph/refill full-12 campaign
completed in note 310. Redundant run-002/run-003 raw logs were removed while all summaries and one raw trace per
workload were retained. Future cleanup remains dry-run/allowlist driven.
