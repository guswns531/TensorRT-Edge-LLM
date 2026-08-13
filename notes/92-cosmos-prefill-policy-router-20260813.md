SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Cosmos 128-graph / 256-graph-off prefill policy router

The router selects between two already validated engine contracts and writes a versioned runner manifest.

- `chunk128_graph`: use when a phase-shape profile exists and available memory covers graph budgets plus the configured
  free-memory reserve.
- `chunk256_graph_off`: use for a high-backlog trace when graph memory is unsafe. The larger chunk amortizes prefill
  launches without paying graph-cache memory.
- `chunk128_graph_off`: memory-constrained short trace; avoids both graph memory and the 256-engine's larger workspace.

The policy is intentionally memory-first. Current post-I/O-reduction balanced conditions expose 927MiB before serving;
240MiB graph budgets plus a 256MiB reserve fit, so the router selects `chunk128_graph`. This agrees with the current
balanced measurements: 128+graph produced about 4,630--4,645 token/s, while the earlier 256+graph-off median was
4,580.5 token/s. If available memory falls below 496MiB, the same 288-request trace selects 256+graph-off.

Implementation and validation:

- `scripts/cosmos_reason2/select_prefill_serving_policy.py`
- `tests/python-unittests/test_prefill_serving_policy.py` (3 cases)
- example manifest: `.local/cosmos-reason2-2b/prefill-policy-router-20260813/balanced-policy.json`

