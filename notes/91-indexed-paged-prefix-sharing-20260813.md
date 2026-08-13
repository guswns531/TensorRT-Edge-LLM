SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Indexed-paged immutable prefix sharing and private tail

Indexed-paged KV now supports explicit active-request prefix reuse. Complete 128-token page bundles are attached to
the target slot by reference count. A prefix ending inside a page allocates one private target bundle and copies that
source K/V page on the caller's CUDA stream. The target can then continue prefill/decode without modifying the source.

```text
source: [bundle 0][bundle 1: 32 valid tokens ...]
                 prefix length = 160

target: [bundle 0 shared, refcount 2][bundle 2 private copy of bundle 1]
                                              ^ future writes affect target only
```

Releasing the source decrements shared references but does not return bundle 0 while the target uses it. Physical
pool statistics count unique allocated bundles, so a 256-token page-aligned prefix shared by two requests consumes
two bundles rather than four. Non-aligned reuse consumes shared full pages plus one private tail page.

`PhaseContextServingFacade::submitWithPagedPrefix()` is the production-facing explicit API. The caller supplies the
active source request and verified matching prefix length. The lifecycle starts the target prefill at that offset and
requires at least one uncached prompt token, preserving the existing final-prefill sampling contract. Automatic token
hash matching and eviction policy remain frontend policy rather than allocator policy.

Validation covers refcount release order, deterministic reuse, partial-tail ownership, device page-table update,
device global length update, K/V tail copy, and mutation isolation between source and target.

