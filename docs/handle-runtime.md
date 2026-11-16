# Native handle runtime and caller-owned bypass

The exact JAX/C handle entry points share a configurable BOSDI-owned Rayon pool. `osdi_runtime` additionally offers
mutable native buffer bindings for simulator loops outside JAX transformations. Binding is the foundation from issue
#39; issue #38's optional bypass uses the same buffers and generation contract. The intended landing order is #39, then
#38. No consumer matrix is accessed.

## Threads and dispatch

```python
from osdi_runtime import set_num_threads, set_parallel_policy

set_num_threads(4)
set_parallel_policy(min_lanes=1024, min_chunk=256)
# For a measured expensive-model group:
set_parallel_policy(min_lanes=32, min_chunk=16, handle=handle)
# Reset that handle to the global policy:
set_parallel_policy(0, 0, handle=handle)
```

Defaults read `RAYON_NUM_THREADS`, `BOSDI_RAYON_THRESHOLD` and `BOSDI_RAYON_MIN_CHUNK` once. Absent overrides, the last
two are 1024 and 256. `set_num_threads(0)` restores the initial thread default. All handle setup and parallel handle
evaluations execute in the BOSDI pool, without initializing or configuring Rayon's global pool or changing the
environment. The legacy non-handle entry points retain their global-pool execution.

Each evaluation captures its configuration. Concurrent setters safely replace the pool; calls already in progress finish
on their captured pool. Host threads acquire exclusive workspace access before entering Rayon, preventing worker
starvation when several threads evaluate the same qualified handle.

The defaults avoid scheduling small cheap-model groups. Expensive models can benefit from lower per-handle thresholds;
BOSDI does not infer a model's evaluation cost from its name or silently calibrate it. Thread count/policy never changes
entry order within a lane or numerical outputs.

## Exact evaluation and qualification

Unqualified handles retain immutable post-setup snapshots and restore them for each evaluation. Node maps and dense
Jacobian destinations are precomputed at setup. Worker scratch is reused and borrowed once per chunk, rather than once
per lane. Dense scatter still adds entries in their original order, including duplicate entries and equality-row
overwrites.

`qualify_handle(handle)` enables in-place workspaces for the shared JAX/C path. This is an explicit assertion by a
caller who has qualified the compiled provider: `eval` reads no previous evaluation/history, fully recomputes everything
used by the extraction callbacks, preserves the node map and reads `prev_solve` without modifying it. Metadata alone
cannot establish these guarantees. Providers exposing ABI state slots are refused, including limiter slots. The local
PSP103 artifact used below exposes no such slots. Generic binaries with hidden mutable history must not be qualified
even if their state count is zero.

Each qualified workspace owns separately aligned model/instance allocations. No shared immutable snapshot is cast to
mutable storage, and odd byte sizes cannot misalign subsequent lanes. A host mutex grants exclusive access; Rayon splits
mutable lane slices. Tiled/vmap calls always restore snapshots, including after qualifying the original handle.
Disabling qualification drops the mutable workspace and resumes restoration from the original snapshots.

## Buffer binding

The C interface is declared in [`include/bosdi_runtime.h`](../include/bosdi_runtime.h). Python obtains function
addresses from the extension, so it does not depend on platform-specific exported-symbol visibility. The extension
retains references to the C entry points on every platform.

```python
import numpy as np
from osdi_runtime import BoundBatch

# voltages: contiguous float64 [handle.num_devices, model.num_nodes]
with BoundBatch(handle, voltages) as batch:
    batch.evaluate(check_finite=True)  # exact; qualification is optional
    currents = batch.currents
    charges = batch.charges
    raw_g = batch.conductances
    raw_c = batch.capacitances
```

Bindings do not tile handles. Input widths include every raw internal unknown. F/Q storage is `[B, N+1]`: the last slot
is a private ground sink. The public Python F/Q views expose `[B, N]`; these views are strided. OSDI's accumulating
residual loads write directly into each caller lane. Grounded/collapsed raw slots receive the same equality residuals as
the existing path.

Jacobians use the raw-entry layout from #36 and the handle's `sparse_pattern`. BOSDI aims `write_jacobian_array_*`
directly at the registered lane blocks. This already removes intermediate Jacobian arrays and scatter. It intentionally
uses those overwrite callbacks instead of redirecting accumulating `load_jacobian_*` pointers: the raw-entry API must
preserve signed zeros and entry order bitwise. Dense callers retain the existing exact API with precomputed scatter
targets. Qualified providers without ground-sink references read voltages directly; other cases use reusable voltage
scratch.

The C caller must keep all registered storage alive, disjoint and exclusively available during calls. Registration
checks lengths, alignment, overflow and intra-binding overlap. Distinct bindings may not concurrently use overlapping
storage. The Python wrapper retains all backing arrays and serializes its own methods; direct access to arrays from
another thread during evaluation remains the caller's responsibility.

`batch.rebind(new_voltages)` registers new output storage and advances a unique generation. Old arrays remain ordinary
Python arrays but no longer receive writes. A stale generation or freed handle fails evaluation. Failed native rebinding
preserves the existing binding. Native unbind waits for in-flight work before the caller can release storage. Exact
residual-only calls preserve both Jacobian arrays.

## Caller-owned bypass and measurement

Exact evaluation is the default; JAX entry points never bypass. Enable bypass only for intermediate linearizations after
qualifying the compiled provider. `qualify_pure_eval()` makes the same provider assertion described above for one
binding, independently of the handle's JAX/C workspace qualification.

```python
with BoundBatch(handle, voltages) as batch:
    batch.qualify_pure_eval()  # only after provider qualification
    batch.evaluate()  # exact full F/Q/G/C
    cache = batch.new_cache()
    cache.previous_voltages[:] = batch.voltages
    cache.valid[:] = 3

    # Caller changes batch.voltages for the next intermediate linearization.
    stats, bypassed = batch.evaluate_bypass(cache=cache, reltol=1e-3, vntol=1e-6)
    # Only if the caller accepts these evaluated outputs as its new cache:
    evaluated = ~bypassed.astype(bool)
    cache.previous_voltages[evaluated] = batch.voltages[evaluated]
    cache.valid[evaluated] = 3

    batch.evaluate()  # exact final residual/history audit
```

For a lane to bypass, every raw input must be finite and move strictly less than
`reltol * max(abs(v), abs(previous)) + vntol`. This includes internal and collapsed unknowns. NaN/Inf inputs never
bypass; invalid tolerances are rejected. Validity bit 1 covers F/Q and bit 2 covers G/C; full evaluation requires both.
BOSDI never updates previous inputs or validity. On a skip it leaves every output bit untouched, including the private
sink. No whole-batch zeroing occurs. The returned mask identifies skipped lanes; statistics count evaluated, bypassed
and eligible lanes. Optional finiteness checks also check skipped cached outputs. First-order extrapolation is not
implemented.

Never move the cached input of a skipped lane to its latest input: doing so would hide accumulated drift from the
original evaluation. After exact residual-only evaluation at a new bias, mark validity 1, since the retained Jacobian
was not recomputed there. Clear validity after any change in parameters, bindings, temperature, analysis or other
context. Registration context is immutable; parameter/context changes normally require a new handle. Rebind and
qualification changes advance a unique generation and reject earlier caches, even across different bindings.

For issue #38's measurement-first workflow, use `batch.evaluate_bypass(cache=cache, dry_run=True)` **without qualifying
the provider**. It counts eligible lanes but evaluates all lanes exactly and sets no bypass-mask bits. Capture those
statistics at every c6288 intermediate linearization, time provider evaluation and consumer assembly separately, and
update previous inputs only to the exact evaluation represented by the caller's outputs. Ordinary calls with no cache
have no bypass diagnostics. There is no automatic activation.

## Short diagnostic benchmark

Run after building and compiling the test artifacts:

```bash
PYTHONPATH=src:tests pixi run python scripts/benchmark_handle_runtime.py --seconds 8
# Optional pre-change extension for bitwise regression tests:
BOSDI_BASELINE_LIBRARY=/absolute/path/to/original.so pixi run pytest tests/test_osdi_runtime.py
```

The benchmark uses native C timing loops and an independent scalar OSDI adapter with per-instance prepared memory.
Python/JAX dispatch and setup are excluded. Scalar F/Q and raw Jacobians are checked bitwise against the bound outputs
before timing. Three short trials per cell share an eight-second timing budget; setup adds wall time. Scalar uses the
raw-entry extraction layout; dense handle timing also includes dense scatter. No simulator or assembly is measured.

The [2026-11-16 measurements](benchmarks/handle-runtime-2026-11-16.json) use the default policy and 1/12/24 BOSDI
workers. Rounded ns/lane:

| Model    |  Lanes | Scalar (1 thread) |   Qualified dense, 1 / 12 / 24 | Bound raw, 1 / 12 / 24 |
| -------- | -----: | ----------------: | -----------------------------: | ---------------------: |
| Resistor |    100 |              7.68 |             24.7 / 24.8 / 24.8 |         12 / 12.1 / 12 |
| Resistor |  1,000 |              7.55 |               23.1 / 23.1 / 23 |         11.1 / 11 / 11 |
| Resistor | 10,000 |              8.18 |             23.9 / 5.04 / 5.89 |     11.4 / 2.97 / 3.52 |
| PSP103   |    100 |               817 |                947 / 952 / 944 |        822 / 820 / 819 |
| PSP103   |  1,000 |               875 | 1.07e+03 / 1.09e+03 / 1.07e+03 |        864 / 869 / 866 |
| PSP103   | 10,000 |           1.4e+03 |           1.57e+03 / 267 / 240 |   1.34e+03 / 198 / 172 |

The resistor remains around 1.5× the scalar raw-entry cost on one thread; #39's approximately 1.2× target is **not
achieved** on this particularly simple artifact. Small batches avoid the earlier scheduling penalty. The conservative
policy deliberately keeps 100/1,000-lane groups serial; expensive PSP103 groups can use a measured per-handle policy to
regain parallelism at smaller sizes. An all-idle synthetic bypass measures the skip path only; it does not establish the
fraction of idle lanes or speedup in a running c6288 circuit.

Actual c6288 per-linearization idle fractions and evaluation/assembly timing remain consumer-side work. The available
local archived voltage traces contain only selected circuit outputs, not every per-device internal unknown/Newton
iterate, so they cannot supply that evidence. Neither issue should be considered fully qualified for c6288 from these
provider-only measurements.
