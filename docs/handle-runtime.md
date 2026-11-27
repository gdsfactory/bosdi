# Native handle runtime and caller-owned bypass

The exact JAX/C handle entry points share a configurable BOSDI-owned Rayon pool. `osdi_runtime` additionally offers
mutable native buffer bindings for simulator loops outside JAX transformations. Binding is the foundation from issue
#39; issue #38's optional bypass uses the same buffers and generation contract. The intended landing order is #39, then
#38. No consumer matrix is accessed.

## Threads and dispatch

```python
from osdi_runtime import set_num_threads, set_parallel_policy

set_num_threads(4)
set_parallel_policy(min_lanes=128, min_chunk=32)
# For a measured expensive-model group:
set_parallel_policy(min_lanes=32, min_chunk=16, handle=handle)
# Reset that handle to the global policy:
set_parallel_policy(0, 0, handle=handle)
```

Defaults read `RAYON_NUM_THREADS`, `BOSDI_RAYON_THRESHOLD` and `BOSDI_RAYON_MIN_CHUNK` once. Absent overrides, the last
two are 128 and 32. `set_num_threads(0)` restores the initial thread default. All handle setup and parallel handle
evaluations execute in the BOSDI pool, without initializing or configuring Rayon's global pool or changing the
environment. The legacy non-handle entry points retain their global-pool execution.

Each evaluation captures its configuration. Concurrent setters safely replace the pool; calls already in progress finish
on their captured pool. Host threads acquire exclusive workspace access before entering Rayon, preventing worker
starvation when several threads evaluate the same qualified handle.

The defaults keep 32-lane cheap-model groups serial and admit 128–512-lane groups. Expensive models can benefit from
lower per-handle thresholds; BOSDI does not infer a model's evaluation cost from its name or silently calibrate it.
Thread count/policy never changes entry order within a lane or numerical outputs.

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

## Embedder ABI v1

`include/bosdi_runtime.h` documents the additive native interface for issue #44. Check
`bosdi_runtime_abi_version() == 1` before resolving the checked entry points. Existing JAX, handle and bound functions
retain their analysis mode, time zero, DC charge suppression and qualification/bypass behavior.

`bosdi_setup_batch_checked` returns a status and publishes a handle only when every lane passes both setup stages. It
checks initialization flags, error counts and error/fatal provider logs. `bosdi_eval_handle_checked` and
`bosdi_eval_bound_checked` report eval failures, error/fatal logs and optionally non-finite outputs. The informational
OSDI limiting return bit alone is accepted; fatal, finish, stop and unknown return bits fail the call. A
`BosdiCallError` contains the lowest failing lane and the first diagnostic for that lane, truncated to 255 bytes.
Argument/stale-handle errors and success use `SIZE_MAX` as the lane. Discard all outputs on failure; they may have been
partially written.

Pass a `BosdiEvalContext` on each evaluation. Its `analysis` is 0 (DC/static), 1 (AC/static), or 2 (transient), with
finite, non-negative `abstime` in seconds. `BOSDI_REACTIVE_AT_DC` requests reactive flags and Q/C extraction even at DC.
`BOSDI_RESIST_ALWAYS` requests resistive flags and F extraction even with no G entries, for constant sources. Options
default to zero and unknown bits fail validation. The context never changes the model registry or another call. Checked
evaluation restores setup snapshots even for qualified providers and supports full or residual-only raw-entry output;
residual-only leaves G/C alone. It uses the same buffer lengths and rules for disjoint arrays as binding. Checked bound
calls validate the generation and hold the binding lock for the whole evaluation. No checked call consumes bypass
caches. Hosts must invalidate cached outputs when changing context or options before using the legacy bypass entry
point.

The optional logging callback receives borrowed text, lane and provider severity. It runs synchronously on the executing
lane's thread, so it must be thread-safe and must not reenter BOSDI. There is no implicit stderr logging or global
failure counter. Error records, context and callback user storage must remain valid for the complete call and must not
alias mutable evaluation buffers.

`bosdi_handle_node_maps` exports the actual setup-selected maps in device-major order, with `-1` for ground. The node
and lane counts must match exactly. This uses the same stored topology as `osdi_handle_topology_ffi`. `bosdi_free_model`
releases the registry entry and its library reference after handles are freed. Unbind buffers first. It returns false
for unknown models, live handles or concurrent registry activity; retry after quiescing the host. Retire the model ID
from any host-side caches before unloading. Zero-byte model structs receive one byte of internal storage, including on
the legacy path.

`tests/test_embedder_runtime.py` uses synthetic OSDI 0.5 providers to exercise setup status/log failures, eval
status/log failures, non-finite output, DC Q/C, constant F without G, time, zero-byte model structs, actual per-lane
collapse, concurrent contexts and optional callbacks. `tests/test_osdi_runtime.py` retains the legacy
dense/sparse/residual and bound bitwise parity suite; set `BOSDI_BASELINE_LIBRARY` to a pre-change native library for an
independent audit.

## Checked compatible groups: delay observation and lane replacement

The CG2 capability query `bosdi_runtime_capabilities()` returned **31**; the explicit-context extension below raises the
current mask to **95**. `bosdi_runtime_abi_version()` remains **1**. The capability registry preserves all CG1
assignments (first checked-groups candidate, mask 7); CG2 is the integration-review follow-up, mask 31, both dated
2026-11-24.

| Value | Symbol                                  | Governed entry points / contract                                                                                             | Introduced |
| ----: | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------- |
|     1 | `BOSDI_CAP_DELAY_OBSERVATION`           | `bosdi_model_delays`, `bosdi_handle_prepared_delays`, `bosdi_setup_batch_delays_checked`, `bosdi_eval_handle_delays_checked` | CG1, ABI 1 |
|     2 | `BOSDI_CAP_OBSERVED_WORKERS`            | `bosdi_handle_observed_workers`                                                                                              | CG1, ABI 1 |
|     4 | `BOSDI_CAP_TRANSACTIONAL_SETUP`         | `bosdi_update_lanes_checked` (finite parameter rows)                                                                         | CG1, ABI 1 |
|     8 | `BOSDI_CAP_STAGED_SETUP`                | `bosdi_stage_lanes_checked`, `bosdi_stage_prepared_delays`, `bosdi_commit_stage`, `bosdi_discard_stage`                      | CG2, ABI 1 |
|    16 | `BOSDI_CAP_NAN_DEFAULT_SETUP`           | NaN-default contract for `bosdi_stage_lanes_checked`; `bosdi_update_lanes_defaults_checked`                                  | CG2, ABI 1 |
|    32 | `BOSDI_RESERVED_CHECKED_BOUND_RETAINED` | Reserved only; no implementation or advertised support                                                                       | CG2, ABI 1 |

Only enabled rows (1 through 16) contribute to the returned mask. Assigned/reserved bits must never be reused for
another feature. **Unknown bits confer no guarantees; gate every feature on both its bit and its required symbols.** The
unregistered local candidate `98626261` used bit 1 for `BOSDI_CHECKED_BOUND_RETAINED`. That name/contract is retired: it
is not this registry's bit 1. Retained-storage support, if ever shipped, is permanently reassigned to bit 32 and must
use new symbols. Bit 32 is not advertised here, and every checked evaluation continues to restore snapshots. Hosts using
the old literal bit value must audit their capability checks; ABI version 1 alone cannot distinguish that historical
local contract from this registry.

All v1 signatures and snapshot, context, residual-only and error semantics remain available. New functions are also
retained in the Python extension's `runtime_addresses()` table, including on platforms with hidden dynamic symbols.

Query `bosdi_model_delays(model, NULL, 0, &count)`, allocate `count` `BosdiDelayMetadata` records, then query again with
that exact length. Each record contains raw `y_node`, raw `z_node` and `has_maxdelay` (0/1). Delay order is descriptor
order. OSDI 0.4 has zero delays. The checked path rejects invalid duration offsets, overlapping delay nodes, terminal
delay nodes, and setup-selected maps which collapse a hidden delay node. Unsupported layouts return `-7`; metadata is
never truncated. Explicit `maxdelay` storage is supported and its presence is reported; BOSDI does not interpret its
history policy or expose a history buffer. No model names or physics are special-cased.

`bosdi_handle_prepared_delays` reads each prepared instance's duration slots. Durations are lane-major:
`durations[lane * count + delay_index]` corresponds to `metadata[delay_index]`. A four-delay waveguide therefore has
four adjacent entries per lane; different lanes may have different lengths and durations.

The compiler used for this qualification writes even fixed durations during **eval**, leaving their setup slots zero.
The v1 `bosdi_setup_batch_checked` preserves those literal setup values. For this compiler, use the new
`bosdi_setup_batch_delays_checked`: supply initial full raw X and a context. It runs ordinary checked setup and a full
checked preparation evaluation on restored scratch, then copies **only td slots** into the prepared instances. The rest
of model and instance storage retains its setup snapshot. It returns no handle on failure. This explicit preparation
also defines the initial voltage/context for variable and mode-dependent durations. A raw nonfinite td is reported;
finite F/Q/G/C validation remains enabled. The host validates prepared durations before accepting the group.

`bosdi_eval_handle_delays_checked` extends the v1 checked call with optional duration output. NULL/0 disables it;
otherwise provide exactly `lanes * count` doubles, aligned and disjoint from every input/output array. Each lane reads
td from **evaluated instance scratch**, before that scratch is restored for another lane. It returns raw values,
including NaN/Inf and negative values; host policy validates finiteness, sign and equality to the accepted prepared
values after every evaluation. This applies to full and residual-only calls in DC, AC and transient. Failed calls retain
the existing lowest-lane error convention; discard all their outputs. Worker callbacks retain the v1 synchronous logging
contract. There is no delay bypass or history algorithm.

The compiled fixtures confirm that y/z appear in the full X stride and have allocated raw residual rows; y has its
provider equation and z has a zero residual row awaiting host history treatment. y appears as a raw Jacobian row and z
as a raw Jacobian column. The exported pairs retain both nodes; the maps preserve their identity. No z equality/history
law or condensation is fabricated by BOSDI.

`bosdi_handle_observed_workers(handle)` reports distinct executing threads in the most recent checked evaluation pass,
including v1 checked direct and bound calls. It counts threads which actually evaluated lanes, not pool capacity. A
successful serial pass reports 1; a failed evaluation pass reports 0. Preparation and successful updates reset it to 0.
Serialize checked calls on the same handle when using this last-call query. Independent handles remain concurrent.
Initial validation failures before a call resolves its handle do not constitute an evaluation pass.

`bosdi_update_lanes_checked` accepts unique lane indices, parameter rows, full raw validation X rows and a required
context. Rows follow **lane-list order**; the list need not be sorted. Every supplied parameter and voltage must be
finite; this operation does not accept NaN defaults. Reuse resolved parameter rows from the host's accepted recipe. It
allocates fresh model and instance storage, runs both setup callbacks, compares node maps **and collapse flags**, then
checks full F/Q/G/C and evaluated durations at the supplied X/context. Validation uses scratch so provider mutations do
not contaminate prepared snapshots; only prepared td slots are refreshed from that evaluation. The lowest failing raw
lane wins, independent of list order or worker scheduling. Failure publishes nothing. `-7` means structural change or
unsupported delay metadata; use a fresh handle for intentional topology changes.

Publication holds a nonblocking global handle write lock, preventing overlap with any evaluation that has acquired a
handle. `-8` means busy; quiesce evaluations and retry. This conservative lock also excludes evaluations of unrelated
handles during staging. Callbacks must not reenter BOSDI. A successful update atomically installs all selected
snapshots, keeps the handle ID, and never reloads the library. It clears the handle's qualified workspace; requalify
that workspace if needed. Existing buffer bindings keep their memory addresses but their old generations become
unusable. Rebind each binding to refresh its lane workspaces and obtain a new generation; cached outputs must be
invalidated. Failed updates preserve snapshots, workspaces and binding generations. Prepared durations can be queried
again after publication.

The checked hot path reduces errors without collecting a per-lane Vec, checks disjoint regions without allocation and
borrows TLS scratch once per chunk. Worker participation uses reusable, cache-line-separated flags. Snapshot restoration
and output initialization remain exact for unqualified providers. A test provider mutates both model and instance data
to guard that contract. The warmed serial full checked call has a Rust allocation-counting regression test. Initial
worker/model/pool warm-up may allocate.

The default parallel policy uses a threshold of 128 lanes and chunks of 32 lanes; explicit environment, global and
per-handle overrides remain available. Host history, delay condensation and end-to-end admission remain caller
responsibilities. Performance should be evaluated through a reproducible open consumer workflow, such as Circulax.

## Owned staged lane candidates (CG2)

A host can prepare a candidate without publishing it: call `bosdi_stage_lanes_checked`, inspect durations through
`bosdi_stage_prepared_delays`, then call `bosdi_commit_stage` or `bosdi_discard_stage`. All input pointers, X, context,
callback and user storage are borrowed only for the stage call. On failure, the output token is zero. A successful token
owns model/instance bytes, topology plans, lane indices and prepared durations; it holds no lock or library reference
between calls. Tokens are process-local and must be committed or discarded, including after becoming stale.

Staging runs both setup callbacks in fresh storage, checks full F/Q/G/C and evaluated td at the supplied X/context,
compares node maps and collapse flags, and copies only evaluated td slots into the prepared instances. It uses the same
preparation/publication helpers as immediate update. It changes no accepted bytes, bindings, generations, qualified
workspace or observed-worker value. Accepted evaluation remains allowed while a stage exists **and while it is being
created**: creation takes a nonblocking global handle read lock, sharing it with evaluations. It returns `-8` when that
read lock is unavailable. The registry stays read-locked during setup/eval to keep the provider alive.

The duration query returns `changed_lane_count * delay_count` doubles in **supplied lane-list order**, then descriptor
delay order. The query takes a handle read lock followed by the stage-table lock. It returns `-2` for invalid length or
alignment, `-1` for unknown/consumed tokens, and `-9` for stale tokens. NULL is valid for zero length. Host policy still
checks duration sign and suitability for its history before committing.

Commit takes a nonblocking global handle write lock followed by the stage-table lock. It performs no setup, evaluation,
validation or allocation. Its failure modes are `-8` busy (token intact, retryable), `-9` stale (never committable) and
`-1` unknown/consumed. A successful commit publishes all staged lanes atomically for that handle and consumes the token.
It clears the qualified workspace and observed-worker value; old binding generations fail until rebind refreshes their
lane storage and returns a new generation. Accepted-handle evaluations cannot overlap publication.

Several stages may coexist for one handle. Any successful update/commit, or handle free, invalidates all remaining
stages of that handle. Their payloads are released immediately; small stale-token tombstones remain until discard, so
commit can consistently return `-9`. `bosdi_discard_stage` takes only the stage-table lock, releases token storage, and
is idempotent for zero, unknown, consumed, stale and freed-handle tokens. It is safe to race discard with commit; the
stage-table lock orders the winner. Lock order is handle table, provider registry when needed, then stage table. No
stage owns any of those guards between calls. Creating or discarding a stage never reloads a library.

These are per-handle commits. For a host transaction spanning handles/native providers, validate every candidate and
honor cancellation before the first publication. Quiesce host readers, retain exclusive ownership of the handles and
serialize writers across the commit sequence. Busy commits can be retried after quiescence. Cancellation should be
deferred once publication begins; BOSDI does not provide a cross-handle commit barrier. The global write lock also
excludes unrelated handles during each commit and during the original immediate update call.

### NaN provider defaults

The existing `bosdi_update_lanes_checked` keeps its finite-parameter-only semantics. Use the new
`bosdi_update_lanes_defaults_checked` for immediate updates accepting defaults, or use stage/commit. Capability bit 16
advertises NaN-default handling for both the new stage API and the new immediate API. Bit 8 by itself only guarantees
the staging interface; gate NaN handling on bit 16 as well.

A NaN parameter means exactly the same thing as in v1 setup: skip that parameter's write and let the provider resolve
its default during ordinary model/instance setup. It is never a missing-change marker or a cached old value. All other
parameters must be finite; X must always be finite. Explicit `-0.0` is passed to the provider and is never treated as a
default. Outputs and evaluated/prepared durations must still be finite, so a provider's numerically invalid default is
rejected with no token or accepted-state change. Provider defaults can depend on other overrides and geometry; a single
per-model defaults vector cannot reproduce that lifecycle. No defaults-query shortcut or host reimplementation is used.

Compiled tests cover canonical waveguide, all delay fixtures, resistor and PSP103, including full default rows, bitwise
F/Q/G/C, raw snapshot bytes, maps and durations, signed zero, rollback, simultaneous stages, stale tokens, binding
invalidation, in-flight staging, busy commit/retry, freed handles and repeated discard. The Linux Rust fixture tests
load libm explicitly, matching the math-symbol environment supplied by Python/native hosts. No bound delay-observation
variant or parallel-policy change was added in this follow-up.

### Explicit context follow-up (CG3)

Capability 64 (`BOSDI_CAP_EXPLICIT_CONTEXT`) and the additive `bosdi_runtime_context_options()` query advertise optional
context bits. The mask is 15; existing bits 1/2 and all legacy calls retain their behavior. Bit 4
(`BOSDI_AC_WITHOUT_STATIC`) requests AC alone, matching scalar OSDI hosts; it is invalid outside AC. Default AC remains
AC|STATIC. No purity or model-name inference selects this flag.

Bit 8 (`BOSDI_DELAY_INPUT_SCOPE`) requests checked input-only finite validation for full delay-observing
evaluation/setup. Diagnostics and evaluation status remain checked. Only independent hidden input rows F/Q and their raw
Jacobians must be finite; physical rows may be nonfinite until the host gathers y/z. Raw durations are still reported
for host validation. This permits zero-X preparation of reciprocal-delay laws without replacing their physics or
inventing another preparation point. The host must validate the subsequent complete physical pass before publication.
No-delay, residual-only or no-duration-output use rejects. Staged updates reject this option and keep complete
finite-output validation; hosts supply gathered hidden coordinates to stage. ABI version remains 1, feature mask becomes
95, and bit 32 remains reserved and unadvertised.
