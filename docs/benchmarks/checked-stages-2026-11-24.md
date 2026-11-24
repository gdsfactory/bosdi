# CG2 staged setup and checked-call measurements, 2026-11-24

This is the Circulox review follow-up to [the CG1 measurements](checked-groups-2026-11-24.md). All work remains
uncommitted on `codex/checked-delay-groups`, based on `343b468`. ABI version remains **1**; the capability mask is now
**31**. The [header](../../include/bosdi_runtime.h) and
[runtime guide](../handle-runtime.md#owned-staged-lane-candidates-cg2) define ownership, locking, staleness, defaults
and the capability registry.

## What changed

- `bosdi_stage_lanes_checked` prepares/validates owned candidates without changing accepted snapshots, maps, bindings,
  generations, qualified workspace or observed workers. Creation may run concurrently with accepted evaluation.
- `bosdi_stage_prepared_delays` returns durations in supplied lane-list order. `bosdi_commit_stage` publishes one handle
  without setup/eval/allocation, consumes the successful token and invalidates other candidates. `bosdi_discard_stage`
  is idempotent, including after handle free. Busy commits (`-8`) are retryable; stale tokens (`-9`) can never publish.
- Handle updates, commits and free release stale payloads immediately. The caller discards remaining token tombstones.
  No token holds a lock between calls or reloads a library.
- NaN means provider default in the new stage API and `bosdi_update_lanes_defaults_checked`, gated by capability 16. The
  existing `bosdi_update_lanes_checked` preserves its finite-only parameter contract. Both immediate routes and staging
  share the same validation/publication implementation; finite rows retain identical numerical behavior.
- Capability bits 1/2/4 retain their CG1 meanings; 8 adds staging, 16 adds the defaults-aware contract. Historical local
  `98626261` used bit 1 for `BOSDI_CHECKED_BOUND_RETAINED`: that name/contract is retired. Any future retained-storage
  support is permanently reassigned reserved bit 32 with new symbols. Bit 32 is not advertised. Unknown bits confer no
  guarantees; consumers must gate on **bit and required symbols**. Checked calls continue to restore snapshots.

## Method

Linux x86-64, Ryzen AI 9 HX 370, CPUs **0–3**, Rust 1.94.0 and GCC 13.3.0, CPU boost enabled. Native C timing loops
exclude Python/JAX dispatch and library registration. Each cell is the median of three trials lasting at least 100 ms;
raw trials and source/artifact hashes are in [the companion JSON](checked-stages-2026-11-24.json). The unchanged default
128/32 dispatch policy and 1/2/4 configured workers are used. No further policy tuning is included.

The workload is the unchanged canonical four-delay waveguide. Parameter rows contain NaN defaults plus per-lane lengths
from 100 to 200 um; candidates increase these lengths by 20%. Full raw validation X is 0.25, context is transient with
both checked options enabled for preparation. Both preparation routes run both setup callbacks and a full finite F/Q/G/C
and duration preparation. The accepted handle remains unchanged across stage/discard trials.

## Stage + discard versus fresh validated setup + free

Values are **ns/lane**, including the complete ownership/lifetime operation in each column. Libraries are already
registered in both cases. Fresh setup uses `bosdi_setup_batch_delays_checked`; the legacy setup-only route does not
perform equivalent validation and is not the reference here.

| Lanes | Workers | Fresh + free | Stage + discard | Stage/fresh |
| ----: | ------: | -----------: | --------------: | ----------: |
|    32 |       1 |        481.6 |           358.3 |        0.74 |
|    32 |       2 |        506.2 |           362.1 |        0.72 |
|    32 |       4 |        502.8 |           359.2 |        0.71 |
|   128 |       1 |        420.5 |           340.7 |        0.81 |
|   128 |       2 |        365.9 |           268.7 |        0.73 |
|   128 |       4 |        291.1 |           173.1 |        0.59 |
|   512 |       1 |        399.1 |           388.7 |        0.97 |
|   512 |       2 |        304.3 |           249.1 |        0.82 |
|   512 |       4 |        233.7 |           146.5 |        0.63 |
|  2048 |       1 |        619.4 |           379.1 |        0.61 |
|  2048 |       2 |        269.6 |           223.9 |        0.83 |
|  2048 |       4 |        207.1 |           140.1 |        0.68 |

Stage/discard is cheaper in every measured median. The 512-lane serial comparison is close (about 3%); short laptop
measurements do not establish a universal timing bound. The larger parallel groups show more substantial savings. This
measures BOSDI pre-check preparation only, without Circulox's orchestration, history or condensation.

## Explicit-buffer versus bound checked evaluation

Values are **ns per call**. The delta is `bound - explicit`; negative favors bound. Both routes use the same accepted
handle, caller buffers, context and restored-snapshot numerical core. The driver checks bit-identical F/Q/G/C before
timing. Timing disables finite-output auditing equally for both; context is transient with options zero. Full calls
extract raw F/Q/G/C, while residual-only preserves the Jacobian arrays. Neither call returns a duration buffer in this
comparison, as requested. No bound delay-observation variant was added.

| Lanes | Workers | Call     | Explicit | Bound | Bound - explicit |
| ----: | ------: | -------- | -------: | ----: | ---------------: |
|    32 |       1 | Residual |     2456 |  2423 |              -33 |
|    32 |       1 | Full     |     2647 |  2631 |              -15 |
|    32 |       2 | Residual |     2491 |  2449 |              -41 |
|    32 |       2 | Full     |     2646 |  2615 |              -31 |
|    32 |       4 | Residual |     2474 |  2436 |              -38 |
|    32 |       4 | Full     |     2653 |  2636 |              -16 |
|   128 |       1 | Residual |     9658 |  9486 |             -172 |
|   128 |       1 | Full     |    10452 | 10282 |             -170 |
|   128 |       2 | Residual |     7418 |  7441 |              +23 |
|   128 |       2 | Full     |     7722 |  7909 |             +187 |
|   128 |       4 | Residual |     5547 |  5613 |              +66 |
|   128 |       4 | Full     |     5894 |  5808 |              -86 |
|   512 |       1 | Residual |    39598 | 38655 |             -944 |
|   512 |       1 | Full     |    42303 | 41150 |            -1153 |
|   512 |       2 | Residual |    22700 | 22596 |             -104 |
|   512 |       2 | Full     |    24256 | 24301 |              +45 |
|   512 |       4 | Residual |    13694 | 14091 |             +397 |
|   512 |       4 | Full     |    14785 | 15005 |             +220 |

There is no consistent bound advantage across worker counts in these short trials. The largest median separation is
about 1.15 us for a full 512-lane serial call; parallel differences vary in sign. These comparisons do not establish
statistical significance or justify retained storage. Circulox should make its admission/API decision using its own
measured explicit delay-observation path. No API for bound duration observation is introduced in this round.

## Tests and commands

From `/home/cdaunt/code/bosdi`:

```bash
pixi run build
mkdir -p build/checked-groups
/home/cdaunt/.local/bin/openvaf-r tests/devices/delays/waveguide.va -o build/checked-groups/waveguide.osdi
PYTHONPATH=src:tests taskset -c 0-3 pixi run python scripts/benchmark_staged_groups.py --optical build/checked-groups/waveguide.osdi --output build/checked-groups/staged-feedback-pinned.json
```

The same compiler SHA-256 as CG1 is used: `613fd46951fdca2de255bd3fadcbf927cf5d195c06af274f085981507999c6fc`. The
complete suite selects the packaged 0.4 compiler for existing tests and the explicit 0.5 compiler for delay fixtures:

```bash
PATH=/tmp/bosdi-tools:$PATH CIRCULOX_OPENVAF=/home/cdaunt/.local/bin/openvaf-r BOSDI_OPENVAF_05=/home/cdaunt/.local/bin/openvaf-r BOSDI_BASELINE_LIBRARY=/tmp/bosdi-before.so pixi run test
ln -sf /home/cdaunt/code/bosdi/tests/compiled_osdi/resistor.osdi build/checked-groups/resistor.osdi
ln -sf /home/cdaunt/code/bosdi/tests/compiled_osdi/psp103v4_psp103.osdi build/checked-groups/psp103v4_psp103.osdi
BOSDI_STAGE_FIXTURES=/home/cdaunt/code/bosdi/build/checked-groups pixi run cargo test
pixi run ruff check tests/test_staged_groups.py tests/test_checked_groups.py scripts/benchmark_staged_groups.py
cc -Wall -Wextra -Werror -fsyntax-only -Iinclude scripts/benchmark_checked_groups.c
git diff --check
```

`/tmp/bosdi-tools/openvaf-r` points to the packaged 0.4 compiler, as documented in the CG1 report. The saved baseline
library is also from that report. `BOSDI_STAGE_FIXTURES` enables direct Rust comparisons of model/instance snapshot
bytes for the compiled waveguide, resistor and PSP103. The Rust harness explicitly supplies libm for compiled artifacts;
Python already supplies those math symbols. Without that environment variable, the general synthetic ownership tests
still run, and only the optional compiled-byte check is omitted.

Validation: **561 passed, 33 skipped** in the standalone suite; **5 Rust tests passed** with compiled fixtures enabled.
Tests cover subset/all-lane commit parity, unchanged accepted state across stage/discard, middle-lane failure, unsorted
lane error precedence and duration ordering, concurrent candidates, another commit/update making tokens stale,
multi-handle stage/commit/discard, evaluation between stage and commit, creation during in-flight eval, busy commit then
retry, safe discard after free, immediate stale-payload release and repeated stage/discard without owned-storage leaks.
A loaded fixture is renamed away before repeated staging to prove no file reopen/reload. Registry tests compare the
returned mask with the OR of documented enabled macros and check all corresponding stage symbols.

Full NaN rows for canonical waveguide, a resistor with valid defaults, and PSP103 match fresh prepared setup in raw
model/instance bytes, node maps and F/Q/G/C/durations. A provider default that produces nonfinite outputs returns no
token. Explicit negative-zero waveguide loss matches fresh negative-zero setup and differs from default loss; it is not
treated as NaN/default. Existing finite-only immediate behavior remains tested separately.

## Remaining qualification and host responsibilities

Only trusted local Linux CPU artifacts are qualified. Other hosts, general providers, and end-to-end Circulox optical
sweeps remain unqualified. History and y/z condensation remain entirely host-owned. Default resolution is the provider's
ordinary setup lifecycle, which may depend on other overridden parameters; there is no universal defaults-vector query.

Commit is atomic per handle. A host-wide transaction must validate all candidates first, honor cancellation before
publication, quiesce readers and serialize writes through the commit phase. Busy is retryable; stale means the host lost
its revision/ownership guarantee. Cancellation should be deferred once the first handle publishes. No cross-handle
commit barrier or rollback-by-restoring-parameters is provided. The global handle write lock conservatively excludes
unrelated handles during commit. This follow-up adds no history ABI, bound delay variant, snapshot skipping or policy
work.
