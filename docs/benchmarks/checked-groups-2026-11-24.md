# Checked compatible-group measurements, 2026-11-24

ABI v1 remains available. New capability bits are `7` (delay observation `1`, observed workers `2`, transactional setup
`4`). The API, ownership and error contracts are in [the header](../../include/bosdi_runtime.h) and
[the runtime guide](../handle-runtime.md#checked-compatible-groups-delay-observation-and-lane-replacement).

## Method

Linux x86-64, AMD Ryzen AI 9 HX 370, process affinity CPUs 0–3, Rust 1.94.0, GCC 13.3.0. CPU boost was enabled; these
short diagnostics are not a hardware-independent performance guarantee. Baseline is
`343b46825a0d9d3514cdb4fd1ef68f0ea3346216`; after is the uncommitted `codex/checked-delay-groups` implementation. The
[raw trials and source/compiler hashes](checked-groups-2026-11-24.json) accompany this report.

Native C loops call `bosdi_eval_handle_checked`, residual-only, transient, no finite-output audit, no logging callback.
Python/JAX dispatch, setup and library loading are excluded. Every cell is the median of three trials of at least 100 ms
after 50 warm-up calls. CPU ratio is **process CPU seconds / wall seconds**, sampled in the same trial; sleeping workers
and wake-up/spin costs are included. Workloads are the compiled canonical four-delay optical waveguide, existing
resistor and existing PSP103 card/geometry used by `benchmark_handle_runtime.py`. Full raw X is populated with 0.3.
Delay output is disabled in this v1 hot-path comparison; the separate compiled tests qualify its full and residual
paths.

The default changes from 1024/256 to **128/32** (`min_lanes` / `min_chunk`). Environment/global/per-handle overrides
remain explicit. Pool sizes below describe configured capacity; the after JSON also includes actual participation in the
final checked pass (`observed_last`). A small group may use fewer workers than configured.

## Before / after scaling

Each cell is **ns/lane (CPU/wall)**. Columns are configured worker counts.

### Canonical waveguide

| Lanes |    Before 1 |    Before 2 |    Before 4 |     After 1 |     After 2 |     After 4 |
| ----: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
|    32 | 81.5 (1.00) | 81.5 (1.00) | 81.4 (1.00) | 74.2 (1.00) | 74.4 (1.00) | 74.3 (1.00) |
|   128 | 79.3 (1.00) | 79.6 (1.00) | 79.2 (1.00) | 73.1 (1.00) | 56.4 (2.16) | 43.2 (3.93) |
|   512 | 85.8 (1.00) | 86.0 (1.00) | 86.3 (1.00) | 74.4 (1.00) | 42.6 (2.05) | 26.9 (4.00) |
|  2048 | 85.1 (1.00) | 50.0 (1.99) | 25.7 (3.84) | 76.5 (1.00) | 40.8 (1.99) | 21.9 (3.99) |

### Resistor

| Lanes |    Before 1 |    Before 2 |    Before 4 |     After 1 |     After 2 |     After 4 |
| ----: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
|    32 | 57.4 (1.00) | 57.1 (1.00) | 57.1 (1.00) | 51.5 (1.00) | 51.6 (1.00) | 51.1 (1.00) |
|   128 | 54.8 (1.00) | 54.6 (1.00) | 54.6 (1.00) | 48.7 (1.00) | 47.1 (2.19) | 36.6 (4.00) |
|   512 | 53.8 (1.00) | 53.9 (1.00) | 54.3 (1.00) | 48.3 (1.00) | 29.9 (2.07) | 20.9 (4.00) |
|  2048 | 54.6 (1.00) | 34.3 (2.02) | 17.4 (3.98) | 48.9 (1.00) | 27.2 (1.97) | 15.4 (3.95) |

### PSP103

| Lanes |      Before 1 |     Before 2 |     Before 4 |       After 1 |      After 2 |      After 4 |
| ----: | ------------: | -----------: | -----------: | ------------: | -----------: | -----------: |
|    32 |  952.3 (1.00) | 937.5 (1.00) | 933.1 (1.00) |  959.3 (1.00) | 949.7 (1.00) | 952.8 (1.00) |
|   128 |  961.7 (1.00) | 973.5 (1.00) | 960.8 (1.00) |  971.2 (1.00) | 516.5 (2.02) | 270.3 (3.99) |
|   512 |  968.7 (1.00) | 969.7 (1.00) | 980.5 (1.00) |  978.2 (1.00) | 503.9 (2.00) | 262.7 (3.93) |
|  2048 | 1368.7 (1.00) | 694.6 (1.98) | 391.8 (3.69) | 1350.5 (1.00) | 679.9 (1.98) | 361.1 (3.88) |

## Diagnosis and policy

Timing-only ablations rebuild the **same baseline cdylib** and change one region at a time. The no-copy and no-zero
variants deliberately violate provider semantics and are never production options or qualification evidence. All cells
below are 512 lanes, one worker, ns/lane. Absolute numbers can differ slightly from the extension-linked before/after
table because these are separate builds and runs.

| Baseline variant                      | Waveguide | Resistor | PSP103 |
| ------------------------------------- | --------: | -------: | -----: |
| Unchanged                             |      77.8 |     51.8 |  970.0 |
| Omit model and instance restores      |      68.9 |     51.3 |  833.2 |
| Omit output zero fill                 |      76.5 |     51.7 |  961.7 |
| Reduce errors instead of per-lane Vec |      82.1 |     60.8 |  970.3 |
| Allocation-free disjoint-region check |      79.6 |     54.6 |  965.7 |

- Snapshot copies account for about 9 ns/lane in the optical case and 137 ns/lane in PSP103 in this ablation. The model
  argument and instance argument are both mutable in the provider ABI. The synthetic mutation test demonstrates why
  neither region can be omitted generically. Production restores both on every lane, regardless of qualification.
- Zero filling is a small fraction of these residual-only workloads. Accumulating residual callbacks still require it;
  all output initialization is retained.
- Removing the error Vec alone is not a measured speedup here: copying/reducing the large error record has its own cost.
  Removing it and the region-check allocation satisfies the allocation contract. The combined implementation also
  amortizes the scratch TLS borrow over a chunk and skips copying full error records for successful lanes. The warmed
  serial full checked path has a zero-allocation Rust regression test.
- The two registry/handle read locks together measure about 16.4 ns **per call** in an isolated uncontended loop; they
  cannot explain tens of ns per lane at these batch sizes. Configuration capture and the handle-policy read remain;
  their individual costs were not separately isolated.
- An empty owned-pool `install` measures about 1.07 microseconds per call at four workers (median). The workload-level
  chunk comparisons below include wake-up, scheduling and reduction overhead. The revised evaluator uses explicit chunk
  ranges rather than relying on `with_min_len` to define tasks. It keeps 32-lane groups serial.
- Aligning each caller buffer to 64 bytes does not improve the measured default-policy optical/resistor results below.
  This does not establish that all output layouts are immune to false sharing. Participation flags are separately
  cache-line aligned so that observing workers does not add a shared hot flag line.

Four configured workers, ns/lane; `aligned` retains the new 128/32 policy. The old-policy column uses the new core with
1024/256, isolating policy from implementation. The one-lane override is 1/1.

| Model     | Lanes | New default | Old-policy override | One-lane chunks | 64-byte aligned |
| --------- | ----: | ----------: | ------------------: | --------------: | --------------: |
| Waveguide |   128 |        43.2 |                71.8 |            69.6 |            44.2 |
| Waveguide |   512 |        26.9 |                72.8 |            39.6 |            28.3 |
| Waveguide |  2048 |        21.9 |                22.9 |            31.1 |            23.6 |
| Resistor  |   128 |        36.6 |                49.8 |            65.7 |            37.3 |
| Resistor  |   512 |        20.9 |                49.0 |            33.5 |            22.4 |
| Resistor  |  2048 |        15.4 |                17.4 |            24.9 |            16.9 |

## Transactional setup

This compares an all-lane identity update with a fresh `bosdi_setup_batch_delays_checked` and handle free, using the
same registered library, finite explicit waveguide parameters and validation X/context. Both routes perform full output
and duration preparation. The legacy v1 setup-only route omits that evaluation and is not the reference for this table.
The native timing helper excludes library registration and reports the median of three 50-ms trials. Values are ns/lane.

Staged plans are built once and retained through publication; all-lane identity updates transfer the staged snapshots
without copying them. There is no library reload. Every measured all-lane update is cheaper than fresh validated
preparation; this is a qualification for this workload and host, not a universal timing bound for arbitrary providers.

| Lanes | Workers | Fresh validated preparation | Update all lanes | Update/fresh |
| ----: | ------: | --------------------------: | ---------------: | -----------: |
|    32 |       1 |                       470.3 |            356.8 |         0.76 |
|    32 |       2 |                       486.6 |            359.3 |         0.74 |
|    32 |       4 |                       508.1 |            354.1 |         0.70 |
|   128 |       1 |                       405.5 |            350.0 |         0.86 |
|   128 |       2 |                       393.2 |            270.6 |         0.69 |
|   128 |       4 |                       283.4 |            186.7 |         0.66 |
|   512 |       1 |                       394.2 |            345.8 |         0.88 |
|   512 |       2 |                       297.4 |            249.4 |         0.84 |
|   512 |       4 |                       229.8 |            160.6 |         0.70 |
|  2048 |       1 |                       479.3 |            350.7 |         0.73 |
|  2048 |       2 |                       261.9 |            240.0 |         0.92 |
|  2048 |       4 |                       202.8 |            155.7 |         0.77 |

## Exact commands

From `/home/cdaunt/code/bosdi`, the baseline extension was preserved **before** editing and rebuilding:

```bash
cp src/osdi_shim_nb.cpython-313-x86_64-linux-gnu.so /tmp/bosdi-before.so
mkdir -p build/checked-groups
CIRCULOX_OPENVAF=/home/cdaunt/.local/bin/openvaf-r
"$CIRCULOX_OPENVAF" tests/devices/delays/waveguide.va -o build/checked-groups/waveguide.osdi
PYTHONPATH=src:tests taskset -c 0-3 pixi run python scripts/benchmark_checked_groups.py --library /tmp/bosdi-before.so --optical build/checked-groups/waveguide.osdi --output build/checked-groups/before-pinned.json
pixi run build
PYTHONPATH=src:tests taskset -c 0-3 pixi run python scripts/benchmark_checked_groups.py --library src/osdi_shim_nb.cpython-313-x86_64-linux-gnu.so --optical build/checked-groups/waveguide.osdi --output build/checked-groups/after-pinned.json
PYTHONPATH=src:tests taskset -c 0-3 pixi run python scripts/benchmark_checked_setup.py --optical build/checked-groups/waveguide.osdi --output build/checked-groups/setup-pinned.json
pixi run python scripts/benchmark_checked_ablation.py
```

The ablation builder defaults to the pinned baseline commit and writes only under `/tmp/bosdi-checked-ablation`. For
each `VARIANT` in `baseline`, `no_snapshot_copy`, `no_output_zero`, `no_error_vec`, `no_region_allocation`, run:

```bash
PYTHONPATH=src:tests taskset -c 0-3 pixi run python scripts/benchmark_checked_groups.py --library /tmp/bosdi-checked-ablation/VARIANT.so --optical build/checked-groups/waveguide.osdi --lanes 512 --workers 1 --output build/checked-groups/ablate-VARIANT.json
```

The three policy/alignment diagnostics use this common invocation, with the corresponding extra option and output name:

```bash
PYTHONPATH=src:tests taskset -c 0-3 pixi run python scripts/benchmark_checked_groups.py --library src/osdi_shim_nb.cpython-313-x86_64-linux-gnu.so --optical build/checked-groups/waveguide.osdi --models optical resistor_va.osdi --lanes 128 512 2048 --workers 4 --policy 1024 256 --output build/checked-groups/diagnostic-old-policy.json
PYTHONPATH=src:tests taskset -c 0-3 pixi run python scripts/benchmark_checked_groups.py --library src/osdi_shim_nb.cpython-313-x86_64-linux-gnu.so --optical build/checked-groups/waveguide.osdi --models optical resistor_va.osdi --lanes 128 512 2048 --workers 4 --policy 1 1 --output build/checked-groups/diagnostic-chunk1.json
PYTHONPATH=src:tests taskset -c 0-3 pixi run python scripts/benchmark_checked_groups.py --library src/osdi_shim_nb.cpython-313-x86_64-linux-gnu.so --optical build/checked-groups/waveguide.osdi --models optical resistor_va.osdi --lanes 128 512 2048 --workers 4 --align64 --output build/checked-groups/diagnostic-aligned.json
```

The timing-only baseline exports `bosdi_bench_lock_ns(size_t)` and `bosdi_bench_install_ns(size_t)`, returning double
ns/call. After `bosdi_set_num_threads(4)`, the recorded three trials used 1,000,000 and 10,000 iterations respectively.

The installed default `openvaf-r` emits ABI 0.5; existing legacy tests expect 0.4. A local PATH entry selects the
packaged 0.4 compiler while explicitly selecting the Circulox compiler for new delays and existing optional 0.5 cases:

```bash
mkdir -p /tmp/bosdi-tools
ln -s /home/cdaunt/code/bosdi/.pixi/envs/integration/lib/python3.13/site-packages/vacask_bin/data/bin/openvaf-r /tmp/bosdi-tools/openvaf-r
PATH=/tmp/bosdi-tools:$PATH CIRCULOX_OPENVAF=/home/cdaunt/.local/bin/openvaf-r BOSDI_OPENVAF_05=/home/cdaunt/.local/bin/openvaf-r BOSDI_BASELINE_LIBRARY=/tmp/bosdi-before.so pixi run test
pixi run cargo test
```

Final validation: **528 passed, 33 skipped** in the standalone suite, including independent pre-change native bitwise
comparisons; **3 Rust tests passed**. Skips are optional integration/compiler-IR/platform cases, not missing delay
qualification in this run. Python formatting/lint and `git diff --check` pass.

## Remaining qualification

The supplied Circulox `variable_delay` has maxdelay with a parameter-only duration; `mode_delay` changes the delayed
signal. They are tested unchanged. Separate compiled fixtures exercise voltage-dependent and analysis-dependent td. The
canonical waveguide is copied unchanged with its discipline include. Its four td values agree per lane, vary with lane
length, and are refreshed after changed-lane setup. Both raw hidden-node maps and Jacobian references are checked; z
residual rows remain zero for the host's history relationship.

This qualifies trusted local Linux CPU artifacts. Windows, macOS, browser execution, general OSDI delay providers and
end-to-end Circulox waveguide admission have not been exercised here. BOSDI owns no accepted history, interpolation,
condensation, transactions in Circulox, or alternate numerical backend. The global staging lock is conservative: it
excludes unrelated handle evaluations too. Warm-up allocations and initial model loading remain. No provider has been
newly certified for snapshot omission, and no universal four-worker speedup is claimed. Changes remain uncommitted.
