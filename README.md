# bosdi — Batched OSDI

![CI](https://github.com/gdsfactory/bosdi/actions/workflows/test.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)
![Platform: Linux | macOS | Windows](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey)
![Status: Experimental](https://img.shields.io/badge/status-experimental-orange)

> **Experimental** — bosdi is under active development. The OSDI binary evaluation path is stable and well-tested, but
> the Verilog-A to JAX lowering compiler (`bosdi.va`) is in **alpha** and its API may change without notice. The VA
> lowering depends on a [custom fork of OpenVAF](https://github.com/cdaunt/OpenVAF) that exposes the compiler's
> intermediate representation; this fork is not yet merged upstream.

Evaluate [OSDI](https://github.com/OpenVAF/OpenVAF) device models (Verilog-A compiled to `.osdi` binaries) in batched
parallel via JAX.

## Two evaluation paths

bosdi provides two ways to evaluate Verilog-A compact models inside JAX:

### OSDI binary path (stable)

Loads a pre-compiled `.osdi` binary and evaluates N device instances in parallel via Rayon inside a JAX XLA custom call.
The OSDI ABI provides analytical Jacobians with respect to **node voltages only** (conductances `dI/dV`, capacitances
`dQ/dV`). A `@custom_jvp` rule makes `jax.grad()` work through node voltages — but not through model parameters or
state.

```python
from osdi_loader import load_osdi_model
from osdi_jax import osdi_eval

model = load_osdi_model("path/to/device.osdi")
N = 1024
voltages = jnp.zeros((N, model.num_nodes), dtype=jnp.float64)
params = jnp.full((N, model.num_params), jnp.nan, dtype=jnp.float64)
old_state = jnp.zeros((N, model.num_states), dtype=jnp.float64)

cur, cond, chg, cap, new_state = osdi_eval(model.id, voltages, params, old_state)

# jax.grad works through node voltages
grad_fn = jax.grad(lambda v: osdi_eval(model.id, v, params, old_state)[0].sum())
```

### VA to JAX lowering (alpha)

Compiles Verilog-A source directly into pure JAX/Python, producing a function that is **fully differentiable** through
all inputs — voltages, parameters, and temperature. This enables parameter optimization, sensitivity analysis, and
end-to-end gradient-based design flows that the OSDI path cannot support.

Requires [openvaf-r](https://github.com/cdaunt/OpenVAF) (a custom OpenVAF fork).

```bash
python -m bosdi.va device.va
```

### When to use which

|                           | OSDI binary                                   | VA to JAX                                                                   |
| ------------------------- | --------------------------------------------- | --------------------------------------------------------------------------- |
| **Use case**              | Circuit simulation (Newton solve)             | Parameter fitting, sensitivity analysis, inverse design                     |
| **Differentiable w.r.t.** | Node voltages only                            | Voltages, parameters, and temperature                                       |
| **Performance**           | Fast — Rayon-parallel C/Rust, batched XLA FFI | Pure Python/JAX — slower per-eval, but composable with `jax.jit`/`jax.vmap` |
| **Maturity**              | Stable                                        | Alpha                                                                       |
| **Dependencies**          | None beyond bosdi                             | [openvaf-r](https://github.com/cdaunt/OpenVAF) fork                         |

The OSDI path treats the compiled model as a black box and extracts only what the ABI exposes: currents, charges, and
their Jacobians w.r.t. node voltages. This is exactly what a Newton solver needs, but the parameter axis is opaque to
JAX — you cannot backpropagate through it.

The VA to JAX path exists to remove that limitation. By lowering the Verilog-A source into native JAX operations, every
computation becomes visible to JAX's autodiff, making the model fully differentiable. This is what enables
gradient-based parameter extraction, design-space exploration, and end-to-end optimization of circuits where device
parameters are the degrees of freedom.

## Architecture

```
OSDI path:
  Python: osdi_eval()  →  JAX XLA custom call
    →  C++ (nanobind/XLA FFI): unpack buffers
      →  Rust (Rayon): evaluate N devices in parallel
        →  OSDI binary: currents, conductances, charges, capacitances

VA path:
  Verilog-A source  →  openvaf-r (MIR dump)
    →  bosdi.va lowering + SCCP optimization
      →  Pure JAX/Python function (fully differentiable)
```

## Installation

### Using Pixi (recommended)

```bash
git clone https://github.com/gdsfactory/bosdi && cd bosdi
pixi run build
```

### Using pip

```bash
pip install bosdi
```

## Build & test

```bash
pixi run build   # compile Rust static lib + C++ extension
pixi run test    # standalone pytest suite; Circulax is not required
git submodule update --init tests/pdks/ihp        # pinned IHP device libraries
pixi run --locked -e integration test-integration  # build and test with Circulax + IHP

# single test
pixi run pytest tests/test_osdi.py::test_resistor_dc_evaluation -v
```

The `integration` environment has its own solve group and includes Circulax only for testing. It builds this checkout's
native extension before running public DC/AC/transient, simulator-settings and generated-component checks. Both Linux
and Windows CI run it alongside the standalone suite. Circulax is temporarily pinned to the immutable integration commit
for PR #64; replace that pin with an upstream release once the required public native APIs are released. No Circulax
Verilog-A extra is requested, so tests use this checkout's bosdi rather than installing a second copy. The netlists
extra provides its model-card parser. OpenVAF must be on PATH; JSON lowering tests additionally need the custom
compiler's dump support.

The IHP integration suite uses the pinned `gdsfactory/IHP` submodule in `tests/pdks/ihp`, rather than copied model
fixtures. It enumerates all 34 SG13G2 VACASK subcircuits and checks compiled, JIT-executed DC and small-signal responses
at 1 MHz and 1 GHz against VACASK. Updating the submodule adds a failing catalogue check if new devices need test cases.
VACASK is pinned to an OSDI 0.4-compatible build and is a test-only dependency.

These are typical-corner device checks at explicit sizes and biases, not complete process qualification. The test
harness hoists repeated common includes, grounds implicit BJT substrate terminals, and folds varactor voltage terms only
after verifying their coefficients are zero. Isolation diodes use a nonzero well spacing to avoid an upstream `ln(0)`
expression; the wrapper forwards public geometry parameters while child cards recompute private derived values. The
upstream checkout stays unchanged, and these parser adaptations do not imply support for arbitrary voltage-dependent
model-card expressions. The test-only PDK is excluded from source packages.

## OSDI outputs

The OSDI path returns per-device arrays shaped by `model.num_nodes` (terminals + internal nodes + branch-current
auxiliaries):

| Output | Shape             | Description                                  |
| ------ | ----------------- | -------------------------------------------- |
| `cur`  | `[N, num_nodes]`  | Resistive current residual at each unknown   |
| `cond` | `[N, num_nodes²]` | `G = ∂cur/∂V` Jacobian (flattened row-major) |
| `chg`  | `[N, num_nodes]`  | Charge residual at each unknown              |
| `cap`  | `[N, num_nodes²]` | `C = ∂chg/∂V` Jacobian (flattened row-major) |

Pass `jnp.nan` for any parameter to use its Verilog-A default. Parameters can be addressed by name via
`model.param_names`. See `tests/test_bsim4_model_card.py` for a full example.

## Further reading

- [OSDI technical reference](docs/osdi-technical-reference.md) — parameter handling, model introspection, output layout,
  host-simulator integration (companion method vs MNA/DAE), and debug utilities

## Limitations

- **Platform:** Linux, macOS, and Windows; Python 3.11+; OSDI 0.4 ABI only. `.osdi` binaries are platform-specific —
  compile from `.va` sources via [openvaf-r](https://github.com/cdaunt/OpenVAF) on each target
- **OSDI differentiability:** `jax.grad()` works through node voltages only, not model parameters — use the VA path for
  parameter gradients
- **ABI states** (`num_states > 0`): the descriptor rejects these by default. Audited OpenVAF voltage-limiting slots may
  use the explicit policy described below; generic history-dependent state is unsupported.
- **VA lowering (alpha):** user-defined `analog function` calls and noise contributions are not yet supported

### Native OSDI node collapse

`OsdiModel.num_nodes` includes every raw OSDI node, including internal nodes that instance setup may collapse. Allocate
voltage/state buffers using the model metadata rather than the external terminal count. The evaluator applies only
`setup_instance`'s selected collapse flags, and represents unused raw node slots with voltage-equality equations. This
preserves a fixed shape for batches whose instances have different parasitic resistances. Both cached and uncached
native paths use the same mapping. Physical currents and charges are stamped into the surviving node; equality rows
carry no charge.

### Integral operators and analysis modes

`load_osdi_model(..., analysis="dc" | "ac" | "tran")` and `osdi_component(..., analysis=...)` select an immutable
evaluation mode for a model registration. The default `"ac"` preserves the full current/charge stamp API. DC disables
reactive evaluation so `idt` uses its explicit initial-value equation. AC and transient enable integration consistently
in both full and residual-only evaluation. OpenVAF uses `CALC_REACT_JACOBIAN` to select these integral equations, so
clearing it only for a residual-only call is incorrect.

To reproduce VACASK AC, first solve DC and retain its conductance matrix. Then evaluate the reactive Jacobian in AC mode
at that operating point and solve `(G_dc + j*omega*C_ac) x = rhs`. A new registration/handle is required to change mode.
Transient callers must provide a DC-consistent initial point. This does not add general state-history or `$abstime`
support.

### OpenVAF voltage-limiting slots

OpenVAF derives OSDI `num_states` from its `$limit` slots. These Newton limiting buffers are distinct from physical
`ddt` charges and `idt` unknowns, which are represented by the circuit DAE. For an audited binary whose slots serve only
voltage limiting, use `osdi_component(..., state_policy="limiting_only")`. Bosdi leaves `ENABLE_LIM` disabled, evaluates
the unmodified device equations, and does not propagate ABI state outputs. This can reduce convergence robustness
compared with a simulator that enables limiting. It does not add fictitious delayed unknowns.

The default `state_policy="reject"` remains appropriate for an unaudited binary. The ABI does not describe what its
state slots mean; `limiting_only` is an explicit assertion by the caller, not automatic compiler detection. Generic
history-dependent models, `$abstime`, and enabled voltage limiting still need a simulator lifecycle implementation.

`descriptor.with_analysis("dc" | "ac" | "tran")` returns cached immutable registrations preserving the binary path,
ports, defaults, temperature, and state policy. Circulax uses these registrations to orchestrate native analyses.

### Registration cache lifetime

`load_osdi_model` reuses native registration IDs across newly created descriptors and circuits using a process-local
`@lru_cache(maxsize=128)`. The key includes canonical binary path, filesystem identity/size/timestamps, ABI, temperature
and analysis mode. Concurrent misses are serialized. Callers receive independent metadata copies; port names and
model-card defaults remain descriptor-local. Failed loads are not cached.

This bounds Python cache entries, not the native registry: native IDs remain alive for existing circuits and JIT
executables until the process exits. Rebuild binaries under new/content-addressed paths (as Circulax's compilation cache
does); replacing a live shared-library file in place is not a supported reload strategy. Native IDs and batch handles
cannot be persisted across processes; compiled binary artifacts can.

Batch setup handles remain circuit-owned because they carry instance parameters and expose `free()`. A global LRU of
those mutable handles would require shared ownership and invalidation before it could safely be introduced. The
per-descriptor/per-circuit analysis caches have only three valid modes and retain their local lifetime.

## Releases

The Git tag is the Python and Pixi package version. No manual version bump is needed. `setuptools-scm` writes
`bosdi/_version.py` during the build and preserves the version in source archives. The Cargo package is a private static
library with its own internal version; it is not published to crates.io.

After merging the changes to `main`, tag the release commit and push the tag:

```bash
git switch main
git pull --ff-only
git tag -a v0.1.8 -m "Release 0.1.8"
git push origin v0.1.8
```

Use the next unused version. The release workflow builds Python 3.12/3.13 wheels for Linux, macOS, and Windows, runs the
release tests, and checks every artifact's filename and embedded version against the tag before publishing. Stable tags
(`vX.Y.Z`) publish to PyPI; prereleases (`vX.Y.Za1`, `vX.Y.Zb1`, `vX.Y.Zrc1`, or `vX.Y.Z.dev1`) publish to TestPyPI and
are marked as prereleases on GitHub.

An existing GitHub release is updated and receives the build artifacts. Failed workflow jobs can be rerun; already
published PyPI filenames are skipped. Package contents for a published version are immutable, so code changes require a
new tag. Tagging alone does not update the committed changelog; review its Unreleased section before releasing.

The old `v0.1.7` release did not publish a `0.1.7` package. Publish the repaired setup under a new unused tag rather
than moving the existing release tag.
