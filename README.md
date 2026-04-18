# bosdi — Batched OSDI

![CI](https://github.com/OWNER/bosdi/actions/workflows/test.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)
![Platform: Linux x86-64](https://img.shields.io/badge/platform-linux--x86--64-lightgrey)

Batched evaluation of [OSDI](https://github.com/OpenVAF/OpenVAF) device models (Verilog-A compiled to `.osdi` binaries)
with JAX differentiability.

## What it does

Evaluates batches of N OSDI 0.4 device instances in parallel (via Rayon) inside a JAX XLA custom call, so you can use
`jax.grad()` through the entire batch. Analytical Jacobians (conductances dI/dV, capacitances dQ/dV) come directly from
the OSDI model — no finite differences.

```python
import jax
import jax.numpy as jnp
from osdi_loader import load_osdi_model
from osdi_jax import osdi_eval

model = load_osdi_model("path/to/device.osdi")

# Batch of N devices evaluated in parallel via Rayon.
# Use num_nodes (not num_pins) — this includes internal Kirchhoff nodes and
# branch-current auxiliaries that the solver exposes as unknowns.
N = 1024
voltages = jnp.zeros((N, model.num_nodes), dtype=jnp.float64)

# Pass NaN for parameters you want to leave at Verilog-A defaults.
# Set only the parameters you care about.
params = jnp.full((N, model.num_params), jnp.nan, dtype=jnp.float64)
params = params.at[:, 0].set(1.0)   # $mfactor = 1 for all devices

old_state = jnp.zeros((N, model.num_states), dtype=jnp.float64)

# Returns batched outputs — one row per device
cur, cond, chg, cap, new_state = osdi_eval(model.id, voltages, params, old_state)

# jax.grad works through the batched call — no finite differences
grad_fn = jax.grad(lambda v: osdi_eval(model.id, v, params, old_state)[0].sum())
```

## Architecture

```
Python: osdi_eval()  →  JAX XLA custom call "OsdiEvalCpu"
  →  C++ (nanobind/XLA FFI): unpack buffers
    →  Rust (Rayon): evaluate N devices in parallel
      →  OSDI model: currents, conductances, charges, capacitances
```

- **`src/lib.rs`** — Rust core: loads `.osdi` via `libloading`, batched eval with Rayon
- **`src/osdi_shim.cpp`** — C++ XLA FFI handler + nanobind Python bindings
- **`src/osdi_jax.py`** — JAX wrapper with `@custom_jvp` for autodiff
- **`src/osdi_loader.py`** — Python model loader returning metadata + buffer helpers

## Parameters

OSDI models order parameters as listed in the `param_opvar` table compiled into the binary. bosdi uses the OSDI
`access()` function to write each parameter to the correct slot in the model/instance struct, so parameter ordering is
handled automatically.

Pass `jnp.nan` for any parameter you want to leave at its Verilog-A default — bosdi skips writing NaN values and lets
`setup_model`/`setup_instance` apply the compiled-in defaults. This is the recommended approach for complex models
(BSIM, PSP) where many parameters have safe defaults.

## Outputs

Outputs are one row per *unknown* — terminals first, then internal Kirchhoff nodes and branch-current auxiliaries.
`model.num_nodes` is the width; `model.num_pins` is the terminal count (`num_pins ≤ num_nodes`).

| Output | Shape             | Description                          |
| ------ | ----------------- | ------------------------------------ |
| `cur`  | `[N, num_nodes]`  | Resistive currents at each unknown   |
| `cond` | `[N, num_nodes²]` | dI/dV Jacobian (flattened row-major) |
| `chg`  | `[N, num_nodes]`  | Charges at each unknown              |
| `cap`  | `[N, num_nodes²]` | dQ/dV Jacobian (flattened row-major) |

## Installation

### Using Pixi (recommended for development)

```bash
git clone https://github.com/OWNER/bosdi && cd bosdi
pixi run build
```

### Using pip (Linux x86-64, Python 3.13, Rust toolchain required)

```bash
pip install bosdi
```

## Build & test

```bash
pixi run build   # compile Rust static lib + C++ extension
pixi run test    # run pytest suite

# single test
pixi run pytest tests/test_osdi.py::test_resistor_dc_evaluation -v
```

## Limitations

**Platform:** Linux x86-64, Python 3.13, OSDI 0.4 ABI only.

**Stateful models** (`num_states > 0`, e.g. BSIM3v3, SPICE wrappers): evaluation is skipped and outputs are zeroed.
Stateful model support is not yet implemented.

**Branch-current auxiliary unknowns:** bosdi exposes internal *voltage* nodes as real unknowns in the solver, but models
that define inductive or ideal-source behaviour through auxiliary *current* unknowns (e.g. the compiled inductor's flux
node) will still produce zero outputs for the affected quantities — the Jacobian stamp is present in the full MNA system
but requires a Newton/DAE solve that bosdi does not perform.

**Known crashes:** `bsim4v8.osdi` (crashes in `setup_instance`) and `vbic_vbic_1p3.osdi` (crashes in `eval`) segfault
with default parameters. Root cause is under investigation. The equivalent 5-terminal models (`bsimbulk106`,
`vbic_vbic_4T_et_cf`) and other complex models (PSP103, BSIM3v3) work correctly.
