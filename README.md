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
import jax.numpy as jnp
from osdi_loader import load_osdi_model
from osdi_jax import osdi_eval

model = load_osdi_model("path/to/device.osdi")

# Batch of N devices evaluated in parallel via Rayon
N = 1024
voltages   = jnp.zeros((N, model.num_terminals - 1))  # per-device bias points
params     = jnp.tile(model.default_params, (N, 1))   # per-device parameters
old_state  = jnp.zeros((N, model.num_states))          # per-device internal state

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

- Linux x86-64 only
- Python 3.13 only
- OSDI 0.4 ABI only
