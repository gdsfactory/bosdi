# bodi

Make [OSDI](https://github.com/OpenVAF/OpenVAF) device models (Verilog-A compiled to `.osdi` binaries) differentiable via JAX.

## What it does

Wraps OSDI 0.4 device models in a JAX custom call so you can use `jax.grad()` through them. Analytical Jacobians (conductances dI/dV, capacitances dQ/dV) are provided directly by the OSDI model — no finite differences. Batched evaluation runs in parallel via Rayon.

```python
from osdi_loader import load_osdi_model
from osdi_jax import osdi_eval

model = load_osdi_model("path/to/device.osdi")
cur, cond, chg, cap, new_state = osdi_eval(model.id, voltages, params, old_state)

# Full JAX AD support
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

## Requirements

Linux x86-64, Python 3.13, Rust toolchain. All Python/C++ dependencies managed by [Pixi](https://pixi.sh).

## Build & test

```bash
pixi run build   # compile Rust static lib + C++ extension
pixi run test    # run pytest suite

# single test
pixi run pytest tests/test_osdi.py::test_resistor_dc_evaluation -v
```
