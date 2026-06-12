# bosdi — Batched OSDI

![CI](https://github.com/gdsfactory/bosdi/actions/workflows/test.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)
![Platform: Linux | macOS](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey)
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
pixi run test    # run pytest suite

# single test
pixi run pytest tests/test_osdi.py::test_resistor_dc_evaluation -v
```

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

- **Platform:** Linux and macOS; Python 3.11+; OSDI 0.4 ABI only. `.osdi` binaries are platform-specific — compile from
  `.va` sources via [openvaf-r](https://github.com/cdaunt/OpenVAF) on each target
- **OSDI differentiability:** `jax.grad()` works through node voltages only, not model parameters — use the VA path for
  parameter gradients
- **Stateful models** (`num_states > 0`): evaluation is skipped and outputs are zeroed
- **VA lowering (alpha):** user-defined `analog function` calls and noise contributions are not yet supported
