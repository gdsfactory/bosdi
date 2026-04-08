# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build the C++ extension (compiles Rust static lib + C++ nanobind shim)
pixi run build

# Clean build and rebuild from scratch
pixi run build-clean

# Run tests
pixi run test

# Run a single test
pixi run pytest tests/test_osdi.py::test_resistor_dc_evaluation -v
```

All commands should be run inside the Pixi environment. If Pixi is not activated, prefix with `pixi run`.

## Architecture

BOSDI is a three-layer bridge that makes OSDI device models (Verilog-A compiled to `.osdi` ELF binaries) differentiable
via JAX.

### Layer 1 — Rust (`src/lib.rs`)

Core engine. Exposes two C FFI functions:

- `load_osdi_library(path)`: Dynamically loads an `.osdi` binary using `libloading`, extracts the `OSDI_DESCRIPTORS`
  symbol (OSDI 0.4 ABI), caches the function pointer in a global `HashMap<u32, LoadedOsdi>`, returns model metadata.
- `batched_osdi_eval_ffi()`: Uses Rayon parallel iterators to evaluate N device instances simultaneously, zero-copy by
  zipping input/output array chunks.

Built as a `staticlib` (`libbosdi.a`), linked into the C++ extension at build time.

### Layer 2 — C++ (`src/osdi_shim.cpp`)

JAX XLA FFI bridge. Two responsibilities:

- **XLA FFI handler** (`batched_osdi_eval_impl`): Receives buffers from JAX's XLA runtime, translates to C++ FFI
  structs, calls into Rust.
- **Nanobind bindings**: Wraps Rust functions as Python-callable objects; registers `OsdiEvalCpu` as an XLA custom call
  target.

Compiled via `setup.py` into `osdi_shim_nb.*.so`.

### Layer 3 — Python (`src/osdi_jax.py`, `src/osdi_loader.py`)

- `osdi_loader.py`: `load_osdi_model()` calls the nanobind wrapper, returns an `OsdiModel` dataclass with metadata and a
  `allocate_jax_buffers()` helper.
- `osdi_jax.py`: `osdi_eval()` decorated with `@custom_jvp` — dispatches to the XLA custom call and provides analytical
  JVP using OSDI's built-in Jacobians (conductances dI/dV, capacitances dQ/dV). This is what makes `jax.grad()` work
  through OSDI models.

### Data Flow

```
Python: osdi_eval(model_id, voltages, params, state)
  → JAX XLA dispatch: custom call "OsdiEvalCpu"
    → C++: batched_osdi_eval_impl() unpacks FFI buffers
      → Rust: batched_osdi_eval_ffi() runs N devices in parallel (Rayon)
        → OSDI model: evaluates currents, conductances, charges, capacitances
```

Outputs per device: `[currents, conductances, charges, capacitances]` — all `f64`.

### Build System

`setup.py` orchestrates the two-stage build:

1. `cargo build --release` → produces `target/release/libbosdi.a`
1. Compiles `osdi_shim.cpp` with nanobind and JAX FFI headers, links against `libbosdi.a`

The Pixi environment (Python 3.13, Linux-64) provides JAX, jaxlib, nanobind, and all C++ headers.
