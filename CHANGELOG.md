# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-06-11

### Added

- Verilog-A to JAX lowering compiler (`bosdi.va`) — **alpha**
- Circulax integration subpackage (`bosdi.circulax`) with `@va_component` and `@osdi_component` decorators
- Multi-platform build support (macOS, Windows, Linux)
- Python 3.11–3.14 support
- SCCP optimization pass for constant folding in lowered VA models
- PHI node batching and dominator-based diamond detection

## [0.1.0] - 2025-04-08

### Added

- OSDI 0.4 device model loading via Rust `libloading` with descriptor caching
- Batched parallel evaluation of N device instances via Rayon
- JAX custom call bridge via XLA FFI + nanobind C++ shim
- `@custom_jvp` support using analytical Jacobians (conductances dI/dV, capacitances dQ/dV)
- `osdi_eval()` Python API returning currents, conductances, charges, capacitances, and updated state
- `load_osdi_model()` loader returning `OsdiModel` dataclass with metadata and buffer helpers
- Resistor and capacitor OSDI binaries included for testing
- Full `jax.grad()` / `jax.jit()` composition support through OSDI models

[0.1.0]: https://github.com/gdsfactory/bosdi/releases/tag/v0.1.0
[0.2.0]: https://github.com/gdsfactory/bosdi/releases/tag/v0.2.0
