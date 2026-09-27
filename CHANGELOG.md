# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.6] - 2026-09-27

Consolidates the Verilog-A lowering fixes that downstream consumers (notably
[photonflux](https://github.com/alexsludds/photonflux)) have been carrying as local workarounds against 0.1.5.

### Added

- `lower(collapse_nodes=...)` now accepts a `Collection[tuple[str, str]]` allow-list, not just `bool`. OpenVAF emits a
  `CollapseHint` for every conditional `V(a,b) <+ 0`, but whether a hint fires depends on parameters resolved at setup
  (BSIM4's `rdsmod`/`rgatemod`/`rbodymod`, the diode's `Rs`). Applying every hint unconditionally silently shorted live
  resistance networks in exactly the models collapse is recommended for.
- `lower(safe_divide_mode="mask")` — opt-in mode giving dead branches finite values instead of letting inf/NaN leak into
  the Jacobian. Default stays `"overflow"`.
- Windows platform support re-enabled, with OSDI tests in CI.
- Error propagation through the FFI layers.

### Fixed

- Consistent state names for implicit branch-current unknowns. Emitted Python referenced states that did not exist
  (BSIM4 with `rbodymod=1` read `s.i_sbulk`/`s.i_b`/`s.i_bi` while the states tuple declared `i_br18..i_br22`).
- Keyword-safe aliases for VA parameters named after Python keywords — BSIM4 declares `parameter real as` and
  `parameter real lambda`, both of which previously emitted invalid Python.
- `osdi_component` fills unspecified parameters with NaN rather than `0.0`.
- OSDI function pointers are read from the descriptor first, with a fallback.

### Upgrading from 0.1.5

If you post-process bosdi's emitted source to repair any of the above, **remove those repairs in the same bump** — the
defects they targeted are gone, and the repairs would otherwise be applied to already-correct output.

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
[0.1.6]: https://github.com/gdsfactory/bosdi/releases/tag/v0.1.6
[0.2.0]: https://github.com/gdsfactory/bosdi/releases/tag/v0.2.0
