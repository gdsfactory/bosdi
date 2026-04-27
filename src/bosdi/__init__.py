"""bosdi — Verilog-A support for circulax (compiled OSDI + differentiable MIR).

Two backends are exposed:

- ``bosdi.osdi``  — load OpenVAF-compiled ``.osdi`` shared libraries and
  evaluate them via Rayon-parallel C FFI calls (see ``osdi_loader``,
  ``osdi_jax``, ``osdi_debug``).  This is the static / non-differentiable
  path; the OSDI library does the physics in compiled C.

- ``bosdi.va``    — read the compiler's MIR via the ``openvaf_py`` PyO3
  binding, run SCCP / dead-block elimination on it, and lower to JAX-
  traceable Python code.  Slower per-step but parameters stay as JAX
  inputs so ``jax.grad`` and ``jax.vmap`` work end-to-end.

The two paths share the bosdi Rust core; the VA backend is opt-in (it
needs ``openvaf_py`` installed; if absent only the OSDI path is
available).
"""

# OSDI path — top-level modules are still importable directly
# (``import osdi_loader`` continues to work for back-compat).
__all__ = ["va"]
