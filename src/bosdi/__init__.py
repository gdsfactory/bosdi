"""Verilog-A support through compiled OSDI libraries and differentiable MIR.

Native modules (``osdi_loader``, ``osdi_jax``, ``osdi_debug``) evaluate
OpenVAF-compiled shared libraries through the Rust/C++ JAX FFI. Voltage
Jacobians are supported; compiled model parameters are not differentiable.
Native imports do not require Circulax.

``bosdi.va`` obtains MIR using the external ``openvaf-r --dump-json``
compiler, simplifies it and emits JAX-traceable Python. This path supports
parameter differentiation and requires a compiler with the JSON dump option,
not an ``openvaf_py`` Python binding.
"""

# OSDI path — top-level modules are still importable directly
# (``import osdi_loader`` continues to work for back-compat).
__all__ = ["va"]
