import functools
import jax
import jax.numpy as jnp

# Depending on your exact JAX version, the FFI module might be under 'extend'
try:
    import jax.ffi as jffi
except ImportError:
    import jax.extend.ffi as jffi

import osdi_shim_nb

# 1. CRITICAL: Enable 64-bit precision for SPICE-level accuracy
jax.config.update("jax_enable_x64", True)

# 2. REGISTER THE C++ XLA FFI TARGET
# We use the new FFI registration API, matching the C++ macro name exactly
jffi.register_ffi_target(
    "OsdiEvalCpu", osdi_shim_nb.batched_osdi_eval(), platform="cpu"
)


# 3. DEFINE THE EVALUATOR & JVP (No core.Primitive needed!)
# We mark model_id as non-differentiable (nondiff) since it's an integer ID.
@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def osdi_eval(model_id, voltages, params, old_state):
    """
    Evaluates the OSDI model via the C++ XLA FFI.
    """
    # Cast everything to float64, and wrap the model_id in an array for C++
    v = jnp.asarray(voltages, dtype=jnp.float64)
    p = jnp.asarray(params, dtype=jnp.float64)
    s = jnp.asarray(old_state, dtype=jnp.float64)
    m_id = jnp.asarray([model_id], dtype=jnp.uint32)

    num_devices, num_pins = v.shape
    num_states = s.shape[1]
    jac_size = num_pins * num_pins

    # Tell JAX what shapes and types the C++ handler will return
    out_shapes = (
        jax.ShapeDtypeStruct((num_devices, num_pins), jnp.float64),  # currents
        jax.ShapeDtypeStruct((num_devices, jac_size), jnp.float64),  # conductances
        jax.ShapeDtypeStruct((num_devices, num_pins), jnp.float64),  # charges
        jax.ShapeDtypeStruct((num_devices, jac_size), jnp.float64),  # capacitances
        jax.ShapeDtypeStruct((num_devices, num_states), jnp.float64),  # new_state
    )

    # ffi_call(name, out_shapes) returns a callable; pass inputs to that callable.
    return jffi.ffi_call("OsdiEvalCpu", out_shapes)(m_id, v, p, s)


# 4. DIFFERENTIATION (Analytical chain rule)
@osdi_eval.defjvp
def osdi_eval_jvp(model_id, primals, tangents):
    v, p, s = primals
    t_v, t_p, t_s = tangents

    # Primal evaluation
    currents, conductances, charges, capacitances, new_state = osdi_eval(
        model_id, v, p, s
    )

    # Reshape the flattened Jacobians for matrix multiplication
    num_devices, num_pins = v.shape
    g_matrix = conductances.reshape((num_devices, num_pins, num_pins))
    c_matrix = capacitances.reshape((num_devices, num_pins, num_pins))

    # Calculate analytical directional derivatives using OpenVAF's Jacobian matrices
    t_currents = jnp.einsum("nij,nj->ni", g_matrix, t_v)
    t_charges = jnp.einsum("nij,nj->ni", c_matrix, t_v)

    # For now, we assume dI/dP and dState/dV are zero or handled elsewhere
    t_conductances = jnp.zeros_like(conductances)
    t_capacitances = jnp.zeros_like(capacitances)
    t_new_state = jnp.zeros_like(new_state)

    return (currents, conductances, charges, capacitances, new_state), (
        t_currents,
        t_conductances,
        t_charges,
        t_capacitances,
        t_new_state,
    )
