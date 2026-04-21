import jax
import jax.numpy as jnp
import numpy as np

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


# 3. DEFINE THE RANK-2 KERNEL (one FFI call, num_devices = leading dim)
def _osdi_eval_impl(model_id, voltages, params, old_state):
    """Rank-2 OSDI FFI call. Shapes: ``v`` is ``(N_dev, num_nodes)``, etc."""
    v = jnp.asarray(voltages, dtype=jnp.float64)
    p = jnp.asarray(params, dtype=jnp.float64)
    s = jnp.asarray(old_state, dtype=jnp.float64)
    m_id = jnp.asarray([model_id], dtype=jnp.uint32)

    num_devices, num_nodes = v.shape
    num_states = s.shape[1]
    jac_size = num_nodes * num_nodes

    out_shapes = (
        jax.ShapeDtypeStruct((num_devices, num_nodes), jnp.float64),  # currents
        jax.ShapeDtypeStruct((num_devices, jac_size), jnp.float64),  # conductances
        jax.ShapeDtypeStruct((num_devices, num_nodes), jnp.float64),  # charges
        jax.ShapeDtypeStruct((num_devices, jac_size), jnp.float64),  # capacitances
        jax.ShapeDtypeStruct((num_devices, num_states), jnp.float64),  # new_state
    )

    # vmap_method="sequential" is the fallback for internal vmap usage inside
    # transforms (jacfwd, etc.); the custom_vmap rule below intercepts user-
    # facing jax.vmap calls so they collapse into a single FFI crossing.
    return jffi.ffi_call("OsdiEvalCpu", out_shapes, vmap_method="sequential")(
        m_id, v, p, s
    )


# 4. ATTACH A VMAP RULE THAT FLATTENS (B, N_dev, …) → (B * N_dev, …)
#
# Under jax.vmap, the default (ffi_call vmap_method="sequential") would call
# the FFI once per replica. The rule below flattens the replica axis into the
# leading device axis, so the existing rank-2 handler sees B * N_dev devices
# in one crossing and Rayon parallelises across all of them.
_osdi_eval_vmap = jax.custom_batching.custom_vmap(_osdi_eval_impl)


@_osdi_eval_vmap.def_vmap
def _osdi_eval_vmap_rule(axis_size, in_batched, model_id, voltages, params, old_state):
    m_batched, v_batched, p_batched, s_batched = in_batched

    if m_batched:
        raise ValueError("osdi_eval: cannot vmap over model_id.")

    def _lift(x, batched):
        if batched:
            return x
        return jnp.broadcast_to(x[None], (axis_size,) + x.shape)

    v = _lift(voltages, v_batched)
    p = _lift(params, p_batched)
    s = _lift(old_state, s_batched)

    B, N_dev, N_pins = v.shape
    N_params = p.shape[-1]
    N_states = s.shape[-1]
    jac_size = N_pins * N_pins

    v_flat = v.reshape(B * N_dev, N_pins)
    p_flat = p.reshape(B * N_dev, N_params)
    s_flat = s.reshape(B * N_dev, N_states)

    cur, cond, chg, cap, new_s = _osdi_eval_impl(model_id, v_flat, p_flat, s_flat)

    cur = cur.reshape(B, N_dev, N_pins)
    cond = cond.reshape(B, N_dev, jac_size)
    chg = chg.reshape(B, N_dev, N_pins)
    cap = cap.reshape(B, N_dev, jac_size)
    new_s = new_s.reshape(B, N_dev, N_states)

    return (cur, cond, chg, cap, new_s), (True, True, True, True, True)


# 5. ATTACH THE ANALYTICAL JVP ON TOP (OSDI Jacobians → JAX autodiff)
#
# Order matters: custom_jvp is the OUTER wrapper so ``jax.grad`` finds the JVP
# rule before descending; custom_vmap is the inner wrapper so ``jax.vmap``
# finds its rule when it traces through custom_jvp.
osdi_eval = jax.custom_jvp(_osdi_eval_vmap, nondiff_argnums=(0,))


@osdi_eval.defjvp
def _osdi_eval_jvp_rule(model_id, primals, tangents):
    v, p, s = primals
    t_v, t_p, t_s = tangents

    currents, conductances, charges, capacitances, new_state = _osdi_eval_vmap(
        model_id, v, p, s
    )

    num_devices, num_nodes = v.shape
    g_matrix = conductances.reshape((num_devices, num_nodes, num_nodes))
    c_matrix = capacitances.reshape((num_devices, num_nodes, num_nodes))

    t_currents = jnp.einsum("nij,nj->ni", g_matrix, t_v)
    t_charges = jnp.einsum("nij,nj->ni", c_matrix, t_v)

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


osdi_eval.__doc__ = """
Evaluate an OSDI model via the C++ XLA FFI.

Shape contract: ``voltages`` must be ``(N_dev, model.num_nodes)`` — one entry
per *unknown*, including internal Kirchhoff nodes and branch-current
auxiliaries, in OSDI node-index order (terminals first, then internals).
``OsdiModel.allocate_jax_buffers()`` is the sanctioned way to get a
correctly-sized buffer.

Outputs ``cur`` and ``chg`` are ``(N_dev, num_nodes)``; ``cond`` and ``cap``
are flattened ``(N_dev, num_nodes**2)`` row-major Jacobians.

Under ``jax.vmap``, the replica axis is fused into the device axis so each
evaluation produces exactly one FFI crossing regardless of batch size.
"""


# =============================================================================
# RESIDUAL-ONLY ENTRY POINT
# =============================================================================
#
# For Newton inner iterations that reuse a frozen Jacobian from the first iter
# of the timestep, the ∂/∂V stamps (cond, cap) aren't needed — only the residual
# (currents, charges) and the next state. Skipping the CALC_*_JACOBIAN flags and
# the write_jacobian_* calls roughly halves per-device OSDI work for strongly-
# nonlinear transients on BSIM4/PSP103-sized models.
#
# No custom_jvp is attached: this is for inner-loop use where differentiability
# through OSDI isn't needed (the Jacobian is cached from a prior full eval).

jffi.register_ffi_target(
    "OsdiResidualEvalCpu",
    osdi_shim_nb.batched_osdi_residual_eval(),
    platform="cpu",
)


def _osdi_residual_eval_impl(model_id, voltages, params, old_state):
    """Rank-2 residual-only FFI call. Returns (cur, chg, new_state)."""
    v = jnp.asarray(voltages, dtype=jnp.float64)
    p = jnp.asarray(params, dtype=jnp.float64)
    s = jnp.asarray(old_state, dtype=jnp.float64)
    m_id = jnp.asarray([model_id], dtype=jnp.uint32)

    num_devices, num_nodes = v.shape
    num_states = s.shape[1]

    out_shapes = (
        jax.ShapeDtypeStruct((num_devices, num_nodes), jnp.float64),  # currents
        jax.ShapeDtypeStruct((num_devices, num_nodes), jnp.float64),  # charges
        jax.ShapeDtypeStruct((num_devices, num_states), jnp.float64),  # new_state
    )

    return jffi.ffi_call("OsdiResidualEvalCpu", out_shapes, vmap_method="sequential")(
        m_id, v, p, s
    )


osdi_residual_eval = jax.custom_batching.custom_vmap(_osdi_residual_eval_impl)


@osdi_residual_eval.def_vmap
def _osdi_residual_eval_vmap_rule(
    axis_size, in_batched, model_id, voltages, params, old_state
):
    m_batched, v_batched, p_batched, s_batched = in_batched

    if m_batched:
        raise ValueError("osdi_residual_eval: cannot vmap over model_id.")

    def _lift(x, batched):
        if batched:
            return x
        return jnp.broadcast_to(x[None], (axis_size,) + x.shape)

    v = _lift(voltages, v_batched)
    p = _lift(params, p_batched)
    s = _lift(old_state, s_batched)

    B, N_dev, N_pins = v.shape
    N_params = p.shape[-1]
    N_states = s.shape[-1]

    v_flat = v.reshape(B * N_dev, N_pins)
    p_flat = p.reshape(B * N_dev, N_params)
    s_flat = s.reshape(B * N_dev, N_states)

    cur, chg, new_s = _osdi_residual_eval_impl(model_id, v_flat, p_flat, s_flat)

    cur = cur.reshape(B, N_dev, N_pins)
    chg = chg.reshape(B, N_dev, N_pins)
    new_s = new_s.reshape(B, N_dev, N_states)

    return (cur, chg, new_s), (True, True, True)


osdi_residual_eval.__doc__ = """
Residual-only OSDI evaluator: returns ``(currents, charges, new_state)`` and
skips the conductance/capacitance Jacobian pass.

Use inside Newton inner iterations where the Jacobian stamp from the first
iter of the timestep is being reused. Cuts per-device OSDI work roughly in
half for Jacobian-heavy models (BSIM4, PSP103, PSP, HiSIM, …).

Shape contract and vmap semantics match ``osdi_eval``: ``voltages`` is
``(N_dev, model.num_nodes)`` and ``jax.vmap`` collapses the replica axis into
the device axis (one FFI crossing per call). No ``custom_jvp`` is attached —
this path is not intended for autodiff.
"""


# =============================================================================
# BATCH HANDLE API — pay setup_model + setup_instance once per param change
# =============================================================================
#
# ``osdi_eval`` and ``osdi_residual_eval`` above re-run setup_model +
# setup_instance on every call. For Newton inner loops where params are fixed
# across 4–5 iters (strongly-nonlinear transients), that's wasted setup — the
# param-writing loop walks every OSDI param via access() + given_flag_*(), and
# setup_instance recomputes derived quantities that haven't changed.
#
# The handle API splits the workflow:
#   handle = osdi_setup_batch(model_id, params)        # once per param change
#   for iter in newton_loop:
#       cur, cond, chg, cap, ns = osdi_eval_with_handle(handle, voltages, state)
#       # …or the residual variant for inner iters with a frozen Jacobian…
#
# The handle is a regular Python object that calls ``osdi_free_handle`` in its
# ``__del__``. It is NOT a JAX array — pass it as a closed-over Python arg to
# functions you're jit'ing or vmap'ing over voltages.

jffi.register_ffi_target(
    "OsdiEvalHandleCpu",
    osdi_shim_nb.batched_osdi_eval_handle(),
    platform="cpu",
)
jffi.register_ffi_target(
    "OsdiResidualEvalHandleCpu",
    osdi_shim_nb.batched_osdi_residual_eval_handle(),
    platform="cpu",
)


class OsdiBatchHandle:
    """Opaque handle holding pre-setup (model_data, inst_data) snapshots for N
    devices. Returned by :func:`osdi_setup_batch`. Auto-freed on garbage
    collection; call :meth:`free` to release earlier.
    """

    __slots__ = ("handle_id", "model_id", "num_devices", "num_params", "_alive")

    def __init__(self, handle_id, model_id, num_devices, num_params):
        self.handle_id = int(handle_id)
        self.model_id = int(model_id)
        self.num_devices = int(num_devices)
        self.num_params = int(num_params)
        self._alive = True

    def free(self):
        if self._alive:
            osdi_shim_nb.osdi_free_handle(self.handle_id)
            self._alive = False

    def __del__(self):
        # __del__ can run during interpreter shutdown when osdi_shim_nb is
        # already torn down; guard so we don't raise from the GC.
        try:
            self.free()
        except Exception:
            pass

    def __repr__(self):
        return (
            f"OsdiBatchHandle(id={self.handle_id}, model_id={self.model_id}, "
            f"num_devices={self.num_devices}, alive={self._alive})"
        )


def osdi_setup_batch(model_id, params):
    """Run setup_model + setup_instance once for ``N`` devices and cache the
    post-setup state. Returns an :class:`OsdiBatchHandle`.

    ``params`` may be a NumPy array or a JAX array of shape
    ``(num_devices, num_params)`` and dtype ``float64``. NaN entries select
    the Verilog-A default for that parameter.

    The handle's lifetime is independent of the JAX tracing/jitting machinery
    — keep it alive (don't let it go out of scope) for the duration of any
    Newton loop that will call :func:`osdi_eval_with_handle` against it.
    """
    p = np.ascontiguousarray(np.asarray(params, dtype=np.float64))
    if p.ndim != 2:
        raise ValueError(
            f"osdi_setup_batch: params must be (N_dev, num_params); got shape {p.shape}"
        )
    num_devices, num_params = p.shape
    handle_id = osdi_shim_nb.osdi_setup_batch(
        int(model_id), int(num_devices), int(num_params), int(p.ctypes.data)
    )
    if handle_id == 0:
        raise RuntimeError(f"osdi_setup_batch failed (unknown model_id={model_id}?)")
    return OsdiBatchHandle(handle_id, model_id, num_devices, num_params)


def _osdi_eval_handle_impl(handle_id, voltages, old_state):
    """Rank-2 handle-based full eval. Returns (cur, cond, chg, cap, new_state)."""
    v = jnp.asarray(voltages, dtype=jnp.float64)
    s = jnp.asarray(old_state, dtype=jnp.float64)
    h_id = jnp.asarray([handle_id], dtype=jnp.uint64)

    num_devices, num_nodes = v.shape
    num_states = s.shape[1]
    jac_size = num_nodes * num_nodes

    out_shapes = (
        jax.ShapeDtypeStruct((num_devices, num_nodes), jnp.float64),  # currents
        jax.ShapeDtypeStruct((num_devices, jac_size), jnp.float64),  # conductances
        jax.ShapeDtypeStruct((num_devices, num_nodes), jnp.float64),  # charges
        jax.ShapeDtypeStruct((num_devices, jac_size), jnp.float64),  # capacitances
        jax.ShapeDtypeStruct((num_devices, num_states), jnp.float64),  # new_state
    )

    return jffi.ffi_call("OsdiEvalHandleCpu", out_shapes, vmap_method="sequential")(
        h_id, v, s
    )


_osdi_eval_handle_vmap = jax.custom_batching.custom_vmap(_osdi_eval_handle_impl)


@_osdi_eval_handle_vmap.def_vmap
def _osdi_eval_handle_vmap_rule(axis_size, in_batched, handle_id, voltages, old_state):
    h_batched, v_batched, s_batched = in_batched
    if h_batched:
        raise ValueError("osdi_eval_with_handle: cannot vmap over handle_id.")

    def _lift(x, batched):
        if batched:
            return x
        return jnp.broadcast_to(x[None], (axis_size,) + x.shape)

    v = _lift(voltages, v_batched)
    s = _lift(old_state, s_batched)

    B, N_dev, N_pins = v.shape
    N_states = s.shape[-1]
    jac_size = N_pins * N_pins

    v_flat = v.reshape(B * N_dev, N_pins)
    s_flat = s.reshape(B * N_dev, N_states)

    cur, cond, chg, cap, new_s = _osdi_eval_handle_impl(handle_id, v_flat, s_flat)

    return (
        cur.reshape(B, N_dev, N_pins),
        cond.reshape(B, N_dev, jac_size),
        chg.reshape(B, N_dev, N_pins),
        cap.reshape(B, N_dev, jac_size),
        new_s.reshape(B, N_dev, N_states),
    ), (True, True, True, True, True)


def osdi_eval_with_handle(handle, voltages, old_state):
    """Full-stamp OSDI eval reusing a pre-setup handle. No setup_instance
    per call; roughly 1.5–3× faster than :func:`osdi_eval` inside a Newton
    loop on BSIM4/PSP103-sized models.
    """
    if not handle._alive:
        raise RuntimeError("osdi_eval_with_handle: handle has been freed.")
    return _osdi_eval_handle_vmap(handle.handle_id, voltages, old_state)


def _osdi_residual_eval_handle_impl(handle_id, voltages, old_state):
    """Rank-2 handle-based residual-only eval. Returns (cur, chg, new_state)."""
    v = jnp.asarray(voltages, dtype=jnp.float64)
    s = jnp.asarray(old_state, dtype=jnp.float64)
    h_id = jnp.asarray([handle_id], dtype=jnp.uint64)

    num_devices, num_nodes = v.shape
    num_states = s.shape[1]

    out_shapes = (
        jax.ShapeDtypeStruct((num_devices, num_nodes), jnp.float64),  # currents
        jax.ShapeDtypeStruct((num_devices, num_nodes), jnp.float64),  # charges
        jax.ShapeDtypeStruct((num_devices, num_states), jnp.float64),  # new_state
    )

    return jffi.ffi_call(
        "OsdiResidualEvalHandleCpu", out_shapes, vmap_method="sequential"
    )(h_id, v, s)


_osdi_residual_eval_handle_vmap = jax.custom_batching.custom_vmap(
    _osdi_residual_eval_handle_impl
)


@_osdi_residual_eval_handle_vmap.def_vmap
def _osdi_residual_eval_handle_vmap_rule(
    axis_size, in_batched, handle_id, voltages, old_state
):
    h_batched, v_batched, s_batched = in_batched
    if h_batched:
        raise ValueError("osdi_residual_eval_with_handle: cannot vmap over handle_id.")

    def _lift(x, batched):
        if batched:
            return x
        return jnp.broadcast_to(x[None], (axis_size,) + x.shape)

    v = _lift(voltages, v_batched)
    s = _lift(old_state, s_batched)

    B, N_dev, N_pins = v.shape
    N_states = s.shape[-1]

    v_flat = v.reshape(B * N_dev, N_pins)
    s_flat = s.reshape(B * N_dev, N_states)

    cur, chg, new_s = _osdi_residual_eval_handle_impl(handle_id, v_flat, s_flat)

    return (
        cur.reshape(B, N_dev, N_pins),
        chg.reshape(B, N_dev, N_pins),
        new_s.reshape(B, N_dev, N_states),
    ), (True, True, True)


def osdi_residual_eval_with_handle(handle, voltages, old_state):
    """Residual-only OSDI eval reusing a pre-setup handle. Skips both
    setup_instance AND the Jacobian pass — the leanest path available and
    the intended entry point for Newton inner iters with a frozen stamp.
    """
    if not handle._alive:
        raise RuntimeError("osdi_residual_eval_with_handle: handle has been freed.")
    return _osdi_residual_eval_handle_vmap(handle.handle_id, voltages, old_state)
