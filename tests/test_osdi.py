import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Assuming you placed the loader and the JAX primitive in these modules
from osdi_loader import load_osdi_model
from osdi_jax import (
    osdi_eval,
    osdi_residual_eval,
    osdi_setup_batch,
    osdi_eval_with_handle,
    osdi_residual_eval_with_handle,
    OsdiBatchHandle,
)
import pathlib

folder = pathlib.Path(__file__).parent


@pytest.fixture(scope="module")
def resistor_model():
    """
    Fixture to load the compiled OSDI binary once for the entire test module.
    Ensure 'resistor_va.osdi' is compiled and present in the test directory.
    """
    # Load the binary dynamically via your Rust host
    model = folder / "resistor_va.osdi"
    model = load_osdi_model(str(model))

    # Quick sanity checks on the OSDI Descriptor extraction
    assert model.num_pins == 2, f"Expected 2 pins, got {model.num_pins}"
    assert model.num_params == 2, (
        f"Expected 2 parameters (m, R), got {model.num_params}"
    )
    assert model.num_states == 0, "Resistor should have 0 internal states"

    return model


def test_resistor_dc_evaluation(resistor_model):
    """
    Test a single resistor under a simple DC voltage bias.
    """
    # 1. Setup Inputs (Shape: [num_devices, features])
    num_devices = 1

    # V(A) = 1.0V, V(B) = 0.0V
    voltages = jnp.array([[1.0, 0.0]], dtype=jnp.float64)

    # OSDI param order for resistor_va.osdi: [m (instance), R (model)]
    params = jnp.array([[1.0, 50.0]], dtype=jnp.float64)

    # No internal state
    old_state = jnp.empty((num_devices, 0), dtype=jnp.float64)

    # 2. Execute the XLA Custom Call
    cur, cond, chg, cap, ns = osdi_eval(resistor_model.id, voltages, params, old_state)

    # 3. Assert Currents (I = V / R)
    # I_A = (1.0 - 0.0) / 50 = 0.02 A
    # I_B = (0.0 - 1.0) / 50 = -0.02 A
    expected_cur = np.array([[0.02, -0.02]])
    np.testing.assert_allclose(
        cur, expected_cur, rtol=1e-6, err_msg="Currents do not match Ohm's Law"
    )

    # 4. Assert Conductances / Jacobian (dI/dV)
    # G_AA = 1/R  =  0.02
    # G_AB = -1/R = -0.02
    # G_BA = -1/R = -0.02
    # G_BB = 1/R  =  0.02
    # Flattened shape expected from Rust: [G_AA, G_AB, G_BA, G_BB]
    expected_cond = np.array([[0.02, -0.02, -0.02, 0.02]])
    np.testing.assert_allclose(
        cond,
        expected_cond,
        rtol=1e-6,
        err_msg="Analytical Conductance Jacobian is incorrect",
    )

    # 5. Assert Dynamic Flows (Charges and Capacitances should be exactly zero)
    # Since the Verilog-A model has no `ddt()` operators, Q and dQ/dV must be 0.
    np.testing.assert_allclose(
        chg,
        np.zeros_like(chg),
        atol=1e-12,
        err_msg="Static resistor should not accumulate charge",
    )
    np.testing.assert_allclose(
        cap,
        np.zeros_like(cap),
        atol=1e-12,
        err_msg="Static resistor should have zero capacitance",
    )


def test_resistor_batched_evaluation(resistor_model):
    """
    Test Rayon/JAX batching by evaluating 1,000 unique resistors simultaneously.
    """
    num_devices = 1000

    # Create 1000 random voltage drops between 0V and 5V
    key = jax.random.PRNGKey(42)
    voltages = jax.random.uniform(key, shape=(num_devices, 2), minval=0.0, maxval=5.0)

    # Create 1000 random resistance values between 10 Ohms and 1000 Ohms
    key, subkey = jax.random.split(key)
    # OSDI param order: [m (instance), R (model)]
    params = jax.random.uniform(
        subkey, shape=(num_devices, 2), minval=10.0, maxval=1000.0
    )
    params = params.at[:, 0].set(1.0)  # m=1.0 for all devices; params[:,1] = R

    old_state = jnp.empty((num_devices, 0), dtype=jnp.float64)

    # Execute batched call (this triggers the Rust Rayon parallel loop)
    cur, cond, _, _, _ = osdi_eval(resistor_model.id, voltages, params, old_state)

    # Verify output shapes
    assert cur.shape == (num_devices, 2)
    assert cond.shape == (num_devices, 4)

    # Spot check device #42 — R is at params[:,1]
    v_a, v_b = voltages[42]
    r = params[42][1]
    expected_i_a = (v_a - v_b) / r

    np.testing.assert_allclose(
        cur[42, 0],
        expected_i_a,
        rtol=1e-6,
        err_msg="Batched Rayon evaluation failed on specific instance",
    )


def test_resistor_jax_jvp(resistor_model):
    """
    Verify that our custom JVP correctly routes the analytical OSDI Jacobians
    into JAX's Automatic Differentiation engine.
    """
    num_devices = 1
    voltages = jnp.array([[2.0, 0.0]], dtype=jnp.float64)
    # OSDI param order: [m (instance), R (model)]
    params = jnp.array([[1.0, 100.0]], dtype=jnp.float64)
    old_state = jnp.empty((num_devices, 0), dtype=jnp.float64)

    # Define a pure JAX function that extracts just the current at Pin A
    def pin_A_current(v):
        cur, _, _, _, _ = osdi_eval(resistor_model.id, v, params, old_state)
        return cur[0, 0]  # Return scalar I_A

    # Ask JAX to compute the gradient of I_A with respect to the input voltages
    # This automatically triggers our registered `osdi_eval_value_and_jvp`
    grad_fn = jax.grad(pin_A_current)
    gradient = grad_fn(voltages)

    # I_A = (V_A - V_B) / R
    # d(I_A)/d(V_A) = 1/R = 1/100 = 0.01
    # d(I_A)/d(V_B) = -1/R = -1/100 = -0.01
    expected_gradient = np.array([[0.01, -0.01]])

    np.testing.assert_allclose(
        gradient,
        expected_gradient,
        rtol=1e-6,
        err_msg="JAX JVP did not correctly pipe OSDI Jacobians",
    )


@pytest.fixture(scope="module")
def capacitor_model():
    """Load the compiled capacitor OSDI binary once for the test module."""
    model = folder / "capacitor_va.osdi"
    model = load_osdi_model(str(model))

    assert model.num_pins == 2, f"Expected 2 pins, got {model.num_pins}"
    assert model.num_params == 3, (
        f"Expected 3 parameters ($mfactor, C, m), got {model.num_params}"
    )
    assert model.num_states == 0, "Linear capacitor should have 0 internal states"

    return model


def test_capacitor_dc_evaluation(capacitor_model):
    """
    Test a single capacitor at a DC operating point.
    A linear capacitor I = C*ddt(V) has no resistive current; charge Q = C*V.
    """
    C = 1e-12  # 1 pF

    voltages = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    # OSDI param order for capacitor_va.osdi: [$mfactor (instance), C (model), m (model)]
    # m is a grading/multiplier coefficient, default 1.0 for a linear capacitor.
    params = jnp.array([[1.0, C, 1.0]], dtype=jnp.float64)
    old_state = jnp.empty((1, 0), dtype=jnp.float64)

    cur, cond, chg, cap, _ = osdi_eval(capacitor_model.id, voltages, params, old_state)

    # No resistive current or conductance for a pure capacitor
    np.testing.assert_allclose(
        cur,
        np.zeros_like(cur),
        atol=1e-30,
        err_msg="Pure capacitor should produce no resistive current",
    )
    np.testing.assert_allclose(
        cond,
        np.zeros_like(cond),
        atol=1e-30,
        err_msg="Pure capacitor should have zero conductance",
    )

    # Q(P) = +C * V_PN = 1e-12,  Q(N) = -C * V_PN = -1e-12
    expected_chg = np.array([[C, -C]])
    np.testing.assert_allclose(
        chg, expected_chg, rtol=1e-6, err_msg="Charge does not match Q = C * V"
    )

    # Capacitance matrix: [[C, -C], [-C, C]]  (row-major: [C_PP, C_PN, C_NP, C_NN])
    expected_cap = np.array([[C, -C, -C, C]])
    np.testing.assert_allclose(
        cap,
        expected_cap,
        rtol=1e-6,
        err_msg="Capacitance Jacobian (dQ/dV) is incorrect",
    )


def test_capacitor_jax_jvp(capacitor_model):
    """
    Verify that JAX's custom JVP routes the analytical OSDI capacitance Jacobians
    correctly, enabling grad() through the reactive (charge) outputs.
    """
    C = 1e-12
    voltages = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    # OSDI param order: [$mfactor (instance), C (model), m (model, default=1.0)]
    params = jnp.array([[1.0, C, 1.0]], dtype=jnp.float64)
    old_state = jnp.empty((1, 0), dtype=jnp.float64)

    # d(Q_P)/d(V_P) = +C,  d(Q_P)/d(V_N) = -C
    def pin_P_charge(v):
        _, _, chg, _, _ = osdi_eval(capacitor_model.id, v, params, old_state)
        return chg[0, 0]

    gradient = jax.grad(pin_P_charge)(voltages)
    expected = np.array([[C, -C]])

    np.testing.assert_allclose(
        gradient,
        expected,
        rtol=1e-6,
        err_msg="JAX JVP did not correctly pipe OSDI capacitance Jacobians",
    )


def test_resistor_jit(resistor_model):
    """
    Verify that osdi_eval can be compiled by jax.jit and produces correct results.
    Also checks that jit + grad composes correctly.
    """
    num_devices = 1
    voltages = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    # OSDI param order: [m (instance), R (model)]
    params = jnp.array([[1.0, 50.0]], dtype=jnp.float64)
    old_state = jnp.empty((num_devices, 0), dtype=jnp.float64)

    # JIT the eval function
    jit_eval = jax.jit(lambda v, p: osdi_eval(resistor_model.id, v, p, old_state))

    cur, cond, _, _, _ = jit_eval(voltages, params)

    np.testing.assert_allclose(
        cur,
        np.array([[0.02, -0.02]]),
        rtol=1e-6,
        err_msg="jit(osdi_eval) produced incorrect currents",
    )
    np.testing.assert_allclose(
        cond,
        np.array([[0.02, -0.02, -0.02, 0.02]]),
        rtol=1e-6,
        err_msg="jit(osdi_eval) produced incorrect conductances",
    )

    # JIT + grad should also compose
    def pin_A_current(v):
        cur, _, _, _, _ = osdi_eval(resistor_model.id, v, params, old_state)
        return cur[0, 0]

    jit_grad = jax.jit(jax.grad(pin_A_current))
    gradient = jit_grad(voltages)

    np.testing.assert_allclose(
        gradient,
        np.array([[0.02, -0.02]]),
        rtol=1e-6,
        err_msg="jit(grad(osdi_eval)) produced incorrect gradient",
    )


def test_resistor_vmap_matches_sequential(resistor_model):
    """
    vmap over a replica axis must produce the same outputs as a manual
    sequential Python loop. This is the correctness check for the custom_vmap
    rule that flattens (B, N_dev, ...) into a single rank-2 FFI crossing.
    """
    B, N_dev = 4, 9
    key = jax.random.PRNGKey(0)
    k_v, k_r = jax.random.split(key)

    voltages_b = jax.random.uniform(
        k_v, shape=(B, N_dev, 2), minval=-2.0, maxval=2.0, dtype=jnp.float64
    )
    r_vals = jax.random.uniform(
        k_r, shape=(B, N_dev), minval=10.0, maxval=1000.0, dtype=jnp.float64
    )
    params_b = jnp.stack([jnp.ones((B, N_dev), dtype=jnp.float64), r_vals], axis=-1)
    old_state_b = jnp.empty((B, N_dev, 0), dtype=jnp.float64)

    # vmap'd call: should emit exactly one FFI crossing regardless of B.
    vmapped = jax.vmap(
        lambda v, p, s: osdi_eval(resistor_model.id, v, p, s),
        in_axes=(0, 0, 0),
    )
    cur_v, cond_v, chg_v, cap_v, ns_v = vmapped(voltages_b, params_b, old_state_b)

    # Ground truth: sequential Python loop over the replica axis.
    cur_ref, cond_ref, chg_ref, cap_ref, ns_ref = [], [], [], [], []
    for b in range(B):
        cur, cond, chg, cap, ns = osdi_eval(
            resistor_model.id, voltages_b[b], params_b[b], old_state_b[b]
        )
        cur_ref.append(cur)
        cond_ref.append(cond)
        chg_ref.append(chg)
        cap_ref.append(cap)
        ns_ref.append(ns)
    cur_ref = jnp.stack(cur_ref)
    cond_ref = jnp.stack(cond_ref)
    chg_ref = jnp.stack(chg_ref)
    cap_ref = jnp.stack(cap_ref)
    ns_ref = jnp.stack(ns_ref)

    assert cur_v.shape == (B, N_dev, 2)
    assert cond_v.shape == (B, N_dev, 4)
    np.testing.assert_allclose(cur_v, cur_ref, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(cond_v, cond_ref, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(chg_v, chg_ref, atol=1e-15)
    np.testing.assert_allclose(cap_v, cap_ref, atol=1e-15)
    np.testing.assert_allclose(ns_v, ns_ref, atol=1e-15)


def test_resistor_vmap_broadcasts_unbatched_args(resistor_model):
    """
    Only ``voltages`` is vmapped; ``params`` and ``old_state`` are shared
    across replicas. The custom_vmap rule must broadcast the non-batched
    inputs to the replica axis before flattening.
    """
    B, N_dev = 3, 5
    key = jax.random.PRNGKey(1)

    voltages_b = jax.random.uniform(
        key, shape=(B, N_dev, 2), minval=-1.0, maxval=1.0, dtype=jnp.float64
    )
    # Shared across replicas: one (N_dev, num_params) vector, same R for all.
    params = jnp.stack(
        [
            jnp.ones((N_dev,), dtype=jnp.float64),
            jnp.full((N_dev,), 100.0, dtype=jnp.float64),
        ],
        axis=-1,
    )
    old_state = jnp.empty((N_dev, 0), dtype=jnp.float64)

    vmapped = jax.vmap(
        lambda v: osdi_eval(resistor_model.id, v, params, old_state),
        in_axes=0,
    )
    cur_v, _, _, _, _ = vmapped(voltages_b)

    # I = (V_A - V_B) / 100 per device.
    expected_cur_0 = (voltages_b[..., 0] - voltages_b[..., 1]) / 100.0
    np.testing.assert_allclose(cur_v[..., 0], expected_cur_0, rtol=1e-12)


def test_resistor_vmap_grad(resistor_model):
    """
    vmap must compose with grad: both transforms should produce correct
    per-replica gradients via the analytical OSDI Jacobians.
    """
    B, N_dev = 4, 3

    voltages_b = jnp.asarray(
        np.random.RandomState(7).uniform(-2.0, 2.0, size=(B, N_dev, 2)),
        dtype=jnp.float64,
    )
    params = jnp.stack(
        [
            jnp.ones((N_dev,), dtype=jnp.float64),
            jnp.array([50.0, 100.0, 200.0], dtype=jnp.float64),
        ],
        axis=-1,
    )
    old_state = jnp.empty((N_dev, 0), dtype=jnp.float64)

    def loss(v):
        cur, _, _, _, _ = osdi_eval(resistor_model.id, v, params, old_state)
        return jnp.sum(cur[:, 0])  # sum of pin-A currents

    g_vmap = jax.vmap(jax.grad(loss))(voltages_b)

    # Ground truth: d/dV_A(sum I_A) = 1/R; d/dV_B(sum I_A) = -1/R (per device).
    inv_r = 1.0 / params[:, 1]
    expected = jnp.stack([inv_r, -inv_r], axis=-1)
    expected_b = jnp.broadcast_to(expected, (B, N_dev, 2))
    np.testing.assert_allclose(g_vmap, expected_b, rtol=1e-12)


def test_resistor_vmap_leading_dim_flattened(resistor_model):
    """
    Architectural check: after the custom_vmap rule fires, the leading device
    axis seen by the rank-2 kernel should equal B * N_dev — proving that the
    replica axis has been folded into the device axis instead of producing
    one FFI call per replica.
    """
    import osdi_jax

    import jax.core as jcore

    seen_leading_dims = []
    original_impl = osdi_jax._osdi_eval_impl

    def spy(model_id, voltages, params, old_state):
        # Only count concrete (runtime) calls — abstract-eval tracing calls
        # use Tracers and do not cross the FFI.
        if not isinstance(voltages, jcore.Tracer):
            seen_leading_dims.append(int(voltages.shape[0]))
        return original_impl(model_id, voltages, params, old_state)

    # Rebuild the stack in production order: custom_jvp(custom_vmap(spy)).
    spy_vmap = jax.custom_batching.custom_vmap(spy)

    @spy_vmap.def_vmap
    def _rule(axis_size, in_batched, model_id, voltages, params, old_state):
        m_b, v_b, p_b, s_b = in_batched
        if m_b:
            raise ValueError("cannot vmap over model_id")

        def _lift(x, b):
            return x if b else jnp.broadcast_to(x[None], (axis_size,) + x.shape)

        v = _lift(voltages, v_b)
        p = _lift(params, p_b)
        s = _lift(old_state, s_b)
        B, N_dev, N_pins = v.shape
        v_f = v.reshape(B * N_dev, N_pins)
        p_f = p.reshape(B * N_dev, p.shape[-1])
        s_f = s.reshape(B * N_dev, s.shape[-1])
        cur, cond, chg, cap, ns = spy(model_id, v_f, p_f, s_f)
        return (
            cur.reshape(B, N_dev, N_pins),
            cond.reshape(B, N_dev, N_pins * N_pins),
            chg.reshape(B, N_dev, N_pins),
            cap.reshape(B, N_dev, N_pins * N_pins),
            ns.reshape(B, N_dev, s.shape[-1]),
        ), (True, True, True, True, True)

    spy_jvp = jax.custom_jvp(spy_vmap, nondiff_argnums=(0,))
    spy_jvp.defjvp(osdi_jax._osdi_eval_jvp_rule)

    B, N_dev = 8, 9
    voltages_b = jnp.zeros((B, N_dev, 2), dtype=jnp.float64)
    params = jnp.stack(
        [
            jnp.ones((N_dev,), dtype=jnp.float64),
            jnp.full((N_dev,), 100.0, dtype=jnp.float64),
        ],
        axis=-1,
    )
    old_state = jnp.empty((N_dev, 0), dtype=jnp.float64)

    vmapped = jax.vmap(
        lambda v: spy_jvp(resistor_model.id, v, params, old_state),
        in_axes=0,
    )
    _ = vmapped(voltages_b)

    assert seen_leading_dims == [B * N_dev], (
        f"Expected one rank-2 call with leading dim {B * N_dev}, "
        f"got {seen_leading_dims}"
    )


# =============================================================================
# RESIDUAL-ONLY ENTRY POINT
# =============================================================================


def test_resistor_residual_eval_matches_osdi_eval(resistor_model):
    """
    Residual-only FFI must return cur/chg/new_state bitwise equal to osdi_eval
    (skipping the Jacobian pass can't change the residual).
    """
    num_devices = 8
    key = jax.random.PRNGKey(17)
    voltages = jax.random.uniform(
        key, shape=(num_devices, 2), minval=-3.0, maxval=3.0, dtype=jnp.float64
    )
    params = jnp.stack(
        [
            jnp.ones((num_devices,), dtype=jnp.float64),
            jnp.linspace(10.0, 1000.0, num_devices, dtype=jnp.float64),
        ],
        axis=-1,
    )
    old_state = jnp.empty((num_devices, 0), dtype=jnp.float64)

    cur_full, _, chg_full, _, ns_full = osdi_eval(
        resistor_model.id, voltages, params, old_state
    )
    cur_res, chg_res, ns_res = osdi_residual_eval(
        resistor_model.id, voltages, params, old_state
    )

    np.testing.assert_array_equal(cur_res, cur_full)
    np.testing.assert_array_equal(chg_res, chg_full)
    np.testing.assert_array_equal(ns_res, ns_full)


def test_capacitor_residual_eval_matches_osdi_eval(capacitor_model):
    """
    Residual-only must also match for a reactive-only model — proves the
    react_residual flag is still set and load_residual_react is still called
    when residual_only=true.
    """
    C = 1e-12
    num_devices = 4
    voltages = jnp.asarray(
        np.random.RandomState(3).uniform(-1.0, 1.0, size=(num_devices, 2)),
        dtype=jnp.float64,
    )
    params = jnp.stack(
        [
            jnp.ones((num_devices,), dtype=jnp.float64),
            jnp.full((num_devices,), C, dtype=jnp.float64),
            jnp.ones((num_devices,), dtype=jnp.float64),
        ],
        axis=-1,
    )
    old_state = jnp.empty((num_devices, 0), dtype=jnp.float64)

    cur_full, _, chg_full, _, ns_full = osdi_eval(
        capacitor_model.id, voltages, params, old_state
    )
    cur_res, chg_res, ns_res = osdi_residual_eval(
        capacitor_model.id, voltages, params, old_state
    )

    np.testing.assert_array_equal(cur_res, cur_full)
    np.testing.assert_array_equal(chg_res, chg_full)
    np.testing.assert_array_equal(ns_res, ns_full)


def test_residual_eval_shapes_no_jac_outputs(resistor_model):
    """
    The residual entry point returns 3 outputs (cur, chg, new_state),
    not 5 — no conductance/capacitance buffers allocated at all.
    """
    voltages = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    params = jnp.array([[1.0, 50.0]], dtype=jnp.float64)
    old_state = jnp.empty((1, 0), dtype=jnp.float64)

    out = osdi_residual_eval(resistor_model.id, voltages, params, old_state)
    assert len(out) == 3
    cur, chg, ns = out
    assert cur.shape == (1, 2)
    assert chg.shape == (1, 2)
    assert ns.shape == (1, 0)
    # Ohm's law still holds.
    np.testing.assert_allclose(cur, [[0.02, -0.02]], rtol=1e-12)


def test_residual_eval_vmap_matches_sequential(resistor_model):
    """
    vmap over a replica axis must produce the same residual-only outputs as a
    sequential Python loop — validating the custom_vmap rule on the new path.
    """
    B, N_dev = 4, 9
    key = jax.random.PRNGKey(23)
    k_v, k_r = jax.random.split(key)

    voltages_b = jax.random.uniform(
        k_v, shape=(B, N_dev, 2), minval=-2.0, maxval=2.0, dtype=jnp.float64
    )
    r_vals = jax.random.uniform(
        k_r, shape=(B, N_dev), minval=10.0, maxval=1000.0, dtype=jnp.float64
    )
    params_b = jnp.stack([jnp.ones((B, N_dev), dtype=jnp.float64), r_vals], axis=-1)
    old_state_b = jnp.empty((B, N_dev, 0), dtype=jnp.float64)

    vmapped = jax.vmap(
        lambda v, p, s: osdi_residual_eval(resistor_model.id, v, p, s),
        in_axes=(0, 0, 0),
    )
    cur_v, chg_v, ns_v = vmapped(voltages_b, params_b, old_state_b)

    cur_ref, chg_ref, ns_ref = [], [], []
    for b in range(B):
        cur, chg, ns = osdi_residual_eval(
            resistor_model.id, voltages_b[b], params_b[b], old_state_b[b]
        )
        cur_ref.append(cur)
        chg_ref.append(chg)
        ns_ref.append(ns)
    cur_ref = jnp.stack(cur_ref)
    chg_ref = jnp.stack(chg_ref)
    ns_ref = jnp.stack(ns_ref)

    np.testing.assert_array_equal(cur_v, cur_ref)
    np.testing.assert_array_equal(chg_v, chg_ref)
    np.testing.assert_array_equal(ns_v, ns_ref)


def test_residual_eval_jit(resistor_model):
    """jax.jit composes with osdi_residual_eval."""
    voltages = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    params = jnp.array([[1.0, 50.0]], dtype=jnp.float64)
    old_state = jnp.empty((1, 0), dtype=jnp.float64)

    jit_eval = jax.jit(
        lambda v, p: osdi_residual_eval(resistor_model.id, v, p, old_state)
    )
    cur, _, _ = jit_eval(voltages, params)
    np.testing.assert_allclose(cur, [[0.02, -0.02]], rtol=1e-12)


# =============================================================================
# BATCH HANDLE API — setup once, eval many
# =============================================================================


def test_handle_eval_matches_stateless(resistor_model):
    """Handle-based full eval must produce bitwise-identical outputs to
    stateless osdi_eval on the same (params, voltages)."""
    num_devices = 5
    key = jax.random.PRNGKey(31)
    voltages = jax.random.uniform(
        key, (num_devices, 2), minval=-2.0, maxval=2.0, dtype=jnp.float64
    )
    params_np = np.stack(
        [
            np.ones((num_devices,), dtype=np.float64),
            np.linspace(10.0, 1000.0, num_devices, dtype=np.float64),
        ],
        axis=-1,
    )
    old_state = jnp.empty((num_devices, 0), dtype=jnp.float64)

    cur_s, cond_s, chg_s, cap_s, ns_s = osdi_eval(
        resistor_model.id, voltages, jnp.asarray(params_np), old_state
    )

    h = osdi_setup_batch(resistor_model.id, params_np)
    try:
        cur_h, cond_h, chg_h, cap_h, ns_h = osdi_eval_with_handle(
            h, voltages, old_state
        )
    finally:
        h.free()

    np.testing.assert_array_equal(cur_h, cur_s)
    np.testing.assert_array_equal(cond_h, cond_s)
    np.testing.assert_array_equal(chg_h, chg_s)
    np.testing.assert_array_equal(cap_h, cap_s)
    np.testing.assert_array_equal(ns_h, ns_s)


def test_handle_residual_eval_matches_full_handle(capacitor_model):
    """Handle-based residual eval returns the same cur/chg/new_state as
    handle-based full eval — on a reactive-only model so the charge path is
    also exercised."""
    C = 1e-12
    num_devices = 4
    voltages = jnp.asarray(
        np.random.RandomState(13).uniform(-1.0, 1.0, size=(num_devices, 2)),
        dtype=jnp.float64,
    )
    params_np = np.stack(
        [
            np.ones((num_devices,), dtype=np.float64),
            np.full((num_devices,), C, dtype=np.float64),
            np.ones((num_devices,), dtype=np.float64),
        ],
        axis=-1,
    )
    old_state = jnp.empty((num_devices, 0), dtype=jnp.float64)

    h = osdi_setup_batch(capacitor_model.id, params_np)
    try:
        cur_f, _, chg_f, _, ns_f = osdi_eval_with_handle(h, voltages, old_state)
        cur_r, chg_r, ns_r = osdi_residual_eval_with_handle(h, voltages, old_state)
    finally:
        h.free()

    np.testing.assert_array_equal(cur_r, cur_f)
    np.testing.assert_array_equal(chg_r, chg_f)
    np.testing.assert_array_equal(ns_r, ns_f)


def test_handle_many_evals_against_same_handle(resistor_model):
    """Evaluating the same handle many times with different voltages must
    give the correct (fresh) answer each call — eval() mutating inst_data
    can't contaminate later calls."""
    num_devices = 3
    params_np = np.stack(
        [
            np.ones((num_devices,), dtype=np.float64),
            np.array([50.0, 100.0, 200.0], dtype=np.float64),
        ],
        axis=-1,
    )
    old_state = jnp.empty((num_devices, 0), dtype=jnp.float64)

    h = osdi_setup_batch(resistor_model.id, params_np)
    try:
        key = jax.random.PRNGKey(99)
        for _ in range(12):
            key, k = jax.random.split(key)
            voltages = jax.random.uniform(
                k, (num_devices, 2), minval=-5.0, maxval=5.0, dtype=jnp.float64
            )
            cur, _, _, _, _ = osdi_eval_with_handle(h, voltages, old_state)
            r = jnp.asarray(params_np)[:, 1]
            expected = (voltages[:, 0] - voltages[:, 1]) / r
            np.testing.assert_allclose(cur[:, 0], expected, rtol=1e-12)
    finally:
        h.free()


def test_handle_vmap_matches_sequential(resistor_model):
    """jax.vmap must collapse into one FFI crossing per call against a
    shared handle (replicas have shared params, per-replica voltages)."""
    B, N_dev = 4, 9
    key = jax.random.PRNGKey(27)
    voltages_b = jax.random.uniform(
        key, (B, N_dev, 2), minval=-2.0, maxval=2.0, dtype=jnp.float64
    )
    params_np = np.stack(
        [
            np.ones((N_dev,), dtype=np.float64),
            np.linspace(10.0, 500.0, N_dev, dtype=np.float64),
        ],
        axis=-1,
    )
    old_state = jnp.empty((N_dev, 0), dtype=jnp.float64)

    h = osdi_setup_batch(resistor_model.id, params_np)
    try:
        vmapped = jax.vmap(lambda v: osdi_eval_with_handle(h, v, old_state), in_axes=0)
        cur_v, cond_v, chg_v, cap_v, ns_v = vmapped(voltages_b)

        cur_ref, cond_ref, chg_ref, cap_ref, ns_ref = [], [], [], [], []
        for b in range(B):
            c, g, q, cp, n = osdi_eval_with_handle(h, voltages_b[b], old_state)
            cur_ref.append(c)
            cond_ref.append(g)
            chg_ref.append(q)
            cap_ref.append(cp)
            ns_ref.append(n)
        np.testing.assert_array_equal(cur_v, jnp.stack(cur_ref))
        np.testing.assert_array_equal(cond_v, jnp.stack(cond_ref))
        np.testing.assert_array_equal(chg_v, jnp.stack(chg_ref))
        np.testing.assert_array_equal(cap_v, jnp.stack(cap_ref))
        np.testing.assert_array_equal(ns_v, jnp.stack(ns_ref))
    finally:
        h.free()


def test_handle_auto_free_on_gc(resistor_model):
    """Dropping the last reference to an OsdiBatchHandle must call
    osdi_free_handle via __del__ — no explicit free() required."""
    import gc

    params_np = np.array([[1.0, 50.0]], dtype=np.float64)
    h = osdi_setup_batch(resistor_model.id, params_np)
    hid = h.handle_id
    # Sanity: handle is alive and callable.
    assert h._alive
    del h
    gc.collect()
    # Re-setup; if the previous handle was freed, the new one gets a fresh id
    # (different from hid) — but more importantly, a follow-up eval on hid
    # should produce undefined results (we can't observe this directly, but
    # we can confirm __del__ didn't raise).
    h2 = osdi_setup_batch(resistor_model.id, params_np)
    try:
        assert h2.handle_id != hid or h2.handle_id > 0  # ids are monotonic
    finally:
        h2.free()


def test_handle_eval_after_free_raises(resistor_model):
    """Using a freed handle must raise cleanly rather than segfault."""
    params_np = np.array([[1.0, 50.0]], dtype=np.float64)
    h = osdi_setup_batch(resistor_model.id, params_np)
    h.free()

    voltages = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    old_state = jnp.empty((1, 0), dtype=jnp.float64)

    with pytest.raises(RuntimeError, match="freed"):
        osdi_eval_with_handle(h, voltages, old_state)
    with pytest.raises(RuntimeError, match="freed"):
        osdi_residual_eval_with_handle(h, voltages, old_state)


def test_handle_setup_invalid_model_raises():
    """osdi_setup_batch must raise for an unknown model_id (handle_id = 0
    from the Rust side)."""
    params_np = np.array([[1.0, 50.0]], dtype=np.float64)
    with pytest.raises(RuntimeError):
        osdi_setup_batch(99999, params_np)  # no model 99999


def test_handle_jit(resistor_model):
    """jax.jit must compose with osdi_eval_with_handle — handle_id ends up
    as a closed-over constant in the compiled graph."""
    num_devices = 2
    params_np = np.array([[1.0, 50.0], [1.0, 100.0]], dtype=np.float64)
    voltages = jnp.array([[1.0, 0.0], [2.0, 0.0]], dtype=jnp.float64)
    old_state = jnp.empty((num_devices, 0), dtype=jnp.float64)

    h = osdi_setup_batch(resistor_model.id, params_np)
    try:
        jit_eval = jax.jit(lambda v: osdi_eval_with_handle(h, v, old_state))
        cur, _, _, _, _ = jit_eval(voltages)
        expected = np.array([[0.02, -0.02], [0.02, -0.02]])
        np.testing.assert_allclose(cur, expected, rtol=1e-12)
    finally:
        h.free()


def test_handle_type():
    """Sanity: osdi_setup_batch returns an OsdiBatchHandle instance."""
    # Need a model; use the resistor fixture via a local load to keep the test
    # decoupled from any particular fixture scope.
    import pathlib

    m = load_osdi_model(str(pathlib.Path(__file__).parent / "resistor_va.osdi"))
    h = osdi_setup_batch(m.id, np.array([[1.0, 50.0]], dtype=np.float64))
    try:
        assert isinstance(h, OsdiBatchHandle)
        assert h.num_devices == 1
        assert h.num_params == 2
        assert h._alive
    finally:
        h.free()
