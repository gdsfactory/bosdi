import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Assuming you placed the loader and the JAX primitive in these modules
from osdi_loader import load_osdi_model
from osdi_jax import osdi_eval
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
