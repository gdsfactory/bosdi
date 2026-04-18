"""
Behavioural tests for tests/compiled_osdi/diode.osdi.

Validates that the diode produces physically sensible forward-biased behaviour
and that JAX autodiff correctly routes the analytical OSDI Jacobian.

A golden-value comparison is deliberately avoided because bosdi does not expose
the diode's default Is/n to the test — instead we assert:

  * currents grow strictly monotonically with forward bias,
  * conductance grows strictly monotonically with forward bias,
  * the DC current is well approximated by a Shockley exponential,
  * jax.grad of the anode current w.r.t. voltage matches the OSDI-reported
    conductance to machine precision (the real Phase-1 regression guard —
    any num_pins/num_nodes shape mismatch would corrupt this gradient).
"""

import pathlib
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from osdi_loader import load_osdi_model
from osdi_jax import osdi_eval


DIODE_PATH = pathlib.Path(__file__).parent / "compiled_osdi" / "diode.osdi"


@pytest.fixture(scope="module")
def diode():
    return load_osdi_model(str(DIODE_PATH))


def _default_params(model):
    """All params at Verilog-A defaults except $mfactor=1 (always safe)."""
    p = jnp.full((1, model.num_params), jnp.nan, dtype=jnp.float64)
    return p.at[0, 0].set(1.0)


def _eval_at(model, v_forward):
    V = jnp.array([[v_forward, 0.0]], dtype=jnp.float64)
    P = _default_params(model)
    S = jnp.empty((1, model.num_states), dtype=jnp.float64)
    return osdi_eval(model.id, V, P, S)


def test_diode_current_monotonic_in_bias(diode):
    vs = [0.3, 0.5, 0.7]
    currents = [float(_eval_at(diode, v)[0][0, 0]) for v in vs]
    # I_A > 0 for forward bias, and strictly increasing
    assert all(i > 0 for i in currents), (
        f"Forward currents should be positive: {currents}"
    )
    assert currents[0] < currents[1] < currents[2], (
        f"I_A should grow monotonically with Vf; got {currents}"
    )


def test_diode_conductance_monotonic_in_bias(diode):
    vs = [0.3, 0.5, 0.7]
    gs = [float(_eval_at(diode, v)[1][0, 0]) for v in vs]
    # dI/dV at the anode row-col must be positive (diode is a conductance element)
    # and increase with bias.
    assert all(g > 0 for g in gs), f"G_AA should be positive: {gs}"
    assert gs[0] < gs[1] < gs[2], f"G_AA should grow monotonically with Vf; got {gs}"


def test_diode_currents_kcl(diode):
    """KCL: current into anode + current into cathode == 0 for a two-terminal device."""
    for v in [0.3, 0.5, 0.7]:
        cur = _eval_at(diode, v)[0]
        np.testing.assert_allclose(
            cur[0, 0] + cur[0, 1],
            0.0,
            atol=1e-15,
            err_msg=f"KCL violated at Vf={v}: I_A + I_C != 0 ({cur[0]})",
        )


def test_diode_follows_shockley_ratio(diode):
    """
    Shockley: I ≈ Is · exp(V/(n·Vt)). Therefore log(I(V2)/I(V1)) ≈ (V2-V1)/(n·Vt).
    We do not know n or Vt exactly, but the slope d(log I)/dV must be positive
    and roughly in the physically sensible range (10 – 50 /V), i.e. V_T between
    0.02 V (cold) and 0.1 V (very hot or very high n).
    """
    I1 = float(_eval_at(diode, 0.4)[0][0, 0])
    I2 = float(_eval_at(diode, 0.6)[0][0, 0])
    slope = (np.log(I2) - np.log(I1)) / (0.6 - 0.4)  # per volt
    assert 10.0 < slope < 60.0, (
        f"d(ln I)/dV = {slope:.2f}/V is outside the physical range [10, 60]"
    )


def test_diode_jax_grad_matches_conductance(diode):
    """
    Critical: jax.grad(I_A wrt V_anode) must equal G_AA reported by the model.
    A num_pins/num_nodes shape mismatch in osdi_jax.py would reshape the
    conductance buffer wrong and this test would return garbage.
    """
    Vf = 0.6
    V = jnp.array([[Vf, 0.0]], dtype=jnp.float64)
    P = _default_params(diode)
    S = jnp.empty((1, diode.num_states), dtype=jnp.float64)

    # Primal eval to read G_AA from the OSDI-reported conductance matrix
    cur, cond, _, _, _ = osdi_eval(diode.id, V, P, S)
    g_aa_osdi = float(cond[0, 0])
    assert g_aa_osdi > 0, "Diode G_AA at Vf=0.6 should be positive"

    def i_anode(v):
        c, _, _, _, _ = osdi_eval(diode.id, v, P, S)
        return c[0, 0]

    grad_v = jax.grad(i_anode)(V)
    g_aa_grad = float(grad_v[0, 0])
    # Must match the OSDI-reported value to near machine precision — the JVP
    # just routes the analytical G_AA through.
    np.testing.assert_allclose(
        g_aa_grad,
        g_aa_osdi,
        rtol=1e-12,
        err_msg=(
            f"jax.grad(I_A wrt V_A) = {g_aa_grad} disagrees with OSDI G_AA "
            f"= {g_aa_osdi}. Most likely a voltage-width / Jacobian-shape "
            "mismatch in osdi_jax.osdi_eval."
        ),
    )


def test_diode_zero_bias_zero_current(diode):
    """At V=0, Shockley gives I = Is·(exp(0)-1) = 0. Verify near-zero current."""
    cur = _eval_at(diode, 0.0)[0]
    np.testing.assert_allclose(
        cur,
        np.zeros_like(cur),
        atol=1e-14,
        err_msg=f"Zero bias should give zero current; got {cur[0]}",
    )


def test_diode_reactive_jacobian_scatter(diode):
    """
    Regression guard: when cj0 > 0, the diode junction capacitance must appear
    in the reactive Jacobian `cap` and in the charges `chg`. This exercises the
    flag-parse + scatter path for dual-flagged Jacobian entries (RESIST|REACT
    in the OSDI 0.4 OsdiJacobianEntry.flags) — a previous positional-layout
    assumption silently dropped every reactive contribution for diode, PSP103,
    BSIM4, BSIMBULK and any model where entries are dual-flagged.
    """
    # cj0 is the 11th parameter in the diode OSDI param table (index 10 —
    # verified empirically; the first 2 are instance params, 10 is cj0).
    CJ0_INDEX = 10
    CJ0 = 1.0e-10  # 100 pF — well above numerical noise

    V = jnp.array([[0.3, 0.0]], dtype=jnp.float64)  # mild forward bias
    P = jnp.full((1, diode.num_params), jnp.nan, dtype=jnp.float64)
    P = P.at[0, 0].set(1.0)  # $mfactor = 1
    P = P.at[0, CJ0_INDEX].set(CJ0)
    S = jnp.empty((1, diode.num_states), dtype=jnp.float64)

    _, _, chg, cap, _ = osdi_eval(diode.id, V, P, S)

    # Charge must be non-zero and KCL-consistent (Q_A + Q_C = 0).
    assert abs(float(chg[0, 0])) > 1e-15, (
        f"Junction charge should be non-zero with cj0={CJ0}; got {chg[0, 0]}. "
        "If this regresses, check lib.rs jacobian_entries flag parsing."
    )
    np.testing.assert_allclose(
        chg[0, 0] + chg[0, 1],
        0.0,
        atol=1e-18,
        err_msg=f"Charge KCL violated: Q_A + Q_C != 0 ({chg[0]})",
    )

    # 2x2 capacitance stamp is [[C, -C], [-C, C]] for any pure two-port cap.
    C_AA = float(cap[0, 0])
    assert C_AA > 0, f"C_AA must be positive; got {C_AA}"
    np.testing.assert_allclose(
        cap[0],
        [C_AA, -C_AA, -C_AA, C_AA],
        rtol=1e-12,
        err_msg=f"Capacitance stamp does not match 2×2 [[C,-C],[-C,C]]: {cap[0]}",
    )


def test_diode_batched_consistent_with_single(diode):
    """Batched eval of 3 identical biases must match single-device evals."""
    vs = jnp.array([[0.3, 0.0], [0.5, 0.0], [0.7, 0.0]], dtype=jnp.float64)
    P = jnp.tile(_default_params(diode), (3, 1))
    S = jnp.empty((3, diode.num_states), dtype=jnp.float64)

    cur_batch, cond_batch, _, _, _ = osdi_eval(diode.id, vs, P, S)

    for i, v in enumerate([0.3, 0.5, 0.7]):
        cur_single, cond_single, _, _, _ = _eval_at(diode, v)
        np.testing.assert_allclose(
            cur_batch[i],
            cur_single[0],
            rtol=1e-12,
            err_msg=f"Batched current at Vf={v} disagrees with single-device eval",
        )
        np.testing.assert_allclose(
            cond_batch[i],
            cond_single[0],
            rtol=1e-12,
            err_msg=f"Batched conductance at Vf={v} disagrees with single-device eval",
        )
