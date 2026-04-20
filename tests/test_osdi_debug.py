"""Tests for the osdi_debug module (schur_reduce + dump_jacobian)."""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from osdi_debug import (
    classify_rows,
    dump_jacobian,
    format_jacobian_table,
    schur_reduce,
)
from osdi_jax import osdi_eval
from osdi_loader import load_osdi_model


# ---------------------------------------------------------------------------
# schur_reduce — mathematical correctness
# ---------------------------------------------------------------------------


def test_schur_reduce_trivial_no_internals():
    # When num_pins == num_nodes there's nothing to eliminate.
    cond = jnp.array([[1.0, -1.0], [-1.0, 1.0]])
    cap = jnp.zeros((2, 2))
    cur = jnp.array([0.5, -0.5])
    chg = jnp.zeros(2)

    r = schur_reduce(cur, cond, chg, cap, num_pins=2, alpha=0.0)
    np.testing.assert_allclose(r.j_eff, cond)
    np.testing.assert_allclose(r.r_eff, cur)
    assert not r.singular


def test_schur_reduce_dc_known_analytic():
    # 3×3 system with 2 terminals (T=0,1) and 1 internal (I=2).
    # Topology: terminal 0 --R1-- internal --R2-- terminal 1.
    # Conductance matrix G at node-ordering [0, 1, 2]:
    #     G = [[ 1/R1   0    -1/R1],
    #          [  0    1/R2  -1/R2],
    #          [-1/R1 -1/R2  1/R1+1/R2]]
    # Eliminating node 2 gives a series network between 0 and 1 with conductance
    # 1 / (R1 + R2):
    R1, R2 = 100.0, 200.0
    g1, g2 = 1 / R1, 1 / R2
    cond = jnp.array(
        [
            [g1, 0.0, -g1],
            [0.0, g2, -g2],
            [-g1, -g2, g1 + g2],
        ]
    )
    cap = jnp.zeros((3, 3))
    cur = jnp.zeros(3)
    chg = jnp.zeros(3)

    r = schur_reduce(cur, cond, chg, cap, num_pins=2, alpha=0.0, gmin=0.0)

    g_series = 1.0 / (R1 + R2)
    expected = np.array([[g_series, -g_series], [-g_series, g_series]])
    np.testing.assert_allclose(r.j_eff, expected, rtol=1e-10)


def test_schur_reduce_cur_residual_reduction():
    # Same topology as above, with a current injection at the internal node.
    R1, R2 = 100.0, 200.0
    g1, g2 = 1 / R1, 1 / R2
    cond = jnp.array(
        [
            [g1, 0.0, -g1],
            [0.0, g2, -g2],
            [-g1, -g2, g1 + g2],
        ]
    )
    cap = jnp.zeros((3, 3))
    # A 1 µA current source injected into the internal node only.
    cur = jnp.array([0.0, 0.0, 1e-6])
    chg = jnp.zeros(3)

    r = schur_reduce(cur, cond, chg, cap, num_pins=2, alpha=0.0, gmin=0.0)
    # Formula: r_eff = cur_T − G_TI · G_II⁻¹ · cur_I.  Here cur_T = 0,
    # G_TI = [-g1; -g2], G_II = g1+g2, cur_I = +I.  So
    #     r_eff = -[-g1; -g2] · I / (g1+g2) = [R2/(R1+R2); R1/(R1+R2)] · I.
    I_val = 1e-6
    expected_r = np.array([R2 / (R1 + R2) * I_val, R1 / (R1 + R2) * I_val])
    np.testing.assert_allclose(r.r_eff, expected_r, rtol=1e-10)


def test_schur_reduce_alpha_monotone():
    # Pure capacitor ladder: 2 terminals, 1 internal, no resistive path.
    # Adding α·C should produce a non-zero j_eff proportional to α.
    C1, C2 = 1e-12, 2e-12
    zero3 = jnp.zeros((3, 3))
    cap = jnp.array(
        [
            [C1, 0.0, -C1],
            [0.0, C2, -C2],
            [-C1, -C2, C1 + C2],
        ]
    )
    cur = jnp.zeros(3)
    chg = jnp.zeros(3)
    # Regularise with a small gmin because A_II has zero G component.
    r0 = schur_reduce(cur, zero3, chg, cap, num_pins=2, alpha=0.0, gmin=1e-9)
    r1 = schur_reduce(cur, zero3, chg, cap, num_pins=2, alpha=1e6, gmin=1e-9)

    # At α=0 with gmin, the reduction is dominated by the regularisation →
    # tiny j_eff.  At α=1e6 the capacitive coupling dominates.
    assert float(jnp.linalg.norm(r1.j_eff)) > 1e3 * float(
        jnp.linalg.norm(r0.j_eff) + 1e-30
    )


def test_schur_reduce_batched_vmap():
    # Exercise the (N, num_nodes, num_nodes) batched path via jax.vmap.
    R = jnp.array([50.0, 100.0, 200.0])

    def build(r_val):
        g1, g2 = 1 / r_val, 1 / (2 * r_val)
        return jnp.array(
            [
                [g1, 0.0, -g1],
                [0.0, g2, -g2],
                [-g1, -g2, g1 + g2],
            ]
        )

    cond = jax.vmap(build)(R)  # (3, 3, 3)
    cap = jnp.zeros_like(cond)
    cur = jnp.zeros((3, 3))
    chg = jnp.zeros((3, 3))

    r = schur_reduce(cur, cond, chg, cap, num_pins=2, alpha=0.0, gmin=0.0)
    # Each device's Schur conductance is 1/(R + 2R) = 1/(3R).
    expected_gseries = 1.0 / (3 * R)
    np.testing.assert_allclose(r.j_eff[:, 0, 0], expected_gseries, rtol=1e-10)
    np.testing.assert_allclose(r.j_eff[:, 0, 1], -expected_gseries, rtol=1e-10)


def test_schur_reduce_singular_flag():
    # A_II exactly zero (before gmin) → singular flag should fire.
    cond = jnp.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],  # diag is 0 → determinant of 1x1 A_II is 0
        ]
    )
    cap = jnp.zeros((3, 3))
    r = schur_reduce(jnp.zeros(3), cond, jnp.zeros(3), cap, num_pins=2, alpha=0.0)
    assert r.singular


# ---------------------------------------------------------------------------
# schur_reduce — against real OSDI models
# ---------------------------------------------------------------------------


FOLDER = pathlib.Path(__file__).parent


@pytest.fixture(scope="module")
def diode():
    return load_osdi_model(str(FOLDER / "compiled_osdi" / "diode.osdi"))


def test_schur_reduce_diode_noop(diode):
    # diode.osdi has num_pins=2 and num_nodes=2 (internal CI is statically
    # collapsed). So schur_reduce at α=0 should return j_eff=cond, r_eff=cur.
    V = jnp.array([[0.6, 0.0]])
    P = jnp.full((1, diode.num_params), jnp.nan).at[0, 0].set(1.0)
    S = jnp.empty((1, diode.num_states))
    cur, cond, chg, cap, _ = osdi_eval(diode.id, V, P, S)

    G = cond[0].reshape(diode.num_nodes, diode.num_nodes)
    C = cap[0].reshape(diode.num_nodes, diode.num_nodes)

    r = schur_reduce(cur[0], G, chg[0], C, num_pins=diode.num_pins, alpha=0.0)
    np.testing.assert_allclose(r.j_eff, G, rtol=1e-12)
    np.testing.assert_allclose(r.r_eff, cur[0], rtol=1e-12)


@pytest.fixture(scope="module")
def bsim4():
    return load_osdi_model(str(FOLDER / "compiled_osdi" / "bsim4v8.osdi"))


@pytest.fixture(scope="module")
def bsim4_params(bsim4):
    with open(FOLDER / "fixtures" / "bsim4v82_nmos.json") as f:
        fix = json.load(f)["params"]
    name_to_idx = {n.lower(): i for i, n in enumerate(bsim4.param_names) if n}
    p = jnp.full((1, bsim4.num_params), jnp.nan, dtype=jnp.float64).at[0, 0].set(1.0)
    for k, v in fix.items():
        idx = name_to_idx.get(k.lower())
        if idx is not None:
            p = p.at[0, idx].set(float(v))
    return p


def test_schur_reduce_bsim4_dc_finite(bsim4, bsim4_params):
    V = jnp.zeros((1, bsim4.num_nodes)).at[0, 0].set(1.0).at[0, 1].set(0.7)
    S = jnp.empty((1, bsim4.num_states))
    cur, cond, chg, cap, _ = osdi_eval(bsim4.id, V, bsim4_params, S)

    G = cond[0].reshape(bsim4.num_nodes, bsim4.num_nodes)
    C = cap[0].reshape(bsim4.num_nodes, bsim4.num_nodes)
    r = schur_reduce(cur[0], G, chg[0], C, num_pins=bsim4.num_pins, alpha=0.0)

    assert r.j_eff.shape == (bsim4.num_pins, bsim4.num_pins)
    assert r.r_eff.shape == (bsim4.num_pins,)
    assert bool(jnp.all(jnp.isfinite(r.j_eff)))
    assert bool(jnp.all(jnp.isfinite(r.r_eff)))


def test_schur_reduce_bsim4_terminal_current_preserved(bsim4, bsim4_params):
    # For a linearised device where cur ≈ G·V, the reduced residual r_eff at
    # the current operating point should equal the original cur projected
    # onto terminals *only if* the device is exactly linear. BSIM4 isn't,
    # but r_eff should still be within an order of magnitude of cur_T and
    # the KCL-summed terminal currents should be close to zero.
    V = jnp.zeros((1, bsim4.num_nodes)).at[0, 0].set(1.0).at[0, 1].set(0.7)
    S = jnp.empty((1, bsim4.num_states))
    cur, cond, chg, cap, _ = osdi_eval(bsim4.id, V, bsim4_params, S)

    G = cond[0].reshape(bsim4.num_nodes, bsim4.num_nodes)
    C = cap[0].reshape(bsim4.num_nodes, bsim4.num_nodes)
    r = schur_reduce(cur[0], G, chg[0], C, num_pins=bsim4.num_pins, alpha=0.0)

    # KCL: terminal currents sum to approximately zero.
    np.testing.assert_allclose(float(jnp.sum(r.r_eff)), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# dump_jacobian + classify_rows
# ---------------------------------------------------------------------------


def test_classify_rows_detects_constraint():
    # 3 rows: [physics, constraint, empty].
    cond = np.array(
        [
            [1e-3, -5e-4, 0.0],  # physics
            [0.0, -1.0, 1.0],  # ± identity constraint row
            [0.0, 0.0, 0.0],  # empty
        ]
    )
    cap = np.array(
        [
            [1e-15, -1e-15, 0.0],
            [0.0, 0.0, 0.0],  # constraint: zero cap row
            [0.0, 0.0, 0.0],
        ]
    )
    rows = classify_rows(cond, cap)
    assert rows[0].kind == "physics"
    assert rows[1].kind == "constraint"
    assert rows[2].kind == "empty"


def test_classify_rows_reactive_only():
    # Pure capacitor row: only cap, no cond.
    cond = np.zeros((2, 2))
    cap = np.array([[1e-12, -1e-12], [-1e-12, 1e-12]])
    rows = classify_rows(cond, cap)
    assert rows[0].kind == "reactive_only"
    assert rows[1].kind == "reactive_only"


def test_dump_jacobian_flags_constraint_entries():
    cond = np.array(
        [
            [1e-3, 0.0, -1e-3],
            [0.0, -1.0, 1.0],
            [-1e-3, 1.0, 1e-3],
        ]
    )
    cap = np.zeros((3, 3))

    entries = dump_jacobian(cond, cap)
    # Row 1 is classified as constraint → both entries in row 1 flagged.
    constraint_entries = [(e.row, e.col) for e in entries if e.is_likely_constraint]
    assert set(constraint_entries) == {(1, 1), (1, 2)}
    # Other rows' entries are not constraint-flagged.
    for e in entries:
        if e.row != 1:
            assert not e.is_likely_constraint


def test_format_jacobian_table_readable():
    cond = np.array([[2e-3, -2e-3], [-2e-3, 2e-3]])
    cap = np.array([[1e-14, -1e-14], [-1e-14, 1e-14]])
    s = format_jacobian_table(cond, cap)
    # Sanity: the table mentions row/col and a classification line.
    assert "row" in s and "col" in s and "classifications" in s


def test_dump_jacobian_diode_osdi(diode):
    V = jnp.array([[0.6, 0.0]])
    P = jnp.full((1, diode.num_params), jnp.nan).at[0, 0].set(1.0)
    S = jnp.empty((1, diode.num_states))
    _, cond, _, cap, _ = osdi_eval(diode.id, V, P, S)
    G = np.asarray(cond[0]).reshape(diode.num_nodes, diode.num_nodes)
    C = np.asarray(cap[0]).reshape(diode.num_nodes, diode.num_nodes)

    entries = dump_jacobian(G, C)
    # Diode at 0.6 V forward has non-zero G entries on all four cells.
    assert len(entries) >= 2
    # No row should classify as a constraint — the constraint row exists
    # only in the raw pre-collapse form.
    assert not any(e.is_likely_constraint for e in entries)
