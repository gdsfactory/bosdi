"""Behavioural test for bsim4v8.osdi driven with a realistic model card.

Loads tests/fixtures/bsim4v82_nmos.json — the NMOS 60nm parameter set from
VACASK's demo/spice/bsim4v82.inc (model 'n1') — and feeds it to bosdi's
OSDI evaluator. Verifies that the MOSFET produces physically sensible
current in saturation and that the Jacobian is consistent (g_m = dI_d/dV_g).

Background: at default NaN params BSIM4 still evaluates (setup_instance
accepts the compiled-in defaults) but the current magnitude is governed by
unrealistic internal defaults. With a real model card the saturation drain
current should be on the order of 100 uA at Vds=1V, Vgs=0.7V for
W=10um, L=60nm, $mfactor=2 — matching the VACASK demo geometry.
"""

import json
import pathlib
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from osdi_loader import load_osdi_model
from osdi_jax import osdi_eval


BSIM4_PATH = pathlib.Path(__file__).parent / "compiled_osdi" / "bsim4v8.osdi"
FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "bsim4v82_nmos.json"


@pytest.fixture(scope="module")
def bsim4():
    return load_osdi_model(str(BSIM4_PATH))


@pytest.fixture(scope="module")
def nmos_param_vec(bsim4):
    """Build a (1, num_params) float64 param row for VACASK's n1 NMOS card.

    Values are keyed by name into the OSDI param table (case-insensitive).
    Unmatched names — there shouldn't be any for bsim4v8 — are reported so
    a future refresh of the .inc fixture highlights missing params instead of
    silently falling back to defaults.
    """
    with open(FIXTURE_PATH) as f:
        fixture = json.load(f)
    params_by_name = fixture["params"]

    name_to_idx = {n.lower(): i for i, n in enumerate(bsim4.param_names) if n}
    p = jnp.full((1, bsim4.num_params), jnp.nan, dtype=jnp.float64)
    p = p.at[0, 0].set(1.0)  # $mfactor

    missing = []
    for name, val in params_by_name.items():
        idx = name_to_idx.get(name.lower())
        if idx is None:
            missing.append(name)
            continue
        p = p.at[0, idx].set(float(val))

    assert not missing, (
        f"VACASK param names not found in bosdi's bsim4v8 OSDI table: {missing}. "
        "The .osdi binary and the .inc fixture are from different BSIM4 sources."
    )
    return p


def _eval(bsim4, params, vd, vg, vs=0.0, vb=0.0):
    """Bias a 4-terminal MOSFET. num_nodes > 4 for BSIM4 (7 internals + aux),
    so pad with zeros — the Newton iterate starts the internal nodes at 0 V."""
    V = jnp.zeros((1, bsim4.num_nodes), dtype=jnp.float64)
    V = V.at[0, 0].set(vd).at[0, 1].set(vg).at[0, 2].set(vs).at[0, 3].set(vb)
    S = jnp.empty((1, bsim4.num_states), dtype=jnp.float64)
    return osdi_eval(bsim4.id, V, params, S)


def test_bsim4_saturation_current(bsim4, nmos_param_vec):
    """At Vds=1V, Vgs=0.7V with the VACASK NMOS card, the drain current should
    be in the tens-to-hundreds of microamps range — not zero, not amps."""
    cur, cond, *_ = _eval(bsim4, nmos_param_vec, vd=1.0, vg=0.7)
    id_drain = float(cur[0, 0])
    assert id_drain > 1e-6, f"I_d should be > 1 uA at Vds=1, Vgs=0.7; got {id_drain}"
    assert id_drain < 1e-2, f"I_d should be < 10 mA for this geometry; got {id_drain}"


def test_bsim4_current_monotonic_in_vgs(bsim4, nmos_param_vec):
    """Above threshold, I_d must grow with V_gs."""
    ids = []
    for vgs in [0.3, 0.5, 0.7, 0.9]:
        cur, *_ = _eval(bsim4, nmos_param_vec, vd=1.0, vg=vgs)
        ids.append(float(cur[0, 0]))
    assert ids[0] < ids[1] < ids[2] < ids[3], (
        f"I_d should be monotonically increasing in V_gs; got {ids}"
    )


def test_bsim4_conductance_matches_jax_grad(bsim4, nmos_param_vec):
    """dI_d/dV_g from jax.grad must match the OSDI Jacobian's G[d,g] row entry.

    This is the real regression guard: if the num_nodes-sized Jacobian buffer
    is mis-routed or mis-shaped, this will diverge.
    """
    vd, vg = 1.0, 0.7
    V = jnp.zeros((1, bsim4.num_nodes), dtype=jnp.float64)
    V = V.at[0, 0].set(vd).at[0, 1].set(vg)
    P = nmos_param_vec
    S = jnp.empty((1, bsim4.num_states), dtype=jnp.float64)

    _, cond, *_ = osdi_eval(bsim4.id, V, P, S)
    # cond is flattened num_nodes x num_nodes, row-major. Row=d (0), Col=g (1).
    g_dg_osdi = float(cond[0, 0 * bsim4.num_nodes + 1])

    def i_drain(v):
        c, *_ = osdi_eval(bsim4.id, v, P, S)
        return c[0, 0]

    g_dg_grad = float(jax.grad(i_drain)(V)[0, 1])
    # Tolerance relaxed to 1e-10 because JAX's reverse-mode AD accumulates in
    # a different order than the direct OSDI read; some models return exactly
    # equal, others have floating rounding at the last bit.
    np.testing.assert_allclose(
        g_dg_grad,
        g_dg_osdi,
        rtol=1e-10,
        err_msg=f"jax.grad disagrees with OSDI: grad={g_dg_grad} osdi={g_dg_osdi}",
    )


def test_bsim4_nmos_param_fixture_complete(bsim4):
    """Refresh-time check: every VACASK param in the fixture must map to a
    canonical OSDI param name. If a future fixture bump adds a param that
    doesn't exist in bosdi's binary, we want to fail loudly, not silently."""
    with open(FIXTURE_PATH) as f:
        fixture = json.load(f)
    osdi_names = {n.lower() for n in bsim4.param_names if n}
    missing = [k for k in fixture["params"] if k.lower() not in osdi_names]
    assert not missing, (
        f"{len(missing)} VACASK params not in bsim4v8.osdi name table: "
        f"{missing[:10]}… consider rebuilding .osdi or updating the fixture."
    )
