"""Native OSDI collapse must follow setup flags for each batched instance."""

import shutil
import subprocess

import jax.numpy as jnp
import numpy as np
import pytest

from osdi_loader import load_osdi_model
from osdi_jax import (
    osdi_eval,
    osdi_eval_with_handle,
    osdi_residual_eval_with_handle,
    osdi_setup_batch,
)


@pytest.fixture(scope="module")
def series_rc(tmp_path_factory):
    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r is not installed")
    root = tmp_path_factory.mktemp("instance-collapse")
    source = root / "series_rc.va"
    source.write_text("""`include "disciplines.vams"
module series_rc(p, n);
  inout p, n;
  electrical p, n, internal;
  parameter real r = 0 from [0:inf);
  parameter real c = 1e-12;
  analog begin
    if (r > 0) I(p, internal) <+ V(p, internal)/r;
    else V(p, internal) <+ 0;
    I(internal, n) <+ ddt(c * V(internal, n));
  end
endmodule
""")
    target = root / "series_rc.osdi"
    subprocess.run(
        [compiler, str(source), "-o", str(target)], check=True, capture_output=True
    )
    return load_osdi_model(str(target))


@pytest.mark.parametrize("cached", [False, True])
def test_batched_series_resistance_is_preserved(series_rc, cached):
    """One instance collapses; its neighbour retains the same internal node."""
    model = series_rc
    assert model.collapsible_pairs
    params = np.full((2, model.num_params), np.nan)
    params[:, model.param_names.index("r")] = [0, 1000]
    params[:, model.param_names.index("c")] = 1e-12
    states = jnp.empty((2, model.num_states))
    voltages = jnp.zeros((2, model.num_nodes))
    handle = osdi_setup_batch(model.id, params) if cached else None
    if cached:
        _, g, _, c, _ = osdi_eval_with_handle(handle, voltages, states)
    else:
        _, g, _, c, _ = osdi_eval(model.id, voltages, jnp.asarray(params), states)
    omega = 2 * np.pi * 1e9
    g = np.asarray(g).reshape(2, model.num_nodes, model.num_nodes)
    c = np.asarray(c).reshape(g.shape)
    for index, resistance in enumerate([0, 1000]):
        stamp = g[index] + 1j * omega * c[index]
        # Fix external terminals to 1 V and 0 V, then solve all internals.
        matrix = stamp.copy()
        matrix[:2] = 0
        matrix[0, 0] = matrix[1, 1] = 1
        rhs = np.zeros(model.num_nodes, dtype=complex)
        rhs[0] = 1
        solution = np.linalg.solve(matrix, rhs)
        admittance = stamp[0] @ solution
        expected = 1j * omega * 1e-12 / (1 + 1j * omega * resistance * 1e-12)
        np.testing.assert_allclose(admittance, expected, rtol=1e-12, atol=1e-15)
    if cached:
        # Artificially violate every equality to exercise the constraint rows
        # through the residual-only path used by Circulax line searches.
        voltages = jnp.arange(2 * model.num_nodes, dtype=jnp.float64).reshape(2, -1)
        f, _, _, _, _ = osdi_eval_with_handle(handle, voltages, states)
        residual, _, _ = osdi_residual_eval_with_handle(handle, voltages, states)
        np.testing.assert_allclose(residual, f, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("cached", [False, True])
def test_short_voltage_buffer_is_rejected(series_rc, cached):
    model = series_rc
    params = jnp.full((1, model.num_params), jnp.nan)
    states = jnp.empty((1, model.num_states))
    voltages = jnp.zeros((1, model.num_pins))
    assert model.num_nodes > model.num_pins
    with pytest.raises(Exception, match="voltage width must equal"):
        if cached:
            handle = osdi_setup_batch(model.id, params)
            osdi_eval_with_handle(handle, voltages, states)[0].block_until_ready()
        else:
            osdi_eval(model.id, voltages, params, states)[0].block_until_ready()


@pytest.fixture(scope="module")
def conditional_branch(tmp_path_factory):
    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r is not installed")
    root = tmp_path_factory.mktemp("ground-collapse")
    source = root / "conditional_branch.va"
    source.write_text("""`include "disciplines.vams"
module conditional_branch(p, n);
  inout p, n;
  electrical p, n, mid;
  parameter integer enabled = 0;
  parameter real r = 1000;
  analog begin
    if (enabled) V(p, mid) <+ r * I(p, mid);
    else V(p, mid) <+ 0;
    I(mid, n) <+ V(mid, n) / r;
  end
endmodule
""")
    target = root / "conditional_branch.osdi"
    subprocess.run(
        [compiler, str(source), "-o", str(target)], check=True, capture_output=True
    )
    return target


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("analysis", ["dc", "ac", "tran"])
def test_inactive_branch_collapses_to_ground(conditional_branch, cached, analysis):
    """Ground a disabled flow unknown; retain and solve the enabled branch."""
    from osdi_jax import osdi_residual_eval

    model = load_osdi_model(str(conditional_branch), analysis=analysis)
    assert any(
        second == -1 or second == 2**32 - 1 for _, second in model.collapsible_pairs
    )
    assert model.num_nodes == 4
    params = np.full((2, model.num_params), np.nan)
    params[:, model.param_names.index("enabled")] = [0, 1]
    params = jnp.asarray(params)
    states = jnp.zeros((2, model.num_states))
    voltages = jnp.asarray([[1.0, 0.0, 5.0, 5.0], [1.0, 0.0, 5.0, 5.0]])
    handle = osdi_setup_batch(model.id, params) if cached else None
    full = (
        osdi_eval_with_handle(handle, voltages, states)
        if cached
        else osdi_eval(model.id, voltages, params, states)
    )
    residual = (
        osdi_residual_eval_with_handle(handle, voltages, states)
        if cached
        else osdi_residual_eval(model.id, voltages, params, states)
    )
    np.testing.assert_allclose(residual[0], full[0])
    np.testing.assert_allclose(np.asarray(full[0])[0, -1], 5)
    g = np.asarray(full[1]).reshape(2, 4, 4)
    np.testing.assert_allclose(g[0, -1], [0, 0, 0, 1])
    np.testing.assert_allclose(np.asarray(full[2])[0], 0)
    np.testing.assert_allclose(np.asarray(full[3])[0], 0)
    # Linear stamp: constrain terminals, solve all raw internal slots. No
    # least-squares fallback or arbitrary regularisation should be necessary.
    for index, expected_current in enumerate([0.001, 0.0005]):
        matrix = g[index].copy()
        matrix[:2] = 0
        matrix[0, 0] = matrix[1, 1] = 1
        solution = np.linalg.solve(matrix, [1, 0, 0, 0])
        np.testing.assert_allclose(g[index, 0] @ solution, expected_current, atol=1e-14)
