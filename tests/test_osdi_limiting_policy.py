"""OpenVAF limiting buffers must not be mistaken for physical time history."""

import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bosdi.circulax import osdi_component
from osdi_jax import osdi_eval, osdi_eval_with_handle, osdi_setup_batch
from osdi_loader import load_osdi_model


@pytest.fixture(scope="module")
def limiting_binary(tmp_path_factory):
    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r is not installed")
    root = tmp_path_factory.mktemp("limiting-policy")
    source = root / "limited.va"
    source.write_text("""`include "disciplines.vams"
module limited(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1000;
  parameter real c = 1e-12;
  real v;
  analog begin
    v = $limit(V(p,n), "pnjlim", 0.026, 0.6);
    I(p,n) <+ v/r + ddt(c*V(p,n));
  end
endmodule
""")
    target = root / "limited.osdi"
    subprocess.run(
        [compiler, str(source), "-o", str(target)], check=True, capture_output=True
    )
    return target


@pytest.mark.parametrize("analysis", ["dc", "ac", "tran"])
@pytest.mark.parametrize("cached", [False, True])
def test_disabled_limiting_does_not_change_physics(limiting_binary, analysis, cached):
    model = load_osdi_model(str(limiting_binary), analysis=analysis)
    assert model.num_states > 0
    params = jnp.full((1, model.num_params), jnp.nan)
    voltage = jnp.array([[1.0, 0.0]])
    handle = osdi_setup_batch(model.id, params) if cached else None
    outputs = []
    for old in [0.0, -1e6, 1e6]:
        states = jnp.full((1, model.num_states), old)
        result = (
            osdi_eval_with_handle(handle, voltage, states)
            if cached
            else osdi_eval(model.id, voltage, params, states)
        )
        outputs.append([np.asarray(item) for item in result[:4]])
    for output in outputs:
        np.testing.assert_allclose(output[0][0], [0.001, -0.001], atol=1e-15)
        for actual, expected in zip(output, outputs[0], strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert np.any(output[3] != 0) == (analysis != "dc")


def test_policy_is_explicit_and_preserved(limiting_binary):
    with pytest.raises(NotImplementedError, match="Stateful OSDI"):
        osdi_component(str(limiting_binary), ("p", "n"))
    with pytest.raises(ValueError, match="state_policy"):
        osdi_component(str(limiting_binary), ("p", "n"), state_policy="automatic")
    descriptor = osdi_component(
        str(limiting_binary),
        ("p", "n"),
        default_params={"r": 2000},
        state_policy="limiting_only",
        analysis="dc",
        temperature=310,
    )
    ac = descriptor.with_analysis("ac")
    assert ac is descriptor.with_analysis("ac")
    assert ac.model.analysis == "ac"
    assert ac.model.temperature == 310
    assert ac.state_policy == "limiting_only"
    assert ac.default_params["r"] == 2000
    assert descriptor.model.analysis == "dc"


def test_parameter_updates_use_uncached_evaluation_under_jit(limiting_binary):
    from bosdi.circulax.osdi_component import OsdiComponentGroup

    model = load_osdi_model(str(limiting_binary), analysis="dc")
    params = jnp.full((1, model.num_params), jnp.nan)
    states = jnp.zeros((1, model.num_states))
    group = OsdiComponentGroup(
        name="limited",
        model_id=model.id,
        num_pins=model.num_pins,
        num_nodes=model.num_nodes,
        num_params=model.num_params,
        num_states=model.num_states,
        params=params,
        states=states,
        var_indices=jnp.array([[0, 1]]),
        eq_indices=jnp.array([[0, 1]]),
        jac_rows=jnp.array([0, 0, 1, 1]),
        jac_cols=jnp.array([0, 1, 0, 1]),
        reg_diag=jnp.zeros((2, 2)),
        handle=osdi_setup_batch(model.id, params),
    )
    column = model.param_names.index("r")
    eager = group.with_params(params.at[0, column].set(2000))
    assert eager.handle is not None

    def current(resistance):
        updated = group.with_params(params.at[0, column].set(resistance))
        assert updated.handle is None
        return osdi_eval(
            updated.model_id, jnp.array([[1.0, 0.0]]), updated.params, updated.states
        )[0][0, 0]

    assert float(jax.jit(current)(2000.0)) == pytest.approx(0.0005)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(current))(jnp.array([1000.0, 2000.0])),
        [0.001, 0.0005],
        atol=1e-15,
    )
