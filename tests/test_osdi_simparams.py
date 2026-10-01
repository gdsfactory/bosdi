"""Simulator settings survive native setup, evaluation and registration reuse."""

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
def binary(tmp_path_factory):
    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r is not installed")
    root = tmp_path_factory.mktemp("simparams")
    source = root / "settings.va"
    source.write_text("""`include "disciplines.vams"
module settings(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1000;
  parameter real c = 1e-9;
  real factor;
  analog begin
    @(initial_step) factor = $simparam("setup_scale", 1);
    I(p,n) <+ factor * $simparam("scale", 1) * V(p,n)/r;
    I(p,n) <+ ddt($simparam("charge_scale", 1) * c * V(p,n));
  end
endmodule
""")
    target = root / "settings.osdi"
    subprocess.run(
        [compiler, str(source), "-o", str(target)], check=True, capture_output=True
    )
    return target


@pytest.mark.parametrize("mode", ["dc", "ac", "tran"])
@pytest.mark.parametrize("cached", [False, True])
def test_native_settings_and_device_updates(binary, mode, cached):
    settings = {"setup_scale": 2, "scale": 3, "charge_scale": 4}
    descriptor = osdi_component(
        str(binary), ("p", "n"), analysis=mode, simparams=settings
    )
    settings["scale"] = 99
    instance = descriptor.make_instance({"r": 2000, "c": 2e-9})
    params = np.array([[instance[name] for name in descriptor.param_names]])
    voltages = jnp.array([[1.0, 0.0]])
    states = jnp.empty((1, 0))
    if cached:
        handle = osdi_setup_batch(descriptor.model.id, params)
        result = jax.jit(lambda v: osdi_eval_with_handle(handle, v, states))(voltages)
    else:
        result = jax.jit(lambda p: osdi_eval(descriptor.model.id, voltages, p, states))(
            jnp.asarray(params)
        )
    current, conductance, charge, capacitance, _ = result
    np.testing.assert_allclose(current, [[0.003, -0.003]], atol=1e-15)
    np.testing.assert_allclose(
        conductance.reshape(2, 2), [[0.003, -0.003], [-0.003, 0.003]], atol=1e-15
    )
    # DC intentionally omits reactive residuals and Jacobians.
    expected = 0 if mode == "dc" else 8e-9
    np.testing.assert_allclose(charge, [[expected, -expected]], atol=1e-20)
    np.testing.assert_allclose(
        capacitance.reshape(2, 2),
        [[expected, -expected], [-expected, expected]],
        atol=1e-20,
    )
    assert dict(descriptor.with_analysis("tran").model.simparams)["scale"] == 3


def test_settings_cache_identity_and_defaults(binary):
    default = load_osdi_model(str(binary))
    first = load_osdi_model(str(binary), simparams={"scale": 2, "setup_scale": 3})
    same = load_osdi_model(str(binary), simparams={"setup_scale": 3.0, "scale": 2.0})
    different = load_osdi_model(str(binary), simparams={"scale": 4, "setup_scale": 3})
    assert first.id == same.id
    assert len({default.id, first.id, different.id}) == 3
    assert default.id == load_osdi_model(str(binary), simparams={}).id
    for model, expected in [(default, 0.001), (first, 0.006), (different, 0.012)]:
        params = jnp.full((1, model.num_params), jnp.nan)
        current = osdi_eval(
            model.id, jnp.array([[1.0, 0.0]]), params, jnp.empty((1, 0))
        )[0]
        np.testing.assert_allclose(current, [[expected, -expected]], atol=1e-15)


@pytest.mark.parametrize(
    "settings, error",
    [
        ({"": 1}, ValueError),
        ({"bad\0name": 1}, ValueError),
        ({1: 2}, ValueError),
        ({"scale": float("nan")}, ValueError),
        ({"scale": float("inf")}, ValueError),
        ({"scale": "2"}, TypeError),
        ([1, 2], TypeError),
    ],
)
def test_invalid_settings(binary, settings, error):
    with pytest.raises(error, match="(parameter|simparams)"):
        load_osdi_model(str(binary), simparams=settings)
