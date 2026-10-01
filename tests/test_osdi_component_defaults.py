"""Unspecified OSDI parameters must be NaN ("not given"), not 0.0."""

import math
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

try:
    from bosdi.circulax.osdi_component import _BOSDI_AVAILABLE, osdi_component
except ImportError:
    _BOSDI_AVAILABLE = False
    osdi_component = None

folder = pathlib.Path(__file__).parent

pytestmark = pytest.mark.skipif(
    not _BOSDI_AVAILABLE, reason="bosdi native extension not available"
)


@pytest.fixture(scope="module")
def resistor_descriptor():
    osdi = folder / "compiled_osdi" / "resistor_va.osdi"
    if not osdi.exists():
        pytest.skip("resistor_va.osdi not compiled (openvaf-r not found)")
    return osdi_component(
        osdi_path=str(osdi), ports=("A", "B"), default_params={"R": 50.0}
    )


def test_unspecified_params_are_nan_not_zero(resistor_descriptor):
    inst = resistor_descriptor.make_instance({})
    assert inst["R"] == 50.0
    unspecified = {k: v for k, v in inst.items() if k != "R"}
    assert unspecified, "resistor model should expose more than one parameter"
    for name, value in unspecified.items():
        assert math.isnan(value), f"param {name!r} = {value!r}, expected NaN"


def test_nan_defaults_evaluate_like_explicit_defaults(resistor_descriptor):
    """1 V / 50 ohm = 20 mA; fails if the unspecified multiplier resolves to 0."""
    from osdi_jax import osdi_eval

    inst = resistor_descriptor.make_instance({})
    params = jnp.array(
        [[inst[k] for k in resistor_descriptor.param_names]], dtype=jnp.float64
    )
    voltages = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    old_state = jnp.empty((1, 0), dtype=jnp.float64)

    cur, cond, chg, cap, ns = osdi_eval(
        resistor_descriptor.model.id, voltages, params, old_state
    )
    np.testing.assert_allclose(cur, np.array([[0.02, -0.02]]), rtol=1e-6)


def test_setup_accepts_simulator_parameter_queries(tmp_path):
    """SPICE capacitor reads $simparam in setup; a null struct used to crash."""
    import shutil
    import subprocess
    from osdi_jax import osdi_eval_with_handle, osdi_setup_batch

    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r is not installed")
    source = folder / "devices/spice/capacitor.va"
    target = tmp_path / "capacitor.osdi"
    subprocess.run(
        [compiler, str(source), "-o", str(target)], check=True, capture_output=True
    )
    descriptor = osdi_component(str(target), ports=("p", "n"))
    instance = descriptor.make_instance({"capacitance": 1e-12})
    params = np.array([[instance[key] for key in descriptor.param_names]])
    handle = osdi_setup_batch(descriptor.model.id, params)
    _, _, charge, capacitance, _ = osdi_eval_with_handle(
        handle, jnp.array([[1.0, 0.0]]), jnp.empty((1, 0))
    )
    np.testing.assert_allclose(charge, [[1e-12, -1e-12]], rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(
        capacitance.reshape(1, 2, 2),
        [[[1e-12, -1e-12], [-1e-12, 1e-12]]],
        rtol=1e-12,
        atol=1e-24,
    )
