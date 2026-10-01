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


@pytest.fixture(scope="module")
def temperature_model(tmp_path_factory):
    import shutil
    import subprocess

    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r is not installed")
    root = tmp_path_factory.mktemp("temperature-model")
    source = root / "temperature_resistor.va"
    source.write_text("""`include "disciplines.vams"
module temperature_resistor(p, n);
    inout p, n;
    electrical p, n;
    parameter real resistance=1000;
    real effective_resistance;
    analog begin
        @(initial_step) effective_resistance = resistance * $temperature / 300;
        I(p, n) <+ V(p, n) / effective_resistance;
    end
endmodule
""")
    target = root / "temperature_resistor.osdi"
    subprocess.run(
        [compiler, str(source), "-o", str(target)], check=True, capture_output=True
    )
    return target


def test_temperature_isolated_across_models_and_evaluation_paths(temperature_model):
    from osdi_jax import osdi_eval, osdi_eval_with_handle, osdi_setup_batch

    # Two registrations of the same binary must not alter each other's setup.
    cold = osdi_component(str(temperature_model), ports=("p", "n"))
    hot = osdi_component(str(temperature_model), ports=("p", "n"), temperature=600.0)
    assert cold.model.temperature == 300.0
    assert hot.model.temperature == 600.0
    voltage = jnp.array([[1.0, 0.0]])
    state = jnp.empty((1, 0))
    for descriptor, temperature in [(hot, 600.0), (cold, 300.0), (hot, 600.0)]:
        for resistance in [1000.0, 2000.0]:
            instance = descriptor.make_instance({"resistance": resistance})
            params = np.array([[instance[key] for key in descriptor.param_names]])
            expected = 300.0 / temperature / resistance
            handle = osdi_setup_batch(descriptor.model.id, params)
            cached = osdi_eval_with_handle(handle, voltage, state)[0]
            uncached = osdi_eval(descriptor.model.id, voltage, params, state)[0]
            np.testing.assert_allclose(cached, [[expected, -expected]], rtol=1e-12)
            np.testing.assert_allclose(uncached, cached, rtol=1e-12)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(temperature):
    from osdi_loader import load_osdi_model

    with pytest.raises(ValueError, match="temperature"):
        load_osdi_model("unused.osdi", temperature=temperature)
