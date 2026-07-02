"""Unspecified OSDI parameters must be "not given" (NaN), not "given as 0".

NaN is the OSDI runtime's documented not-given marker (``osdi_setup_batch``):
the loader skips writing NaN entries, so ``setup_model``/``setup_instance``
run the model's own ``$param_given`` default resolution — exactly like an
ngspice ``.model`` card.  Zero-filling instead marks every unnamed parameter
as explicitly given as zero, including ``$mfactor`` (the device multiplier),
which silently scales every current and charge the device contributes to 0.
"""

import math
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from bosdi.circulax.osdi_component import _BOSDI_AVAILABLE, osdi_component

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
    # R was given -> concrete value.  m (the multiplier) was not -> NaN,
    # so the OSDI runtime applies the model's own default (1.0), instead of
    # treating it as explicitly given as 0 and scaling all currents to zero.
    assert inst["R"] == 50.0
    unspecified = {k: v for k, v in inst.items() if k != "R"}
    assert unspecified, "resistor model should expose more than one parameter"
    for name, value in unspecified.items():
        assert math.isnan(value), (
            f"unspecified param {name!r} filled with {value!r}; expected NaN "
            "(the OSDI 'not given' marker) so the Verilog-A default applies"
        )


def test_nan_defaults_evaluate_like_explicit_defaults(resistor_descriptor):
    """End-to-end: NaN-filled defaults must produce the model's own defaults.

    A 1 V bias across the 50-ohm resistor must yield 20 mA — which requires
    the multiplier ``m`` (unspecified, NaN) to resolve to its Verilog-A
    default of 1.0 inside ``setup_instance``.  With a 0.0 fill this returns
    exactly 0 A.
    """
    from osdi_jax import osdi_eval

    inst = resistor_descriptor.make_instance({})
    params = jnp.array([[inst[k] for k in resistor_descriptor.param_names]],
                       dtype=jnp.float64)
    voltages = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    old_state = jnp.empty((1, 0), dtype=jnp.float64)

    cur, cond, chg, cap, ns = osdi_eval(
        resistor_descriptor.model.id, voltages, params, old_state
    )
    np.testing.assert_allclose(cur, np.array([[0.02, -0.02]]), rtol=1e-6)
