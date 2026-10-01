"""Integral equations must use consistent full/residual analysis semantics."""

import shutil
import subprocess

import jax.numpy as jnp
import numpy as np
import pytest

from osdi_loader import load_osdi_model
from osdi_jax import (
    osdi_eval,
    osdi_eval_with_handle,
    osdi_residual_eval,
    osdi_residual_eval_with_handle,
    osdi_setup_batch,
)


@pytest.fixture(scope="module")
def integral_binary(tmp_path_factory):
    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r is not installed")
    root = tmp_path_factory.mktemp("integral-analysis")
    source = root / "integral.va"
    source.write_text("""`include "disciplines.vams"
module integral(p, n);
  inout p, n;
  electrical p, n;
  analog I(p, n) <+ idt(V(p, n), 2);
endmodule
""")
    target = root / "integral.osdi"
    subprocess.run(
        [compiler, str(source), "-o", str(target)], check=True, capture_output=True
    )
    return target


@pytest.mark.parametrize("analysis", ["dc", "ac", "tran"])
@pytest.mark.parametrize("cached", [False, True])
def test_integral_residual_matches_full_stamp(integral_binary, analysis, cached):
    model = load_osdi_model(str(integral_binary), analysis=analysis)
    params = jnp.full((1, model.num_params), jnp.nan)
    states = jnp.empty((1, model.num_states))
    assert model.num_nodes == 3
    voltage = jnp.array([[0.2, 0.0, 1.5]])
    if cached:
        handle = osdi_setup_batch(model.id, params)
        current, _, charge, capacitance, _ = osdi_eval_with_handle(
            handle, voltage, states
        )
        residual, residual_charge, _ = osdi_residual_eval_with_handle(
            handle, voltage, states
        )
    else:
        current, _, charge, capacitance, _ = osdi_eval(
            model.id, voltage, params, states
        )
        residual, residual_charge, _ = osdi_residual_eval(
            model.id, voltage, params, states
        )
    np.testing.assert_allclose(residual, current, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(residual_charge, charge, rtol=1e-12, atol=1e-15)
    if analysis == "dc":
        # In DC the integral equals its explicit initial value (2), rather
        # than requiring its input (0.2 V) to vanish in a steady-state solve.
        assert abs(float(current[0, -1])) == pytest.approx(0.5)
        np.testing.assert_array_equal(capacitance, 0)
    else:
        assert abs(float(current[0, -1])) == pytest.approx(0.2)
        assert np.any(np.asarray(capacitance) != 0)


def test_invalid_analysis_is_rejected(integral_binary):
    with pytest.raises(ValueError, match="analysis must be"):
        load_osdi_model(str(integral_binary), analysis="invalid")
