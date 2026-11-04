"""Exercise the experimental 0.5 header independently of an OpenVAF install.

Header copied (with whitespace normalized) from cdaunt/OpenVAF mb-experimental at
c63c4872ffb95e651f11311b0221f7e6f8121d16, openvaf/osdi/header/osdi_0_5.h.
Set BOSDI_OPENVAF_05 to also test a local experimental compiler.
"""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bosdi.circulax import osdi_component
from osdi_loader import load_osdi_model
from osdi_jax import (
    osdi_eval,
    osdi_eval_with_handle,
    osdi_residual_eval,
    osdi_residual_eval_with_handle,
    osdi_setup_batch,
)


@pytest.fixture(scope="module", params=["header", "openvaf"])
def binary(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("osdi05")
    target = root / "rc.osdi"
    if request.param == "header":
        compiler = shutil.which("cc")
        if compiler is None:
            pytest.skip("C compiler not installed")
        tests = Path(__file__).parent
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-shared",
                "-fPIC",
                f"-I{tests / 'headers'}",
                str(tests / "fixtures/osdi_0_5.c"),
                "-o",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
    else:
        compiler = os.environ.get("BOSDI_OPENVAF_05")
        if not compiler:
            pytest.skip("Set BOSDI_OPENVAF_05 to an experimental OpenVAF compiler")
        source = root / "rc.va"
        source.write_text("""`include "disciplines.vams"
module rc(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1000;
  parameter real c = 1e-9;
  analog begin
    I(p,n) <+ V(p,n)/r;
    I(p,n) <+ ddt(c * V(p,n));
  end
endmodule
""")
        subprocess.run(
            [compiler, str(source), "-o", str(target)], check=True, capture_output=True
        )
    lib = ctypes.CDLL(str(target))
    assert ctypes.c_uint32.in_dll(lib, "OSDI_VERSION_MINOR").value == 5
    return target


@pytest.mark.parametrize("analysis", ["dc", "ac", "tran"])
@pytest.mark.parametrize("cached", [False, True])
def test_rc_stamp_and_voltage_gradient(binary, analysis, cached):
    descriptor = osdi_component(
        str(binary), ("p", "n"), version="0.5", analysis=analysis
    )
    model = descriptor.model
    assert model.osdi_version == "0.5"
    assert (model.num_pins, model.num_nodes) == (2, 2)
    assert {"r", "c"}.issubset(model.param_names)
    assert model.num_resist_jac == model.num_react_jac == 4
    values = {"r": 2000.0, "c": 2e-9}
    params = jnp.array([[values.get(name, float("nan")) for name in model.param_names]])
    states = jnp.zeros((1, model.num_states))
    voltage = jnp.array([[1.0, 0.0]])
    if cached:
        handle = osdi_setup_batch(model.id, params)

        def full(v):
            return osdi_eval_with_handle(handle, v, states)

        def residual(v):
            return osdi_residual_eval_with_handle(handle, v, states)
    else:

        def full(v):
            return osdi_eval(model.id, v, params, states)

        def residual(v):
            return osdi_residual_eval(model.id, v, params, states)

    current, conductance, charge, capacitance, _ = jax.jit(full)(voltage)
    f, q, _ = jax.jit(residual)(voltage)
    g = 1 / values["r"]
    c = 0 if analysis == "dc" else values["c"]
    np.testing.assert_allclose(current, [[g, -g]])
    np.testing.assert_allclose(charge, [[c, -c]], atol=1e-20)
    np.testing.assert_allclose(conductance.reshape(2, 2), [[g, -g], [-g, g]])
    np.testing.assert_allclose(
        capacitance.reshape(2, 2), [[c, -c], [-c, c]], atol=1e-20
    )
    np.testing.assert_allclose(f, current)
    np.testing.assert_allclose(q, charge)
    if not cached:
        gradient = jax.grad(lambda v: full(v)[0][0, 0])(voltage)
        np.testing.assert_allclose(gradient, [[g, -g]])


def test_defaults_and_variants(binary):
    descriptor = osdi_component(str(binary), ("p", "n"), version="0.5")
    for variant in [
        descriptor.with_analysis("dc"),
        descriptor.with_simparams({"gmin": 1e-12}),
    ]:
        assert variant.model.osdi_version == "0.5"
        model = variant.model
        result = osdi_eval(
            model.id,
            jnp.array([[1.0, 0.0]]),
            jnp.full((1, model.num_params), jnp.nan),
            jnp.zeros((1, model.num_states)),
        )
        np.testing.assert_allclose(result[0], [[0.001, -0.001]])


def test_wrong_version_rejected(binary):
    with pytest.raises(RuntimeError, match=r"binary exports ABI 0.5, requested 0.4"):
        load_osdi_model(str(binary), version="0.4")
