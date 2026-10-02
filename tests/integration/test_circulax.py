"""Public Circulax analyses exercise the native bosdi checkout end to end."""

import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("circulax")
from circulax import compile_circuit
from circulax.components.electronic import CurrentSource
from bosdi.circulax import osdi_component


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.fail("Integration tests require openvaf-r on PATH")
    root = tmp_path_factory.mktemp("circulax-integration")
    source = root / "rc.va"
    source.write_text("""`include "disciplines.vams"
module native_rc(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1000;
  parameter real c = 1e-9;
  real factor, v;
  analog begin
    @(initial_step) factor = $simparam("setup_scale", 1);
    v = $limit(V(p,n), "pnjlim", 0.026, 0.6);
    if (analysis("ac")) I(p,n) <+ 9*factor*$simparam("scale", 1)*v/r;
    else I(p,n) <+ factor*$simparam("scale", 1)*v/r;
    I(p,n) <+ ddt($simparam("charge_scale", 1)*c*V(p,n));
  end
endmodule
""")
    target = root / "rc.osdi"
    subprocess.run(
        [compiler, str(source), "-o", str(target)], check=True, capture_output=True
    )
    return target


@pytest.mark.parametrize("initial_mode", ["dc", "ac", "tran"])
def test_public_native_analyses_and_settings_sweep(binary, initial_mode):
    descriptor = osdi_component(
        str(binary), ("p", "n"), analysis=initial_mode, state_policy="limiting_only"
    )
    netlist = {
        "instances": {
            "r": {"component": "native"},
            "bias": {"component": "isource", "settings": {"I": -1e-3}},
            "gnd": {"component": "ground"},
        },
        "connections": {"r,p": "bias,p1", "r,n": ["bias,p2", "gnd,p1"]},
        "ports": {"out": "r,p"},
    }
    circuit = compile_circuit(
        netlist,
        {"native": descriptor, "isource": CurrentSource},
        backend="dense",
        is_complex=False,
        g_leak=0,
        rtol=1e-8,
        atol=1e-12,
        simparams={"setup_scale": 2, "scale": 1, "charge_scale": 4},
    )
    old_dc = jax.jit(circuit.dc)
    original = old_dc()
    assert float(circuit.port(original, "out")) == pytest.approx(0.5)
    updated = circuit.set_simparams(scale=2)
    assert float(updated.port(updated.dc(), "out")) == pytest.approx(0.25)
    frequencies = jnp.array([1e3, 1e6])
    # AC uses DC conductance despite the model's deliberately different AC G.
    expected = 2 / (1 + 50 * (4e-3 + 2j * np.pi * frequencies * 4e-9)) - 1
    actual = jax.jit(lambda f: updated.sp(ports="out", freqs=f))(frequencies)
    np.testing.assert_allclose(actual[:, 0, 0], expected, atol=1e-10)
    parameter_updated = jax.jit(
        lambda r: updated.sp(ports="out", freqs=frequencies, params={"r.r": r})
    )(2000.0)
    expected = 2 / (1 + 50 * (2e-3 + 2j * np.pi * frequencies * 4e-9)) - 1
    np.testing.assert_allclose(parameter_updated[:, 0, 0], expected, atol=1e-10)
    y0 = updated.dc().at[updated.port_map["r,p"]].add(1)
    times = jnp.linspace(0, 1e-6, 11)
    solution = updated.transient(
        t0=0, t1=1e-6, dt0=1e-9, y0=y0, saveat=times, max_steps=10000
    )
    np.testing.assert_allclose(
        updated.port(solution.ys, "out"),
        0.25 + np.exp(-np.asarray(times) / 1e-6),
        rtol=2e-3,
        atol=1e-4,
    )
    np.testing.assert_allclose(old_dc(), original, atol=1e-12)


def test_va_decorator_jacobian_and_parameter_gradient():
    from bosdi.circulax import va_component

    def jacobian(signals, s, r=1000):
        conductance = jnp.array([[1, -1], [-1, 1]]) / r
        return conductance, jnp.zeros((2, 2))

    @va_component(ports=("p", "n"), jacobian_fn=jacobian, differentiable_params=("r",))
    def resistor(signals, s, r=1000):
        current = (signals.p - signals.n) / r
        return {"p": current, "n": -current}, {}

    voltage = jnp.array([1.0, 0.0])
    conductance = jax.jit(
        jax.jacfwd(lambda y: resistor._fast_physics(y, resistor(), 0.0)[0])
    )(voltage)
    np.testing.assert_allclose(conductance, [[1e-3, -1e-3], [-1e-3, 1e-3]], atol=1e-15)
    gradient = jax.jit(
        jax.grad(lambda r: resistor._fast_physics(voltage, resistor(r=r), 0.0)[0][0])
    )(2000.0)
    assert float(gradient) == pytest.approx(-1 / 2000**2)
