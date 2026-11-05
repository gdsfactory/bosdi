"""Native OSDI collapse must follow setup flags for each batched instance."""

import os
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


@pytest.fixture(scope="module", params=["0.4", "0.5"])
def abi(request):
    compiler = (
        shutil.which("openvaf-r")
        if request.param == "0.4"
        else os.environ.get("BOSDI_OPENVAF_05")
    )
    if compiler is None:
        pytest.skip("Compiler missing; set BOSDI_OPENVAF_05 for experimental 0.5")
    return request.param, compiler


@pytest.fixture(scope="module")
def series_rc(tmp_path_factory, abi):
    version, compiler = abi
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
    return load_osdi_model(str(target), version=version)


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
def conditional_branch(tmp_path_factory, abi):
    _, compiler = abi
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
def test_inactive_branch_collapses_to_ground(conditional_branch, cached, analysis, abi):
    """Ground a disabled flow unknown; retain and solve the enabled branch."""
    from osdi_jax import osdi_residual_eval

    model = load_osdi_model(str(conditional_branch), version=abi[0], analysis=analysis)
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


@pytest.fixture(scope="module")
def reduced_assembly():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "docs/examples/reduced_batch.py"
    spec = importlib.util.spec_from_file_location("reduced_batch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("analysis", ["dc", "tran", "ac"])
@pytest.mark.parametrize("ground_branch", [False, True])
def test_reduced_stamps_and_solutions(
    series_rc, conditional_branch, reduced_assembly, analysis, ground_branch
):
    """Full equality-constrained DAE and reduced DC/BE/AC solve agree."""
    model = load_osdi_model(
        str(conditional_branch) if ground_branch else series_rc.path,
        version=series_rc.osdi_version,
        analysis=analysis,
    )
    params = np.full((2, model.num_params), np.nan)
    name = "enabled" if ground_branch else "r"
    params[:, model.param_names.index(name)] = [0, 1 if ground_branch else 1000]
    handle = osdi_setup_batch(model.id, params)
    topology = handle.topology
    mapping = topology.raw_to_representative
    assert mapping.shape == (2, model.num_nodes)
    assert topology.num_terminals == model.num_pins
    assert not topology.physical_row_mask[0].all()
    assert topology.physical_row_mask[1].all()
    if ground_branch:
        assert mapping[0, -1] == -1
    else:
        assert mapping[0, 2] == 0  # zero series resistance shorts internal to p
    assert np.all(mapping[1] == np.arange(model.num_nodes))
    for row in mapping:
        for rep in row[row >= 0]:
            assert row[rep] == rep
    # Deliberately share terminal 0 between devices and omit global ground.
    indices, size = reduced_assembly.allocate_raw_indices(topology, [[0, -1]] * 2, 1)
    y = np.linspace(0.2, 0.8, size)
    v = reduced_assembly.gather_voltages(y, indices)
    states = np.zeros((2, model.num_states))
    f, g, q, c, _ = osdi_eval_with_handle(handle, v, states)
    f, g, q, c = map(np.asarray, (f, g, q, c))
    stamps = reduced_assembly.scatter_stamps(topology, indices, f, g, q, c, size)
    fr, gr, qr, cr = stamps
    n = model.num_nodes
    # Compare S F(P y), S J P against full formulation P.T F, P.T J P.
    # Constraint Jacobians cancel exactly under tied column indices.
    projection = np.zeros((2 * n, size))
    for d, row in enumerate(indices):
        for r, idx in enumerate(row):
            if idx >= 0:
                projection[d * n + r, idx] = 1
    full_g = np.zeros((2 * n, 2 * n))
    full_c = full_g.copy()
    for d in range(2):
        sl = slice(d * n, (d + 1) * n)
        full_g[sl, sl] = g[d].reshape(n, n)
        full_c[sl, sl] = c[d].reshape(n, n)
    for reduced, full in [(fr, f.ravel()), (qr, q.ravel())]:
        np.testing.assert_allclose(reduced, projection.T @ full, atol=1e-15)
    for reduced, full in [(gr, full_g), (cr, full_c)]:
        np.testing.assert_allclose(
            reduced, projection.T @ full @ projection, atol=1e-15
        )
    # Numerical derivatives of reduced physical residual/charge.
    for j in range(size):
        delta = np.zeros(size)
        delta[j] = 1e-6

        def evaluate(y):
            v = reduced_assembly.gather_voltages(y, indices)
            ff, gg, qq, cc, _ = osdi_eval_with_handle(handle, v, states)
            return reduced_assembly.scatter_stamps(
                topology, indices, ff, gg, qq, cc, size
            )

        plus, minus = evaluate(y + delta), evaluate(y - delta)
        np.testing.assert_allclose((plus[0] - minus[0]) / 2e-6, gr[:, j], atol=1e-12)
        np.testing.assert_allclose((plus[2] - minus[2]) / 2e-6, cr[:, j], atol=1e-21)
    # Solve DC, backward Euler with nonzero previous charge, or AC.
    alpha = {"dc": 0, "tran": 1e9, "ac": 2j * np.pi * 1e9}[analysis]
    reduced_matrix = (gr + alpha * cr).copy()
    reduced_rhs = alpha * qr
    reduced_matrix[0] = 0
    reduced_matrix[0, 0] = 1
    reduced_rhs = reduced_rhs.copy()
    reduced_rhs[0] = 1
    reduced_solution = np.linalg.solve(reduced_matrix, reduced_rhs)
    for d in range(2):
        full_matrix = (g[d] + alpha * c[d]).reshape(n, n).copy()
        rhs = alpha * q[d].copy()
        full_matrix[:2] = 0
        full_matrix[0, 0] = full_matrix[1, 1] = 1
        rhs[:2] = [1, 0]
        solution = np.linalg.solve(full_matrix, rhs)
        gathered = reduced_assembly.gather_voltages(reduced_solution, indices)[d]
        np.testing.assert_allclose(solution, gathered, rtol=1e-12, atol=1e-14)


def test_topology_lifetime_and_updates(series_rc):
    import jax
    import osdi_shim_nb

    model = series_rc
    params = np.full((1, model.num_params), np.nan)
    handle = osdi_setup_batch(model.id, params)
    old = handle.topology
    assert old.handle_id == handle.handle_id and old.model_id == model.id
    with pytest.raises(ValueError):
        old.raw_to_representative[0, 0] = 99
    with pytest.raises(ValueError):
        old.raw_to_representative.setflags(write=True)
    params[:, model.param_names.index("r")] = 1000
    updated = handle.with_params(params)
    assert updated.handle_id != handle.handle_id
    assert not np.array_equal(
        old.raw_to_representative, updated.topology.raw_to_representative
    )
    np.testing.assert_array_equal(
        old.raw_to_representative, handle.topology.raw_to_representative
    )
    # Updating caller params never changes either immutable native snapshot.
    params[:] = 0
    assert updated.topology.physical_row_mask.all()
    states = np.zeros((1, model.num_states))
    voltage = np.ones((1, model.num_nodes))
    evaluate = jax.jit(lambda v: osdi_eval_with_handle(handle, v, states))
    evaluate(voltage)[0].block_until_ready()
    np.testing.assert_array_equal(
        old.raw_to_representative, handle.topology.raw_to_representative
    )
    handle.free()
    with pytest.raises(RuntimeError, match="freed"):
        _ = handle.topology
    with pytest.raises(RuntimeError, match="freed"):
        handle.with_params(params)
    with pytest.raises(ValueError, match="freed"):
        osdi_shim_nb.osdi_handle_topology(handle.handle_id)
    assert old.raw_to_representative.shape == (1, model.num_nodes)


def test_descriptor_setup_batch_defaults_and_names(series_rc):
    from bosdi.circulax import osdi_component

    descriptor = osdi_component(
        series_rc.path,
        ("p", "n"),
        version=series_rc.osdi_version,
        default_params={"r": 1000},
    )
    handle = descriptor.setup_batch([{}, {"R": 0}])
    assert handle.num_params == descriptor.model.num_params
    assert handle.topology.physical_row_mask[0].all()
    assert not handle.topology.physical_row_mask[1].all()
    assert descriptor.with_analysis("dc").setup_batch([{}]).model_id != handle.model_id

    # Legacy names rename the complete native parameter vector positionally.
    names = tuple(f"column{i}" for i in range(series_rc.num_params))
    resistance_name = names[series_rc.param_names.index("r")]
    renamed = osdi_component(
        series_rc.path,
        ("p", "n"),
        param_names=names,
        version=series_rc.osdi_version,
        default_params={resistance_name: 1000},
    )
    np.testing.assert_array_equal(
        renamed.setup_batch([{}, {resistance_name: 0}]).topology.raw_to_representative,
        handle.topology.raw_to_representative,
    )


def test_topology_uses_registration_temperature_and_settings(tmp_path):
    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r is not installed")
    source = tmp_path / "configured_collapse.va"
    source.write_text("""`include "disciplines.vams"
module configured_collapse(p, n);
  inout p, n;
  electrical p, n, mid;
  parameter real r = 0;
  analog begin
    if (r > 0)
      I(p, mid) <+ V(p, mid)/r * $temperature/300 * $simparam("scale", 1);
    else V(p, mid) <+ 0;
    I(mid, n) <+ V(mid, n)/1000;
  end
endmodule
""")
    binary = tmp_path / "configured_collapse.osdi"
    subprocess.run(
        [compiler, str(source), "-o", str(binary)], check=True, capture_output=True
    )
    from bosdi.circulax import osdi_component

    handles = []
    for temperature, settings in [(280, {}), (330, {}), (280, {"scale": 2})]:
        descriptor = osdi_component(
            str(binary), ("p", "n"), temperature=temperature, simparams=settings
        )
        handle = descriptor.setup_batch([{}, {"r": 1000}])
        handles.append(handle)
        mapping = handle.topology.raw_to_representative
        assert not handle.topology.physical_row_mask[0].all()
        assert handle.topology.physical_row_mask[1].all()
        voltages = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        assert mapping[0, 2] == 0
        f = osdi_eval_with_handle(handle, voltages, np.empty((2, 0)))[0]
        np.testing.assert_allclose(
            f[:, 0], [0.001, temperature / 300 / 1000 * settings.get("scale", 1)]
        )
    assert len({h.model_id for h in handles}) == 3
    # Re-read after configuring the other registrations: no shared mutable map.
    assert not handles[0].topology.physical_row_mask.all()


@pytest.mark.parametrize("analysis", ["dc", "tran", "ac"])
def test_c6288_psp103_cards(analysis, reduced_assembly):
    """Actual setup removes seven parasitic slots and retains both noise slots."""
    import re
    from pathlib import Path
    from bosdi.circulax import osdi_component

    binary = Path(__file__).parent / "compiled_osdi/psp103v4_psp103.osdi"
    if not binary.exists():
        pytest.skip("PSP103 binary not compiled")
    cards = (Path(__file__).parent / "fixtures/c6288_models.inc").read_text()
    blocks = re.findall(r"model\s+\w+\s+psp103va\s*\(([^)]*)\)", cards, re.S)
    assert len(blocks) == 2
    all_maps = []
    for d, block in enumerate(blocks):
        settings = {
            k: float(v) for k, v in re.findall(r"(\w+)\s*=\s*([+-]?[\d.eE+-]+)", block)
        }
        # c6288's NMOS/PMOS wrappers and gate instance dimensions.
        w, length, ld = (0.5e-6 if d == 0 else 1e-6), 0.2e-6, 0.5e-6
        settings.update(
            w=w, l=length, ad=w * ld, **{"as": w * ld}, pd=2 * (w + ld), ps=2 * (w + ld)
        )
        descriptor = osdi_component(
            str(binary),
            ("d", "g", "s", "b"),
            default_params=settings,
            analysis=analysis,
        )
        handle = descriptor.setup_batch([{}])
        topology = handle.topology
        mapping = topology.raw_to_representative
        # Count is measured from setup, never used to construct the mapping.
        assert descriptor.model.num_nodes == 13
        assert np.count_nonzero(~topology.physical_row_mask) == 7
        assert np.count_nonzero(topology.physical_row_mask[:, 4:]) == 2
        all_maps.append(mapping.copy())
        indices, size = reduced_assembly.allocate_raw_indices(
            topology, [[0, 1, -1, -1]], 2
        )
        y = np.zeros(size)
        y[:2] = [1.0, 0.8] if d == 0 else [-1.0, -0.8]
        v = reduced_assembly.gather_voltages(y, indices)
        states = np.zeros((1, descriptor.model.num_states))
        full = osdi_eval_with_handle(handle, v, states)
        f, g, q, c = map(np.asarray, full[:4])
        reduced = reduced_assembly.scatter_stamps(topology, indices, f, g, q, c, size)
        assert all(np.isfinite(stamp).all() for stamp in reduced)
        n = descriptor.model.num_nodes
        projection = np.zeros((n, size))
        for raw, idx in enumerate(indices[0]):
            if idx >= 0:
                projection[raw, idx] = 1
        for result, raw, matrix in zip(
            reduced, [f, g, q, c], [False, True, False, True]
        ):
            expected = (
                projection.T @ raw.reshape(n, n) @ projection
                if matrix
                else projection.T @ raw.ravel()
            )
            np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-15)
    np.testing.assert_array_equal(all_maps[0], all_maps[1])


def test_external_terminal_aliases_and_ground(tmp_path, abi, reduced_assembly):
    """Terminal collapses require global net unions before unknown allocation."""
    version, compiler = abi
    source = tmp_path / "terminal_shorts.va"
    source.write_text("""`include "disciplines.vams"
module terminal_shorts(p, n);
  inout p, n;
  electrical p, n;
  parameter integer mode = 0;
  analog begin
    if (mode == 1) V(p, n) <+ 0;
    else I(p, n) <+ V(p, n)/1000;
    if (mode == 2) V(p) <+ 0;
  end
endmodule
""")
    binary = tmp_path / "terminal_shorts.osdi"
    subprocess.run(
        [compiler, str(source), "-o", str(binary)], check=True, capture_output=True
    )
    from bosdi.circulax import osdi_component

    descriptor = osdi_component(str(binary), ("p", "n"), version=version)
    handle = descriptor.setup_batch([{"mode": 0}, {"mode": 1}, {"mode": 2}])
    mapping = handle.topology.raw_to_representative
    np.testing.assert_array_equal(mapping[:, :2], [[0, 1], [0, 0], [-1, 1]])
    with pytest.raises(ValueError, match="Union"):
        reduced_assembly.allocate_raw_indices(handle.topology, [[0, 1]] * 3, 2)
    indices, size = reduced_assembly.allocate_raw_indices(
        handle.topology, [[0, 1], [0, 0], [-1, 1]], 2
    )
    np.testing.assert_array_equal(indices[:, :2], [[0, 1], [0, 0], [-1, 1]])
    assert size == 2  # all extra flow slots were disabled by setup


@pytest.mark.parametrize("count", [0, 65])
def test_empty_and_parallel_mixed_topology(series_rc, count):
    params = np.full((count, series_rc.num_params), np.nan)
    params[:, series_rc.param_names.index("r")] = np.arange(count) % 2 * 1000
    handle = osdi_setup_batch(series_rc.id, params)
    mapping = handle.topology.raw_to_representative
    assert mapping.shape == (count, series_rc.num_nodes)
    for d, row in enumerate(mapping):
        np.testing.assert_array_equal(row, [0, 1, 2] if d % 2 else [0, 1, 0])
