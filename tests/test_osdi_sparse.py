"""Raw-entry handle stamps preserve the dense reference without dense buffers."""

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import osdi_shim_nb
from osdi_jax import (
    osdi_eval_sparse_with_handle,
    osdi_eval_with_handle,
    osdi_residual_eval_with_handle,
    osdi_setup_batch,
)
from osdi_loader import load_osdi_model
import test_osdi_instance_collapse as collapse_fixtures
import test_osdi_0_5 as osdi05_fixtures

abi = collapse_fixtures.abi
series_rc = collapse_fixtures.series_rc
conditional_branch = collapse_fixtures.conditional_branch
binary = osdi05_fixtures.binary

ROOT = Path(__file__).parent


def reconstruct(values, pairs, nodes):
    """Sum from zero in raw-entry order, just like the dense native path."""
    result = np.zeros((values.shape[0], nodes, nodes))
    for d, entries in enumerate(pairs):
        for k, (row, col) in enumerate(entries):
            if row >= 0 and col >= 0:
                result[d, row, col] += values[d, k]
    return result


def assert_equivalent(handle, v, s):
    dense = tuple(map(np.asarray, osdi_eval_with_handle(handle, v, s)))
    sparse = tuple(map(np.asarray, osdi_eval_sparse_with_handle(handle, v, s)))
    for index in (0, 2, 4):
        np.testing.assert_array_equal(sparse[index], dense[index], strict=True)
        assert sparse[index].tobytes() == dense[index].tobytes()
    pattern = handle.sparse_pattern
    nodes = v.shape[1]
    g = reconstruct(sparse[1], pattern.resist_pairs, nodes)
    # Dense equality rows overwrite these positions after the raw scatter.
    for d, entries in enumerate(pattern.equality_pairs):
        for k, (row, col) in enumerate(entries):
            if row >= 0:
                g[d, row, col] = pattern.equality_values[d, k]
    c = reconstruct(sparse[3], pattern.react_pairs, nodes)
    assert g.tobytes() == dense[1].tobytes()
    assert c.tobytes() == dense[3].tobytes()
    residual = osdi_residual_eval_with_handle(handle, v, s, sparse=True)
    for actual, expected in zip(residual, (dense[0], dense[2], dense[4])):
        assert np.asarray(actual).tobytes() == expected.tobytes()
    return dense, sparse


@pytest.mark.parametrize("count", [1, 3, 65, 257])
def test_sparse_mixed_collapse(series_rc, count):
    model = series_rc
    params = np.full((count, model.num_params), np.nan)
    params[:, model.param_names.index("r")] = np.arange(count) % 2 * 1000
    handle = osdi_setup_batch(model.id, params)
    v = jnp.arange(count * model.num_nodes, dtype=jnp.float64).reshape(count, -1) / 10
    s = jnp.zeros((count, model.num_states))
    assert_equivalent(handle, v, s)
    pattern = handle.sparse_pattern
    assert pattern.resist_pairs.shape == (count, model.num_resist_jac, 2)
    assert pattern.react_pairs.shape == (count, model.num_react_jac, 2)
    with pytest.raises(ValueError):
        pattern.resist_pairs.setflags(write=True)


@pytest.mark.parametrize("binary", ["resistor_va", "capacitor_va"])
@pytest.mark.parametrize("analysis", ["dc", "ac", "tran"])
def test_sparse_zero_width_and_analysis(binary, analysis):
    model = load_osdi_model(
        str(ROOT / f"compiled_osdi/{binary}.osdi"), analysis=analysis
    )
    handle = osdi_setup_batch(model.id, np.full((65, model.num_params), np.nan))
    v = jnp.ones((65, model.num_nodes))
    s = jnp.zeros((65, model.num_states))
    _, sparse = assert_equivalent(handle, v, s)
    assert 0 in (sparse[1].shape[1], sparse[3].shape[1])


@pytest.mark.parametrize("analysis", ["dc", "ac", "tran"])
@pytest.mark.parametrize("card", [0, 1])
def test_sparse_c6288_psp103(analysis, card):
    cards = (ROOT / "fixtures/c6288_models.inc").read_text()
    blocks = re.findall(r"model\s+\w+\s+psp103va\s*\(([^)]*)\)", cards, re.S)
    settings = {
        k: float(v)
        for k, v in re.findall(r"(\w+)\s*=\s*([+-]?[\d.eE+-]+)", blocks[card])
    }
    w, length, ld = (0.5e-6 if card == 0 else 1e-6), 0.2e-6, 0.5e-6
    settings.update(
        w=w, l=length, ad=w * ld, **{"as": w * ld}, pd=2 * (w + ld), ps=2 * (w + ld)
    )
    model = load_osdi_model(
        str(ROOT / "compiled_osdi/psp103v4_psp103.osdi"), analysis=analysis
    )
    params = np.full((65, model.num_params), np.nan)
    for name, value in settings.items():
        params[:, [n.lower() for n in model.param_names].index(name.lower())] = value
    handle = osdi_setup_batch(model.id, params)
    assert model.num_nodes == 13
    assert np.count_nonzero(handle.topology.physical_row_mask[0, 4:]) == 2
    v = (
        jnp.zeros((65, model.num_nodes))
        .at[:, :2]
        .set(jnp.array([1.0, 0.8]) * (-1 if card else 1))
    )
    assert_equivalent(handle, v, jnp.zeros((65, model.num_states)))


def test_sparse_jit_vmap_and_jvp(series_rc):
    model = series_rc
    params = np.full((2, model.num_params), np.nan)
    params[:, model.param_names.index("r")] = [0, 1000]
    handle = osdi_setup_batch(model.id, params)
    v = jnp.arange(2 * model.num_nodes, dtype=jnp.float64).reshape(2, -1)
    s = jnp.zeros((2, model.num_states))

    def evaluate(x):
        return osdi_eval_sparse_with_handle(handle, x, s)

    dense, _ = assert_equivalent(handle, v, s)
    mapped = jax.jit(jax.vmap(evaluate))(jnp.stack([v, 2 * v, 3 * v]))
    for index, scale in enumerate((1, 2, 3)):
        direct = evaluate(scale * v)
        for result, expected in zip(mapped, direct):
            np.testing.assert_array_equal(result[index], expected)
    tangent = jnp.cos(v)
    for sparse_residual in (False, True):
        fn = (
            evaluate
            if not sparse_residual
            else lambda x: osdi_residual_eval_with_handle(handle, x, s, sparse=True)
        )
        _, derivatives = jax.jit(lambda x, t: jax.jvp(fn, (x,), (t,)))(v, tangent)
        q_index = 1 if sparse_residual else 2
        np.testing.assert_allclose(
            derivatives[0],
            np.einsum("bij,bj->bi", dense[1].reshape(2, model.num_nodes, -1), tangent),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            derivatives[q_index],
            np.einsum("bij,bj->bi", dense[3].reshape(2, model.num_nodes, -1), tangent),
            atol=1e-25,
        )
        grad = jax.jit(jax.grad(lambda x: fn(x)[0].sum()))(v)
        np.testing.assert_allclose(
            grad, dense[1].reshape(2, model.num_nodes, -1).sum(axis=1), atol=1e-15
        )


def test_sparse_invalid_shapes_and_freed_handle(series_rc):
    model = series_rc
    handle = osdi_setup_batch(model.id, np.full((2, model.num_params), np.nan))
    state = jnp.zeros((2, model.num_states))
    with pytest.raises(Exception, match="voltage width must equal"):
        osdi_eval_sparse_with_handle(handle, jnp.zeros((2, model.num_pins)), state)[
            0
        ].block_until_ready()
    with pytest.raises(Exception, match="shapes do not match"):
        osdi_eval_sparse_with_handle(
            handle, jnp.zeros((3, model.num_nodes)), jnp.zeros((3, model.num_states))
        )[0].block_until_ready()
    pattern = handle.sparse_pattern
    handle.free()
    assert pattern.resist_pairs.shape[0] == 2
    with pytest.raises(RuntimeError, match="freed"):
        _ = handle.sparse_pattern
    with pytest.raises(RuntimeError, match="freed"):
        osdi_eval_sparse_with_handle(handle, jnp.zeros((2, model.num_nodes)), state)
    with pytest.raises(ValueError, match="freed"):
        osdi_shim_nb.osdi_handle_jac_pairs(handle.handle_id, False)


@pytest.mark.parametrize("analysis", ["dc", "ac", "tran"])
def test_sparse_grounded_auxiliary(conditional_branch, analysis, abi):
    model = load_osdi_model(str(conditional_branch), version=abi[0], analysis=analysis)
    params = np.full((65, model.num_params), np.nan)
    params[:, model.param_names.index("enabled")] = np.arange(65) % 2
    handle = osdi_setup_batch(model.id, params)
    assert (handle.topology.raw_to_representative == -1).any()
    v = jnp.tile(jnp.array([[1.0, 0.0, 5.0, 5.0]]), (65, 1))
    state = jnp.zeros((65, model.num_states))
    dense, _ = assert_equivalent(handle, v, state)
    assert (handle.sparse_pattern.resist_pairs == -1).any()
    derivative = jax.jacfwd(
        lambda x: osdi_eval_sparse_with_handle(handle, x, state)[0]
    )(v)
    for d in range(65):
        np.testing.assert_allclose(
            derivative[d, :, d, :], dense[1].reshape(65, 4, 4)[d]
        )


@pytest.mark.parametrize("analysis", ["dc", "ac", "tran"])
def test_sparse_05_header(binary, analysis):
    model = load_osdi_model(str(binary), version="0.5", analysis=analysis)
    handle = osdi_setup_batch(model.id, np.full((3, model.num_params), np.nan))
    assert_equivalent(
        handle, jnp.ones((3, model.num_nodes)), jnp.zeros((3, model.num_states))
    )
