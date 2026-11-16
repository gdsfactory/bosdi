"""Exact handle parity, buffer lifetime and caller-owned bypass semantics."""

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re

import numpy as np
import pytest

from _native_runtime import Native
from osdi_jax import OsdiBatchHandle, osdi_setup_batch
from osdi_loader import load_osdi_model
from osdi_runtime import (
    BoundBatch,
    BypassCache,
    num_threads,
    qualify_handle,
    set_num_threads,
    set_parallel_policy,
    _Buffers,
    _bind,
    _rebind,
    _eval,
    _generation,
    _unbind,
)
import test_osdi_limiting_policy as limiting_fixtures
import test_osdi_instance_collapse as collapse

limiting_binary = limiting_fixtures.limiting_binary
abi = collapse.abi
series_rc = collapse.series_rc
conditional_branch = collapse.conditional_branch
ROOT = Path(__file__).parent
CPUS = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else os.cpu_count() or 1
)
THREADS = sorted({1, max(1, CPUS // 2), CPUS})


def model_params(name, count, analysis="dc"):
    model = load_osdi_model(str(ROOT / "compiled_osdi" / name), analysis=analysis)
    params = np.full((count, model.num_params), np.nan)
    if name.startswith("psp"):
        cards = (ROOT / "fixtures/c6288_models.inc").read_text()
        block = re.findall(r"model\s+\w+\s+psp103va\s*\(([^)]*)\)", cards, re.S)[0]
        settings = {
            k: float(v) for k, v in re.findall(r"(\w+)\s*=\s*([+-]?[\d.eE+-]+)", block)
        }
        settings.update(
            w=0.5e-6, l=0.2e-6, ad=0.25e-12, **{"as": 0.25e-12}, pd=2e-6, ps=2e-6
        )
        names = [n.lower() for n in model.param_names]
        for name, value in settings.items():
            params[:, names.index(name.lower())] = value
    elif name == "resistor_va.osdi":
        params[:, model.param_names.index("R")] = np.linspace(100, 1000, count)
    return model, params


@pytest.fixture(scope="module", params=THREADS)
def threads(request):
    before = dict(os.environ)
    set_num_threads(request.param)
    assert num_threads() == request.param
    assert dict(os.environ) == before
    yield request.param
    set_parallel_policy()
    set_num_threads(0)


@pytest.mark.parametrize("count", [1, 31, 32, 100, 1000, 10000])
@pytest.mark.parametrize("name", ["resistor_va.osdi", "psp103v4_psp103.osdi"])
@pytest.mark.parametrize("analysis", ["dc", "tran"])
def test_bitwise_parity(threads, count, name, analysis):
    # Supplying the saved pre-change shared library audits against the original
    # implementation; ordinary CI compares the restored-snapshot reference.
    current = Native()
    reference = Native(os.environ.get("BOSDI_BASELINE_LIBRARY"))
    model, params = model_params(name, count, analysis)
    metadata = current.model(name, 0 if analysis == "dc" else 2)
    old_metadata = reference.model(name, 0 if analysis == "dc" else 2)
    handle = current.setup(metadata.id, count, model.num_params, params.ctypes.data)
    old_handle = reference.setup(
        old_metadata.id, count, model.num_params, params.ctypes.data
    )
    qualify = current.function(
        "bosdi_qualify_handle", ct.c_bool, ct.c_uint64, ct.c_bool
    )
    try:
        expected = []
        for scale in (0.8, 0.3, 0.8):
            voltage = np.zeros((count, model.num_nodes))
            voltage[:, :2] = [scale, scale * 0.8]
            for tiled in (False, True):
                v = np.tile(voltage, (2, 1)) if tiled else voltage
                for mode in ("dense", "sparse", "residual"):
                    want = reference.evaluate(old_handle, old_metadata, v, mode)
                    got = current.evaluate(handle, metadata, v, mode)
                    assert [a.tobytes() for a in got] == [a.tobytes() for a in want]
                    expected.append((v, mode, want))
        if not metadata.states:
            assert qualify(handle, True)
            for v, mode, want in expected:
                got = current.evaluate(handle, metadata, v, mode)
                assert [a.tobytes() for a in got] == [a.tobytes() for a in want]
        else:
            assert not qualify(handle, True)
        wrapped = OsdiBatchHandle(handle, metadata.id, count, params.shape[1])
        with BoundBatch(wrapped, np.zeros((count, model.num_nodes))) as bound:
            if not metadata.states:
                bound.qualify_pure_eval()
            for v, mode, want in expected:
                if mode != "sparse" or len(v) != count:
                    continue
                bound.rebind(v.copy())
                bound.evaluate(check_finite=True)
                actual = (
                    bound.currents,
                    bound.conductances,
                    bound.charges,
                    bound.capacitances,
                )
                assert [a.tobytes() for a in actual] == [a.tobytes() for a in want[:4]]
    finally:
        current.free(handle)
        reference.free(old_handle)


@pytest.mark.parametrize("analysis", ["dc", "tran"])
def test_bound_collapse_and_rebind(threads, series_rc, analysis):
    model = load_osdi_model(
        series_rc.path, analysis=analysis, version=series_rc.osdi_version
    )
    count = 1000
    params = np.full((count, model.num_params), np.nan)
    params[:, model.param_names.index("r")] = np.arange(count) % 2 * 1000
    handle = osdi_setup_batch(model.id, params)
    native = Native()
    # Metadata is used only for dimensions here, not registration.
    from _native_runtime import Metadata

    metadata = Metadata(
        model.id,
        model.num_pins,
        model.num_nodes,
        model.num_params,
        model.num_states,
        4,
        True,
    )
    voltage = np.arange(count * model.num_nodes, dtype=float).reshape(count, -1) / 10000
    with BoundBatch(handle, voltage) as bound:
        bound.qualify_pure_eval()
        for scale in (1.0, 2.0, 0.5):
            bound.voltages[:] = voltage * scale
            want = native.evaluate(handle.handle_id, metadata, bound.voltages, "sparse")
            bound.evaluate(check_finite=True)
            for actual, expected in zip(
                (bound.currents, bound.conductances, bound.charges, bound.capacitances),
                want,
            ):
                assert actual.tobytes() == expected.tobytes()
        with pytest.raises(AttributeError):
            bound.voltages = np.zeros_like(voltage)
        with pytest.raises(AttributeError):
            bound.currents = np.zeros_like(voltage)
        cache = bound.new_cache()
        bound.conductances.fill(123.0)
        saved = bound.conductances.tobytes()
        bound.evaluate(residual_only=True)
        assert bound.conductances.tobytes() == saved
        old_id, old_generation = bound._id, bound.generation
        bound.rebind(np.zeros_like(voltage))
        assert bound.generation > old_generation
        assert _eval(old_id, old_generation, False, False) == -1
        with pytest.raises(ValueError, match="Stale"):
            bound.evaluate_bypass(cache=cache)
        bound.evaluate(check_finite=True)
    assert _eval(old_id, bound.generation, False, False) == -1


def test_bypass_validity_and_exactness(threads):
    model, params = model_params("resistor_va.osdi", 1000)
    handle = osdi_setup_batch(model.id, params)
    set_parallel_policy(1, 16, handle=handle)
    voltage = np.ones((1000, model.num_nodes))
    voltage[:, 1] = 0.0
    with BoundBatch(handle, voltage) as bound:
        assert bound.evaluate() == {"evaluated": 1000}
        cache = bound.new_cache()
        cache.previous_voltages[:] = voltage
        cache.valid[:] = 3
        stats, mask = bound.evaluate_bypass(cache=cache, dry_run=True)
        assert stats == {"evaluated": 1000, "bypassed": 0, "eligible": 1000}
        assert not mask.any()
        with pytest.raises(ValueError, match="qualified"):
            bound.evaluate_bypass(cache=cache)
        bound.qualify_pure_eval()
        cache.generation = bound.generation
        # Exactly half the lanes are idle; invalid/full-vs-residual distinctions
        # further exclude lanes independently of their voltages.
        voltage[::2, 0] += 0.1
        cache.valid[1] = 0
        cache.valid[3] = 1
        before = [a.copy() for a in bound._arrays[1:]]
        stats, mask = bound.evaluate_bypass(cache=cache, check_finite=True)
        assert stats == {"evaluated": 502, "bypassed": 498, "eligible": 498}
        assert not mask[0::2].any()
        for actual, old in zip(bound._arrays[1:], before):
            assert (
                actual[mask.astype(bool)].tobytes() == old[mask.astype(bool)].tobytes()
            )
        assert np.array_equal(cache.previous_voltages[:, 0], np.ones(1000))
        # A residual-only cache covers residual requests; Jacobians stay intact.
        g = bound.conductances.copy()
        stats, _ = bound.evaluate_bypass(cache=cache, residual_only=True)
        assert stats["bypassed"] == 499
        assert bound.conductances.tobytes() == g.tobytes()
        # Exact calls ignore the cache and recompute every lane.
        voltage[5, 0] += 1e-7
        f5 = bound.currents[5].copy()
        stats, _ = bound.evaluate_bypass(cache=cache)
        assert np.array_equal(bound.currents[5], f5)
        assert bound.evaluate()["evaluated"] == 1000
        assert not np.array_equal(bound.currents[5], f5)
        cache.valid[:] = 0
        assert bound.evaluate_bypass(cache=cache)[0]["bypassed"] == 0
        for reltol, vntol in [(-1.0, 1e-6), (np.inf, 1e-6), (1e-3, np.nan)]:
            with pytest.raises(ValueError, match="Invalid cache"):
                bound.evaluate_bypass(cache=cache, reltol=reltol, vntol=vntol)
        invalid = BypassCache(bound.voltages, cache.valid, bound.generation)
        with pytest.raises(ValueError, match="aliasing"):
            bound.evaluate_bypass(cache=invalid)
        bound.currents[1] = np.nan
        cache.valid[1] = 3
        with pytest.raises(ValueError, match="Nonfinite"):
            bound.evaluate_bypass(cache=cache, check_finite=True)


def test_psp_qualification_and_dry_run(threads):
    model, params = model_params("psp103v4_psp103.osdi", 100)
    handle = osdi_setup_batch(model.id, params)
    with BoundBatch(handle, np.zeros((100, model.num_nodes))) as bound:
        bound.evaluate(check_finite=True)
        cache = bound.new_cache()
        cache.valid[:] = 3
        assert bound.evaluate_bypass(cache=cache, dry_run=True)[0]["eligible"] == 100
        # This compiled PSP103 artifact exposes no active ABI state slots.
        assert model.num_states == 0
        bound.qualify_pure_eval()
        cache.generation = bound.generation
        assert bound.evaluate_bypass(cache=cache)[0]["bypassed"] == 100
        # Internal unknowns participate in the tolerance test too.
        bound.voltages[0, -1] += 0.1
        assert bound.evaluate_bypass(cache=cache)[0]["evaluated"] == 1


def test_native_binding_rejects_bad_buffers():
    model, params = model_params("resistor_va.osdi", 2)
    handle = osdi_setup_batch(model.id, params)
    arrays = [
        np.zeros((2, model.num_nodes)),
        np.zeros((2, model.num_nodes + 1)),
        np.zeros((2, model.num_resist_jac)),
        np.zeros((2, model.num_nodes + 1)),
        np.zeros((2, model.num_react_jac)),
    ]
    desc = _Buffers(*(v for a in arrays for v in (a.ctypes.data, a.size)))
    desc.currents = desc.voltages
    assert _bind(handle.handle_id, ct.byref(desc)) == 0
    desc.currents = arrays[1].ctypes.data + 1
    assert _bind(handle.handle_id, ct.byref(desc)) == 0
    desc.currents = arrays[1].ctypes.data
    desc.currents_len -= 1
    assert _bind(handle.handle_id, ct.byref(desc)) == 0
    desc.currents_len += 1
    bid = _bind(handle.handle_id, ct.byref(desc))
    assert bid
    generation = _generation(bid)
    desc.currents_len -= 1
    assert _rebind(bid, ct.byref(desc)) == 0
    assert _generation(bid) == generation
    handle.free()
    assert _eval(bid, _generation(bid), False, False) == -1
    _unbind(bid)


def test_thread_reconfiguration_and_host_concurrency():
    model, params = model_params("resistor_va.osdi", 1000)
    handle = osdi_setup_batch(model.id, params)
    qualify_handle(handle)
    native = Native()
    from _native_runtime import Metadata

    metadata = Metadata(
        model.id, model.num_pins, model.num_nodes, model.num_params, 0, 4, True
    )
    voltage = np.tile([1.0, 0.0], (1000, 1))
    expected = native.evaluate(handle.handle_id, metadata, voltage, "dense")
    env = dict(os.environ)

    def run(index):
        if index % 3 == 0:
            set_num_threads(index % 4 + 1)
        set_parallel_policy(1, 16, handle=handle)
        result = native.evaluate(handle.handle_id, metadata, voltage, "dense")
        assert [a.tobytes() for a in result] == [a.tobytes() for a in expected]

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(run, range(24)))
        assert dict(os.environ) == env
        set_num_threads(2)
        set_num_threads(3)
        assert num_threads() == 3
        with pytest.raises(ValueError):
            set_parallel_policy(1, 0)
        with pytest.raises(ValueError):
            set_parallel_policy(1, 0, handle=handle)
    finally:
        set_num_threads(0)


def test_limiting_provider_refuses_bypass(limiting_binary):
    model = load_osdi_model(str(limiting_binary))
    handle = osdi_setup_batch(model.id, np.full((1, model.num_params), np.nan))
    with BoundBatch(handle, np.zeros((1, model.num_nodes))) as bound:
        with pytest.raises(ValueError, match="ABI-state"):
            bound.qualify_pure_eval()
        with pytest.raises(ValueError, match="ABI-state"):
            qualify_handle(handle)
        assert bound.evaluate() == {"evaluated": 1}
        cache = bound.new_cache()
        cache.valid[:] = 3
        assert bound.evaluate_bypass(cache=cache, dry_run=True)[0]["eligible"] == 1
        with pytest.raises(ValueError, match="qualified"):
            bound.evaluate_bypass(cache=cache)


def test_grounded_bound_outputs(threads, conditional_branch, abi):
    model = load_osdi_model(str(conditional_branch), version=abi[0], analysis="tran")
    count = 1000
    params = np.full((count, model.num_params), np.nan)
    params[:, model.param_names.index("enabled")] = np.arange(count) % 2
    handle = osdi_setup_batch(model.id, params)
    assert (handle.topology.raw_to_representative == -1).any()
    from _native_runtime import Metadata

    metadata = Metadata(
        model.id,
        model.num_pins,
        model.num_nodes,
        model.num_params,
        model.num_states,
        4,
        True,
    )
    voltage = np.tile([1.0, 0.0, 5.0, 5.0], (count, 1))
    with BoundBatch(handle, voltage) as bound:
        bound.qualify_pure_eval()
        want = Native().evaluate(handle.handle_id, metadata, voltage, "sparse")
        bound.evaluate(check_finite=True)
        actual = (bound.currents, bound.conductances, bound.charges, bound.capacitances)
        assert [a.tobytes() for a in actual] == [a.tobytes() for a in want[:4]]
