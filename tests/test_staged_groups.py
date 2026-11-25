"""Owned staged candidates, stale tokens, default parity and public capability registry."""

import ctypes as ct
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from threading import Event

import numpy as np
import pytest

from _native_runtime import Native, PTR, SIZE, U64
from test_checked_groups import Group
import test_checked_groups as groups
from test_embedder_runtime import Context, Error, CALLBACK, Bridge
import test_embedder_runtime as embedder
from test_osdi_runtime import model_params

compiled_delays = groups.compiled_delays
provider = embedder.provider
synthetic_delay = groups.synthetic_delay


class Staging:
    def __init__(self, g):
        self.g = g
        fn = g.n.function
        self.stage = fn(
            "bosdi_stage_lanes_checked",
            ct.c_int,
            U64,
            SIZE,
            PTR,
            SIZE,
            PTR,
            PTR,
            SIZE,
            PTR,
            PTR,
            PTR,
            PTR,
            PTR,
        )
        self.td = fn("bosdi_stage_prepared_delays", ct.c_int, U64, PTR, SIZE)
        self.commit = fn("bosdi_commit_stage", ct.c_int, U64)
        self.discard = fn("bosdi_discard_stage", None, U64)
        self.defaults = fn(
            "bosdi_update_lanes_defaults_checked",
            ct.c_int,
            U64,
            SIZE,
            PTR,
            SIZE,
            PTR,
            PTR,
            SIZE,
            PTR,
            PTR,
            PTR,
            PTR,
        )
        self.tokens = []

    def create(self, lanes, params, analysis=2):
        lanes = np.array(lanes, np.uintp)
        params = np.ascontiguousarray(params, dtype=np.float64)
        x = np.ascontiguousarray(self.g.arrays[0][lanes])
        token = U64(99)
        error = Error()
        status = self.stage(
            self.g.h,
            len(lanes),
            lanes.ctypes.data,
            self.g.model.params,
            params.ctypes.data,
            x.ctypes.data,
            x.size,
            ct.byref(Context(analysis, 3, 0)),
            ct.byref(token),
            ct.byref(error),
            None,
            None,
        )
        if token.value:
            self.tokens.append(token.value)
        return status, token.value, error

    def close(self):
        for token in self.tokens:
            self.discard(token)
            self.discard(token)


def bits(g):
    return [a.tobytes() for a in g.run(2)]


@pytest.fixture(
    params=[
        "waveguide",
        "fixed_delay",
        "variable_delay",
        "mode_delay",
        "reciprocal_delay",
        "duration_voltage",
        "duration_mode",
        "resistor",
        "psp103",
    ]
)
def staged_case(request, compiled_delays):
    name = request.param
    if name in ("resistor", "psp103"):
        filename = "resistor_va.osdi" if name == "resistor" else "psp103v4_psp103.osdi"
        md, p = model_params(filename, 4, "tran")
        # All-NaN defaults are qualified separately; use existing finite card overrides here.
        path = Path(md.path)
        version = 4
        g = Group(path, p, 4, version, 2)
        from osdi_loader import load_osdi_model

        names = load_osdi_model(str(path), version="0.4").param_names
        key = "R" if name == "resistor" else "w"
        changed = p.copy()
        changed[:, [n.lower() for n in names].index(key.lower())] *= np.linspace(
            1.1, 1.4, 4
        )
    else:
        path = compiled_delays / (name + ".osdi")
        version = 5
        from osdi_loader import load_osdi_model

        md = load_osdi_model(str(path), version="0.5")
        p = np.full((4, md.num_params), np.nan)
        key = "length_um" if name == "waveguide" else "tau"
        p[:, md.param_names.index(key)] = 100 if name == "waveguide" else 1
        changed = p.copy()
        changed[:, md.param_names.index(key)] *= np.linspace(1.1, 1.4, 4)
        g = Group(path, p, 4, version, 2)
    yield g, path, version, changed
    g.close()


@pytest.mark.parametrize("lanes", [[3, 1], [0, 1, 2, 3]])
def test_staged_discard_commit_defaults_parity(staged_case, lanes):
    g, path, version, changed = staged_case
    api = Staging(g)
    ref = Group(path, g.params.copy(), g.count, version, 2)
    fresh_params = g.params.copy()
    fresh_params[lanes] = changed[lanes]
    fresh = Group(path, fresh_params, g.count, version, 2)
    try:
        before = bits(g)
        bind = g.n.function("bosdi_bind_buffers", U64, U64, PTR)
        generation = g.n.function("bosdi_binding_generation", U64, U64)
        unbind = g.n.function("bosdi_unbind_buffers", None, U64)
        binding = bind(g.h, ct.byref(g.b))
        assert binding
        try:
            old_generation = generation(binding)
            old_observed = g.observed(g.h)
            status, token, error = api.create(lanes, changed[lanes])
            assert status == 0, (error.status, error.lane, error.message)
            assert generation(binding) == old_generation
            assert g.observed(g.h) == old_observed
            td = np.empty((len(lanes), len(g.delays)))
            assert api.td(token, td.ctypes.data, td.size) == 0
            assert fresh.prepared(fresh.h, fresh.td.ctypes.data, fresh.td.size) == 0
            np.testing.assert_array_equal(td, fresh.td[lanes])
            assert bits(g) == before  # accepted evaluation while a candidate exists
            api.discard(token)
            assert api.commit(token) == -1
            assert generation(binding) == old_generation
            assert bits(g) == before
            status, token, error = api.create(lanes, changed[lanes])
            assert status == 0, error.message
            assert api.commit(token) == 0
            assert api.commit(token) == -1
            assert bits(g) == bits(fresh)
            # Defaults-aware immediate route is the same shared preparation/publication.
            ref.update = api.defaults
            assert ref.change(lanes, changed[lanes])[0] == 0
            assert bits(g) == bits(ref)
            bound = g.n.function(
                "bosdi_eval_bound_checked",
                ct.c_int,
                U64,
                U64,
                PTR,
                ct.c_bool,
                ct.c_bool,
                PTR,
                PTR,
                PTR,
            )
            e = Error()
            assert (
                bound(
                    binding,
                    old_generation,
                    ct.byref(Context(2, 3, 0)),
                    False,
                    True,
                    ct.byref(e),
                    None,
                    None,
                )
                == -1
            )
        finally:
            unbind(binding)
    finally:
        api.close()
        ref.close()
        fresh.close()


def test_stale_stages_multigroup_and_handle_free(compiled_delays):
    path = compiled_delays / "fixed_delay.osdi"
    a = Group(path, count=4, analysis=2)
    b = Group(path, count=4, analysis=2)
    aa, bb = Staging(a), Staging(b)
    try:
        original_b = bits(b)
        p = np.full_like(a.params, np.nan)
        from osdi_loader import load_osdi_model

        tau = load_osdi_model(str(path), version="0.5").param_names.index("tau")
        p[:, tau] = 2
        _, first, _ = aa.create([1], p[[1]])
        _, other, _ = aa.create([2], p[[2]])
        _, second, _ = bb.create([1], p[[1]])
        assert aa.commit(first) == 0
        accepted = bits(a)
        assert aa.commit(other) == -9
        assert aa.td(other, None, 0) == -9
        assert bits(a) == accepted
        bb.discard(second)
        assert bits(b) == original_b
        _, token, _ = aa.create([0], p[[0]])
        assert a.change([0], np.ones((1, a.model.params)))[0] == 0
        after_update = bits(a)
        assert aa.commit(token) == -9
        assert bits(a) == after_update
        _, token, _ = aa.create([0], p[[0]])
        a.n.free(a.h)
        assert aa.commit(token) == -9
        assert aa.td(token, None, 0) == -9
        aa.discard(token)
        aa.discard(token)
    finally:
        aa.close()
        bb.close()
        a.close()
        b.close()


def test_stage_failure_and_legacy_nan_contract(provider):
    b = Bridge(provider)
    try:
        _, h, _ = b.create(r=(1000,) * 3, c=(1e-9,) * 3)
        g = Group.__new__(Group)
        g.h = h
        g.n = b.native
        g.model = b.model
        g.arrays, g.b = b.buffers(h, 3)
        g.update = b.native.function(
            "bosdi_update_lanes_checked",
            ct.c_int,
            U64,
            SIZE,
            PTR,
            SIZE,
            PTR,
            PTR,
            SIZE,
            PTR,
            PTR,
            PTR,
            PTR,
        )
        api = Staging(g)
        try:
            before = [
                x.tobytes()
                for x in b.native.evaluate(h, b.model, g.arrays[0], "sparse")
            ]
            for p, want in [
                ([[1000, 1e-9], [-1, 1e-9], [1000, 1e-9]], -5),
                ([[1000, 1e-9], [42, 1e-9], [1000, 1e-9]], -7),
                ([[1000, 1e-9], [np.inf, 1e-9], [1000, 1e-9]], -4),
            ]:
                status, token, error = api.create([0, 1, 2], p)
                assert (status, token, error.lane) == (want, 0, 1)
                assert [
                    x.tobytes()
                    for x in b.native.evaluate(h, b.model, g.arrays[0], "sparse")
                ] == before
            status, token, error = api.create([2, 0], [[-1, 1e-9], [-2, 1e-9]])
            assert (status, token, error.lane) == (-5, 0, 0)
            assert (
                g.change([0], [[np.nan, 1e-9]])[0] == -4
            )  # existing candidate semantics retained
        finally:
            api.close()
    finally:
        b.close()


def test_stage_while_eval_and_busy_retry(provider):
    b = Bridge(provider)
    entered, release = Event(), Event()
    try:
        _, h, _ = b.create(r=(1000, -4, 1000), c=(1e-9,) * 3)
        arrays, buffers = b.buffers(h, 3)
        g = Group.__new__(Group)
        g.h = h
        g.n = b.native
        g.model = b.model
        g.arrays = arrays
        api = Staging(g)

        @CALLBACK
        def callback(user, lane, message, level):
            entered.set()
            assert release.wait(10)

        try:
            with ThreadPoolExecutor(1) as pool:
                f = pool.submit(
                    b.evaluate, h, buffers, Context(2, 3, 0), False, False, callback
                )
                try:
                    assert entered.wait(10)
                    # Creation shares the accepted read lock and may run in flight.
                    status, token, error = api.create([1], [[2000, 1e-9]])
                    assert status == 0, error.message
                    assert api.commit(token) == -8
                finally:
                    release.set()
                assert f.result()[0] == -6
            assert api.commit(token) == 0
            assert b.evaluate(h, buffers, Context(2, 3, 0))[0] == 0
        finally:
            api.close()
    finally:
        b.close()


def test_registry_equals_documented_enabled_bits():
    header = (Path(__file__).parent.parent / "include/bosdi_runtime.h").read_text()
    assigned = dict(
        (name, int(value))
        for name, value in re.findall(
            r"^#define (BOSDI_CAP_\w+) (\d+)ull$", header, re.M
        )
    )
    assert assigned == {
        "BOSDI_CAP_DELAY_OBSERVATION": 1,
        "BOSDI_CAP_OBSERVED_WORKERS": 2,
        "BOSDI_CAP_TRANSACTIONAL_SETUP": 4,
        "BOSDI_CAP_STAGED_SETUP": 8,
        "BOSDI_CAP_NAN_DEFAULT_SETUP": 16,
        "BOSDI_CAP_EXPLICIT_CONTEXT": 64,
    }
    native = Native()
    mask = 0
    for value in assigned.values():
        mask |= value
    assert native.function("bosdi_runtime_capabilities", ct.c_uint64)() == mask == 95
    for symbol in [
        "bosdi_stage_lanes_checked",
        "bosdi_stage_prepared_delays",
        "bosdi_commit_stage",
        "bosdi_discard_stage",
        "bosdi_update_lanes_defaults_checked",
    ]:
        assert symbol in native.addresses
    assert mask & 32 == 0


def test_signed_zero_is_explicit_parameter(compiled_delays):
    path = compiled_delays / "waveguide.osdi"
    from osdi_loader import load_osdi_model

    md = load_osdi_model(str(path), version="0.5")
    p = np.full((2, md.num_params), np.nan)
    p[:, md.param_names.index("loss_dB_cm")] = -0.0
    g = Group(path, count=2, analysis=2)
    fresh = Group(path, p, 2, analysis=2)
    api = Staging(g)
    try:
        before = bits(g)
        status, token, error = api.create([0, 1], p)
        assert status == 0, error.message
        assert api.commit(token) == 0
        assert bits(g) == bits(fresh)
        assert bits(g) != before  # treating -0.0 as default loss=2 would be wrong
    finally:
        api.close()
        g.close()
        fresh.close()


@pytest.mark.parametrize("name", ["waveguide", "resistor", "psp103"])
def test_all_nan_defaults_match_fresh_and_immediate(compiled_delays, name):
    if name == "waveguide":
        path = compiled_delays / "waveguide.osdi"
        version = 5
    else:
        path = (
            Path(__file__).parent
            / "compiled_osdi"
            / ("resistor.osdi" if name == "resistor" else "psp103v4_psp103.osdi")
        )
        version = 4
    default = Group(path, count=4, version=version, analysis=2)
    from osdi_loader import load_osdi_model

    md = load_osdi_model(str(path), version=f"0.{version}")
    p = default.params.copy()
    p[:, md.param_names.index("$mfactor")] = 2
    g = Group(path, p, 4, version, 2)
    api = Staging(g)
    try:
        status, token, error = api.create(range(4), default.params)
        assert status == 0, error.message
        assert api.commit(token) == 0
        assert bits(g) == bits(default)
        maps = np.empty((4, g.model.nodes), np.int32)
        expected = np.empty_like(maps)
        assert g.maps(g.h, 4, g.model.nodes, maps.ctypes.data) == 0
        assert (
            default.maps(default.h, 4, default.model.nodes, expected.ctypes.data) == 0
        )
        assert maps.tobytes() == expected.tobytes()
        g.update = api.defaults
        assert g.change(range(4), p)[0] == 0
        assert g.change(range(4), default.params)[0] == 0
        assert bits(g) == bits(default)
    finally:
        api.close()
        g.close()
        default.close()


def test_invalid_default_outputs_do_not_return_token():
    path = Path(__file__).parent / "compiled_osdi/resistor_va.osdi"
    md, p = model_params("resistor_va.osdi", 4, "tran")
    g = Group(path, p, 4, 4, 2)
    api = Staging(g)
    try:
        before = bits(g)
        status, token, error = api.create(range(4), np.full_like(p, np.nan))
        assert (status, token, error.lane) == (
            -4,
            0,
            0,
        )  # provider default R=0 is not numerically valid
        assert bits(g) == before
    finally:
        api.close()
        g.close()


def test_finite_stage_matches_original_immediate(compiled_delays):
    from osdi_loader import load_osdi_model

    path = compiled_delays / "waveguide.osdi"
    md = load_osdi_model(str(path), version="0.5")
    values = {
        "$mfactor": 1,
        "r0": 1,
        "length_um": 150,
        "neff": 2.31,
        "n_group": 4.34,
        "loss_dB_cm": 2,
        "wavelength_nm": 1550,
    }
    p = np.array([[values[n] for n in md.param_names]] * 4)
    a = Group(path, count=4, analysis=2)
    b = Group(path, count=4, analysis=2)
    api = Staging(a)
    try:
        for lanes in ([3, 1], [0, 1, 2, 3]):
            status, token, error = api.create(lanes, p[lanes])
            assert status == 0, error.message
            assert api.commit(token) == 0
            assert b.change(lanes, p[lanes])[0] == 0
            assert bits(a) == bits(b)
            p[:, md.param_names.index("length_um")] += 25
    finally:
        api.close()
        a.close()
        b.close()


@pytest.mark.skipif(
    os.name != "posix", reason="POSIX permits renaming a loaded shared library"
)
def test_stage_does_not_reopen_library(synthetic_delay):
    path = synthetic_delay()
    g = Group(path, np.array([[1000, 1e-9]] * 4), 4)
    api = Staging(g)
    try:
        path.rename(path.with_suffix(".moved"))
        before = bits(g)
        for _ in range(30):
            status, token, error = api.create([3, 1], [[2000, 1e-9], [3000, 1e-9]])
            assert status == 0, error.message
            api.discard(token)
        assert bits(g) == before
    finally:
        api.close()
        g.close()
