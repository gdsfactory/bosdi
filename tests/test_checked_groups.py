"""Compiled OSDI 0.5 delay observation and transactional checked updates."""

import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from _native_runtime import Native, PTR, SIZE, U64
from osdi_runtime import _Buffers
from test_embedder_runtime import Context, Error, Bridge
import test_embedder_runtime as embedder

provider = embedder.provider

ROOT = Path(__file__).parent


class Delay(ct.Structure):
    _fields_ = [("y", ct.c_uint32), ("z", ct.c_uint32), ("maxdelay", ct.c_uint32)]


def test_jacobian_pairs_available_without_exported_symbols():
    native = Native()
    assert native.addresses["osdi_handle_jac_pairs_ffi"]
    pairs = native.function(
        "osdi_handle_jac_pairs_ffi", ct.c_int, U64, ct.c_bool, PTR, SIZE
    )
    assert pairs(0, False, None, 0) == -1


class Group:
    def __init__(self, path, params=None, count=32, version=5, analysis=0):
        self.n = Native()
        fn = self.n.function
        self.model = self.n.load(str(path).encode(), version, 300, 0)
        assert self.model.success
        self.count = count
        self.setup = fn(
            "bosdi_setup_batch_delays_checked",
            ct.c_int,
            ct.c_uint32,
            SIZE,
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
        self.metadata = fn("bosdi_model_delays", ct.c_int, ct.c_uint32, PTR, SIZE, PTR)
        self.prepared = fn("bosdi_handle_prepared_delays", ct.c_int, U64, PTR, SIZE)
        self.eval = fn(
            "bosdi_eval_handle_delays_checked",
            ct.c_int,
            U64,
            PTR,
            PTR,
            ct.c_bool,
            ct.c_bool,
            PTR,
            SIZE,
            PTR,
            PTR,
            PTR,
        )
        self.update = fn(
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
        self.observed = fn("bosdi_handle_observed_workers", SIZE, U64)
        self.policy = fn("bosdi_set_handle_parallel_policy", ct.c_bool, U64, SIZE, SIZE)
        self.threads = fn("bosdi_set_num_threads", ct.c_bool, SIZE)
        self.maps = fn("bosdi_handle_node_maps", ct.c_int, U64, SIZE, SIZE, PTR)
        self.pairs = fn(
            "osdi_handle_jac_pairs_ffi", ct.c_int, U64, ct.c_bool, PTR, SIZE
        )
        self.params = (
            np.full((count, self.model.params), np.nan) if params is None else params
        )
        h, e = U64(), Error()
        v = np.full((count, self.model.nodes), 0.25)
        assert (
            self.setup(
                self.model.id,
                count,
                self.model.params,
                self.params.ctypes.data,
                v.ctypes.data,
                v.size,
                ct.byref(Context(analysis, 3, 0)),
                ct.byref(h),
                ct.byref(e),
                None,
                None,
            )
            == 0
        ), e.message
        self.h = h.value
        num = SIZE()
        assert self.metadata(self.model.id, None, 0, ct.byref(num)) == 0
        self.delays = (Delay * num.value)()
        assert self.metadata(self.model.id, self.delays, num, ct.byref(num)) == 0
        self.td = np.empty((count, num.value))
        self.arrays = [
            np.full((count, self.model.nodes), 0.25),
            np.zeros((count, self.model.nodes + 1)),
            np.zeros((count, self.n.jac_count(self.h, False))),
            np.zeros((count, self.model.nodes + 1)),
            np.zeros((count, self.n.jac_count(self.h, True))),
        ]
        self.b = _Buffers(*(v for a in self.arrays for v in (a.ctypes.data, a.size)))

    def close(self):
        self.n.free(self.h)
        self.threads(0)
        assert self.n.function("bosdi_free_model", ct.c_bool, ct.c_uint32)(
            self.model.id
        )

    def run(self, analysis=0, residual=False, finite=True):
        e = Error()
        status = self.eval(
            self.h,
            ct.byref(self.b),
            ct.byref(Context(analysis, 3, 0)),
            residual,
            finite,
            self.td.ctypes.data,
            self.td.size,
            ct.byref(e),
            None,
            None,
        )
        assert status == 0, (status, e.lane, e.message)
        return [a.copy() for a in self.arrays[1:]] + [self.td.copy()]

    def change(self, lanes, params, analysis=2):
        lanes = np.array(lanes, dtype=np.uintp)
        params = np.ascontiguousarray(params, dtype=np.float64)
        v = np.ascontiguousarray(self.arrays[0][lanes])
        e = Error()
        status = self.update(
            self.h,
            len(lanes),
            lanes.ctypes.data,
            self.model.params,
            params.ctypes.data,
            v.ctypes.data,
            v.size,
            ct.byref(Context(analysis, 3, 0)),
            ct.byref(e),
            None,
            None,
        )
        return status, e


@pytest.fixture(scope="module")
def compiled_delays(tmp_path_factory):
    requested = os.environ.get("BOSDI_OPENVAF_05")
    compiler = requested or shutil.which("openvaf-r")
    if not compiler:
        pytest.skip("BOSDI_OPENVAF_05 / openvaf-r required")
    dest = tmp_path_factory.mktemp("checked-delays")
    probe = subprocess.run(
        [
            compiler,
            str(ROOT / "devices/delays/fixed_delay.va"),
            "-o",
            str(dest / "probe.osdi"),
        ],
        capture_output=True,
    )
    if probe.returncode != 0:
        if requested:
            pytest.fail(probe.stderr.decode(errors="replace"))
        pytest.skip("OSDI 0.5 absdelay compiler required; set BOSDI_OPENVAF_05")
    artifact = ct.CDLL(str(dest / "probe.osdi"))
    if ct.c_uint32.in_dll(artifact, "OSDI_VERSION_MINOR").value != 5:
        if requested:
            pytest.fail("Explicit delay compiler must emit OSDI 0.5")
        pytest.skip("OSDI 0.5 compiler required; set BOSDI_OPENVAF_05")
    for src in (ROOT / "devices/delays").glob("*.va"):
        subprocess.run(
            [compiler, str(src), "-o", str(dest / (src.stem + ".osdi"))],
            check=True,
            capture_output=True,
        )
    return dest


@pytest.mark.parametrize(
    "name",
    [
        "fixed_delay",
        "variable_delay",
        "mode_delay",
        "reciprocal_delay",
        "waveguide",
        "duration_voltage",
        "duration_mode",
    ],
)
@pytest.mark.parametrize("residual", [False, True])
def test_compiled_delay_observation(compiled_delays, name, residual):
    g = Group(compiled_delays / (name + ".osdi"))
    try:
        assert len(g.delays) == (
            4 if name == "waveguide" else 2 if name == "mode_delay" else 1
        )
        assert bool(g.delays[0].maxdelay) == (
            name in ["variable_delay", "duration_voltage"]
        )
        prepared = np.empty_like(g.td)
        assert g.prepared(g.h, prepared.ctypes.data, prepared.size) == 0
        assert np.isfinite(prepared).all()
        maps = np.empty((g.count, g.model.nodes), np.int32)
        assert g.maps(g.h, g.count, g.model.nodes, maps.ctypes.data) == 0
        pairs = np.empty((g.count, g.n.jac_count(g.h, False), 2), np.int32)
        assert g.pairs(g.h, False, pairs.ctypes.data, pairs.size) == 0
        for d in g.delays:
            assert g.model.pins <= d.y < g.model.nodes
            assert g.model.pins <= d.z < g.model.nodes
            np.testing.assert_array_equal(maps[:, d.y], d.y)
            np.testing.assert_array_equal(maps[:, d.z], d.z)
            assert (pairs == d.y).any() and (pairs == d.z).any()
            assert (pairs[:, :, 0] == d.y).any()
            assert (pairs[:, :, 1] == d.z).any()
        g.arrays[0][:, 0] = np.linspace(0.1, 0.9, g.count)
        baseline = {}
        assert g.policy(g.h, 1, 1)
        for workers in [1, 2, 4]:
            assert g.threads(workers)
            for analysis in [0, 1, 2]:
                for _ in range(3):
                    actual = g.run(analysis, residual)
                    if workers == 1:
                        baseline[analysis] = [a.tobytes() for a in actual]
                    else:
                        assert [a.tobytes() for a in actual] == baseline[analysis]
                    assert 1 <= g.observed(g.h) <= workers
                    expected = (
                        1 + g.arrays[0][:, 0]
                        if name == "duration_voltage"
                        else 2
                        if name == "duration_mode" and analysis == 2
                        else 1
                    )
                    if name not in ["waveguide", "mode_delay"]:
                        np.testing.assert_array_equal(g.td[:, 0], expected)
                    if name in [
                        "fixed_delay",
                        "variable_delay",
                        "reciprocal_delay",
                        "waveguide",
                        "mode_delay",
                    ]:
                        np.testing.assert_array_equal(g.td, prepared)
        # y has the provider equation; z is a host-supplied delayed value.
        # Its allocated raw residual row is zero, without a fabricated history law.
        before = g.run(2, residual)[0]
        for d in g.delays:
            np.testing.assert_array_equal(before[:, d.z], 0)
            g.arrays[0][:, d.y] += 1
            after = g.run(2, residual)[0]
            np.testing.assert_allclose(after[:, d.y] - before[:, d.y], -1, atol=1e-15)
            g.arrays[0][:, d.y] -= 1
            g.arrays[0][:, d.z] += 1
            after = g.run(2, residual)[0]
            np.testing.assert_array_equal(after[:, d.z], 0)
            assert np.any(after != before)
            g.arrays[0][:, d.z] -= 1
    finally:
        g.close()


def test_delay_length_update_and_fresh_parity(compiled_delays):
    path = compiled_delays / "waveguide.osdi"
    # Resolve descriptor-order names without assuming compiler order.
    from osdi_loader import load_osdi_model

    md = load_osdi_model(str(path), version="0.5")
    values = {
        "$mfactor": 1,
        "r0": 1,
        "length_um": 100,
        "neff": 2.31,
        "n_group": 4.34,
        "loss_dB_cm": 2,
        "wavelength_nm": 1550,
    }
    p = np.array([[values[n] for n in md.param_names]] * 4)
    length = md.param_names.index("length_um")
    p[:, length] = [50, 100, 150, 200]
    g = Group(path, p.copy(), 4)
    try:
        p[[1, 3], length] = [250, 400]
        assert g.change([3, 1], p[[3, 1]])[0] == 0
        prepared = np.empty_like(g.td)
        assert g.prepared(g.h, prepared.ctypes.data, prepared.size) == 0
        np.testing.assert_array_equal(
            prepared[:, 0], p[:, length] * 1e-6 * 4.34 / 299792458.0
        )
        np.testing.assert_array_equal(prepared, np.repeat(prepared[:, :1], 4, axis=1))
        fresh = Group(path, p.copy(), 4)
        try:
            assert [a.tobytes() for a in g.run(2)] == [
                a.tobytes() for a in fresh.run(2)
            ]
        finally:
            fresh.close()
        p[:, length] += 50
        assert g.change(range(4), p)[0] == 0
        fresh = Group(path, p.copy(), 4)
        try:
            assert [a.tobytes() for a in g.run(2)] == [
                a.tobytes() for a in fresh.run(2)
            ]
        finally:
            fresh.close()
    finally:
        g.close()


def test_update_rollback_collapse_and_binding(provider):
    b = Bridge(provider)
    try:
        _, h, _ = b.create(r=(1000, 1000, 1000), c=(1e-9,) * 3)
        g = Group.__new__(Group)
        g.n = b.native
        g.model = b.model
        g.h = h
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
        before = b.native.evaluate(h, b.model, g.arrays[0], "sparse")
        binding = b.bind(h, ct.byref(g.b))
        b.bindings.append(binding)
        generation = b.generation(binding)
        for params, want in [
            ([[2000, 1e-9], [-1, 1e-9], [3000, 1e-9]], -5),
            ([[2000, 1e-9], [42, 1e-9], [3000, 1e-9]], -7),
        ]:
            status, e = g.change([0, 1, 2], params)
            assert (status, e.lane) == (want, 1)
            assert b.generation(binding) == generation
            assert [
                a.tobytes()
                for a in b.native.evaluate(h, b.model, g.arrays[0], "sparse")
            ] == [a.tobytes() for a in before]
        assert g.change([1], [[2000, 1e-9]])[0] == 0
        error = Error()
        assert (
            b.bound(
                binding,
                generation,
                ct.byref(Context(2, 3, 0)),
                False,
                True,
                ct.byref(error),
                None,
                None,
            )
            == -1
        )
        rebind = b.native.function("bosdi_rebind_buffers", U64, U64, PTR)
        new = rebind(binding, ct.byref(g.b))
        assert new and new != generation
        assert (
            b.bound(
                binding,
                new,
                ct.byref(Context(2, 3, 0)),
                False,
                True,
                ct.byref(error),
                None,
                None,
            )
            == 0
        )
        assert g.arrays[1][1, 0] == 0.0005
    finally:
        b.close()


@pytest.fixture
def synthetic_delay(tmp_path):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler required")

    def compile(flags=()):
        target = tmp_path / (str(len(list(tmp_path.iterdir()))) + ".osdi")
        subprocess.run(
            [
                compiler,
                "-shared",
                "-fPIC",
                "-DDELAY_FIXTURE",
                *flags,
                "-I" + str(ROOT / "headers"),
                str(ROOT / "fixtures/embedder.c"),
                "-o",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
        return target

    return compile


@pytest.mark.parametrize(
    "flags", [["-DDELAY_Y=0"], ["-DDELAY_Z=2"], ["-DDELAY_TD=UINT32_MAX"]]
)
def test_unsupported_delay_layout(synthetic_delay, flags):
    n = Native()
    md = n.load(str(synthetic_delay(flags)).encode(), 5, 300, 0)
    assert md.success  # legacy registration semantics preserved
    count = SIZE()
    query = n.function("bosdi_model_delays", ct.c_int, ct.c_uint32, PTR, SIZE, PTR)
    assert query(md.id, None, 0, ct.byref(count)) == -7
    assert n.function("bosdi_free_model", ct.c_bool, ct.c_uint32)(md.id)


def test_raw_nonfinite_td_and_setup_storage(synthetic_delay):
    g = Group(synthetic_delay(), np.array([[1000, 1e-9]] * 32), 32)
    try:
        prepared = np.empty_like(g.td)
        assert g.prepared(g.h, prepared.ctypes.data, prepared.size) == 0
        np.testing.assert_array_equal(prepared, 1000)
        assert g.policy(g.h, 1, 1)
        for workers in [1, 2, 4]:
            assert g.threads(workers)
            for analysis in [0, 1, 2]:
                for residual in [False, True]:
                    e = Error()
                    assert (
                        g.eval(
                            g.h,
                            ct.byref(g.b),
                            ct.byref(Context(analysis, 3, 7)),
                            residual,
                            True,
                            g.td.ctypes.data,
                            g.td.size,
                            ct.byref(e),
                            None,
                            None,
                        )
                        == 0
                    )
                    assert np.isnan(g.td).all()  # finite F/Q/G/C, raw nonfinite td
                    assert g.prepared(g.h, prepared.ctypes.data, prepared.size) == 0
                    np.testing.assert_array_equal(prepared, 1000)
        # v1 setup reads the actual provider setup slot, without preparation eval.
        setup = g.n.function(
            "bosdi_setup_batch_checked",
            ct.c_int,
            ct.c_uint32,
            SIZE,
            SIZE,
            PTR,
            PTR,
            PTR,
            PTR,
            PTR,
        )
        h = U64()
        e = Error()
        assert (
            setup(
                g.model.id,
                32,
                2,
                g.params.ctypes.data,
                ct.byref(h),
                ct.byref(e),
                None,
                None,
            )
            == 0
        )
        try:
            assert g.prepared(h, prepared.ctypes.data, prepared.size) == 0
            np.testing.assert_array_equal(prepared, 2000)
        finally:
            g.n.free(h)
    finally:
        g.close()


def test_delay_collapse_rejected(synthetic_delay):
    g = Group(synthetic_delay(), np.array([[1000, 1e-9]] * 4), 4)
    try:
        before = [a.tobytes() for a in g.run()]
        status, e = g.change([1], [[42, 1e-9]])
        assert (status, e.lane) == (-7, 1)
        assert [a.tobytes() for a in g.run()] == before
    finally:
        g.close()


def test_update_busy_and_lowest_lane(provider):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from test_embedder_runtime import CALLBACK

    b = Bridge(provider)
    entered, release = Event(), Event()
    try:
        _, h, _ = b.create(r=(1000, -4, 1000), c=(1e-9,) * 3)
        arrays, buffers = b.buffers(h, 3)
        update = b.native.function(
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
        lanes = np.array([2, 0, 1], np.uintp)
        params = np.array([[1000, 1e-9]] * 3)
        e = Error()

        @CALLBACK
        def callback(user, lane, message, level):
            entered.set()
            assert release.wait(10)

        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(
                b.evaluate, h, buffers, Context(2, 3, 0), False, False, callback
            )
            try:
                assert entered.wait(10)
                assert (
                    update(
                        h,
                        3,
                        lanes.ctypes.data,
                        2,
                        params.ctypes.data,
                        arrays[0].ctypes.data,
                        arrays[0].size,
                        ct.byref(Context(2, 3, 0)),
                        ct.byref(e),
                        None,
                        None,
                    )
                    == -8
                )
            finally:
                release.set()
            assert future.result()[0] == -6
        params[:, 0] = [-1, -2, -1]
        assert (
            update(
                h,
                3,
                lanes.ctypes.data,
                2,
                params.ctypes.data,
                arrays[0].ctypes.data,
                arrays[0].size,
                ct.byref(Context(2, 3, 0)),
                ct.byref(e),
                None,
                None,
            )
            == -5
        )
        assert e.lane == 0  # lane-list order cannot choose the error
    finally:
        b.close()


def test_delay_buffer_validation_and_worker_reset(compiled_delays):
    g = Group(compiled_delays / "fixed_delay.osdi")
    try:
        assert g.n.function("bosdi_runtime_capabilities", ct.c_uint64)() & 7 == 7
        assert g.threads(1)
        g.run()
        assert g.observed(g.h) == 1
        e = Error()
        assert (
            g.eval(
                g.h,
                ct.byref(g.b),
                ct.byref(Context(0, 3, 0)),
                False,
                True,
                g.arrays[1].ctypes.data,
                g.td.size,
                ct.byref(e),
                None,
                None,
            )
            == -2
        )
        assert g.observed(g.h) == 0
        assert (
            g.eval(
                g.h,
                ct.byref(g.b),
                ct.byref(Context(0, 3, 0)),
                False,
                True,
                None,
                0,
                ct.byref(e),
                None,
                None,
            )
            == 0
        )
        assert g.observed(g.h) == 1
        assert (
            g.eval(
                g.h,
                ct.byref(g.b),
                ct.byref(Context(0, 3, 0)),
                False,
                True,
                g.td.ctypes.data,
                g.td.size - 1,
                ct.byref(e),
                None,
                None,
            )
            == -2
        )
    finally:
        g.close()


def test_mutable_provider_restores_both_snapshots(synthetic_delay):
    g = Group(
        synthetic_delay(["-DMUTATING_FIXTURE"]), np.array([[1000, 1e-9]] * 128), 128
    )
    try:
        assert g.policy(g.h, 1, 1)
        baseline = None
        for workers in [1, 2, 4]:
            assert g.threads(workers)
            for _ in range(5):
                actual = [a.tobytes() for a in g.run(2)]
                if baseline is None:
                    baseline = actual
                else:
                    assert actual == baseline
        np.testing.assert_array_equal(g.td, 1001)
    finally:
        g.close()
