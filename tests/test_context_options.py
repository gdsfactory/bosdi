"""Explicit shared-library CG3 regressions; no Python extension is needed."""

import ctypes as ct
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).parent
P = ct.c_void_p
S = ct.c_size_t
U = ct.c_uint64


class Metadata(ct.Structure):
    _fields_ = [
        ("id", ct.c_uint32),
        ("pins", S),
        ("nodes", S),
        ("params", S),
        ("states", S),
        ("version", ct.c_uint32),
        ("success", ct.c_bool),
    ]


class Context(ct.Structure):
    _fields_ = [
        ("analysis", ct.c_uint32),
        ("options", ct.c_uint32),
        ("time", ct.c_double),
    ]


class Error(ct.Structure):
    _fields_ = [("status", ct.c_int32), ("lane", S), ("message", ct.c_char * 256)]


class Buffers(ct.Structure):
    _fields_ = [
        (name, kind)
        for name in ("x", "f", "g", "q", "c")
        for kind, name in ((P, name), (S, name + "_len"))
    ]


@pytest.fixture
def runtime():
    path = os.environ.get("BOSDI_TEST_LIBRARY")
    if not path:
        pytest.skip("explicit BOSDI_TEST_LIBRARY required")
    lib = ct.CDLL(path)

    def fn(name, result, *args):
        f = getattr(lib, name)
        f.restype = result
        f.argtypes = args
        return f

    assert fn("bosdi_runtime_capabilities", U)() & 64
    assert fn("bosdi_runtime_context_options", ct.c_uint32)() == 15
    return fn


def group(fn, path, delays=False, options=3):
    model = fn(
        "load_osdi_library_with_analysis",
        Metadata,
        ct.c_char_p,
        ct.c_uint32,
        ct.c_double,
        ct.c_uint32,
    )(str(path).encode(), 5, 300.0, 0)
    assert model.success
    params = np.full(model.params, np.nan)
    x = np.zeros(model.nodes)
    h = U()
    e = Error()
    if delays:
        status = fn(
            "bosdi_setup_batch_delays_checked",
            ct.c_int32,
            ct.c_uint32,
            S,
            S,
            P,
            P,
            S,
            P,
            P,
            P,
            P,
            P,
        )(
            model.id,
            1,
            model.params,
            params.ctypes.data,
            x.ctypes.data,
            x.size,
            ct.byref(Context(0, options, 0)),
            ct.byref(h),
            ct.byref(e),
            None,
            None,
        )
    else:
        status = fn(
            "bosdi_setup_batch_checked", ct.c_int32, ct.c_uint32, S, S, P, P, P, P, P
        )(
            model.id,
            1,
            model.params,
            params.ctypes.data,
            ct.byref(h),
            ct.byref(e),
            None,
            None,
        )
    if status:
        assert fn("bosdi_free_model", ct.c_bool, ct.c_uint32)(model.id)
        return status, None
    count = fn("osdi_handle_jac_count_ffi", S, U, ct.c_bool)
    arrays = [
        x,
        np.zeros(model.nodes + 1),
        np.zeros(count(h, False)),
        np.zeros(model.nodes + 1),
        np.zeros(count(h, True)),
    ]
    buffers = Buffers(*(v for a in arrays for v in (a.ctypes.data, a.size)))
    return 0, (model, h, params, arrays, buffers)


def close(fn, g):
    model, h, *_ = g
    fn("osdi_free_handle_ffi", None, U)(h)
    assert fn("bosdi_free_model", ct.c_bool, ct.c_uint32)(model.id)


def test_ac_static_is_preserved_by_default_and_explicitly_omitted(runtime, tmp_path):
    fn = runtime
    target = tmp_path / "context.osdi"
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-DCONTEXT_FLAGS_FIXTURE",
            "-I",
            str(ROOT / "headers"),
            str(ROOT / "fixtures/embedder.c"),
            "-o",
            str(target),
        ],
        check=True,
    )
    status, g = group(fn, target)
    assert status == 0
    try:
        _, h, _, arrays, buffers = g
        evaluate = fn(
            "bosdi_eval_handle_checked",
            ct.c_int32,
            U,
            P,
            P,
            ct.c_bool,
            ct.c_bool,
            P,
            P,
            P,
        )
        e = Error()
        assert (
            evaluate(
                h,
                ct.byref(buffers),
                ct.byref(Context(1, 3, 0)),
                False,
                True,
                ct.byref(e),
                None,
                None,
            )
            == 0
        )
        assert arrays[1][0] == 42.0
        assert (
            evaluate(
                h,
                ct.byref(buffers),
                ct.byref(Context(1, 7, 0)),
                False,
                True,
                ct.byref(e),
                None,
                None,
            )
            == 0
        )
        assert arrays[1][0] == 0.0
        assert (
            evaluate(
                h,
                ct.byref(buffers),
                ct.byref(Context(0, 7, 0)),
                False,
                True,
                ct.byref(e),
                None,
                None,
            )
            == -2
        )
    finally:
        close(fn, g)


def test_reciprocal_delay_input_scope_and_full_physical_validation(runtime, tmp_path):
    fn = runtime
    compiler = os.environ.get("BOSDI_OPENVAF_05")
    if not compiler:
        pytest.skip("explicit OSDI 0.5 compiler required")
    target = tmp_path / "reciprocal.osdi"
    subprocess.run(
        [compiler, str(ROOT / "devices/delays/reciprocal_delay.va"), "-o", str(target)],
        check=True,
    )
    assert group(fn, target, True, 3)[0] == -4
    status, g = group(fn, target, True, 11)
    assert status == 0
    try:
        _, h, params, arrays, buffers = g
        evaluate = fn(
            "bosdi_eval_handle_delays_checked",
            ct.c_int32,
            U,
            P,
            P,
            ct.c_bool,
            ct.c_bool,
            P,
            S,
            P,
            P,
            P,
        )
        td = np.zeros(1)
        e = Error()

        def call(options, residual=False):
            return evaluate(
                h,
                ct.byref(buffers),
                ct.byref(Context(0, options, 0)),
                residual,
                True,
                td.ctypes.data,
                td.size,
                ct.byref(e),
                None,
                None,
            )

        assert call(11) == 0
        assert td[0] == 1.0
        assert np.isinf(arrays[1]).any()
        assert call(11, True) == -2
        assert (
            evaluate(
                h,
                ct.byref(buffers),
                ct.byref(Context(0, 11, 0)),
                False,
                True,
                None,
                0,
                ct.byref(e),
                None,
                None,
            )
            == -2
        )
        assert call(3) == -4
        arrays[0][2:] = 1.0  # independent input/output hidden coordinates
        assert call(3) == 0
        assert np.isfinite(arrays[1]).all()
        stage = fn(
            "bosdi_stage_lanes_checked", ct.c_int32, U, S, P, S, P, P, S, P, P, P, P, P
        )
        lane = np.array([0], dtype=np.uintp)
        token = U()
        assert (
            stage(
                h,
                1,
                lane.ctypes.data,
                params.size,
                params.ctypes.data,
                arrays[0].ctypes.data,
                arrays[0].size,
                ct.byref(Context(0, 11, 0)),
                ct.byref(token),
                ct.byref(e),
                None,
                None,
            )
            == -2
        )
        assert not token.value
        assert (
            stage(
                h,
                1,
                lane.ctypes.data,
                params.size,
                params.ctypes.data,
                arrays[0].ctypes.data,
                arrays[0].size,
                ct.byref(Context(0, 3, 0)),
                ct.byref(token),
                ct.byref(e),
                None,
                None,
            )
            == 0
        )
        fn("bosdi_discard_stage", None, U)(token)
    finally:
        close(fn, g)
