"""Versioned embedder ABI: diagnostics, context isolation and opt-in stamps."""

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from _native_runtime import Native, PTR, SIZE, U64
from osdi_runtime import _Buffers


class Context(ct.Structure):
    _fields_ = [
        ("analysis", ct.c_uint32),
        ("options", ct.c_uint32),
        ("abstime", ct.c_double),
    ]


class Error(ct.Structure):
    _fields_ = [("status", ct.c_int32), ("lane", SIZE), ("message", ct.c_char * 256)]


CALLBACK = ct.CFUNCTYPE(None, PTR, SIZE, ct.c_char_p, ct.c_uint32)


@pytest.fixture(scope="module", params=["normal", "zero_model", "zero_resist"])
def provider(request, tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler not installed")
    root = Path(__file__).parent
    target = tmp_path_factory.mktemp("embedder") / "fixture.osdi"
    flags = [] if request.param == "normal" else [f"-D{request.param.upper()}"]
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            *flags,
            f"-I{root / 'headers'}",
            str(root / "fixtures/embedder.c"),
            "-o",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target, request.param


class Bridge:
    def __init__(self, provider):
        self.native = Native()
        self.kind = provider[1]
        self.model = self.native.load(str(provider[0]).encode(), 5, 300, 0)
        assert self.model.success
        self.handles, self.bindings = [], []
        fn = self.native.function
        self.setup = fn(
            "bosdi_setup_batch_checked",
            ct.c_int32,
            ct.c_uint32,
            SIZE,
            SIZE,
            PTR,
            PTR,
            PTR,
            PTR,
            PTR,
        )
        self.eval = fn(
            "bosdi_eval_handle_checked",
            ct.c_int32,
            U64,
            PTR,
            PTR,
            ct.c_bool,
            ct.c_bool,
            PTR,
            PTR,
            PTR,
        )
        self.bound = fn(
            "bosdi_eval_bound_checked",
            ct.c_int32,
            U64,
            U64,
            PTR,
            ct.c_bool,
            ct.c_bool,
            PTR,
            PTR,
            PTR,
        )
        self.bind = fn("bosdi_bind_buffers", U64, U64, PTR)
        self.generation = fn("bosdi_binding_generation", U64, U64)
        self.unbind = fn("bosdi_unbind_buffers", None, U64)
        self.unload = fn("bosdi_free_model", ct.c_bool, ct.c_uint32)
        self.maps = fn("bosdi_handle_node_maps", ct.c_int32, U64, SIZE, SIZE, PTR)
        self.version = fn("bosdi_runtime_abi_version", ct.c_uint32)
        self.policy = fn("bosdi_set_handle_parallel_policy", ct.c_bool, U64, SIZE, SIZE)

    def create(self, r=(1000, 1000), c=(1e-9, 1e-9), callback=None):
        params = np.array(list(zip(r, c)), dtype=np.float64)
        handle, error = U64(), Error()
        status = self.setup(
            self.model.id,
            len(params),
            2,
            params.ctypes.data,
            ct.byref(handle),
            ct.byref(error),
            callback,
            123,
        )
        if handle.value:
            self.handles.append(handle.value)
        return status, handle.value, error

    def buffers(self, handle, count=2):
        n = self.model.nodes
        arrays = [
            np.ones((count, n)),
            np.zeros((count, n + 1)),
            np.zeros((count, self.native.jac_count(handle, False))),
            np.zeros((count, n + 1)),
            np.zeros((count, self.native.jac_count(handle, True))),
        ]
        arrays[0][:, 1:] = 0
        return arrays, _Buffers(*(v for a in arrays for v in (a.ctypes.data, a.size)))

    def evaluate(
        self, handle, buffers, context, bound=False, residual=False, callback=None
    ):
        error = Error()
        if bound:
            binding = self.bind(handle, ct.byref(buffers))
            assert binding
            self.bindings.append(binding)
            status = self.bound(
                binding,
                self.generation(binding),
                ct.byref(context),
                residual,
                True,
                ct.byref(error),
                callback,
                123,
            )
        else:
            status = self.eval(
                handle,
                ct.byref(buffers),
                ct.byref(context),
                residual,
                True,
                ct.byref(error),
                callback,
                123,
            )
        assert status == error.status
        return status, error

    def close(self):
        for b in self.bindings:
            self.unbind(b)
        for h in self.handles:
            self.native.free(h)
        assert self.unload(self.model.id)


@pytest.fixture
def bridge(provider):
    b = Bridge(provider)
    yield b
    b.close()


@pytest.mark.parametrize(
    "r,c,message",
    [
        ((1000, -1), (1e-9, 1e-9), b"setup"),
        ((1000, -6), (1e-9, 1e-9), b"setup"),
        ((1000, -2), (1e-9, 1e-9), b"instance error"),
        ((1000, 1000), (1e-9, -1), b"setup"),
        ((1000, 1000), (1e-9, -2), b"setup"),
        ((1000, 1000), (1e-9, -3), b"model error"),
    ],
)
def test_setup_failures(bridge, r, c, message):
    if bridge.kind == "zero_model" and min(c) < 0:
        pytest.skip("zero-size fixture has no model parameters")
    status, handle, error = bridge.create(r, c)
    assert (status, handle, error.lane) == (-5, 0, 1)
    assert message in error.message
    assert bridge.create()[0] == 0  # no sticky failure counter


@pytest.mark.parametrize("r,status", [(-3, -6), (-4, -6), (-5, -4), (-8, -6)])
@pytest.mark.parametrize("bound", [False, True])
def test_eval_failures(bridge, r, status, bound):
    if bridge.kind == "zero_resist" and r == -5:
        pytest.skip("non-finite resistive residual requires extraction")
    _, handle, _ = bridge.create(r=(1000, r))
    arrays, buffers = bridge.buffers(handle)
    got, error = bridge.evaluate(handle, buffers, Context(2, 3, 0), bound)
    assert (got, error.lane) == (status, 1)
    assert error.message


@pytest.mark.parametrize("bound", [False, True])
def test_options_context_maps_and_lifetime(bridge, bound):
    assert bridge.version() == 1
    _, handle, _ = bridge.create(r=(1000, 42))
    assert not bridge.unload(bridge.model.id)
    maps = np.empty((2, bridge.model.nodes), dtype=np.int32)
    assert bridge.maps(handle, 2, bridge.model.nodes, maps.ctypes.data) == 0
    np.testing.assert_array_equal(maps, [[0, 1, 2], [0, 1, 0]])
    assert bridge.maps(handle, 1, bridge.model.nodes, maps.ctypes.data) == -2
    arrays, buffers = bridge.buffers(handle)
    assert bridge.evaluate(handle, buffers, Context(0, 0, 0), bound)[0] == 0
    assert not arrays[3].any() and not arrays[4].any()
    arrays[2].fill(17)
    arrays[4].fill(19)
    assert (
        bridge.evaluate(handle, buffers, Context(0, 3, 0), bound, residual=True)[0] == 0
    )
    np.testing.assert_array_equal(arrays[2], 17)
    np.testing.assert_array_equal(arrays[4], 19)
    assert bridge.evaluate(handle, buffers, Context(0, 3, 0), bound)[0] == 0
    np.testing.assert_array_equal(arrays[3][:, :2], [[1e-9, -1e-9]] * 2)
    np.testing.assert_array_equal(arrays[4], [[1e-9, -1e-9, -1e-9, 1e-9]] * 2)
    assert bridge.evaluate(handle, buffers, Context(2, 3, 2.5), bound)[0] == 0
    np.testing.assert_array_equal(arrays[1][:, 0], [2.501, 2.5 + 1 / 42])
    # Legacy path still uses the load-time DC mode and time=0.
    v = arrays[0].copy()
    old = bridge.native.evaluate(handle, bridge.model, v, "sparse")
    assert not old[2].any() and not old[3].any()
    assert old[0][0, 0] == (0 if bridge.kind == "zero_resist" else 0.001)


@pytest.mark.parametrize(
    "context",
    [
        Context(3, 0, 0),
        Context(0, 4, 0),
        Context(0, 0, -1),
        Context(2, 0, np.nan),
        Context(2, 0, np.inf),
    ],
)
def test_invalid_context(bridge, context):
    _, handle, _ = bridge.create()
    arrays, buffers = bridge.buffers(handle)
    status, error = bridge.evaluate(handle, buffers, context)
    assert status == -2 and error.lane == SIZE(-1).value


def test_concurrent_contexts_and_callback(bridge):
    _, first, _ = bridge.create(r=(1000, -4))
    _, second, _ = bridge.create()
    arrays1, buffers1 = bridge.buffers(first)
    arrays2, buffers2 = bridge.buffers(second)
    messages = []

    @CALLBACK
    def callback(user, lane, message, level):
        messages.append((user, lane, message, level))

    for handle in (first, second):
        assert bridge.policy(handle, 1, 1)
    with ThreadPoolExecutor(2) as pool:
        for _ in range(10):
            a = pool.submit(
                bridge.evaluate,
                first,
                buffers1,
                Context(2, 3, 0.5),
                False,
                False,
                callback,
            )
            b = pool.submit(bridge.evaluate, second, buffers2, Context(2, 3, 7))
            assert (a.result()[0], b.result()[0]) == (-6, 0)
            np.testing.assert_array_equal(arrays2[1][:, 0], 7.001)
    assert len(messages) == 10
    assert all(m[:3] == (123, 1, b"eval error") for m in messages)


def test_default_bitwise_parity_and_error_reset(bridge):
    _, handle, _ = bridge.create()
    arrays, buffers = bridge.buffers(handle)
    reference = bridge.native.evaluate(handle, bridge.model, arrays[0], "sparse")
    for bound in (False, True):
        status, error = bridge.evaluate(handle, buffers, Context(0, 0, 0), bound)
        assert status == 0 and error.lane == SIZE(-1).value and not error.message
        actual = (
            arrays[1][:, : bridge.model.nodes],
            arrays[2],
            arrays[3][:, : bridge.model.nodes],
            arrays[4],
        )
        assert [a.tobytes() for a in actual] == [a.tobytes() for a in reference[:4]]


def test_standalone_c_embedder(provider, tmp_path):
    if provider[1] != "normal":
        return
    root = Path(__file__).resolve().parent.parent
    static_library = root / "target/release/libbosdi.a"
    if not static_library.exists():
        pytest.skip("source build static library required")
    exe = tmp_path / "embedder"
    subprocess.run(
        [
            shutil.which("cc"),
            f"-I{root / 'include'}",
            str(root / "tests/fixtures/embedder_smoke.c"),
            str(static_library),
            "-ldl",
            "-lpthread",
            "-lm",
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(exe), str(provider[0])], check=True, capture_output=True)
