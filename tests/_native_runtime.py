"""C ABI test helpers; no JAX dispatch or asynchronous timing."""

import ctypes as ct
from pathlib import Path

import numpy as np
import osdi_shim_nb

ROOT = Path(__file__).parent
PTR = ct.c_void_p
SIZE = ct.c_size_t
U64 = ct.c_uint64


class Metadata(ct.Structure):
    _fields_ = [
        ("id", ct.c_uint32),
        ("pins", SIZE),
        ("nodes", SIZE),
        ("params", SIZE),
        ("states", SIZE),
        ("version", ct.c_uint32),
        ("success", ct.c_bool),
    ]


class Native:
    def __init__(self, path=None):
        self.lib = ct.CDLL(str(path or osdi_shim_nb.__file__))
        self.addresses = None if path else osdi_shim_nb.runtime_addresses()
        self.load = self.function(
            "load_osdi_library_with_analysis",
            Metadata,
            ct.c_char_p,
            ct.c_uint32,
            ct.c_double,
            ct.c_uint32,
        )
        self.setup = self.function(
            "osdi_setup_batch_ffi", U64, ct.c_uint32, SIZE, SIZE, PTR
        )
        self.free = self.function("osdi_free_handle_ffi", None, U64)
        self.full = self.function(
            "batched_osdi_eval_handle_ffi", None, U64, SIZE, SIZE, SIZE, *([PTR] * 7)
        )
        self.sparse = self.function(
            "batched_osdi_eval_handle_sparse_ffi",
            None,
            U64,
            SIZE,
            SIZE,
            SIZE,
            *([PTR] * 7),
        )
        self.residual = self.function(
            "batched_osdi_residual_eval_handle_ffi",
            None,
            U64,
            SIZE,
            SIZE,
            SIZE,
            *([PTR] * 5),
        )
        self.jac_count = self.function(
            "osdi_handle_jac_count_ffi", SIZE, U64, ct.c_bool
        )

    def function(self, name, result, *args):
        addresses = getattr(self, "addresses", None)
        address = addresses.get(name) if addresses is not None else None
        fn = (
            ct.CFUNCTYPE(result, *args)(address) if address else getattr(self.lib, name)
        )
        fn.restype, fn.argtypes = result, args
        return fn

    def model(self, filename, analysis=0, version=4):
        model = self.load(
            str(ROOT / "compiled_osdi" / filename).encode(), version, 300.0, analysis
        )
        assert model.success
        return model

    def evaluate(self, handle, model, voltages, mode):
        n = len(voltages)
        nodes = model.nodes
        state = np.zeros((n, model.states))
        f, q = np.empty((n, nodes)), np.empty((n, nodes))
        args = [handle, n, nodes, model.states, voltages.ctypes.data, state.ctypes.data]
        if mode == "residual":
            self.residual(*args, f.ctypes.data, q.ctypes.data, state.ctypes.data)
            return f, q, state
        widths = (
            [self.jac_count(handle, False), self.jac_count(handle, True)]
            if mode == "sparse"
            else [nodes**2] * 2
        )
        g, c = (np.empty((n, w)) for w in widths)
        fn = self.sparse if mode == "sparse" else self.full
        fn(
            *args,
            f.ctypes.data,
            g.ctypes.data,
            q.ctypes.data,
            c.ctypes.data,
            state.ctypes.data,
        )
        return f, g, q, c, state
