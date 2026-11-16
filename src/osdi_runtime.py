"""Native caller-owned handle buffers and thread control.

JAX entry points remain exact and retain their tiling semantics. This mutable
API is for native-style simulator loops, outside JAX transformations. A binding
owns references to its arrays; callers must not access them during evaluation.
"""

import ctypes as ct
from threading import RLock

import numpy as np
import osdi_shim_nb


class _Buffers(ct.Structure):
    _fields_ = [
        (name, kind)
        for label in ("voltages", "currents", "conductances", "charges", "capacitances")
        for name, kind in ((label, ct.c_void_p), (label + "_len", ct.c_size_t))
    ]


_addresses = osdi_shim_nb.runtime_addresses()


def _function(name, result, *args):
    return ct.CFUNCTYPE(result, *args)(_addresses[name])


_set_threads = _function("set_num_threads", ct.c_bool, ct.c_size_t)
_num_threads = _function("num_threads", ct.c_size_t)
_set_policy = _function("set_parallel_policy", ct.c_bool, ct.c_size_t, ct.c_size_t)
_set_handle_policy = _function(
    "set_handle_parallel_policy", ct.c_bool, ct.c_uint64, ct.c_size_t, ct.c_size_t
)
_qualify_handle = _function("qualify_handle", ct.c_bool, ct.c_uint64, ct.c_bool)
_bind = _function("bind_buffers", ct.c_uint64, ct.c_uint64, ct.POINTER(_Buffers))
_rebind = _function("rebind_buffers", ct.c_uint64, ct.c_uint64, ct.POINTER(_Buffers))
_unbind = _function("unbind_buffers", None, ct.c_uint64)
_generation = _function("binding_generation", ct.c_uint64, ct.c_uint64)
_qualify = _function("qualify_binding", ct.c_bool, ct.c_uint64, ct.c_bool)
_eval = _function(
    "eval_bound",
    ct.c_int32,
    ct.c_uint64,
    ct.c_uint64,
    ct.c_bool,
    ct.c_bool,
)


def _count(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < (0 if zero else 1) or value > ct.c_size_t(-1).value:
        raise ValueError(f"{name} is out of range")
    return int(value)


def set_num_threads(threads=0):
    """Replace the BOSDI-owned pool. Zero restores once-read environment defaults."""
    if not _set_threads(_count(threads, "threads", zero=True)):
        raise RuntimeError("Cannot create BOSDI thread pool")


def num_threads():
    return _num_threads()


def set_parallel_policy(min_lanes=1024, min_chunk=256, *, handle=None):
    """Set dispatch granularity globally or for one handle (both zero resets it)."""
    allow_zero = handle is not None
    args = (
        _count(min_lanes, "min_lanes", zero=allow_zero),
        _count(min_chunk, "min_chunk", zero=allow_zero),
    )
    okay = (
        _set_policy(*args)
        if handle is None
        else _set_handle_policy(handle.handle_id, *args)
    )
    if not okay:
        raise ValueError("Invalid policy or freed handle")


def qualify_handle(handle, qualified=True):
    """Certify a history-free, fully recomputing provider for in-place JAX/C eval.

    ABI-state models are refused; tiled calls still restore immutable snapshots.
    """
    if not _qualify_handle(handle.handle_id, qualified):
        raise ValueError("Cannot qualify handle (freed or ABI-state model)")


def _array(array, shape, dtype, *, writable=False):
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != dtype
        or array.shape != shape
    ):
        raise ValueError(f"Expected ndarray with shape {shape} and dtype {dtype}")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("Buffers must be contiguous and aligned")
    if writable and not array.flags.writeable:
        raise ValueError("Output buffers must be writable")
    return array


class BoundBatch:
    """Register native arrays for one handle; retain their storage until close.

    Raw F/Q storage has a private final ground slot. ``currents`` and ``charges``
    expose only physical/equality rows, as strided views. Jacobians use the
    handle's sparse pattern. No arrays are replaced during evaluation.
    """

    def __init__(self, handle, voltages):
        self._lock = RLock()
        self._id = 0
        self.handle = handle
        self.rebind(voltages)

    def rebind(self, voltages):
        """Allocate/register new outputs and advance the binding generation."""
        with self._lock:
            if self.handle.handle_id == 0:
                raise ValueError("Handle is freed")
            n = self.handle.num_devices
            nodes = self.handle.topology.raw_to_representative.shape[1]
            voltages = _array(voltages, (n, nodes), np.dtype("float64"))
            pattern = self.handle.sparse_pattern
            arrays = [
                voltages,
                np.empty((n, nodes + 1)),
                np.empty((n, pattern.resist_pairs.shape[1])),
                np.empty((n, nodes + 1)),
                np.empty((n, pattern.react_pairs.shape[1])),
            ]
            desc = _Buffers(*(v for a in arrays for v in (a.ctypes.data, a.size)))
            if self._id:
                generation = _rebind(self._id, ct.byref(desc))
                if not generation:
                    raise ValueError("Cannot rebind buffers")
            else:
                self._id = _bind(self.handle.handle_id, ct.byref(desc))
                if not self._id:
                    raise ValueError("Cannot bind buffers")
            self._arrays = arrays
            self._currents_view = arrays[1][:, :nodes]
            self._charges_view = arrays[3][:, :nodes]
            self.generation = _generation(self._id)

    # Buffer identities are read-only: addresses change only through rebind.
    @property
    def voltages(self):
        return self._arrays[0]

    @property
    def _currents(self):
        return self._arrays[1]

    @property
    def currents(self):
        return self._currents_view

    @property
    def conductances(self):
        return self._arrays[2]

    @property
    def _charges(self):
        return self._arrays[3]

    @property
    def charges(self):
        return self._charges_view

    @property
    def capacitances(self):
        return self._arrays[4]

    def qualify_pure_eval(self, qualified=True):
        """Certify this provider overwrites intermediates and reads no history.

        Use only after qualifying the compiled provider. ABI-state models and
        providers that read history cannot qualify.
        """
        with self._lock:
            if not _qualify(self._id, qualified):
                raise ValueError("Cannot qualify binding (freed or ABI-state model)")
            self.generation = _generation(self._id)

    def evaluate(self, *, residual_only=False, check_finite=False):
        """Evaluate every lane exactly into the registered output arrays."""
        with self._lock:
            status = _eval(
                self._id,
                self.generation,
                residual_only,
                check_finite,
            )
            errors = {-1: "Stale or unbound buffers", -4: "Nonfinite model outputs"}
            if status:
                raise ValueError(
                    errors.get(status, f"Native evaluation failed ({status})")
                )
            return {"evaluated": self.voltages.shape[0]}

    def close(self):
        with self._lock:
            if self._id:
                _unbind(self._id)
                self._id = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_id", 0):
            self.close()
