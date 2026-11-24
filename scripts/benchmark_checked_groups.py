"""Native checked residual timing; run with PYTHONPATH=src:tests."""

import argparse
import ctypes as ct
import json
from pathlib import Path
import subprocess
import numpy as np
from _native_runtime import Native, PTR, SIZE, U64
from test_osdi_runtime import model_params
from osdi_runtime import _Buffers

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--library", required=True)
p.add_argument("--optical", required=True)
p.add_argument("--output", required=True)
p.add_argument("--policy", nargs=2, type=int)
p.add_argument(
    "--models",
    nargs="+",
    default=["optical", "resistor_va.osdi", "psp103v4_psp103.osdi"],
)
p.add_argument("--lanes", nargs="+", type=int, default=[32, 128, 512, 2048])
p.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
p.add_argument("--seconds", type=float, default=0.1)
p.add_argument("--align64", action="store_true")
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
subprocess.run(
    [
        "cc",
        "-O3",
        "-shared",
        "-fPIC",
        "-I" + str(root / "include"),
        str(root / "scripts/benchmark_checked_groups.c"),
        "-o",
        "/tmp/bosdi-checked-timing.so",
    ],
    check=True,
)
t = ct.CDLL("/tmp/bosdi-checked-timing.so").time_checked
t.argtypes = [PTR, U64, PTR, SIZE, ct.c_double, PTR]
t.restype = ct.c_int
native = Native(a.library)
threads = native.function("bosdi_set_num_threads", ct.c_bool, SIZE)
policy = native.function("bosdi_set_handle_parallel_policy", ct.c_bool, U64, SIZE, SIZE)
evaluate = native.function(
    "bosdi_eval_handle_checked",
    ct.c_int,
    U64,
    PTR,
    PTR,
    ct.c_bool,
    ct.c_bool,
    PTR,
    PTR,
    PTR,
)
try:
    observed = native.function("bosdi_handle_observed_workers", SIZE, U64)
except AttributeError:
    observed = None
rows = []
for name in a.models:
    for count in a.lanes:
        if name == "optical":
            md = native.load(str(Path(a.optical).resolve()).encode(), 5, 300, 2)
            assert md.success
            params = np.full((count, md.params), np.nan)
        else:
            model, params = model_params(name, count, "tran")
            md = native.model(name, 2)
        h = native.setup(md.id, count, md.params, params.ctypes.data)
        assert h
        if a.policy:
            assert policy(h, *a.policy)
        arrays = [
            np.full((count, md.nodes), 0.3),
            np.zeros((count, md.nodes + 1)),
            np.zeros((count, native.jac_count(h, False))),
            np.zeros((count, md.nodes + 1)),
            np.zeros((count, native.jac_count(h, True))),
        ]
        if a.align64:
            aligned = []
            for array in arrays:
                backing = np.empty(array.size + 8)
                offset = (-backing.ctypes.data % 64) // 8
                view = backing[offset : offset + array.size].reshape(array.shape)
                view[:] = array
                aligned.append(view)
            arrays = aligned
        b = _Buffers(*(v for x in arrays for v in (x.ctypes.data, x.size)))
        for workers in a.workers:
            assert threads(workers)
            trials = []
            for _ in range(3):
                out = np.zeros(2)
                assert (
                    t(
                        ct.cast(evaluate, PTR),
                        h,
                        ct.byref(b),
                        count,
                        a.seconds,
                        out.ctypes.data,
                    )
                    == 0
                )
                trials.append(out.tolist())
            best = sorted(trials)[1]
            rows.append(
                dict(
                    model=name,
                    lanes=count,
                    workers=workers,
                    ns_lane=best[0],
                    cpu_wall=best[1],
                    observed_last=observed(h) if observed else None,
                    trials=trials,
                )
            )
        native.free(h)
Path(a.output).write_text(json.dumps(rows, indent=2) + "\n")
