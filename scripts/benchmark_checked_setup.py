"""All-lane transactional update versus fresh validated preparation; native timing."""

import argparse
import ctypes as ct
import json
from pathlib import Path
import subprocess
import numpy as np
from _native_runtime import Native, PTR, SIZE, U64
from osdi_loader import load_osdi_model
from test_embedder_runtime import Context, Error

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--optical", required=True)
p.add_argument("--output", required=True)
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
        "/tmp/bosdi-setup-timing.so",
    ],
    check=True,
)
t = ct.CDLL("/tmp/bosdi-setup-timing.so").time_prepare
t.argtypes = [
    PTR,
    PTR,
    PTR,
    ct.c_uint32,
    U64,
    PTR,
    SIZE,
    PTR,
    SIZE,
    SIZE,
    ct.c_bool,
    ct.c_double,
    PTR,
]
t.restype = ct.c_int
n = Native()
setup = n.function(
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
update = n.function(
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
threads = n.function("bosdi_set_num_threads", ct.c_bool, SIZE)
md = load_osdi_model(str(Path(a.optical).resolve()), version="0.5")
values = {
    "$mfactor": 1,
    "r0": 1,
    "length_um": 100,
    "neff": 2.31,
    "n_group": 4.34,
    "loss_dB_cm": 2,
    "wavelength_nm": 1550,
}
rows = []
for count in [32, 128, 512, 2048]:
    params = np.array([[values[name] for name in md.param_names]] * count)
    x = np.full((count, md.num_nodes), 0.25)
    h = U64()
    e = Error()
    assert (
        setup(
            md.id,
            count,
            md.num_params,
            params.ctypes.data,
            x.ctypes.data,
            x.size,
            ct.byref(Context(2, 3, 0)),
            ct.byref(h),
            ct.byref(e),
            None,
            None,
        )
        == 0
    )
    for workers in [1, 2, 4]:
        assert threads(workers)
        row = dict(lanes=count, workers=workers)
        for replace in [False, True]:
            trials = []
            for _ in range(3):
                out = np.zeros(1)
                assert (
                    t(
                        ct.cast(setup, PTR),
                        ct.cast(update, PTR),
                        ct.cast(n.free, PTR),
                        md.id,
                        h,
                        params.ctypes.data,
                        md.num_params,
                        x.ctypes.data,
                        x.size,
                        count,
                        replace,
                        0.05,
                        out.ctypes.data,
                    )
                    == 0
                )
                trials.append(out[0])
            row["update_ns_lane" if replace else "fresh_ns_lane"] = float(
                np.median(trials)
            )
        rows.append(row)
    n.free(h)
Path(a.output).write_text(json.dumps(rows, indent=2) + "\n")
