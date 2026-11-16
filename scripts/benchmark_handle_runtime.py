"""Budgeted native C ABI benchmark, including an independent scalar OSDI loop.

Run: PYTHONPATH=src:tests pixi run python scripts/benchmark_handle_runtime.py
The C timing loops exclude Python and JAX dispatch and model setup. The optional
baseline is a pre-change extension/shared library. Results are diagnostic only.
"""

import argparse
import ctypes as ct
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from _native_runtime import Native, Metadata
from test_osdi_runtime import model_params, THREADS
from osdi_jax import osdi_setup_batch
from osdi_runtime import (
    BoundBatch,
    qualify_handle,
    set_num_threads,
    _Bypass,
    _addresses,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        help="Approximate total timing budget; setup is additional",
    )
    parser.add_argument("--baseline")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        library = Path(directory) / "timing.so"
        subprocess.run(
            [
                "cc",
                "-O3",
                "-shared",
                "-fPIC",
                str(root / "scripts/handle_benchmark.c"),
                "-o",
                str(library),
            ],
            check=True,
        )
        helper = Native.__new__(Native)
        helper.lib = ct.CDLL(str(library))
        helper.addresses = None
        create = helper.function(
            "scalar_create",
            ct.c_void_p,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_uint32,
        )
        output = helper.function(
            "scalar_output", ct.POINTER(ct.c_double), ct.c_void_p, ct.c_uint32
        )
        free = helper.function("scalar_free", None, ct.c_void_p)
        run = helper.function("scalar_run", None, ct.c_void_p)
        scalar_time = helper.function(
            "time_scalar", ct.c_double, ct.c_void_p, ct.c_size_t, ct.c_double
        )
        handle_time = helper.function(
            "time_handle",
            ct.c_double,
            ct.c_void_p,
            ct.c_uint64,
            *([ct.c_size_t] * 3),
            *([ct.c_void_p] * 6),
            ct.c_double,
        )
        bound_time = helper.function(
            "time_bound",
            ct.c_double,
            ct.c_void_p,
            ct.c_uint64,
            ct.c_uint64,
            ct.c_size_t,
            ct.c_double,
        )
        bypass_time = helper.function(
            "time_bound_bypass",
            ct.c_double,
            ct.c_void_p,
            ct.c_uint64,
            ct.c_uint64,
            ct.c_size_t,
            ct.POINTER(_Bypass),
            ct.c_double,
        )
        native = Native()
        baseline = Native(args.baseline) if args.baseline else None
        rows = []
        # scalar + dense snapshots + qualified dense + bound raw + idle bound,
        # plus an optional old dense path. Each cell gets three short trials.
        cell_budget = args.seconds / (2 * 3 * len(THREADS) * (6 if baseline else 5) * 3)
        for name in ("resistor_va.osdi", "psp103v4_psp103.osdi"):
            for count in (100, 1000, 10000):
                model, params = model_params(name, count, "tran")
                metadata = Metadata(
                    model.id,
                    model.num_pins,
                    model.num_nodes,
                    model.num_params,
                    model.num_states,
                    4,
                    True,
                )
                handle = osdi_setup_batch(model.id, params)
                voltage = np.zeros((count, model.num_nodes))
                voltage[:, :2] = [0.8, 0.6]
                artifact = ct.CDLL(model.path)
                descriptor = ct.addressof(
                    ct.c_byte.in_dll(artifact, "OSDI_DESCRIPTORS")
                )
                scalar = create(
                    descriptor, count, params.ctypes.data, voltage.ctypes.data, 2
                )
                assert scalar
                f, g, q, c, state = native.evaluate(
                    handle.handle_id, metadata, voltage, "dense"
                )
                if baseline:
                    old_model = baseline.model(name, 2)
                    old_handle = baseline.setup(
                        old_model.id, count, model.num_params, params.ctypes.data
                    )
                with BoundBatch(handle, voltage) as bound:
                    bound.qualify_pure_eval()
                    bound.evaluate()
                    run(scalar)
                    # Independent scalar adapter must agree bitwise, including
                    # collapsed equality rows and raw Jacobian signed zeros.
                    for index, actual in enumerate(
                        (
                            bound._currents,
                            bound.conductances,
                            bound._charges,
                            bound.capacitances,
                        )
                    ):
                        expected = np.ctypeslib.as_array(
                            output(scalar, index), shape=(actual.size,)
                        ).reshape(actual.shape)
                        assert actual.tobytes() == expected.tobytes(), (name, index)
                    cache = bound.new_cache()
                    cache.previous_voltages[:] = voltage
                    cache.valid[:] = 3
                    mask = np.zeros(count, np.uint8)
                    options = _Bypass(
                        cache.previous_voltages.ctypes.data,
                        voltage.size,
                        cache.valid.ctypes.data,
                        count,
                        mask.ctypes.data,
                        count,
                        bound.generation,
                        1e-3,
                        1e-6,
                        False,
                    )
                    for threads in THREADS:
                        set_num_threads(threads)

                        def time_dense(provider, hid, md):
                            return handle_time(
                                ct.cast(provider.full, ct.c_void_p),
                                hid,
                                count,
                                model.num_nodes,
                                md.states,
                                voltage.ctypes.data,
                                f.ctypes.data,
                                g.ctypes.data,
                                q.ctypes.data,
                                c.ctypes.data,
                                state.ctypes.data,
                                cell_budget,
                            )

                        samples = {
                            k: []
                            for k in (
                                "scalar",
                                "dense_snapshot",
                                "dense_qualified",
                                "bound_raw",
                                "bound_idle",
                            )
                        }
                        if baseline:
                            samples["baseline_dense"] = []
                        for _ in range(3):
                            qualify_handle(handle, False)
                            samples["scalar"].append(
                                scalar_time(scalar, count, cell_budget)
                            )
                            samples["dense_snapshot"].append(
                                time_dense(native, handle.handle_id, metadata)
                            )
                            qualify_handle(handle)
                            samples["dense_qualified"].append(
                                time_dense(native, handle.handle_id, metadata)
                            )
                            samples["bound_raw"].append(
                                bound_time(
                                    _addresses["eval_bound"],
                                    bound._id,
                                    bound.generation,
                                    count,
                                    cell_budget,
                                )
                            )
                            samples["bound_idle"].append(
                                bypass_time(
                                    _addresses["eval_bound_bypass"],
                                    bound._id,
                                    bound.generation,
                                    count,
                                    ct.byref(options),
                                    cell_budget,
                                )
                            )
                            if baseline:
                                samples["baseline_dense"].append(
                                    time_dense(baseline, old_handle, old_model)
                                )
                        row = {
                            "model": name,
                            "lanes": count,
                            "threads": threads,
                            **{
                                k: round(float(np.median(v)), 2)
                                for k, v in samples.items()
                            },
                        }
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                free(scalar)
                handle.free()
                if baseline:
                    baseline.free(old_handle)
        if args.output:
            args.output.write_text(
                json.dumps(
                    {
                        "unit": "ns/lane",
                        "timing_budget_seconds": args.seconds,
                        "rows": rows,
                    },
                    indent=2,
                )
                + "\n"
            )
    set_num_threads(0)


if __name__ == "__main__":
    main()
