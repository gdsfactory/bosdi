"""Native staged-discard/fresh setup and explicit/bound checked comparisons."""

import argparse
import ctypes as ct
import json
from pathlib import Path
import subprocess

import numpy as np
from _native_runtime import PTR, SIZE, U64
from osdi_loader import load_osdi_model
from test_checked_groups import Group
from test_staged_groups import Staging
from test_embedder_runtime import Context, Error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0.1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    binary = root / "build/checked-groups/staged-timing.so"
    subprocess.run(
        [
            "cc",
            "-O3",
            "-shared",
            "-fPIC",
            "-I" + str(root / "include"),
            str(root / "scripts/benchmark_checked_groups.c"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    helper = ct.CDLL(str(binary))
    fresh = helper.time_prepare
    fresh.argtypes = [
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
    fresh.restype = ct.c_int
    staged = helper.time_stage_discard
    staged.argtypes = [PTR, PTR, U64, PTR, SIZE, PTR, SIZE, SIZE, ct.c_double, PTR]
    staged.restype = ct.c_int
    evaluation = helper.time_explicit_bound
    evaluation.argtypes = [
        PTR,
        PTR,
        U64,
        PTR,
        U64,
        U64,
        SIZE,
        ct.c_bool,
        ct.c_bool,
        ct.c_double,
        PTR,
    ]
    evaluation.restype = ct.c_int
    md = load_osdi_model(str(args.optical.resolve()), version="0.5")
    setup_rows = []
    eval_rows = []
    for count in [32, 128, 512, 2048]:
        baseline = np.full((count, md.num_params), np.nan)
        baseline[:, md.param_names.index("length_um")] = np.linspace(100, 200, count)
        candidate = baseline.copy()
        candidate[:, md.param_names.index("length_um")] *= 1.2
        g = Group(args.optical, baseline, count, analysis=2)
        api = Staging(g)
        try:
            for workers in [1, 2, 4]:
                assert g.threads(workers)
                row = dict(lanes=count, workers=workers)
                for name in ["fresh", "stage_discard"]:
                    trials = []
                    for _ in range(3):
                        out = np.zeros(1)
                        if name == "fresh":
                            status = fresh(
                                ct.cast(g.setup, PTR),
                                None,
                                ct.cast(g.n.free, PTR),
                                g.model.id,
                                g.h,
                                candidate.ctypes.data,
                                g.model.params,
                                g.arrays[0].ctypes.data,
                                g.arrays[0].size,
                                count,
                                False,
                                args.seconds,
                                out.ctypes.data,
                            )
                        else:
                            status = staged(
                                ct.cast(api.stage, PTR),
                                ct.cast(api.discard, PTR),
                                g.h,
                                candidate.ctypes.data,
                                g.model.params,
                                g.arrays[0].ctypes.data,
                                g.arrays[0].size,
                                count,
                                args.seconds,
                                out.ctypes.data,
                            )
                        assert status == 0, status
                        trials.append(float(out[0]))
                    row[name + "_ns_lane"] = float(np.median(trials))
                    row[name + "_trials"] = trials
                setup_rows.append(row)
                if count == 2048:
                    continue
                bind = g.n.function("bosdi_bind_buffers", U64, U64, PTR)
                gen = g.n.function("bosdi_binding_generation", U64, U64)
                unbind = g.n.function("bosdi_unbind_buffers", None, U64)
                explicit = g.n.function(
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
                binding = bind(g.h, ct.byref(g.b))
                assert binding
                generation = gen(binding)
                try:
                    for residual in [True, False]:
                        error = Error()
                        ctx = Context(2, 0, 0)
                        assert (
                            explicit(
                                g.h,
                                ct.byref(g.b),
                                ct.byref(ctx),
                                residual,
                                True,
                                ct.byref(error),
                                None,
                                None,
                            )
                            == 0
                        )
                        expected = [a.tobytes() for a in g.arrays[1:]]
                        assert (
                            bound(
                                binding,
                                generation,
                                ct.byref(ctx),
                                residual,
                                True,
                                ct.byref(error),
                                None,
                                None,
                            )
                            == 0
                        )
                        assert [a.tobytes() for a in g.arrays[1:]] == expected
                        row = dict(lanes=count, workers=workers, residual_only=residual)
                        for use_bound in [False, True]:
                            trials = []
                            for _ in range(3):
                                out = np.zeros(1)
                                assert (
                                    evaluation(
                                        ct.cast(explicit, PTR),
                                        ct.cast(bound, PTR),
                                        g.h,
                                        ct.byref(g.b),
                                        binding,
                                        generation,
                                        count,
                                        use_bound,
                                        residual,
                                        args.seconds,
                                        out.ctypes.data,
                                    )
                                    == 0
                                )
                                trials.append(float(out[0]))
                            key = "bound" if use_bound else "explicit"
                            row[key + "_ns_call"] = float(np.median(trials))
                            row[key + "_trials"] = trials
                        row["bound_minus_explicit_ns_call"] = (
                            row["bound_ns_call"] - row["explicit_ns_call"]
                        )
                        eval_rows.append(row)
                finally:
                    unbind(binding)
        finally:
            api.close()
            g.close()
    args.output.write_text(
        json.dumps(dict(setup=setup_rows, evaluation=eval_rows), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
