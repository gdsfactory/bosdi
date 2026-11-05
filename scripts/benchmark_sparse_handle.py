"""Compare synchronized dense/raw-entry handle calls on c6288 PSP103 groups.

Run after ``pixi run build`` and fixture compilation:
    PYTHONPATH=src pixi run python scripts/benchmark_sparse_handle.py
"""

import argparse
import re
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from osdi_jax import (
    osdi_eval_sparse_with_handle,
    osdi_eval_with_handle,
    osdi_setup_batch,
)
from osdi_loader import load_osdi_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices", type=int, default=5056, help="Instances per polarity"
    )
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.devices < 1 or args.repeats < 1 or args.warmup < 1:
        parser.error("counts must be positive")
    root = Path(__file__).resolve().parents[1]
    cards = (root / "tests/fixtures/c6288_models.inc").read_text()
    blocks = re.findall(r"model\s+\w+\s+psp103va\s*\(([^)]*)\)", cards, re.S)
    model = load_osdi_model(str(root / "tests/compiled_osdi/psp103v4_psp103.osdi"))
    names = [name.lower() for name in model.param_names]
    handles, voltages = [], []
    for polarity, block in enumerate(blocks):
        settings = {
            k: float(v) for k, v in re.findall(r"(\w+)\s*=\s*([+-]?[\d.eE+-]+)", block)
        }
        w, length, ld = (0.5e-6 if polarity == 0 else 1e-6), 0.2e-6, 0.5e-6
        settings.update(
            w=w, l=length, ad=w * ld, **{"as": w * ld}, pd=2 * (w + ld), ps=2 * (w + ld)
        )
        params = np.full((args.devices, model.num_params), np.nan)
        for name, value in settings.items():
            params[:, names.index(name.lower())] = value
        handles.append(osdi_setup_batch(model.id, params))
        v = np.zeros((args.devices, model.num_nodes))
        # Vary bias across the group; keep collapsed equalities consistent.
        v[:, 0] = np.linspace(0.1, 1.0, args.devices) * (-1 if polarity else 1)
        v[:, 1] = 0.8 * (-1 if polarity else 1)
        mapping = handles[-1].topology.raw_to_representative
        v = np.where(mapping >= 0, np.take_along_axis(v, np.maximum(mapping, 0), 1), 0)
        voltages.append(jnp.asarray(v))
    states = jnp.zeros((args.devices, model.num_states))
    dense_bytes = 2 * len(handles) * args.devices * model.num_nodes**2 * 8
    sparse_bytes = (
        len(handles) * args.devices * (model.num_resist_jac + model.num_react_jac) * 8
    )
    print(f"{len(handles)} x {args.devices} PSP103 instances, N={model.num_nodes}")
    print(f"Raw entries: G={model.num_resist_jac}, C={model.num_react_jac}")
    print(
        f"Jacobian output bytes: dense={dense_bytes:,}, sparse={sparse_bytes:,} ({sparse_bytes / dense_bytes:.1%})"
    )
    functions = {
        name: [jax.jit(lambda v, h=h, fn=fn: fn(h, v, states)) for h in handles]
        for name, fn in (
            ("dense", osdi_eval_with_handle),
            ("sparse", osdi_eval_sparse_with_handle),
        )
    }

    def run(name):
        return jax.block_until_ready(
            tuple(fn(v) for fn, v in zip(functions[name], voltages))
        )

    for name in functions:
        for _ in range(args.warmup):
            run(name)
    timings = {name: [] for name in functions}
    # Alternate order to reduce warm-cache and thermal bias.
    for i in range(args.repeats):
        for name in ("dense", "sparse") if i % 2 == 0 else ("sparse", "dense"):
            start = perf_counter()
            run(name)
            timings[name].append(perf_counter() - start)
    for name, times in timings.items():
        print(
            f"{name}: median={np.median(times) * 1000:.3f} ms, mean={np.mean(times) * 1000:.3f} ms"
        )
    print(
        f"Median speedup: {np.median(timings['dense']) / np.median(timings['sparse']):.3f}x"
    )
    for h in handles:
        h.free()


if __name__ == "__main__":
    main()
