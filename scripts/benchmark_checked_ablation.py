"""Build timing-only baseline ablations in /tmp; never qualify these providers.

Run with pixi run python scripts/benchmark_checked_ablation.py, then use each
printed shared library in benchmark_checked_groups.py. Requires cached Cargo deps.
"""

import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ref", default="343b46825a0d9d3514cdb4fd1ef68f0ea3346216")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
dest = Path("/tmp/bosdi-checked-ablation")
dest.mkdir(exist_ok=True)
for name in [
    "Cargo.toml",
    "Cargo.lock",
    "src/lib.rs",
    "src/checked_runtime.rs",
    "src/handle_runtime.rs",
]:
    path = dest / name
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_bytes(
        subprocess.check_output(["git", "show", f"{args.ref}:{name}"], cwd=root)
    )
lib = dest / "src/lib.rs"
lib.write_text(
    lib.read_text()
    + r"""
// Timing-only diagnostics, never part of the production ABI.
#[no_mangle]
pub extern "C" fn bosdi_bench_lock_ns(iterations: usize) -> f64 {
    let start=std::time::Instant::now();
    for _ in 0..iterations {
        std::hint::black_box((HANDLES.read().unwrap(),OSDI_REGISTRY.read().unwrap()));
    }
    start.elapsed().as_secs_f64()*1e9/iterations as f64
}
#[no_mangle]
pub extern "C" fn bosdi_bench_install_ns(iterations: usize) -> f64 {
    let config=configuration();
    let start=std::time::Instant::now();
    for _ in 0..iterations { config.pool.install(|| std::hint::black_box(1)); }
    start.elapsed().as_secs_f64()*1e9/iterations as f64
}
"""
)
p = dest / "Cargo.toml"
p.write_text(p.read_text().replace('["staticlib"]', '["cdylib"]'))
base = (dest / "src/checked_runtime.rs").read_text()
handle = (dest / "src/handle_runtime.rs").read_text()
start = base.index("pub(super) unsafe fn evaluate(")
end = base.index("\n#[no_mangle]", start)
prefix, body, suffix = base[:start], base[start:end], base[end:]
new_handle = (root / "src/handle_runtime.rs").read_text()
a = new_handle.index("pub(super) fn disjoint_regions(")
b = new_handle.index("\n}", a) + 2
region_impl = new_handle[a:b].replace("pub(super) fn", "fn", 1)
variants = {
    "baseline": body,
    "no_snapshot_copy": body.replace(
        "scratch.model_data.clear();",
        "if scratch.model_data.len() != m.model_size {\nscratch.model_data.clear();",
    )
    .replace(
        "scratch.inst_data.clear();",
        "}\nif scratch.inst_data.len() != m.instance_size {\nscratch.inst_data.clear();",
    )
    .replace(
        "                let EvalScratch {", "}\n                let EvalScratch {"
    ),
    "no_output_zero": body.replace("cur.fill(0.0);", "")
    .replace("chg.fill(0.0);", "")
    .replace("cond.fill(0.0);", "")
    .replace("cap.fill(0.0);", ""),
    "no_error_vec": body.replace("let errors: Vec<_>", "let errors")
    .replace(
        ".map(work)\n                .collect()",
        ".map(work)\n                .reduce(BosdiCallError::default, lowest)",
    )
    .replace(
        "(0..h.num_devices).map(work).collect()",
        "(0..h.num_devices).map(work).fold(BosdiCallError::default(), lowest)",
    )
    .replace(
        """    errors
        .into_iter()
        .find(|e| e.status != 0)
        .unwrap_or_default()""",
        "    errors",
    ),
    "no_region_allocation": body,
}
lowest = "\nfn lowest(a: BosdiCallError,b: BosdiCallError)->BosdiCallError { if a.status==0 || (b.status!=0 && b.lane<a.lane) {b} else {a} }\n"
for name, code in variants.items():
    (dest / "src/checked_runtime.rs").write_text(prefix + code + suffix + lowest)
    h = handle
    if name == "no_region_allocation":
        a = h.index("fn disjoint_regions(")
        b = h.index("\n}", a) + 2
        h = h[:a] + region_impl + h[b:]
    (dest / "src/handle_runtime.rs").write_text(h)
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--offline",
            "--manifest-path",
            str(dest / "Cargo.toml"),
        ],
        check=True,
        cwd=root,
        capture_output=True,
    )
    target = dest / (name + ".so")
    target.write_bytes((dest / "target/release/libbosdi.so").read_bytes())
    print(target, flush=True)
