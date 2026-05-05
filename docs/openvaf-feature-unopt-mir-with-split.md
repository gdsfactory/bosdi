# Spec: `--dump-unopt-mir-with-split` flag for `openvaf-r`

## 1. Goal

Add a new CLI flag `--dump-unopt-mir-with-split` to `openvaf-r` that emits MIR with **the analog-initial / analog
function split** but **without value-level optimization passes** applied to either function body. The flag's output must
be byte-compatible with the existing `--dump-mir` text format so existing parsers work unchanged.

This bridges the gap between two existing flags:

| Existing flag                           | Init/eval split | Value-level optimizations |
| --------------------------------------- | --------------- | ------------------------- |
| `--dump-unopt-mir`                      | NO              | NO                        |
| `--dump-mir` / `--dump-json`            | YES             | YES                       |
| **`--dump-unopt-mir-with-split` (NEW)** | **YES**         | **NO**                    |

## 2. Definitions

- **Init/eval split**: the transformation that takes a single combined MIR function (containing both `analog initial`
  and `analog` block instructions) and produces:

  - a separate `init_fn` containing only the `analog initial` instructions plus their dependency closure;
  - a separate `eval_fn` containing the `analog` instructions, with additional trailing function arguments ("cslots")
    that receive cached `init_fn` outputs at runtime;
  - a `CachedValues` mapping that records, for each cslot, which `init_fn` SSA produces the value bound to it.

- **Value-level optimization passes**: the per-function transformations in the `mir_opt` pipeline:

  - sparse conditional constant propagation (SCCP)
  - global value numbering / common-subexpression elimination (GVN/CSE)
  - dead-code elimination (DCE)
  - phi-node coalescing / single-edge phi simplification
  - branch / block simplification (`simplify_cfg`)
  - instruction combining (`inst_combine`)
  - taint splitting

  These transform the function body. The init/eval split is structurally separate from these and operates on the
  function-decomposition level.

## 3. Background (problem this fixes)

PSP103 (and likely BSIM4, juncap200 nested-conditional patterns) uses the `expll(x, xlow, expxlow, xhigh, expxhigh)`
macro:

```verilog
if (x < xlow)        result = expxlow * (1.0 + (x - xlow));
else if (x > xhigh)  result = expxhigh * (1.0 + (x - xhigh));
else                 result = exp(x);
```

This produces nested 2-edge phi nodes in the unoptimized MIR. The optimization pipeline collapses them into shapes that
downstream re-lowering tooling cannot reliably recover into the original 3-way select. Numerical impact, measured by
lowering PSP103 from each dump variant and running a single-NMOS DC sweep at Vds=0.6 V, Vgs=0:

| Source                              | Drain current at Vgs=0 V | Relative err vs OSDI |
| ----------------------------------- | ------------------------ | -------------------- |
| OSDI binary (reference)             | 1.83 × 10⁻⁸ A            | —                    |
| `--dump-unopt-mir` (currently used) | 1.84 × 10⁻⁸ A            | 0.66 %               |
| `--dump-mir`                        | 2.04 × 10¹ A             | ~10⁹×                |
| `--dump-json`                       | 2.84 × 10⁵ A             | ~10¹³×               |

Currently the unopt path is the only correct one but lacks the split, costing ~5× per-step time at runtime because all
device-physics "constants" (~600 SSAs derived only from model parameters and temperature) are recomputed on every Newton
step instead of being cached at instance construction.

## 4. CLI definition

Add a new flag entry alongside the existing `--dump-mir` / `--dump-unopt-mir` / `--dump-json` flags:

- **Flag name**: `--dump-unopt-mir-with-split`
- **Argument**: none (boolean toggle)
- **Help text**: `Dump MIR with init/eval split but without value-level optimization passes.`
- **Mutual exclusion**: same group as `--dump-mir`, `--dump-unopt-mir`, `--dump-json`, `--print-expansion` — only one
  dump mode active at a time.
- **Termination**: same as `--dump-mir` — emit to stdout, then exit with code 0.

Likely file: `openvaf-driver/src/cli_def.rs` (existing dump flags live here).

## 5. Compilation pipeline

Conceptual pipeline stages (the implementer should verify exact names and ordering against the current OpenVAF source):

```
┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐  ┌────────────┐  ┌───────────┐
│ Parse VA     │→ │ HIR          │→ │ MIR (combined fn)       │→ │ MIR opt    │→ │ Codegen   │
│ → AST        │  │ → MIR build  │  │ ↓ analog-initial / eval │  │ pipeline   │  │ (OSDI/    │
│              │  │ ──────────── │  │   extraction (split)    │  │ ────────── │  │  LLVM)    │
│              │  │  combined fn │  │ ↓ produces init_fn,     │  │ SCCP, CSE, │  │           │
│              │  │              │  │   eval_fn, CachedValues │  │ DCE, etc.  │  │           │
└──────────────┘  └──────────────┘  └─────────────────────────┘  └────────────┘  └───────────┘
                          │                       │                       │
                  --dump-unopt-mir          (NEW: --dump-unopt-mir-with-split)         --dump-mir
                  emits here                emits here                                 emits here
```

The flag's behaviour:

1. Run the parse → HIR → MIR-build phases as today.
1. Run **only** the init/eval extraction pass (produces `init_fn`, `eval_fn`, `CachedValues`).
1. Skip the value-level optimization passes (SCCP, CSE, DCE, phi-collapse, simplify_cfg, inst_combine, taint_splitting).
1. Serialize using the same text dumper that `--dump-mir` already uses.
1. Exit.

## 6. Implementation requirements

### 6.1. Pass identification

Identify the existing pass in the OpenVAF source that performs the init/eval extraction and `CachedValues` construction.
Expected location: somewhere under `mir_opt/` or its callers. Search hints:

- Functions/methods that touch `init_fn`, `eval_fn`, `CachedValues` fields on the equivalent of `CompiledModule`
- Pipeline driver(s) that compose the per-function passes — the extraction is likely either an early step in that
  pipeline or invoked just before/after it
- Anything that adds the trailing cslot arguments to `eval_fn`

The extraction pass MUST be runnable standalone on raw (post-MIR-build, pre-optimization) MIR. If today it depends on
results from optimization passes (e.g. uses SCCP-determined dead-block info), factor out a minimal-prereq version that
runs only the extraction logic. The ranking of "must-run" prerequisites is:

1. Anything required to enumerate the analog-initial vs analog instructions (e.g. block-tagging from HIR-lowering
   metadata) → keep / re-run.
1. Anything required to compute the dependency closure of init instructions in eval (so cslots can be sized) → keep.
1. Anything that simplifies block structure or folds values for "performance" → omit.

### 6.2. Driver branch

In `openvaf-driver/src/main.rs` (or equivalent), add a branch that:

1. Builds the MIR as today.
1. Runs the extraction pass (and only that pass) on the raw MIR.
1. Calls the existing serialization function used by `--dump-mir` to emit text-format MIR for `init_fn`, `eval_fn`,
   `CachedValues`, `DaeSystem`, etc.
1. Returns / exits with code 0.

The new branch should reuse as much existing infrastructure as possible — especially the serializer. If the serializer
requires running through the full pipeline (e.g. it expects post-optimization invariants), that's a bug in the
serializer that should be fixed in the same change so it emits whatever lattice / metadata fields it has, even if those
are at "unoptimized" levels.

### 6.3. Output format

Byte-compatible with `--dump-mir`. Specifically:

- Same module header (`module ...`, `ports`, `port_nodes`, `internal_nodes`)
- `init_fn:` section using the same syntax as `--dump-mir`
- `eval_fn:` section using the same syntax as `--dump-mir`, including the trailing cslot args
- `setup_fn:` section (whatever shape `--dump-mir` emits — likely empty or minimal for the unopt path; preserve the
  section header even if empty)
- `Cached values during instance setup` section with the cslot → init-SSA mapping
- `DaeSystem { ... }` block as today
- All other sections (`HirInterner` / argument tables, callbacks, small-signal parameters, noise sources, model inputs,
  counters) exactly as `--dump-mir`

The instruction-level differences are exclusively that the bodies of `init_fn` and `eval_fn` are pre-optimization (more
blocks, more SSAs, fully expanded phi nodes preserving 2-edge structure, no SCCP-folded literals).

## 7. Acceptance criteria

A correct implementation MUST satisfy:

### 7.1. Smoke test (must pass)

For the test module in §8, the following are all true:

- Running `openvaf-r --dump-unopt-mir-with-split test_split.va` produces a non-empty stdout output and exits with code
  0\.
- The output begins with `module test_split`.
- The output contains a non-empty `init_fn:` section.
- The output contains a non-empty `eval_fn:` section.
- The output contains a `Cached values during instance setup` section with at least one entry of the form
  `cslot N: <init_ssa>`.
- The text format is byte-compatible with `--dump-mir`'s syntax (no new keywords, no reordered sections).

### 7.2. Parser compatibility (must pass)

The output, when fed to the existing `bosdi/src/bosdi/va/dump_parser.py` `parse_dump()` function (no parser changes),
produces a `DumpFile` where for the test module:

- `len(dumpfile.modules) == 1`
- `len(module.init_fn.blocks) >= 1`
- `len(module.eval_fn.blocks) >= 1`
- `len(module.cached.mapping) >= 1`

### 7.3. Phi-structure preservation (must pass)

For PSP103 (or any model with nested-conditional patterns like `expll`), the `eval_fn` body in the new output must
contain the same phi structure as `--dump-unopt-mir`'s output for the same SSAs (modulo SSA renaming caused by the split
itself). Specifically: chains of 2-edge phi nodes from the original Verilog-A `if`/`else if`/`else` chains must NOT have
been collapsed into single N-edge phis or folded to single-arm phis via SCCP.

A practical check: lowering PSP103 from `--dump-unopt-mir-with-split` output (via bosdi) and running a single-NMOS DC
sweep at Vds=0.6 V, Vgs=0 V should produce a drain current within 1% of the OSDI binary's result (1.83 × 10⁻⁸ A). The
current `--dump-mir` path produces 20.4 A at the same point — a 9-orders-of-magnitude error.

### 7.4. Determinism (must pass)

Two invocations of `openvaf-r --dump-unopt-mir-with-split same_file.va` on the same input MUST produce byte-identical
stdout output.

## 8. Test inputs

### 8.1. Minimal smoke test (`test_split.va`)

```verilog
`include "discipline.h"

module test_split(p, n);
  inout p, n;
  electrical p, n;
  parameter real Is = 1.0e-15 from (0:inf);
  parameter real Vt = 0.025   from (0:inf);
  parameter real Rs = 0.1     from [0:inf);

  real Gleak;
  real Vd;

  analog initial begin
    Gleak = 1.0 / (Rs * 1000.0);
  end

  analog begin
    Vd = V(p, n);
    I(p, n) <+ Is * (exp(Vd / Vt) - 1.0) + Gleak * Vd;
  end
endmodule
```

Expected (via `parse_dump`):

- `init_fn.blocks` populated with the `Gleak = 1/(Rs*1000)` computation
- `eval_fn.blocks` populated with the analog-block computation, using a cslot for the cached `Gleak` value
- `cached.mapping` has exactly one entry mapping the init SSA for `Gleak` to cslot index 0

### 8.2. Nested-conditional regression test

Lower the IHP-PDK psp103v4 model
(`/home/cdaunt/code/gdsfactory/pdks/IHP-Open-PDK/ihp-sg13g2/libs.tech/verilog-a/psp103/psp103.va`) through the new flag.
Verify with bosdi that:

- Parsing succeeds (no `DumpParseError`).
- After lowering with `bosdi.va.lower(..., collapse_nodes=True, static_params=...)`, a single-NMOS DC sweep at Vds=0.6 V
  matches OSDI within 1 % across Vgs ∈ {0, 0.3, 0.5, 0.7, 1.1} V.

This catches the `expll`-collapse regression that affects the current `--dump-mir` path.

### 8.3. Determinism test

Run `openvaf-r --dump-unopt-mir-with-split test_split.va` twice. The two stdout outputs must be byte-identical.

## 9. Out of scope

- Changes to LLVM-IR codegen, OSDI codegen, or the JSON serializer.
- Changes to the existing `--dump-mir`, `--dump-unopt-mir`, `--dump-json` flags. Their behaviour MUST remain unchanged.
- Changes to MIR data structures, lattice types, or other non-text-format internals.
- Performance optimization of the new flag (it's a debug/developer feature; correctness and parser compatibility are the
  only success criteria).

## 10. Files likely involved

Verify these against current source — line numbers will drift:

- `openvaf-driver/src/cli_def.rs` — flag declaration
- `openvaf-driver/src/main.rs` — dispatch on flag, call into pipeline
- `mir_opt/src/lib.rs` (or equivalent) — extract the split pass into a standalone callable if it isn't already
- `mir/src/serialize.rs` (or equivalent) — text serializer; reuse as is unless it has a hard dependency on
  post-optimization invariants

## 11. Reference: existing parser to satisfy

The downstream parser that consumes this format is `bosdi/src/bosdi/va/dump_parser.py`, function `parse_dump(text)`. Its
top-level expected sections (for one module) are:

```
module <name>
ports: ...
port_nodes: ...
internal_nodes: ...
init_fn:
  <blocks, instructions>
eval_fn:
  <blocks, instructions, trailing cslot args>
setup_fn:
  <blocks, instructions or empty>
Cached values during instance setup
  cslot 0: <init_ssa>
  cslot 1: <init_ssa>
  ...
DaeSystem {
  ...
}
HirInterner {
  ...
}
```

Section headers, indentation, and instruction syntax must match `--dump-mir` exactly. The parser's regex-based section
detection is strict about exact header text.
