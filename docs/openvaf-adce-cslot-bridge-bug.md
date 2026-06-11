# Bug: `--dump-unopt-mir-with-split` leaks unbridged init→eval SSA references

## Summary

After the ADCE-only refinement landed, `openvaf-r --dump-unopt-mir-with-split` emits `eval_fn` bodies that reference
SSAs defined in `init_fn` but not listed in `CachedValues` (the cslot bridge). Downstream parsers see "undefined
operand" errors when walking the eval body.

## Reproduction

Three IHP-PDK models, all show the leak:

```
PSP103:    1 undefined ref in eval_fn (v17 — defined in init_fn.args[0])
mosvar:    2 undefined refs
juncap200: 1 undefined ref
```

`--dump-mir` (the original optimized path) shows **0 undefined refs** on all three — so the bridge is correct there but
is leaking specifically in the unopt-with-split path.

## Concrete example: PSP103's `v17`

In the `--dump-unopt-mir-with-split` output:

- `eval_fn`'s entry block:
  ```
  block4772:
    br v17, block2, block3
  ```
- `v17` is **not** in `eval_fn.args`, **not** in `eval_fn.constants`, **not** in `eval_fn` instruction results.
- `v17` **is** `init_fn.args[0]` (a function arg of init_fn).
- `v17` **is not** in `CachedValues` (`cached.mapping`).

Result: `v17` is referenced as a branch condition in eval_fn, but the caller has no way to bind it. Either eval_fn
should have `v17` as a trailing cslot arg with `cached.mapping['init_fn_v17'] = cslot_for_v17`, or the eval reference
should have been replaced by the init-time constant value.

For comparison, the same model under `--dump-mir`:

- All eval_fn references resolve cleanly within eval_fn.args + .constants
  - insts.
- `cached.mapping` has 442 entries, all of which appear as trailing arguments of `eval_fn`.

## Diagnosis

ADCE is the right pass to run for the goal (cache-slot reduction), and it preserves phi structure as expected — the
eval-side block / phi counts match the no-opts version:

```
                    Cache slots  Eval blocks  2-edge phis
v1 (no value opts)  2086         4869         804
v2 (ADCE only)      467          4869         752  ← current
--dump-mir          442          980          431
```

But ADCE on init_fn (or eval_fn) appears to drop the cslot-bridge generation for SSAs that the rewriter no longer keeps
in eval-side form. The cached-value computation runs after some simplification that removed the eval-side use chain that
the bridge was tracking.

## Acceptance criteria for the fix

For all three IHP-PDK models tested (`PSP103`, `mosvar`, `juncap200`), the following Python snippet must print `0`:

```python
import sys; sys.path.insert(0, '/home/cdaunt/code/bosdi/src')
from bosdi.va.ir_client import compile_va_unopt_with_split

m = compile_va_unopt_with_split('<model>.va').modules[0]
defined = set(m.eval_fn.args)
defined.update(c.name for c in m.eval_fn.constants)
for b in m.eval_fn.blocks:
    for inst in b.insts:
        if inst.result: defined.add(inst.result)

referenced = set()
for b in m.eval_fn.blocks:
    for inst in b.insts:
        referenced.update(inst.operands or ())
        referenced.update(e.value for e in (inst.phi_edges or ()))

print(len(referenced - defined))  # must be 0
```

Equivalent: any SSA referenced as an operand or phi-edge value in `eval_fn` must be defined as one of:

- `eval_fn.args` (function argument, including the trailing cslot args)
- A constant in `eval_fn.constants`
- An instruction result in some `eval_fn.blocks[*].insts`

## Files affected (from previous implementation summary)

- `openvaf/sim_back/src/init.rs` — ADCE pass on init/eval; cached-value generation likely needs to track ADCE-removed
  eval uses and either preserve their cslot bridge or eliminate the eval reference.
- `openvaf/sim_back/src/lib.rs` — `Initialization::new_with_opts(..., skip_value_opts)` orchestration.

## Severity

Hard blocker for the unopt-with-split path on the three IHP-PDK models above. Without this fix, downstream parsers (e.g.
`bosdi`'s `dump_parser.parse_dump`) raise "unresolved operand" errors during lowering.

## Workaround until fixed

`bosdi.va.compile_va_unopt` (the original combined-function path) remains correct and compiles cleanly — the partition
analysis in `bosdi.va.lowering._partition_init_eval` derives an equivalent init/eval boundary at the cost of a slightly
larger emitted setup function.
