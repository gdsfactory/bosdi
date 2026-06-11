"""Turn a :class:`circulax.va.lowering.LoweredDevice` into a ``.py`` source file.

One file per Verilog-A module: a ``from ...`` import block, then one
``@component`` / ``@va_component`` (or ``@source`` once we handle
time-dependent devices) function defining the physics. When the device's
``LoweredDevice`` carries a non-empty Jacobian (from
``DaeSystem.jacobian``), we emit a sibling ``_<Name>_jacobian`` function
and wrap the component in ``@va_component(..., jacobian_fn=...)`` so
circulax's Newton path can skip ``jax.jacfwd`` and use OpenVAF's
pre-computed Jacobian expressions directly.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
from pathlib import Path

from .lowering import LoweredDevice, PhiResolution

# Matches the SSA names the lowering emits — bare ``v123`` / ``i_v123``,
# the ``_init_cache[N]`` indices, and any local prefixed with ``v``.
# Used by the emit-time DCE pass to find which SSAs are referenced from
# a given expression text.
_SSA_NAME = re.compile(r"\b(i_v\d+|v\d+)\b")

HEADER_PREAMBLE = """\
\"\"\"Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
\"\"\"

from __future__ import annotations

import jax
import jax.numpy as jnp
"""

# Line added when any device's params include a non-float (``int`` / ``str``),
# since those are emitted as ``equinox.field(static=True, default=...)``.
HEADER_EQX_IMPORT = "import equinox as eqx\n"

HEADER_COMPONENT_IMPORT = "from circulax.components.base_component import PhysicsReturn, Signals, States, component\n"
HEADER_VA_COMPONENT_IMPORT = (
    "from circulax.components.base_component import PhysicsReturn, Signals, States\n"
    "from bosdi.circulax.va_component import va_component\n"
)


def emit_source(devices: list[LoweredDevice]) -> str:
    """Render one or more :class:`LoweredDevice` into a single Python module source string."""
    has_jacobian = any(_has_jacobian(d) for d in devices)
    # Only ``str`` params force an Equinox import — ``int`` params ship as
    # plain Python defaults (JAX traces them as weak arrays with zero
    # gradient, which matches the Verilog-A switch-flag intent and avoids
    # the ``eqx.field`` default-unwrap issue in circulax's dry-run).
    has_str_param = any(ty == "str" for d in devices for _, ty, _ in d.params)

    header = HEADER_PREAMBLE
    if has_str_param:
        header += HEADER_EQX_IMPORT
    header += "\n"
    header += HEADER_VA_COMPONENT_IMPORT if has_jacobian else HEADER_COMPONENT_IMPORT

    parts = [header, *(_emit_device(d) for d in devices)]
    # Blank line between blocks, trailing newline for POSIX friendliness.
    return "\n".join(parts).rstrip() + "\n"


def write_source(
    devices: list[LoweredDevice], out_path: Path, *, run_ruff_format: bool = True
) -> Path:
    """Write the rendered source to disk; optionally run ``ruff format`` on it.

    ``run_ruff_format`` defaults to ``True`` — it smooths over minor
    formatting choices we don't want to hand-tune (trailing commas,
    wrapping of long return-dicts). If ``ruff`` isn't available, we
    silently skip formatting.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = emit_source(devices)
    out_path.write_text(source)
    if run_ruff_format:
        # Missing / failed ruff is non-fatal — the raw output is still valid Python.
        with contextlib.suppress(FileNotFoundError, subprocess.CalledProcessError):
            subprocess.run(  # noqa: S603
                ["ruff", "format", str(out_path)],  # noqa: S607
                check=True,
                capture_output=True,
            )
    return out_path


def _has_jacobian(device: LoweredDevice) -> bool:
    return bool(device.jacobian_resist) or bool(device.jacobian_react)


def _emit_device(device: LoweredDevice) -> str:
    """Render one device: cache fn + (combined → wrappers) + physics function.

    When the device carries a Jacobian we emit one ``_<Name>_combined``
    function with all hoists exactly once and have ``_<Name>_jacobian`` and
    the public physics function be thin wrappers that call it and slice.
    This collapses the duplicated trace cost — at PSP103 scale the
    physics+jacobian functions are each ~17k lines of mostly identical
    hoists, and JAX's tracer doesn't memoize across separate function
    invocations even though XLA might CSE the resulting HLO.
    """
    cache_block = _emit_cache_fn(device)
    if _has_jacobian(device):
        combined_block = _emit_combined(device)
        jac_block = _emit_jacobian_wrapper(device)
        physics_block = _emit_physics_wrapper(device)
    else:
        combined_block = ""
        jac_block = ""
        physics_block = _emit_physics(device)
    return cache_block + combined_block + jac_block + physics_block


def _emit_physics(device: LoweredDevice) -> str:
    """Render the (no-Jacobian) physics function inline.

    Used only when the device has no analytical Jacobian — the body has
    to be emitted directly because there is no ``_combined`` to wrap.
    """
    decorator_args = _render_decorator_args(device)
    signature = _render_signature(device)
    body = _render_body(device)
    decorator = "@source" if device.uses_time else "@component"

    block = (
        f"\n{decorator}({decorator_args})\n"
        f"def {device.class_name}({signature}) -> PhysicsReturn:\n"
        f'    """Auto-generated from Verilog-A."""\n'
        f"{body}\n"
    )
    if device.init_cache_refs:
        # Register the cache-compute function via ``@<Name>.setup``.
        block += (
            f"\n@{device.class_name}.setup\n"
            f"def _{device.class_name}_register_setup(*_a, **_kw):\n"
            f"    return _{device.class_name}_setup(*_a, **_kw)\n"
        )
    return block


def _emit_physics_wrapper(device: LoweredDevice) -> str:
    """Render the public physics function as a thin wrapper around ``_combined``.

    The wrapper exists so callers using the conventional
    ``Component(...)`` API get back ``(f_dict, q_dict)`` as before. On
    the Newton hot path the custom JVP installed by ``@va_component``
    bypasses this wrapper entirely and calls ``_combined`` once per
    Newton iteration to get all four outputs in a single trace.

    When the device has an init cache, the public physics function
    declares ``init`` as the first non-reserved positional argument and
    a separate ``@<Name>.setup`` decorator registers the
    ``_<Name>_setup`` cache-compute function on the resulting class.
    """
    decorator_args = _render_decorator_args(device)
    signature = _render_signature(device)
    arg_forward = _render_forwarded_args(device)

    decorator = "@va_component"
    decorator_args = (
        f"{decorator_args},"
        f" jacobian_fn=_{device.class_name}_jacobian,"
        f" combined_fn=_{device.class_name}_combined"
    )
    # Forward ``differentiable_params`` to the decorator so the caller's
    # choice of which params remain JAX leaves vs eqx-static is honoured
    # in the emitted source.  Default ``()`` — all-static, fastest.
    if device.differentiable_params is None:
        decorator_args += ", differentiable_params=None"
    elif device.differentiable_params:
        names = ", ".join(f'"{n}"' for n in device.differentiable_params)
        decorator_args += f", differentiable_params=({names},)"
    # ``differentiable_params=()`` is the @va_component default — no need
    # to render it explicitly.

    block = (
        f"\n{decorator}({decorator_args})\n"
        f"def {device.class_name}({signature}) -> PhysicsReturn:\n"
        f'    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""\n'
        f"    f, q, _j_f, _j_q = _{device.class_name}_combined({arg_forward})\n"
        f"    return f, q\n"
    )
    if device.init_cache_refs:
        block += (
            f"\n@{device.class_name}.setup\n"
            f"def _{device.class_name}_register_setup(*_a, **_kw):\n"
            f"    return _{device.class_name}_setup(*_a, **_kw)\n"
        )
    return block


def _try_eval_literal(expr: str) -> str | None:
    """Return the canonical Python repr if ``expr`` is a literal or trivially
    foldable (binop on two numeric literals).  Else ``None``.

    Used by the post-lowering const-prop pass to detect hoists whose
    definition can be replaced by a single literal — once that happens,
    every use site can be substituted, opening up further folding.
    """
    s = expr.strip()
    # Direct literal — number, bool, signed number.
    if s in ("True", "False"):
        return s
    try:
        v = float(s) if "." in s or "e" in s.lower() else int(s)
        return repr(v) if isinstance(v, float) else str(v)
    except ValueError:
        pass
    # Strip a single outer paren and retry.
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1]
        if inner.count("(") == inner.count(")"):
            return _try_eval_literal(inner)
    return None


_PEEPHOLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # x + 0 / 0 + x → x  (safe — preserves NaN / Inf)
    (re.compile(r"\(([^()]+) \+ 0\.0\)"), r"(\1)"),
    (re.compile(r"\(0\.0 \+ ([^()]+)\)"), r"(\1)"),
    (re.compile(r"\(([^()]+) - 0\.0\)"), r"(\1)"),
    # x * 1 / 1 * x → x  (safe — preserves NaN / Inf)
    (re.compile(r"\(([^()]+) \* 1\.0\)"), r"(\1)"),
    (re.compile(r"\(1\.0 \* ([^()]+)\)"), r"(\1)"),
    (re.compile(r"\(([^()]+) / 1\.0\)"), r"(\1)"),
    # NB: ``x * 0 → 0`` is NOT safe — ``nan * 0 = nan``, ``inf * 0 = nan``.
    # Folding it changes the result whenever an upstream expression hits
    # NaN or Inf in some dynamic range, which manifested as a KLU factor
    # failure during PSP103 DC homotopy when all 783 model params were
    # promoted to literals. Leave it alone and let XLA's runtime semantics
    # handle the multiplication.
    # jnp.where(True/False, t, f) — simple inline form.  Only fires when
    # the condition arg is a bare ``True`` / ``False`` literal; nested
    # whatever-other-call shapes are left alone.
    (re.compile(r"jnp\.where\(\s*True\s*,\s*([^,]+?),\s*[^()]+?\)"), r"\1"),
    (re.compile(r"jnp\.where\(\s*False\s*,\s*[^,]+?,\s*([^()]+?)\)"), r"\1"),
)


def _apply_peepholes(expr: str) -> str:
    """Run all peephole rules to fixed point on a single expression text.

    Each pass is shallow — a rule matches only at the innermost
    parenthesised level (the ``[^()]+`` constraint).  Iterating to fixed
    point lets nested simplifications propagate outward without writing
    a real expression rewriter.  This catches the common cases (chains
    of ``+ 0.0`` or ``* 1.0`` left over from the MIR after a static
    parameter resolved to ``0`` or ``1``) without being a full SCCP.
    """
    while True:
        before = expr
        for pat, rep in _PEEPHOLE_RULES:
            expr = pat.sub(rep, expr)
        if expr == before:
            return expr


def _constprop_pass(
    cse_hoists: list[tuple[str, str]],
    roots: list[str],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Propagate literal constants through hoist chains and apply peepholes.

    The lowering already substitutes ``static_params`` literals into the
    MIR walk, so simple ``binop fold`` cases collapse to numeric literals
    in ``cse_hoists``.  But chains of those literals — e.g. an SSA whose
    definition is ``a + b`` where both ``a`` and ``b`` are themselves
    literal hoists — don't fold further: the lowering's binop folder
    only fires on direct literal operands, not on hoisted SSAs that
    happen to bind literals.  This pass picks up the leftovers:

    1. Find every hoist whose def is a Python-evaluable literal.
    2. Substitute that literal everywhere ``\\bSSA\\b`` appears, both in
       other hoist defs and in the root expressions.
    3. Run peepholes on the modified defs to drop the now-folded
       ``+ 0.0`` / ``* 1.0`` / ``jnp.where(True, ...)`` artifacts.
    4. Re-evaluate which hoists are now literals (the substitution may
       have unblocked them) and repeat to fixed point.

    Plays well with the rest of the pipeline:
    - Runs **before** DCE — because constprop creates new dead hoists
      (the literals' source SSAs are no longer referenced after we
      substitute their value everywhere).
    - Runs **before** single-use inlining — peepholes shrink chains
      that the inliner would otherwise have to re-discover.

    Returns the updated ``(cse_hoists, roots)`` pair.
    """
    defs: dict[str, str] = {ssa: expr for ssa, expr in cse_hoists}
    order: list[str] = [ssa for ssa, _ in cse_hoists]

    while True:
        changed = False

        # Phase 1: identify literal hoists.
        literal_subst: dict[str, str] = {}
        for ssa in order:
            lit = _try_eval_literal(defs[ssa])
            if lit is not None:
                literal_subst[ssa] = lit

        if literal_subst:
            # Phase 2: substitute literals into every other def + roots.
            big_pat = re.compile(
                r"\b(" + "|".join(re.escape(s) for s in literal_subst) + r")\b"
            )

            def _subst(text: str) -> str:
                return big_pat.sub(lambda m: literal_subst[m.group(0)], text)

            for ssa in order:
                if ssa in literal_subst:
                    continue
                new = _subst(defs[ssa])
                if new != defs[ssa]:
                    defs[ssa] = new
                    changed = True
            for i, r in enumerate(roots):
                new = _subst(r)
                if new != r:
                    roots[i] = new
                    changed = True

        # Phase 3: peephole on every def + root.
        for ssa in order:
            new = _apply_peepholes(defs[ssa])
            if new != defs[ssa]:
                defs[ssa] = new
                changed = True
        for i, r in enumerate(roots):
            new = _apply_peepholes(r)
            if new != r:
                roots[i] = new
                changed = True

        if not changed:
            break

    return [(s, defs[s]) for s in order], roots


def _substitute_static_params(
    cse_hoists: list[tuple[str, str]],
    static_params: dict[str, int | float],
) -> list[tuple[str, str]]:
    """Replace bare references to static-param names with their literal values.

    Walks every hoist's expression text and substitutes ``\\bname\\b``
    with ``repr(value)`` for each ``(name, value)`` in ``static_params``.
    Handles the rare case where the lowering's env-based substitution
    didn't reach every emitted expression (parameter-default fallback
    paths in particular).  Idempotent: a hoist that already uses the
    literal value is unchanged.
    """
    if not static_params:
        return cse_hoists
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in static_params) + r")\b"
    )

    def _sub(text: str) -> str:
        return pattern.sub(lambda m: repr(static_params[m.group(0)]), text)

    return [(ssa, _sub(expr)) for ssa, expr in cse_hoists]


def _live_after_constprop(
    cse_hoists: list[tuple[str, str]],
    roots: list[str],
) -> set[str]:
    """Compute the live set on a (cse_hoists, roots) pair, post-constprop.

    Mirrors ``_live_ssas`` but operates on raw lists so the const-prop
    pass can re-DCE without rebuilding a ``LoweredDevice``.
    """
    defs = {ssa: expr for ssa, expr in cse_hoists}
    live: set[str] = set()
    stack: list[str] = []
    for r in roots:
        for name in _SSA_NAME.findall(r):
            if name in defs and name not in live:
                live.add(name)
                stack.append(name)
    while stack:
        n = stack.pop()
        for ref in _SSA_NAME.findall(defs[n]):
            if ref in defs and ref not in live:
                live.add(ref)
                stack.append(ref)
    return live


def _inline_single_use_ssas(
    cse_hoists: list[tuple[str, str]],
    roots: list[str],
    *,
    init_cache_refs: set[str],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Inline SSAs that are referenced from exactly one site.

    A hoist that has refcount = 1 buys nothing from CSE — keeping it as
    a named local just adds a JAX tracer op and a line of Python source
    to the JIT trace. Inlining such hoists into their unique use site
    is functionally equivalent (no expression duplication, since they
    were used once) and shrinks the trace.

    The algorithm runs to a fixed point: after inlining a hoist, its
    parent's refcount may drop to 1, opening up the next round.

    ``init_cache_refs`` are the SSAs that bind to ``_init_cache[i]``
    rather than to a real expression — those are NOT inlined because
    the emitter wants their assignment line to fire the cheap array
    indexing op exactly once (inlining ``_init_cache[5]`` into N use
    sites would do N array gathers; refcount = 1 hoists are already
    a net win to inline; refcount > 1 cache refs are kept as named
    locals).  Since this routine only inlines refcount = 1 hoists,
    the cache-ref check is a defensive belt-and-braces guard for the
    rare case that DCE leaves a single-use lookup that's already
    handled by the emitter's ``_init_cache[i] = ...`` line.

    Returns ``(new_cse_hoists, new_roots)`` — the surviving hoists
    in their original order, and the (possibly modified) root
    expressions that picked up inlined sub-trees.
    """
    defs: dict[str, str] = {ssa: expr for ssa, expr in cse_hoists}
    order: list[str] = [ssa for ssa, _ in cse_hoists]

    # Build user → list of consumers index once, then maintain incrementally.
    # ``consumers[ssa]`` is the list of places (other defs or root indices,
    # encoded as ``("def", ssa_name)`` or ``("root", idx)``) that reference
    # this SSA. Using a list rather than recomputing refcount on every
    # iteration takes the inliner from O(n²) (PSP103: ~18 s) down to ~O(n).
    consumers: dict[str, list[tuple[str, object]]] = {ssa: [] for ssa in order}
    for ssa in order:
        for name in _SSA_NAME.findall(defs[ssa]):
            if name in consumers:
                consumers[name].append(("def", ssa))
    for i, r in enumerate(roots):
        for name in _SSA_NAME.findall(r):
            if name in consumers:
                consumers[name].append(("root", i))

    # Initial work-queue: SSAs with exactly one consumer.
    queue: list[str] = [
        ssa
        for ssa, cs in consumers.items()
        if len(cs) == 1 and ssa not in init_cache_refs
    ]
    in_queue: set[str] = set(queue)
    dead: set[str] = set()

    while queue:
        target = queue.pop()
        in_queue.discard(target)
        if target in dead or len(consumers.get(target, ())) != 1:
            continue

        kind, where = consumers[target][0]
        rep = f"({defs[target]})"
        pat = re.compile(rf"\b{re.escape(target)}\b")
        if kind == "def":
            new_expr, n = pat.subn(rep, defs[where])  # type: ignore[arg-type]
            if n == 0:
                continue
            defs[where] = new_expr  # type: ignore[index]
            # Re-index: every name in ``rep`` now has ``where`` as a consumer
            # in addition to the old uses inside the (single) target def.
            new_names = set(_SSA_NAME.findall(rep))
            for name in new_names:
                if name in consumers and name != target:
                    cl = consumers[name]
                    cl.append(("def", where))
                    if (
                        name not in dead
                        and len(cl) == 1
                        and name not in init_cache_refs
                        and name not in in_queue
                    ):
                        queue.append(name)
                        in_queue.add(name)
        else:
            idx = where  # type: ignore[assignment]
            new_r, n = pat.subn(rep, roots[idx])  # type: ignore[index]
            if n == 0:
                continue
            roots[idx] = new_r  # type: ignore[index]
            new_names = set(_SSA_NAME.findall(rep))
            for name in new_names:
                if name in consumers and name != target:
                    cl = consumers[name]
                    cl.append(("root", idx))
                    if (
                        name not in dead
                        and len(cl) == 1
                        and name not in init_cache_refs
                        and name not in in_queue
                    ):
                        queue.append(name)
                        in_queue.add(name)

        # Mark target dead and drop its outgoing consumer edges.
        dead.add(target)
        for name in _SSA_NAME.findall(defs[target]):
            if name in consumers:
                # Remove one occurrence of target's old consumer edge from name.
                cl = consumers[name]
                # We can't tell the exact edge tuple to remove; rebuild without target.
                consumers[name] = [
                    c for c in cl if not (c[0] == "def" and c[1] == target)
                ]
                if (
                    name not in dead
                    and len(consumers[name]) == 1
                    and name not in init_cache_refs
                    and name not in in_queue
                ):
                    queue.append(name)
                    in_queue.add(name)

    new_order = [s for s in order if s not in dead]
    return [(s, defs[s]) for s in new_order], roots


def _live_init_cache_slots(device: LoweredDevice) -> list[str]:
    """Return only the ``init_cache_refs`` whose value is referenced by the
    eval body (after constprop / DCE).  Order is preserved.

    The lowering's ``init_cache_refs`` list grows to one entry per cslot
    output, but DCE in the eval body finds many of these are never read.
    For PSP103 with ``static_params={"TYPE": 1}``, 220 of 407 slots are
    dead — they get computed at instantiation and stored in the cache
    array, then never loaded.  Returning a compacted live-only list lets
    ``_emit_cache_fn`` skip the wasted compute and lets the eval body
    use a smaller cache index.
    """
    if not device.init_cache_refs:
        return []
    live = _live_ssas(device)
    seen: set[str] = set()
    return [
        ref
        for ref in device.init_cache_refs
        if ref in live and not (ref in seen or seen.add(ref))
    ]


def _live_ssas(device: LoweredDevice) -> set[str]:
    """Return the set of SSA names that are reachable from any return value.

    Walks back from the final ``f_dict`` / ``q_dict`` / Jacobian-matrix
    expressions through ``cse_hoists`` to find every named local that
    actually contributes to an output. Hoists not in the returned set
    are dead — emit-time DCE drops them. At PSP103 scale this prunes
    several hundred entirely-dead intermediates that survived the
    lowering's CSE because they were *referenced* by other hoists, but
    whose chain ends in a dead-end after static-param substitution.

    Init-cache references (``i_v123 = _init_cache[N]``) that aren't
    reached are dropped along with the matrix entry — the underlying
    cache slot still gets computed in the setup function, but the
    per-step lookup is gone.
    """
    # Build a definition map for fast back-pointer resolution.
    defs: dict[str, str] = {ssa: expr for ssa, expr in device.cse_hoists}

    # Roots: every expression that ends up in a return value or in the
    # emitted matrix construction.
    roots: list[str] = []
    roots.extend(device.f_expressions.values())
    roots.extend(device.q_expressions.values())
    roots.extend(device.jacobian_resist.values())
    roots.extend(device.jacobian_react.values())
    roots.extend(stmt for stmt in device.preamble_stmts)

    live: set[str] = set()
    stack: list[str] = []
    for root_expr in roots:
        for name in _SSA_NAME.findall(root_expr):
            if name in defs and name not in live:
                live.add(name)
                stack.append(name)
    while stack:
        name = stack.pop()
        for ref in _SSA_NAME.findall(defs[name]):
            if ref in defs and ref not in live:
                live.add(ref)
                stack.append(ref)
    return live


def _split_where_args(inner: str) -> list[str]:
    """Split 'cond, true, false' on commas at parenthesis depth 0.

    Handles nested function calls such as ``jnp.where(a, jnp.exp(b), c)``
    where a naive comma-split would yield the wrong result.
    """
    depth = 0
    parts: list[str] = []
    cur: list[str] = []
    for ch in inner:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


_WHERE_PREFIX = "jnp.where("


def _batch_where_hoists(hoists: list[tuple[str, str]]) -> list[str]:
    """Render hoist assignments as lines, batching consecutive same-condition ``jnp.where`` calls.

    Works on the FINAL (post-transform) hoist list.  Groups consecutive
    entries whose expression is ``jnp.where(cond, true, false)`` with the
    SAME condition string into a single ``jax.tree_util.tree_map`` call.
    All other entries are rendered as plain ``    ssa = expr`` lines.

    Two consecutive PHIs with the same condition become:
    ``(v1, v2,) = jax.tree_util.tree_map(lambda _t, _f: jnp.where(cond, _t, _f), (t1, t2,), (f1, f2,))``

    Singleton groups fall through to individual ``jnp.where`` lines.
    """
    lines: list[str] = []
    batch: list[tuple[str, str, str]] = []  # (ssa, true_expr, false_expr)
    batch_cond: str | None = None

    def flush() -> None:
        if not batch:
            return
        if len(batch) == 1:
            ssa, t, f = batch[0]
            lines.append(f"    {ssa} = jnp.where({batch_cond}, {t}, {f})")
        else:
            lhs = "(" + ", ".join(b[0] for b in batch) + ",)"
            true_tup = "(" + ", ".join(b[1] for b in batch) + ",)"
            false_tup = "(" + ", ".join(b[2] for b in batch) + ",)"
            lines.append(
                f"    {lhs} = jax.tree_util.tree_map("
                f"lambda _t, _f: jnp.where({batch_cond}, _t, _f), "
                f"{true_tup}, {false_tup})"
            )
        batch.clear()

    for ssa, expr in hoists:
        if expr.startswith(_WHERE_PREFIX) and expr.endswith(")"):
            args = _split_where_args(expr[len(_WHERE_PREFIX) : -1])
            if len(args) == 3:
                cond, true_e, false_e = args
                if cond != batch_cond:
                    flush()
                    batch_cond = cond
                batch.append((ssa, true_e, false_e))
                continue
        flush()
        batch_cond = None
        lines.append(f"    {ssa} = {expr}")

    flush()
    return lines


def _emit_hoists_batched(
    hoists: list[tuple[str, str]],
    phi_resolutions: dict[str, PhiResolution],
    live: set[str],
    out: list[str],
) -> None:
    """Emit hoist assignments into ``out``, batching consecutive diamond PHIs.

    Consecutive ``jnp.where`` calls that share the same condition are grouped
    into a single ``jax.tree_util.tree_map`` call, which reduces emitted
    source size and JAX trace overhead for models with many PHI nodes at the
    same merge point (e.g. PSP103 has 200-400 diamond PHIs, many sharing
    a common ``if (SW... > 0)`` guard).

    Non-consecutive same-condition PHIs and non-batchable PHIs (Case 1b
    nested diamonds, Case 1.5 SCCP shortcuts, Case 2 fallback) are emitted
    as individual ``ssa = <expr>`` lines in their original SSA order.
    """
    batch: list[tuple[str, PhiResolution]] = []
    batch_key: tuple[str, bool] | None = None

    def flush() -> None:
        if not batch:
            return
        if len(batch) == 1:
            ssa, phi_res = batch[0]
            out.append(
                f"    {ssa} = jnp.where({phi_res.cond_ssa}, {phi_res.true_expr}, {phi_res.false_expr})"
            )
        else:
            cond = batch[0][1].cond_ssa
            lhs = "(" + ", ".join(b[0] for b in batch) + ",)"
            true_tup = "(" + ", ".join(b[1].true_expr for b in batch) + ",)"
            false_tup = "(" + ", ".join(b[1].false_expr for b in batch) + ",)"
            out.append(
                f"    {lhs} = jax.tree_util.tree_map("
                f"lambda _t, _f: jnp.where({cond}, _t, _f), "
                f"{true_tup}, {false_tup})"
            )
        batch.clear()

    for ssa, expr in hoists:
        if ssa not in live:
            continue
        if ssa in phi_resolutions:
            phi_res = phi_resolutions[ssa]
            key = (phi_res.cond_ssa, phi_res.cond_negated)
            if key != batch_key:
                flush()
                batch_key = key
            batch.append((ssa, phi_res))
        else:
            flush()
            batch_key = None
            out.append(f"    {ssa} = {expr}")

    flush()


def _hoist_lines(device: LoweredDevice) -> list[str]:
    """Return the ``    ssa = expr`` lines for the physics / Jacobian preamble.

    When ``device.init_cache_refs`` is non-empty the init-derived hoists are
    replaced by cheap ``_init_cache[i]`` index lookups; the heavy expressions
    live in ``_<Name>_compute_cache`` which runs once at instantiation.

    A ``None`` guard is prepended so the decorator's dry-run (which passes
    ``_init_cache=None``) gets a zero-filled fallback without raising.  At JIT
    trace time the guard is a Python-level ``is`` check that short-circuits
    immediately since the live value is a JAX array, not ``None``.

    Performs emit-time DCE: hoists not transitively used by any output
    expression are dropped.
    """
    live = _live_ssas(device)
    phi_resolutions = device.phi_resolutions
    if device.init_cache_refs:
        # ``init`` is always provided by the framework (either positionally
        # via ``@<Name>.setup``-registered cache fn, or as the empty-dict
        # placeholder during the circulax-side dry-run, where the body's
        # KeyError is caught and suppressed). No guard needed.
        lines: list[str] = []
        seen: set[str] = set()
        for i, ref in enumerate(device.init_cache_refs):
            if ref in seen or ref not in live:
                continue
            lines.append(f"    {ref} = init[{i}]")
            seen.add(ref)
        _emit_hoists_batched(
            device.cse_hoists[device.init_hoist_count :], phi_resolutions, live, lines
        )
        return lines
    lines = []
    _emit_hoists_batched(device.cse_hoists, phi_resolutions, live, lines)
    return lines


def _render_hoists(device: LoweredDevice) -> str:
    """Render hoisted subexpressions as a block of ``    ssa = expr`` lines.

    Emitted at the top of both the physics function and the Jacobian
    sibling.  When an init cache is present the init hoists are replaced by
    ``_init_cache[i]`` index lookups (the heavy computation runs once at
    instantiation via ``_<Name>_compute_cache``).
    """
    lines = _hoist_lines(device)
    return "\n".join(lines) + "\n" if lines else ""


def _emit_cache_fn(device: LoweredDevice) -> str:
    """Render the ``_<Name>_setup`` function when the device has init hoists.

    This function takes the same parameters as the physics function (minus
    ``signals``, ``s``, and ``init`` itself) and returns a
    ``jnp.ndarray`` of shape ``(N,)`` containing the cslot-output values
    that the eval body indexes into via ``init[i]``.

    Registered onto the component class via ``@<Name>.setup`` so circulax
    invokes it inside the JAX trace each evaluation — XLA constant-folds
    when params are static, AD flows when they aren't.

    Runs the same constprop / DCE / single-use-inlining passes the
    eval body gets — for PSP103 this collapses a chunk of the 8 k-line
    setup body that the lowering's binop folder didn't catch.  The
    cache only runs once per instance at construction, so the per-step
    cost is unaffected, but the JIT compile of ``compute_cache`` (which
    happens lazily on first instantiation) benefits.
    """
    if not device.init_cache_refs:
        return ""
    init_hoists = list(device.cse_hoists[: device.init_hoist_count])
    # Only emit / compute / return the cache slots actually referenced
    # by the eval body.  Drops 50%+ of slots on PSP103 — these were
    # being computed at instantiation time and never read.
    refs = _live_init_cache_slots(device)
    roots = list(refs)  # the cache returns these by name
    # Belt-and-braces sweep: replace bare ``static_params`` names with
    # their literal values before constprop / DCE / inlining run.
    # The lowering's env-based substitution misses some MIR paths
    # (parameter-default fallbacks in particular); substituting at
    # source level closes that gap. ``init_cache_refs`` may itself
    # contain bare param names rather than hoisted SSAs, so the same
    # sweep applies to the roots.
    init_hoists = _substitute_static_params(init_hoists, device.static_params)
    if device.static_params:
        pat = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in device.static_params) + r")\b"
        )
        roots = [
            pat.sub(lambda m: repr(device.static_params[m.group(0)]), r) for r in roots
        ]
    init_hoists, roots = _constprop_pass(init_hoists, roots)
    live = _live_after_constprop(init_hoists, roots)
    init_hoists = [(s, e) for s, e in init_hoists if s in live]
    init_hoists, roots = _inline_single_use_ssas(
        init_hoists,
        roots,
        init_cache_refs=set(refs),
    )
    kwargs = [f"{name}: {ty} = {default}" for name, ty, default in device.params]
    sig = ", ".join(kwargs)
    hoist_block = "\n".join(_batch_where_hoists(init_hoists))
    refs_str = ", ".join(roots)
    return (
        f"\ndef _{device.class_name}_setup({sig}) -> jnp.ndarray:\n"
        f"{hoist_block}\n"
        f"    return jnp.array([{refs_str}])\n"
    )


def _emit_combined(device: LoweredDevice) -> str:
    """Render the single ``_<Name>_combined`` function.

    Returns ``(f_dict, q_dict, J_resist, J_react)`` from one hoist block.
    All shared subexpressions between the physics residual and the
    analytical Jacobian appear exactly once, so JAX's tracer materializes
    them once and XLA receives a graph half the size of the
    "two functions side-by-side" layout that preceded this.

    Also runs single-use SSA inlining over the eval-phase hoists: any
    hoist with refcount = 1 gets folded into its unique consumer.  At
    PSP103 scale this drops the named-local count from ~6 k post-DCE
    to ~1.8 k, removing JAX tracer overhead per Newton iter.
    """
    unknowns = device.ports + device.states
    n = len(unknowns)
    idx = {name: i for i, name in enumerate(unknowns)}

    hoists, f_exprs, q_exprs, jac_r, jac_q, preamble_stmts = _prep_combined_body(device)

    j_f_rows = _render_matrix_rows(jac_r, idx, n)
    j_q_rows = _render_matrix_rows(jac_q, idx, n)

    body_lines: list[str] = [f"    {stmt}" for stmt in preamble_stmts]
    body_lines.extend(hoists)
    f_block = _render_dict_literal(f_exprs)
    q_block = _render_dict_literal(q_exprs)
    body_lines.append(f"    j_resist = jnp.array([\n{j_f_rows}    ])")
    body_lines.append(f"    j_react = jnp.array([\n{j_q_rows}    ])")
    body_lines.append(f"    return {f_block}, {q_block}, j_resist, j_react")
    body = "\n".join(body_lines)

    signature = _render_signature(device)
    return (
        f"\ndef _{device.class_name}_combined({signature}) -> tuple:\n"
        f'    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""\n'
        f"{body}\n"
    )


def _prep_combined_body(
    device: LoweredDevice,
) -> tuple[
    list[str],
    dict[str, str],
    dict[str, str],
    dict[tuple[str, str], str],
    dict[tuple[str, str], str],
    list[str],
]:
    """Build the eval-phase hoist block and substituted return expressions.

    Sequence:
    1. Materialise the eval-phase hoist list, replacing each live
       ``init_cache_ref`` with its ``_init_cache[N]`` lookup form so
       the inliner sees one consistent representation.
    2. DCE: drop hoists not transitively used by any return expression
       or preamble statement.
    3. Single-use inlining: fold every refcount = 1 hoist into its
       unique consumer until a fixed point is reached.
    4. Render the surviving hoists into ``    name = expr`` lines (with
       the ``if _init_cache is None`` guard prepended when relevant).

    Returns the tuple consumed by ``_emit_combined``.
    """
    live = _live_ssas(device)

    # Phase 1: build the eval-phase hoist list.  Use compact indices for
    # the live-only cache slots so the eval body's ``_init_cache[i]``
    # lookups match what ``_emit_cache_fn`` returns.
    init_cache_lookup: dict[str, str] = {}
    live_refs = _live_init_cache_slots(device)
    for compact_i, ref in enumerate(live_refs):
        init_cache_lookup[ref] = f"init[{compact_i}]"

    eval_hoists: list[tuple[str, str]] = []
    # Init-cache lookups come first so the inliner sees them as defs.
    for ref, lookup in init_cache_lookup.items():
        if ref in live:
            eval_hoists.append((ref, lookup))
    for ssa, expr in device.cse_hoists[device.init_hoist_count :]:
        if ssa in live:
            eval_hoists.append((ssa, expr))

    # Belt-and-braces: replace any remaining bare reference to a
    # ``static_params`` name with its literal value.  The lowering's
    # ``_inject_static_params`` substitutes via the SSA env, but an
    # expression that bypassed the env (e.g. emitted directly from a
    # parameter-default fallback path) can still mention the param by
    # name.  Catching it here keeps the ``static_params=…`` knob
    # robust against lowering edge cases.
    eval_hoists = _substitute_static_params(eval_hoists, device.static_params)

    # Phase 3: collect roots and run inliner.
    roots: list[str] = []
    f_keys = list(device.f_expressions.keys())
    q_keys = list(device.q_expressions.keys())
    jac_r_keys = list(device.jacobian_resist.keys())
    jac_q_keys = list(device.jacobian_react.keys())
    roots.extend(device.f_expressions[k] for k in f_keys)
    roots.extend(device.q_expressions[k] for k in q_keys)
    roots.extend(device.jacobian_resist[k] for k in jac_r_keys)
    roots.extend(device.jacobian_react[k] for k in jac_q_keys)
    roots.extend(device.preamble_stmts)

    # Belt-and-braces sweep on roots too — same rationale as for the
    # eval_hoists loop above.  Done after the eval-hoists rebind so
    # roots that pull from substituted hoists still get their bare
    # ``CF``-style references resolved.
    if device.static_params:
        pat = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in device.static_params) + r")\b"
        )
        roots = [
            pat.sub(lambda m: repr(device.static_params[m.group(0)]), r) for r in roots
        ]

    # Constprop / peephole first — fold literal chains, simplify ``x + 0``
    # / ``x * 1`` / ``jnp.where(True, ...)`` artifacts left over by the
    # lowering's binop folder, propagate literals across SSA boundaries.
    # Doing this *before* inlining shrinks the expressions inlining will
    # paste into consumers, which keeps the resulting source small.
    eval_hoists, roots = _constprop_pass(eval_hoists, roots)

    # Re-run live-set walk: constprop may have orphaned hoists whose
    # value is no longer referenced (literal chain collapse).
    live_after = _live_after_constprop(eval_hoists, roots)
    eval_hoists = [(s, e) for s, e in eval_hoists if s in live_after]

    # The inliner is conservative — init_cache_refs whose lookup form is
    # ``_init_cache[N]`` are atomic and cheap; folding them into the
    # only consumer is a win, so don't exempt them here.
    eval_hoists, roots = _inline_single_use_ssas(
        eval_hoists,
        roots,
        init_cache_refs=set(),
    )

    # Phase 4: split the substituted roots back into the original dicts.
    n_f, n_q, n_jr, n_jq = len(f_keys), len(q_keys), len(jac_r_keys), len(jac_q_keys)
    cursor = 0
    f_subst = dict(zip(f_keys, roots[cursor : cursor + n_f]))
    cursor += n_f
    q_subst = dict(zip(q_keys, roots[cursor : cursor + n_q]))
    cursor += n_q
    jr_subst = dict(zip(jac_r_keys, roots[cursor : cursor + n_jr]))
    cursor += n_jr
    jq_subst = dict(zip(jac_q_keys, roots[cursor : cursor + n_jq]))
    cursor += n_jq
    pre_subst = roots[cursor:]

    # Phase 5: render the surviving hoists with phi batching.  Consecutive
    # ``jnp.where(cond, …)`` lines sharing the same condition are folded into
    # a single ``jax.tree_util.tree_map`` call, reducing trace overhead.
    hoist_lines = _batch_where_hoists(eval_hoists)

    return hoist_lines, f_subst, q_subst, jr_subst, jq_subst, pre_subst


def _emit_jacobian_wrapper(device: LoweredDevice) -> str:
    """Render ``_<Name>_jacobian`` as a thin wrapper around ``_combined``.

    Kept for backward compatibility with the ``jacobian_fn=`` decorator
    contract — callers that construct a ``@va_component`` from a Python
    physics function and a separately-supplied ``jacobian_fn`` (e.g.
    hand-written tests) still expect a callable returning
    ``(J_resist, J_react)``. The Newton hot path bypasses this wrapper
    by going through ``combined_fn=`` instead.
    """
    signature = _render_signature(device)
    arg_forward = _render_forwarded_args(device)
    return (
        f"\ndef _{device.class_name}_jacobian({signature}) -> tuple:\n"
        f'    """Jacobian wrapper — slices the combined function\'s output."""\n'
        f"    _f, _q, j_resist, j_react = _{device.class_name}_combined({arg_forward})\n"
        f"    return j_resist, j_react\n"
    )


def _render_matrix_rows(
    entries: dict[tuple[str, str], str], idx: dict[str, int], n: int
) -> str:
    """Render a dense ``(n, n)`` matrix as a list-of-lists source snippet.

    Missing ``(row, col)`` pairs default to ``0.0``. Emits one row per
    line, each row wrapped in brackets; ``ruff format`` handles the
    inner wrapping when individual rows exceed the line length.
    """
    # Build dense layout: dense[row][col] = expr_str.
    dense: list[list[str]] = [["0.0"] * n for _ in range(n)]
    for (row, col), expr in entries.items():
        if row in idx and col in idx:
            dense[idx[row]][idx[col]] = expr
    lines = []
    for row_vals in dense:
        joined = ", ".join(row_vals)
        lines.append(f"        [{joined}],")
    return "\n".join(lines) + "\n"


def _render_decorator_args(device: LoweredDevice) -> str:
    """Render the ``ports=(...), states=(...)`` part of the decorator."""
    parts: list[str] = [f"ports={_tuple_literal(device.ports)}"]
    if device.states:
        parts.append(f"states={_tuple_literal(device.states)}")
    return ", ".join(parts)


def _render_signature(device: LoweredDevice) -> str:
    """Render the physics / Jacobian function's formal parameter list.

    Leading positional args are fixed by the circulax decorator contract
    (``signals, s`` for ``@component`` / ``signals, s, t`` for ``@source``),
    followed by the device's declared parameters. Float params get plain
    Python-literal defaults; ``int`` / ``str`` params are wrapped in
    ``eqx.field(static=True, default=...)`` so Equinox marks them as
    trace-time-constant aux data rather than traced leaves.

    When the device has an init cache, ``_init_cache`` is appended as a
    trailing kwarg with a zero-array default of the correct size.  The
    custom ``__init__`` generated by ``_build_component`` always replaces
    this default with the pre-computed values.
    """
    fixed: list[str] = ["signals: Signals", "s: States"]
    if device.uses_time:
        fixed.append("t: float")
    # ``init`` goes between the reserved positionals and the param kwargs
    # so circulax's signature introspection picks it up as the first
    # non-reserved positional — that's the trigger for ``@<Name>.setup``
    # injection. No default: the framework always provides a value
    # (either the registered setup result, or the empty-dict placeholder
    # during the decorator dry-run).
    init_slot = ["init"] if device.init_cache_refs else []
    kwargs = [
        _render_param_kwarg(name, ty, default) for name, ty, default in device.params
    ]
    return ", ".join(fixed + init_slot + kwargs)


def _render_forwarded_args(device: LoweredDevice) -> str:
    """Render the call-site argument list that forwards everything to ``_combined``.

    Used by the wrapper functions: each declared param is forwarded as
    ``name=name`` so the wrapper doesn't have to duplicate default
    values.  Reserved positionals (``signals``, ``s``, optional ``t``)
    and ``_init_cache`` are passed positionally / by keyword as needed.
    """
    parts = ["signals", "s"]
    if device.uses_time:
        parts.append("t")
    # ``init`` is positional, matching its position in the public physics
    # signature emitted by ``_render_signature``.
    if device.init_cache_refs:
        parts.append("init")
    parts.extend(f"{name}={name}" for name, _ty, _d in device.params)
    return ", ".join(parts)


def _render_param_kwarg(name: str, ty: str, default: str) -> str:
    """Render one kwarg in the physics / Jacobian signature.

    - ``float`` params become plain ``name: float = default`` — JAX
      handles these as dynamic leaves (differentiable).
    - ``int`` params become plain ``name: int = default`` too. JAX wraps
      them as ``weak`` arrays on trace; ``jax.grad`` produces a zero
      gradient through them, which matches the ``.va`` intent of an
      integer being a non-differentiable switch without needing Equinox
      static-field machinery.
    - ``str`` params get wrapped in ``eqx.field(static=True, ...)``
      because JAX genuinely can't trace strings — they must land in the
      Equinox aux data or ``jax.jit`` errors out.

    Reserving ``eqx.field`` for strings sidesteps a gotcha in circulax's
    ``@component`` dry-run: the decorator calls the physics function
    with its default kwargs at decoration time, and ``eqx.field(...)``
    evaluates to a ``Field`` object (not the underlying default), which
    breaks any ``<int-default> + 1`` expression inside the body. Strings
    don't participate in arithmetic, so the dry-run tolerates a ``Field``
    passed as the string kwarg.
    """
    if ty == "str":
        return f"{name}: str = eqx.field(static=True, default={default})"
    return f"{name}: {ty} = {default}"


def _render_body(device: LoweredDevice) -> str:
    """Render the body statements and the final ``return (f, q)`` line."""
    lines: list[str] = [f"    {stmt}" for stmt in device.preamble_stmts]
    lines.extend(_hoist_lines(device))
    f_block = _render_dict_literal(device.f_expressions)
    q_block = _render_dict_literal(device.q_expressions)
    lines.append(f"    return {f_block}, {q_block}")
    return "\n".join(lines)


def _render_dict_literal(d: dict[str, str]) -> str:
    if not d:
        return "{}"
    entries = [f'"{k}": {v}' for k, v in d.items()]
    return "{" + ", ".join(entries) + "}"


def _tuple_literal(items: list[str]) -> str:
    if len(items) == 1:
        return f'("{items[0]}",)'
    inner = ", ".join(f'"{x}"' for x in items)
    return f"({inner})"


__all__ = ["emit_source", "write_source"]
