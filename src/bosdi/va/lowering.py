"""MIR → Python expression trees for circulax ``@component`` emission.

Consumes a parsed :class:`circulax.va.mir.CompiledModule`, walks the init and
eval functions, and produces a :class:`LoweredDevice` that names the
circulax ports and states, lists the keyword parameters, and gives one
Python expression per DAE residual contribution (resistive vs reactive).

Handles:

- Single-block eval (resistor / capacitor / inductor) and multi-block
  eval with diamond-shaped control flow from ``analog`` ``if/else``
  (diode). Phi nodes whose predecessor blocks share a common
  ``br``-terminating ancestor lift cleanly to ``jnp.where(cond, t, f)``;
  fan-in phis from the parameter-clamping path fall back to
  "first non-zero resolved edge".
- Builtin callbacks: ``ddt`` / ``ddx_*`` as identity (their DAE
  contributions are captured elsewhere), ``$simparam("name")`` as a
  kwarg reference, ``set_Invalid`` and ``collapse_*`` as silent no-ops.
- Internal nodes declared in the module header — surfaced as circulax
  states named ``v_<.va_name>`` with their residual contributions
  keyed accordingly.
- Parameter defaults, when a companion ``.va`` source is available
  (see :func:`circulax.va.va_defaults.parse_va_defaults`).
- Division routed through ``jnp.divide`` so that ``0.0 / 0.0`` at
  circulax's eager dry-run produces ``nan`` instead of a
  ``ZeroDivisionError`` (relevant when a ``.va`` param defaults to 0,
  e.g. the diode's ``Rs``).

Known shortfalls (deliberate):

- User-defined ``analog function`` calls (``lexp`` / ``spicepnjlim``
  inside the diode) still raise ``LoweringError``. Handling them
  needs either inlining their bodies or emitting helper functions.
- Phi nodes outside the simple-diamond pattern (three-way, nested,
  loop-carried) fall back to the first-non-zero heuristic.
- **Noise contributions** (``DaeSystem.noise_sources``) are not
  lowered. Circulax doesn't yet have a noise-analysis path; when
  it does we'll surface each noise source as an extra output of the
  physics function. Today we just ignore ``noise_sources`` entries.
"""

from __future__ import annotations

import ast
import math
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from .mir import (
    CachedValues,
    CompiledModule,
    Constant,
    CurrentKind,
    DaeResidual,
    EnableLimInput,
    Function,
    HiddenStateInput,
    HirInterner,
    InputKind,
    Inst,
    NewStateInput,
    ParamGivenRef,
    ParamRef,
    ParamSysFunInput,
    PortConnectedInput,
    PrevStateInput,
    TemperatureInput,
    Voltage,
)
from .va_defaults import ParamSpec


class LoweringError(RuntimeError):
    """Raised when the MIR shape can't be lowered to circulax Python."""


# Opcodes the lowering recognises. Anything else raises LoweringError so we
# know what to add next rather than silently emitting wrong code.
_BINOPS: dict[str, tuple[str, int]] = {
    # opcode -> (python operator, precedence)
    # Precedence: 1=low (or/and), 4=comparison, 6=additive, 7=multiplicative, 10=unary/atom
    # Division deliberately omitted — we emit ``jnp.divide(a, b)`` to keep
    # the circulax dry-run (which uses Python floats with default kwargs)
    # from raising ZeroDivisionError on ``0.0 / 0.0`` patterns that appear
    # when a ``.va``-declared default is 0 (e.g. the diode's ``Rs = 0``).
    "fadd": ("+", 6),
    "fsub": ("-", 6),
    "fmul": ("*", 7),
    "frem": ("%", 7),
    "iadd": ("+", 6),
    "isub": ("-", 6),
    "imul": ("*", 7),
    "idiv": ("//", 7),
    "irem": ("%", 7),
    "flt": ("<", 4),
    "fgt": (">", 4),
    "fle": ("<=", 4),
    "fge": (">=", 4),
    "feq": ("==", 4),
    "fne": ("!=", 4),
    "ilt": ("<", 4),
    "igt": (">", 4),
    "ile": ("<=", 4),
    "ige": (">=", 4),
    "ieq": ("==", 4),
    "ine": ("!=", 4),
    "beq": ("==", 4),
    "bne": ("!=", 4),
}

_UNARYOPS: dict[str, str] = {
    "fneg": "-",
    "ineg": "-",
    "bnot": "not ",
    "inot": "~",
}

# Math intrinsics with a single argument.
_MATH1: dict[str, str] = {
    "sqrt": "jnp.sqrt",
    "exp": "jnp.exp",
    "ln": "jnp.log",
    "log": "jnp.log10",
    "floor": "jnp.floor",
    "ceil": "jnp.ceil",
    "sin": "jnp.sin",
    "cos": "jnp.cos",
    "tan": "jnp.tan",
    "asin": "jnp.arcsin",
    "acos": "jnp.arccos",
    "atan": "jnp.arctan",
    "sinh": "jnp.sinh",
    "cosh": "jnp.cosh",
    "tanh": "jnp.tanh",
    "asinh": "jnp.arcsinh",
    "acosh": "jnp.arccosh",
    "atanh": "jnp.arctanh",
}

_MATH2: dict[str, str] = {
    "pow": "jnp.power",
    "hypot": "jnp.hypot",
    "atan2": "jnp.arctan2",
    # Division routed through ``jnp.divide`` to survive ``0.0 / 0.0``
    # during circulax's eager dry-run (produces ``nan`` instead of raising).
    "fdiv": "jnp.divide",
}

# Pure-pass-through: ``v = optbarrier x`` means ``v = x``.
_PASSTHROUGH: set[str] = {"optbarrier"}

# Type-cast opcodes — numeric/boolean reinterpretations that lower to identity
# in a JAX-traceable world. Treated like passthroughs for CSE purposes.
_CAST_OPS: set[str] = {
    "ifcast",
    "ficast",
    "fbcast",
    "bfcast",
    "bicast",
    "ibfcast",
    "sibitcast",
    "fibcast",
}

# Expression text that's already a bare identifier (optionally dotted).
# These are worthless to hoist — ``v_foo = v_bar`` only adds source noise.
# Matches ``v42``, ``signals.A``, ``s.v_CI``, ``_mfactor``, ``jnp.pi`` etc.
_TRIVIAL_EXPR_RE = re.compile(r"^[A-Za-z_][\w.]*$")


@dataclass
class Expr:
    """A Python expression with precedence metadata for parenthesising.

    ``prec`` follows Python's binding rules loosely — atoms are 100,
    multiplicative ops are 7, additive 6, comparisons 4, boolean 1–3.
    A child is wrapped in parens when its prec is below the parent's.
    """

    text: str
    prec: int = 100  # default atom


@dataclass
class CseState:
    """Common-subexpression-elimination bookkeeping for one ``lower()`` run.

    Shared across the init + eval walks so init-hoisted intermediates are
    visible in the eval function's emission (which is crucial: without
    sharing, ``_bind_cslot_args`` would alias an eval kwarg to a local
    like ``v42`` that's only defined in init's scope, producing
    unbound-name errors in the emitted ``.py``).

    ``refcount[ssa]`` counts how many places downstream reference the SSA
    (across the current function's own insts plus the current set of
    roots). ``hoist_defs`` holds ``ssa_name -> expression_text`` for every
    hoisted SSA; ``hoist_order`` is the DFS-post-order emission order.

    ``ssa_prefix`` is an optional prefix applied when hoisting (``"i_"``
    for the init function) so that ``i_v42`` and eval's ``v42`` don't
    collide in the shared table. Init's env holds ``Expr("i_v42")``
    references, which the cslot-bridge carries verbatim into eval.
    """

    refcount: dict[str, int]
    hoist_defs: dict[str, str] = field(default_factory=dict)
    hoist_order: list[str] = field(default_factory=list)
    ssa_prefix: str = ""


def _compute_refcount(fn: Function, roots: list[str]) -> dict[str, int]:
    """Count each SSA's use count: inst operands + root references.

    For an SSA at refcount ≥ 2 the emitter should hoist it as a named
    local instead of inlining at each use site.
    """
    count: dict[str, int] = dict.fromkeys(roots, 1)
    for b in fn.blocks:
        for inst in b.insts:
            operands = list(inst.operands)
            if inst.phi_edges:
                operands.extend(edge.value for edge in inst.phi_edges)
            for op in operands:
                count[op] = count.get(op, 0) + 1
    # Bump every root one extra if it appears in multiple root positions.
    for r in roots:
        count[r] = count.get(r, 0)
    return count


@dataclass
class LoweredDevice:
    """The payload of a lowering, ready for ``emit_component_file``.

    - ``class_name`` is what the emitted ``@component`` / ``@source``
      decorator binds in Python (CamelCase of the VA module name).
    - ``ports`` is the tuple passed to ``@component(ports=...)`` —
      drawn from the ``.va`` port declaration, already resolved to
      their source names (A / B / C / ...).
    - ``states`` are the extra DAE unknowns beyond ports (branch
      currents, implicit equations) — passed to
      ``@component(states=...)``.
    - ``params`` lists the keyword parameters of the physics function
      in their printed order, with Python-literal default expressions.
    - ``f_expressions`` / ``q_expressions`` map each port/state name
      to the Python expression that becomes ``f[name]`` / ``q[name]``
      in the generated physics function. Keys missing from each dict
      are omitted from the return — matching circulax's convention of
      "no entry = zero contribution".
    - ``preamble_stmts`` are Python assignment statements emitted
      before the ``return``, used to bind named locals (one per
      ``HiddenState`` hit in eval) for readability.
    """

    class_name: str
    ports: list[str]
    states: list[str]
    # ``(name, python_type, default_literal)`` per kwarg in declaration order.
    # ``python_type`` is ``"float"``, ``"int"``, or ``"str"``. ``float``
    # params become JAX-differentiable leaves; ``int``/``str`` become
    # ``equinox.field(static=True)`` so they're constant at trace time.
    params: list[tuple[str, str, str]]
    preamble_stmts: list[str] = field(default_factory=list)
    f_expressions: dict[str, str] = field(default_factory=dict)
    q_expressions: dict[str, str] = field(default_factory=dict)
    # Sparse Jacobian entries from ``DaeSystem.jacobian``, keyed by
    # ``(row_name, col_name)`` in circulax's port/state naming. ``row_name``
    # and ``col_name`` are the same identifiers used as keys in
    # ``f_expressions`` / ``q_expressions``. Missing ``(row, col)`` pairs
    # are treated as zero. When either dict is non-empty the emitter writes
    # a sibling ``_<Name>_jacobian`` function and wraps the component in
    # ``@va_component(..., jacobian_fn=...)`` so circulax's solver can use
    # the pre-computed Jacobian instead of ``jax.jacfwd``.
    jacobian_resist: dict[tuple[str, str], str] = field(default_factory=dict)
    jacobian_react: dict[tuple[str, str], str] = field(default_factory=dict)
    # Hoisted shared subexpressions, in dependency order. Each entry is a
    # ``(ssa_name, python_expression)`` pair; the emitter writes them as
    # ``vN = <expr>`` assignments at the top of both the physics function
    # and the ``_jacobian`` sibling. Duplication at the Python source level
    # is deliberate — XLA's CSE deduplicates at compile time when both
    # functions are traced inside a ``custom_jvp`` rule.
    cse_hoists: list[tuple[str, str]] = field(default_factory=list)
    uses_time: bool = False
    # Parameters whose values were baked in as literals during lowering.
    # These are absent from ``params`` (they don't appear in the emitted
    # function signature) and from the XLA graph.  Stored here so callers
    # can inspect which specialisation was applied.
    static_params: dict[str, int | float] = field(default_factory=dict)
    # Init-cache split: ``init_hoist_count`` is the index in ``cse_hoists``
    # where init-derived hoists end and eval-derived hoists begin.
    # ``init_cache_refs`` is the ordered, deduplicated list of hoisted SSA
    # names (``i_v42``, …) that correspond to cached-slot outputs bridged
    # into eval.  When non-empty, the emitter extracts these computations
    # into a ``_<Name>_compute_cache`` function called once at instantiation
    # so they do not re-execute on every Newton step.
    init_hoist_count: int = 0
    init_cache_refs: list[str] = field(default_factory=list)
    # ``differentiable_params`` is forwarded to the emitted ``@va_component``
    # decorator.  ``()`` (default) makes every model parameter an Equinox
    # static field so XLA folds it as a compile-time constant; ``None``
    # makes every parameter a JAX-traced leaf for full ``jax.grad`` support;
    # a tuple of names like ``("VTH0", "TOXE")`` keeps just those params
    # as leaves while baking the rest.  This is the lever for the
    # parameter-fitting use case where a small subset of model parameters
    # need gradients but the bulk should remain folded for speed.
    differentiable_params: tuple[str, ...] | None = ()


# ---------------------------------------------------------------------------
# Compile-time constant folding support.
# ---------------------------------------------------------------------------

# Python callables that evaluate each binary operator at lowering time.
# Keys match the ``py_op`` strings from ``_BINOPS``.  ``fdiv`` is handled
# separately via ``jnp.divide`` and is intentionally absent — we never
# constant-fold division to avoid introducing ``inf``/``nan`` literals.
_BINOP_FOLDS: dict[str, Callable[[object, object], object]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "//": lambda a, b: a // b,
    "%": lambda a, b: a % b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _try_literal(expr: Expr) -> int | float | bool | None:
    """Return the Python value if *expr* is a compile-time numeric/bool literal, else ``None``.

    Uses :func:`ast.literal_eval` which only handles safe Python literals
    (numbers, strings, booleans, None) — no arbitrary code execution.
    """
    try:
        val = ast.literal_eval(expr.text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val
    return None


def _literal_expr(val: int | float | bool) -> Expr:
    """Wrap a Python scalar as an atom :class:`Expr`."""
    if isinstance(val, bool):
        return Expr("True" if val else "False")
    if isinstance(val, int):
        # Emit large integers as float literals — Python ints beyond 2^53
        # cannot be represented exactly in float64 and overflow JAX tracers.
        if abs(val) > 2**53:
            return Expr(_float_literal(float(val)))
        return Expr(str(val))
    return Expr(_float_literal(val))


# Patterns that prove an expression is provably non-zero at runtime.
# Used by ``fdiv`` lowering to skip the safe-divide wrapper (which adds
# 4 HLO ops per use site) when we can mathematically prove the
# denominator can never be zero.
_NONZERO_LIT_FLOOR = re.compile(
    # ``jnp.maximum(_, K)`` where K is a positive numeric literal — clamps
    # the value at K, so result >= K > 0.
    r"^jnp\.maximum\([^,]+,\s*(?P<floor>[\d.+\-eE]+)\)$"
)
_SQRT_MAX_RE = re.compile(
    # ``jnp.sqrt(jnp.maximum(_, K))`` with positive K → strictly positive.
    r"^jnp\.sqrt\(jnp\.maximum\([^,]+,\s*(?P<floor>[\d.+\-eE]+)\)\)$"
)
_EXP_RE = re.compile(r"^jnp\.exp\(.+\)$")
_PARENS_RE = re.compile(r"^\((?P<inner>.*)\)$")
_MUL_RE = re.compile(r"^(?P<a>\S+)\s*\*\s*(?P<b>\S+)$")
_HOIST_REF_RE = re.compile(r"^(?:i_v|v)\d+$")
_DIVIDE_SAFE_RE = re.compile(
    # ``jnp.divide(N, jnp.where((D == 0.0) | ~jnp.isfinite(D), 1e-300, D))`` —
    # the safe-divide pattern itself.  When the numerator is non-zero, the
    # result is non-zero (the where-1e-300 fallback is finite and small,
    # so 1.0/1e-300 = 1e+300 ≠ 0).
    r"^jnp\.divide\((?P<num>[^,]+(?:\([^)]*\))?[^,]*?),"
)


def _is_provably_nonzero(text: str, hoist_defs: dict[str, str], depth: int = 0) -> bool:
    """Heuristic: does ``text`` define an expression that's strictly non-zero at runtime?

    Conservative — returns False whenever unsure.  Recurses through CSE
    hoist references via ``hoist_defs``.  Used to skip the 4-op safe-
    divide wrapper for the ~10 % of denominators we can prove are
    non-zero (sqrt-with-floor, exp, max-with-positive-floor, products of
    non-zero operands).
    """
    if depth > 10:
        return False
    t = text.strip()

    # Numeric literal.
    try:
        v = float(t)
        return v != 0.0 and v == v  # finite, non-zero
    except ValueError:
        pass

    # Strip outermost parens.
    m = _PARENS_RE.match(t)
    if m and "(" not in m.group("inner")[: m.group("inner").rfind(")") + 1]:
        return _is_provably_nonzero(m.group("inner"), hoist_defs, depth + 1)

    # ``jnp.exp(...)`` — strictly positive (clipped or not).
    if _EXP_RE.match(t):
        return True

    # ``jnp.sqrt(jnp.maximum(_, K))`` with K > 0 → strictly positive.
    m = _SQRT_MAX_RE.match(t)
    if m:
        try:
            return float(m.group("floor")) > 0.0
        except ValueError:
            return False

    # ``jnp.maximum(_, K)`` with K > 0 → result ≥ K > 0.
    m = _NONZERO_LIT_FLOOR.match(t)
    if m:
        try:
            return float(m.group("floor")) > 0.0
        except ValueError:
            return False

    # Hoist reference: look up its definition and recurse.
    if _HOIST_REF_RE.match(t):
        defn = hoist_defs.get(t)
        if defn is None:
            return False
        return _is_provably_nonzero(defn, hoist_defs, depth + 1)

    # ``a * b`` where both are non-zero → non-zero (algebraic).  Only handles
    # the simple two-operand case, not ``a * b * c``.
    m = _MUL_RE.match(t)
    if m:
        return _is_provably_nonzero(
            m.group("a"), hoist_defs, depth + 1
        ) and _is_provably_nonzero(m.group("b"), hoist_defs, depth + 1)

    # ``jnp.divide(num, ...)`` where num is non-zero — result is non-zero.
    # PSP103 chains divides through 3-4 levels; recursing here unlocks
    # another tier of safe-divide elimination.
    m = _DIVIDE_SAFE_RE.match(t)
    if m:
        return _is_provably_nonzero(m.group("num").strip(), hoist_defs, depth + 1)

    return False


# Default values for ``$simparam(...)`` queries the MIR may reference. Covers
# the ones we see in the bundled fixtures; new entries can be added as the
# fixture set grows. Used both to synthesise a kwarg in the function signature
# and to emit the reference in the body.
_SIMPARAM_DEFAULTS: dict[str, str] = {
    "gmin": "1e-12",
    "sourceFactor": "1.0",
    "scale": "1.0",
    "shrink": "0.0",
    "imax": "1.0",
    "imelt": "1.0",
}


def _simparam_kwarg_name(sim_name: str) -> str:
    return f"_simparam_{sim_name}"


def _inject_static_params(
    static_params: dict[str, int | float],
    env: dict[str, Expr],
    cm: CompiledModule,
    interner: HirInterner | None = None,
) -> None:
    """Replace ParamRef env entries with compile-time literals for known static params.

    ``interner`` (when supplied) restricts injection to that single
    interner's ParamRefs.  This matters because OpenVAF's MIR uses a
    *per-function* SSA namespace — the same SSA name (``v158``) refers to
    different parameters in init vs eval (``AXL`` vs ``NSUBO``, for
    example, on PSP103).  Walking all three interners against a single
    env dict (the previous behaviour) silently clobbered correct
    injections with the last-iterating interner's wrong value, producing
    cross-function name aliasing that yielded nonsense residuals under
    aggressive ``static_params``.

    The default (``interner=None``) preserves the legacy "walk all"
    behaviour for the few callers that genuinely operate across
    functions; new code should always pass the correct interner.

    Only overrides SSA names already seeded into *env* — silently skips
    params not referenced by this function.
    """
    interners = (
        (interner,)
        if interner is not None
        else (
            cm.setup_interner,
            cm.init_interner,
            cm.eval_interner,
        )
    )
    for it in interners:
        for val, kind in it.parameters.items():
            if isinstance(kind, ParamRef) and kind.name in static_params and val in env:
                env[val] = _literal_expr(static_params[kind.name])


def _sccp_initial_constants(
    fn: Function,
    interner: HirInterner,
    static_params: dict[str, int | float] | None,
    sentinel_params: set[str] | None = None,
) -> dict[str, object]:
    """Resolve ``static_params`` to per-SSA initial constants for the given function.

    SCCP wants ``{ssa_name: python_value}`` keyed by the function's
    argument SSAs.  We walk the interner to find which arg position
    corresponds to each ``ParamRef``, then feed the value in.  The same
    approach as ``_inject_static_params`` but producing a dict for SCCP
    rather than mutating an Expr env.

    Seeds ``$param_given(X) = False`` only for params explicitly flagged in
    ``sentinel_params`` (typically the auto-detected sentinel-defaulted
    floats).  Models like BSIM3v3 use ``$param_given`` to choose between
    user-supplied and computed parameter values — flagging sentinels as
    "not given" lets SCCP eliminate the dead "given" branches and prevents
    sentinel values (−9.9999e−99) from reaching sqrt / log in the inactive
    branch and producing NaN through JAX's both-branches-evaluated semantics.

    For non-sentinel params, ``$param_given`` is left unseeded — the lowering
    treats it as a runtime value.  This matches PSP103's expectation that
    instance params (W, L, AD, …) are always "given" via the user's settings.
    """
    out: dict[str, object] = {}
    for ssa, kind in interner.parameters.items():
        if isinstance(kind, ParamRef) and static_params and kind.name in static_params:
            out[ssa] = static_params[kind.name]
        elif isinstance(kind, ParamGivenRef):
            # Only force False for known sentinels; leave others unseeded so
            # the runtime value (effectively True for concrete user values)
            # is used.  Static-param-listed names are also "given" → True.
            if sentinel_params is not None and kind.name in sentinel_params:
                out[ssa] = False
            elif static_params is not None and kind.name in static_params:
                out[ssa] = True
            # else: leave unseeded
    return out


def _inject_sccp_constants(
    env: dict[str, Expr], sccp_result: object, fn: Function | None = None
) -> int:
    """Pre-load *env* with literal Exprs for every SSA SCCP marked CONSTANT.

    Returns the count of injected entries.  Doesn't override existing env
    entries (they may carry richer info — node-voltage references, time
    inputs, simparams).  The lowering walk's existing fast-path lookup
    (``env.get(ssa)`` early-return in ``_resolve_ssa``) does the rest:
    constant SSAs short-circuit the recursive walk and the existing
    binop folder picks up downstream chains.

    **Phi results are intentionally not injected**, even when SCCP marks
    them CONSTANT.  SCCP's lattice for a phi is the meet over executable
    incoming edges only; if an edge is marked dead, the phi's lattice
    can collapse to a constant that drops voltage-dependent chains the
    runtime path would have selected.  Letting the lowering's diamond
    rule walk the phi (and its existing static-condition fold) keeps
    the ``jnp.where`` emission honest — it sees the full set of edges
    and either folds correctly with literal conditions or emits a
    runtime ``jnp.where`` that evaluates both branches at JAX time.
    Skipping phi-result injection costs a few literals but avoids
    eliminating valid voltage chains.
    """
    from .sccp import (
        SccpResult,
    )  # local import to avoid module-init cycles  # noqa: PLC0415

    if not isinstance(sccp_result, SccpResult):
        return 0
    # Build a set of phi-result SSAs to exclude.
    phi_results: set[str] = set()
    if fn is not None:
        for blk in fn.blocks:
            for inst in blk.insts:
                if inst.phi_edges is not None and inst.result is not None:
                    phi_results.add(inst.result)
    n = 0
    for ssa, lat in sccp_result.lattice.items():
        if not lat.is_constant:
            continue
        if ssa in env:
            continue
        if ssa in phi_results:
            continue
        # Skip non-numeric constants — strings don't take part in the
        # binop folder and the lowering's existing handling of ``sconst``
        # references is fine for them.  Booleans also need careful
        # rendering (Python literal ``True`` / ``False``); fall through
        # to ``_literal_expr`` for ints/floats only.
        if not isinstance(lat.value, (int, float)) or isinstance(lat.value, bool):
            continue
        env[ssa] = _literal_expr(lat.value)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Top-level entry point.
# ---------------------------------------------------------------------------


def lower(
    cm: CompiledModule,
    *,
    va_defaults: dict[str, ParamSpec] | None = None,
    collapse_nodes: bool = False,
    static_params: dict[str, int | float] | None = None,
    class_name: str | None = None,
    differentiable_params: tuple[str, ...] | None = (),
) -> LoweredDevice:
    """Lower a parsed :class:`CompiledModule` into a :class:`LoweredDevice`.

    ``va_defaults`` is an optional ``{param_name: ParamSpec}`` map —
    usually produced by :func:`circulax.va.va_defaults.parse_va_defaults`
    from the ``.va`` source text. When supplied, the emitted function
    signature's parameter types and defaults are sourced from it
    (``float`` / ``int`` / ``str`` with the ``.va``-declared literal).
    When absent, params fall back to ``float`` + ``"0.0"``.

    ``collapse_nodes`` runs :func:`_collapse_trivial_nodes` before the
    rest of lowering — apply OpenVAF's ``CollapseHint`` decisions to
    shrink the DAE to the same shape OSDI emits. Off by default because
    collapse decisions are conditional on user-facing parameters (e.g.
    the diode's ``Rs=0`` triggers ``CI→C`` but ``Rs>0`` doesn't); only
    enable it for devices where the user intends the OSDI-matching
    reduced system (PSP103, BSIM4, etc).

    ``static_params`` is an optional ``{param_name: value}`` dict of
    integer or float parameters whose values are *known at lowering time*
    and should be baked into the emitted XLA graph as compile-time
    literals.  Typically these are integer switch flags (``TYPE``,
    ``SWGIDL``, ``SWIGATE``, …) whose value is fixed for a given device
    instance class.  Substituting them as literals lets XLA constant-fold
    all downstream comparisons (e.g. ``TYPE == 1``) and eliminate dead
    ``jnp.where`` branches, halving the graph size for models like PSP103
    that condition large sections of physics on these flags.  Static
    params are removed from the emitted function signature — callers
    must not pass them as kwargs to the generated class.

    ``class_name`` overrides the default CamelCase class name derived
    from the VA module name.  Use it to emit multiple specialisations of
    the same model under distinct Python identifiers.
    """
    # Bump Python's recursion ceiling so deeply-chained dataflows (BSIM4's
    # ~ 9 k eval ops, PSP103's ~ 20 k) don't trip the default 1 000-frame
    # limit. Aggressive CSE hoisting in ``_resolve_ssa`` caps the *textual*
    # blow-up but the DFS still recurses once per SSA in the worst case.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 200_000))

    # Auto-detect sentinel-default float params and add them to the SCCP seed.
    #
    # BSIM-family models (BSIM3v3, BSIM4, PSP103 JUNCAP …) use −9.9999e−99 as
    # a sentinel meaning "I was not given by the user; compute me from other
    # params."  Their guard pattern is:
    #
    #     if (poxedge <= −1e−99) { use_tox_instead; } else { use poxedge; }
    #
    # In C/SPICE this is a real branch — the ``use poxedge`` path never runs.
    # In JAX, jnp.where evaluates *both* sides: ``x / poxedge`` with
    # poxedge ≈ −1e−98 produces ~1e98, which overflows downstream and gives NaN.
    #
    # Seeding the sentinel value into SCCP tells it the branch condition is
    # always True, so the dead ``use poxedge`` block is eliminated from the
    # rewritten MIR before the Python lowering ever sees it.
    #
    # Which params get auto-seeded?  Any float param whose .va default is the
    # sentinel AND that is not listed in ``differentiable_params`` (the caller
    # explicitly wants gradients through the primary physics params like nch,
    # tox, vth0 — not through derived quantities like poxedge).
    _SENTINEL = -9.9999e-99
    _diff_set: set[str] = set(differentiable_params) if differentiable_params else set()
    _sentinel_seed: dict[str, float] = {}
    if va_defaults is not None:
        for _name, _spec in va_defaults.items():
            if _spec.type_ != "float":
                continue
            _val = float(_spec.default)
            # Exact sentinel check (tolerance allows for float-repr rounding).
            if abs(_val - _SENTINEL) > abs(_SENTINEL) * 1e-4:
                continue
            # Don't override explicit static_params or differentiable_params.
            if _name in _diff_set:
                continue
            if static_params and _name in static_params:
                continue
            _sentinel_seed[_name] = _val

    # Effective static params for SCCP: explicit overrides take precedence,
    # sentinel auto-seeds fill in the rest.
    effective_static: dict[str, int | float] | None = (
        {**_sentinel_seed, **(static_params or {})} if _sentinel_seed else static_params
    )

    if collapse_nodes:
        _collapse_trivial_nodes(cm)

    # Build a constants table that spans all three functions — the DaeSystem
    # block references SSA names that may have been declared in any of them
    # (notably zero constants shared across residuals).
    const_table = _build_const_table(cm)

    # Classify each ``inst<k> = ... fn %name ...`` decl so the call-site
    # handling in ``_resolve_ssa`` can dispatch on the builtin's kind.
    call_kinds = _classify_callbacks(cm)

    # Map any internal node id (``node2``, ...) to its semantic ``.va`` name
    # (``CI``, ...) so voltage probes on internal nodes can reference the
    # circulax state variable ``s.v_<name>`` rather than ``signals.<name>``.
    internal_name = _build_internal_node_name_map(cm)

    # Build a BranchId -> circulax state name map from the eval interner's
    # Current(Branch) entries. This is the single source of truth for how a
    # branch-current unknown shows up in the generated ``states=(...)`` tuple
    # and in ``s.<name>`` references inside the physics body.
    branch_state_name = _build_branch_state_name_map(cm)

    # One CSE table shared across init + eval so the bridge from init's
    # cslot-producing SSAs into eval's extra args keeps working — an init
    # hoist like ``i_v42 = <expr>`` is emitted in the eval function's
    # preamble, and eval expressions that read the bridged input get the
    # same ``i_v42`` name in scope. Init hoists use the ``i_`` prefix so
    # ``i_v42`` and eval's own ``v42`` never collide.
    all_roots = _residual_ssa_names(cm) + _jacobian_ssa_names(cm)
    cse = CseState(refcount=_compute_refcount(cm.eval_fn, all_roots), ssa_prefix="i_")

    # Run SCCP on the init function, then rewrite it: dead blocks dropped,
    # phi nodes whose dead predecessors are pruned simplify to single-edge
    # phis (which become ``optbarrier`` passthroughs).  This replaces the
    # older heuristic phi-fallback in ``_lower_phi`` — the rewritten MIR
    # has *literally* fewer edges, so the phi handler either sees one
    # source (trivial) or sees a real runtime fan-in that emits a clean
    # ``jnp.where``.  ``static_params`` literals propagate through
    # arithmetic chains via the lattice; constant-conditioned branches
    # collapse to unconditional jumps.
    from .sccp import rewrite_function, run_sccp  # noqa: PLC0415

    _sentinel_set: set[str] | None = (
        set(_sentinel_seed.keys()) if _sentinel_seed else None
    )
    init_sccp = run_sccp(
        cm.init_fn,
        _sccp_initial_constants(
            cm.init_fn, cm.init_interner, effective_static, _sentinel_set
        ),
    )
    init_fn = rewrite_function(cm.init_fn, init_sccp)

    # Lower the rewritten init function: its cslot outputs feed eval's extra args.
    init_env = _initial_env_for_function(
        init_fn,
        cm.init_interner,
        branch_state_name,
        internal_name,
        effective_static,
        _sentinel_set,
    )
    init_defs = _defining_insts(init_fn)
    init_cfg = _build_cfg(init_fn)

    _inject_sccp_constants(init_env, init_sccp, init_fn)
    used_simparams: set[str] = set()
    # Merge init's own refcounts into the shared table so init's internal
    # CSE works. Residual / Jacobian roots of eval are in there too; that's
    # fine because any SSA that *only* appears in init has its refcount
    # counted locally here.
    init_refcount = _compute_refcount(init_fn, list(cm.cached.mapping))
    for k, v in init_refcount.items():
        cse.refcount[k] = cse.refcount.get(k, 0) + v
    # Track which cslot values were resolvable.  When ``static_params``
    # are aggressive (e.g. PSP103 all-static), the rewriter can prove an
    # init-side branch unreachable and eliminate its block — taking a
    # cslot-feeder SSA with it.  That's *correct*: the cslot value is
    # genuinely unused at those param settings, and the matching eval-
    # side arg has no consumer either.  Below, ``_bind_cslot_args`` skips
    # those bridges; if eval *does* try to read the orphaned arg, the
    # resolve walk will surface a clean "unresolved operand" error.
    dead_cslots: set[str] = set()
    for val in cm.cached.mapping:
        if val not in init_defs and val not in init_env:
            dead_cslots.add(val)
            continue
        _resolve_ssa(
            val,
            init_env,
            init_defs,
            const_table,
            fn_name=init_fn.name,
            visiting=set(),
            call_kinds=call_kinds,
            used_simparams=used_simparams,
            cfg=init_cfg,
            cse=cse,
            sccp=init_sccp,
        )

    # Record how many hoists came from init before switching to eval prefix.
    _init_hoist_end = len(cse.hoist_order)

    # Switch CSE prefix so eval's hoists land under their plain SSA names,
    # distinguishable from init's ``i_``-prefixed ones.
    cse.ssa_prefix = ""

    # Inject static-param literals before the eval walk so every downstream
    # SSA expression that reads a static param gets the literal value and
    # binary ops (TYPE == 1, SWGIDL != 0, …) constant-fold to True/False.
    if effective_static:
        _inject_static_params(effective_static, init_env, cm, interner=cm.init_interner)

    # Map each eval arg to its source expression (interner input or cslot).
    eval_env = _initial_env_for_function(
        cm.eval_fn,
        cm.eval_interner,
        branch_state_name,
        internal_name,
        effective_static,
        _sentinel_set,
    )
    _bind_cslot_args(cm, eval_env, init_env)

    # SCCP for the eval function — initial constants come from the static
    # params plus the cslot bridge values that were resolved as constants
    # during the init analysis.  Bridging constants across the init→eval
    # boundary is what lets PSP103's parameter-derived setup values flow
    # through to eval-side branch conditions and fold them.
    eval_sccp_init: dict[str, object] = _sccp_initial_constants(
        cm.eval_fn,
        cm.eval_interner,
        effective_static,
        _sentinel_set,
    )
    for init_val, eval_arg in zip(
        list(cm.cached.mapping)[
            : max(0, len(cm.eval_fn.args) - len(cm.eval_interner.parameters))
        ],
        cm.eval_fn.args[len(cm.eval_interner.parameters) :],
        strict=False,
    ):
        lat = init_sccp.lattice_value(init_val)
        if lat.is_constant:
            eval_sccp_init[eval_arg] = lat.value
    eval_sccp = run_sccp(cm.eval_fn, eval_sccp_init)
    eval_fn = rewrite_function(cm.eval_fn, eval_sccp)
    _inject_sccp_constants(eval_env, eval_sccp, eval_fn)

    # Collect the unique cslot-bridge SSA names for the init-cache vector.
    # The bridge maps each cslot's init SSA to an Expr such as Expr("i_v42");
    # we deduplicate so each hoisted name appears exactly once in the cache.
    # SCCP-eliminated cslots (init block dead under aggressive static_params)
    # don't contribute — their eval-side arg has no consumer either.
    _seen_cache_ref: dict[str, int] = {}
    _init_cache_refs: list[str] = []
    for init_val in cm.cached.mapping:
        if init_val not in init_env:
            continue
        ref = init_env[init_val].text
        if ref not in _seen_cache_ref:
            _seen_cache_ref[ref] = len(_init_cache_refs)
            _init_cache_refs.append(ref)

    if effective_static:
        _inject_static_params(effective_static, eval_env, cm, interner=cm.eval_interner)

    # Residual SSAs must resolve — failure here is a real lowering bug.
    eval_defs = _defining_insts(eval_fn)
    eval_cfg = _build_cfg(eval_fn)
    for ssa in _residual_ssa_names(cm):
        _resolve_ssa(
            ssa,
            eval_env,
            eval_defs,
            const_table,
            fn_name=eval_fn.name,
            visiting=set(),
            call_kinds=call_kinds,
            used_simparams=used_simparams,
            cfg=eval_cfg,
            cse=cse,
            sccp=eval_sccp,
        )
    # Jacobian SSAs are allowed to fail — OpenVAF's optimizer sometimes DCEs
    # trivial constants like ``fconst 1.0`` that only appear in the Jacobian
    # block (e.g. ``∂f[branch]/∂V_B = 1`` for an inductor). When that happens
    # we drop the whole Jacobian and the component falls back to
    # ``jax.jacfwd`` via the plain ``@component`` path — correct, just
    # slower on the Newton hot path.
    jacobian_ok = True
    for ssa in _jacobian_ssa_names(cm):
        try:
            _resolve_ssa(
                ssa,
                eval_env,
                eval_defs,
                const_table,
                fn_name=eval_fn.name,
                visiting=set(),
                call_kinds=call_kinds,
                used_simparams=used_simparams,
                cfg=eval_cfg,
                cse=cse,
                sccp=eval_sccp,
            )
        except LoweringError:
            jacobian_ok = False
            break

    # Compose the LoweredDevice.
    ports, states, param_specs = _plan_component_surface(
        cm,
        branch_state_name,
        internal_name,
        va_defaults or {},
        used_simparams,
        static_params=effective_static or {},
    )
    f_exprs, q_exprs = _collect_residuals(
        cm, eval_env, const_table, ports, states, branch_state_name, internal_name
    )
    if jacobian_ok:
        jac_resist, jac_react = _collect_jacobian(
            cm, eval_env, const_table, ports, branch_state_name, internal_name
        )
    else:
        jac_resist, jac_react = {}, {}

    cse_hoists = [(ssa, cse.hoist_defs[ssa]) for ssa in cse.hoist_order]

    return LoweredDevice(
        class_name=class_name or _camel_case(cm.name),
        ports=ports,
        states=states,
        params=param_specs,
        cse_hoists=cse_hoists,
        f_expressions=f_exprs,
        q_expressions=q_exprs,
        jacobian_resist=jac_resist,
        jacobian_react=jac_react,
        static_params=effective_static or {},
        init_hoist_count=_init_hoist_end,
        init_cache_refs=_init_cache_refs,
        differentiable_params=differentiable_params,
    )


_CALLBACK_NAME_RE = re.compile(r"fn\s+%([A-Za-z_][\w]*)")


def _classify_callbacks(cm: CompiledModule) -> dict[tuple[str, str], str]:
    """Build ``{(fn_name, inst_name): builtin_kind}`` for every callback decl.

    ``builtin_kind`` is one of ``"ddt"``, ``"ddx"``, ``"simparam"``,
    ``"collapse"``, ``"set_invalid"``, or ``"user"`` (fallback). Dispatching
    on this lets the call-site lowering drop the no-op cases (``ddt``,
    ``ddx``, ``set_invalid``, ``collapse``) and translate the remaining ones
    (``simparam`` → kwarg ref; ``user`` still raises in Stage 3).
    """
    kinds: dict[tuple[str, str], str] = {}
    for fn in (cm.setup_fn, cm.init_fn, cm.eval_fn):
        for cd in fn.call_decls:
            m = _CALLBACK_NAME_RE.search(cd.raw)
            if not m:
                continue
            builtin = m.group(1)
            if builtin == "ddt":
                kinds[(fn.name, cd.name)] = "ddt"
            elif builtin.startswith("ddx_"):
                kinds[(fn.name, cd.name)] = "ddx"
            elif builtin == "simparam":
                kinds[(fn.name, cd.name)] = "simparam"
            elif builtin.startswith("collapse_"):
                kinds[(fn.name, cd.name)] = "collapse"
            elif builtin == "set_Invalid":
                kinds[(fn.name, cd.name)] = "set_invalid"
            elif builtin in ("StoreLimit", "LimDiscontinuity", "Print"):
                # OpenVAF's $limit infrastructure (BSIM3v3, diode w/ limiting):
                # ``StoreLimit(state)``  records the new iterate (return = arg);
                # ``LimDiscontinuity()`` sets a flag we don't consume;
                # ``Print(...)``        is debug-only.
                # All three are passthrough / no-op for our pipeline.
                kinds[(fn.name, cd.name)] = "limit_passthrough"
            elif builtin in ("WhiteNoise", "FlickerNoise"):
                # Noise sources contribute zero to the deterministic residual.
                kinds[(fn.name, cd.name)] = "noise_zero"
            elif builtin in ("SimParamOpt", "Analysis"):
                # ``$simparam_opt(name, default)`` returns the default when
                # the simulator hasn't supplied the parameter.  ``$analysis``
                # discriminates on the active analysis kind ("dc", "tran",
                # "ac"); circulax always runs DC + transient via the same
                # eval body, so falling through to the second arg matches
                # the runtime behaviour at every operating point.
                kinds[(fn.name, cd.name)] = "default_arg"
            else:
                kinds[(fn.name, cd.name)] = "user"
    return kinds


_COLLAPSE_DECL_RE = re.compile(
    # Match either OpenVAF's internal ``node{N}`` ids (text-parser path)
    # or the semantic node names the binding emits after translation
    # (``CollapseHint(node1, Some(node5))`` → ``collapse_G_Some(GP)``).
    # Both forms hit the same ``cm.dae.unknowns`` lookup downstream, so
    # the regex just needs to capture the surrounding tokens.
    r"%collapse_([A-Za-z_]\w*)_Some\(([A-Za-z_]\w*)\)"
)


def _collapse_trivial_nodes(cm: CompiledModule) -> dict[str, str]:
    """Apply OpenVAF's ``CollapseHint`` decisions to the parsed DAE.

    OpenVAF's ``hir_lower`` pass emits a callback of the form
    ``fn %collapse_node<A>_Some(node<B>)`` whenever it lowers a branch
    ``V(A, B) <+ 0`` — an ideal voltage source pinned at zero potential
    difference. Those callbacks survive verbatim in the MIR dump as
    ``call_decls`` on the init / eval functions; the OSDI backend
    downstream folds them into ``collapsible_pairs`` on the descriptor.

    For our VA → JAX pipeline we apply the same folding: each pair
    ``(A, B)`` merges both nodes into one surviving unknown. When one
    side of the pair is a port we keep the port (its display name
    anchors the circulax netlist); otherwise we keep the
    lower-numbered node as a deterministic tie-breaker. The collapsed
    side is dropped from ``cm.dae.unknowns`` / ``residual`` /
    ``jacobian``; every ``Voltage`` input whose ``hi_node``/``lo_node``
    matches the collapsed id is rewritten to reference the survivor
    instead.

    Mutates ``cm.dae``, ``cm.internal_nodes``, and each interner's
    ``Voltage`` inputs in place. Returns ``{collapsed_node_id →
    survivor_node_id}`` for tests / logging.
    """
    # Gather every collapse decl from all three functions. The same pair
    # appears in both init and eval — dedup by (a, b) content.
    pairs: set[tuple[str, str]] = set()
    for fn in (cm.setup_fn, cm.init_fn, cm.eval_fn):
        for cd in fn.call_decls:
            m = _COLLAPSE_DECL_RE.search(cd.raw)
            if m:
                pairs.add((m.group(1), m.group(2)))
    if not pairs:
        return {}

    # Union-find over the nodes in the pair set — multiple pairs can
    # chain (PSP103: node8→node9, node10→node9, node3→node9, etc.).
    parent: dict[str, str] = {}

    def _find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return
        # Tie-breaker: prefer a port as the survivor; otherwise the
        # lower-numbered node. Ports have short names in cm.port_nodes;
        # everything else sorts lexicographically.
        port_nodes = set(cm.port_nodes)
        ra_is_port = ra in port_nodes
        rb_is_port = rb in port_nodes
        if ra_is_port and not rb_is_port:
            parent[rb] = ra
        elif rb_is_port and not ra_is_port:
            parent[ra] = rb
        else:
            # Compare numeric suffix so "node10" sorts above "node2".
            def _key(n: str) -> int:
                m = re.search(r"\d+", n)
                return int(m.group()) if m else 0

            if _key(ra) <= _key(rb):
                parent[rb] = ra
            else:
                parent[ra] = rb

    for a, b in pairs:
        _union(a, b)

    # Build the final map of {collapsed_id → survivor_id} for every node
    # that ended up pointing at a different root.
    collapses: dict[str, str] = {}
    for node in list(parent.keys()):
        root = _find(node)
        if node != root:
            collapses[node] = root
    if not collapses:
        return {}

    # Remap collapsed unknowns to their survivor's node_repr.
    # We keep the collapsed sim_id in cm.dae.unknowns (pointing at the
    # survivor's node_repr) so that _collect_residuals processes its
    # residual and accumulates it into the same port/state bucket as the
    # survivor via _maybe_record.  Without this, the collapsed node's
    # KCL — which carries the channel current — is silently dropped and
    # the port row of the Jacobian is identically zero.
    node_repr_to_sim: dict[str, str] = {v: k for k, v in cm.dae.unknowns.items()}
    sim_collapses: dict[str, str] = {}
    for sim_id in list(cm.dae.unknowns.keys()):
        node_repr = cm.dae.unknowns[sim_id]
        if node_repr in collapses:
            survivor_repr = collapses[node_repr]
            survivor_sim = node_repr_to_sim.get(survivor_repr)
            if survivor_sim is not None:
                sim_collapses[sim_id] = survivor_sim
            # Remap the unknown to the survivor's node_repr so
            # _collect_residuals funnels the residual into the right bucket.
            # Do NOT delete the residual — it carries the real KCL.
            cm.dae.unknowns[sim_id] = survivor_repr
    # Remap Jacobian entries whose row/col was collapsed onto a survivor.
    # Multiple original entries may map to the same (new_row, new_col) pair;
    # keep all under distinct keys so _collect_jacobian accumulates them.
    for jk in list(cm.dae.jacobian.keys()):
        entry = cm.dae.jacobian[jk]
        new_row = sim_collapses.get(entry.row, entry.row)
        new_col = sim_collapses.get(entry.col, entry.col)
        # An entry is dead only if its row/col refers to a sim_id that
        # was never a valid unknown (shouldn't happen, but guard anyway).
        if new_row not in cm.dae.unknowns or new_col not in cm.dae.unknowns:
            del cm.dae.jacobian[jk]
        elif new_row != entry.row or new_col != entry.col:
            entry.row = new_row
            entry.col = new_col

    # Rewrite every Voltage input referencing a collapsed node.
    display: dict[str, str] = {}
    # Port node_ids take their display from cm.ports (ordered match).
    for nid, pname in zip(cm.port_nodes, cm.ports, strict=False):
        display[nid] = pname
    # Internal node_ids take their display from whichever Voltage input
    # first named them (handled the same way _build_internal_node_name_map
    # discovers semantic names).
    for interner in (cm.eval_interner, cm.init_interner, cm.setup_interner):
        for kind in interner.parameters.values():
            if isinstance(kind, Voltage):
                if kind.hi_node and kind.hi:
                    display.setdefault(kind.hi_node, kind.hi)
                if kind.lo_node and kind.lo:
                    display.setdefault(kind.lo_node, kind.lo)

    for interner in (cm.eval_interner, cm.init_interner, cm.setup_interner):
        for name, kind in list(interner.parameters.items()):
            if not isinstance(kind, Voltage):
                continue
            new_hi = collapses.get(kind.hi_node, kind.hi_node)
            new_lo = collapses.get(kind.lo_node, kind.lo_node) if kind.lo_node else None
            if new_hi == kind.hi_node and new_lo == kind.lo_node:
                continue
            interner.parameters[name] = Voltage(
                hi=display.get(new_hi, kind.hi),
                lo=display.get(new_lo, kind.lo) if new_lo is not None else None,
                hi_node=new_hi,
                lo_node=new_lo,
            )

    cm.internal_nodes = [n for n in cm.internal_nodes if n not in collapses]
    return collapses


def _build_internal_node_name_map(cm: CompiledModule) -> dict[str, str]:
    """Resolve internal-node ids like ``node2`` to their ``.va`` names (``CI``).

    The only place these semantic names surface in the MIR dump is in
    Voltage inputs of the form ``V("hi", "lo") -> vN`` — so we scan every
    interner's voltage probes and match ``hi_node`` / ``lo_node`` to
    internal-node ids.
    """
    mapping: dict[str, str] = {}
    if not cm.internal_nodes:
        return mapping
    internals = set(cm.internal_nodes)
    for interner in (cm.eval_interner, cm.init_interner, cm.setup_interner):
        for kind in interner.parameters.values():
            if not isinstance(kind, Voltage):
                continue
            if kind.hi_node in internals and kind.hi is not None:
                mapping.setdefault(kind.hi_node, kind.hi)
            if (
                kind.lo_node is not None
                and kind.lo_node in internals
                and kind.lo is not None
            ):
                mapping.setdefault(kind.lo_node, kind.lo)
    return mapping


def _build_branch_state_name_map(cm: CompiledModule) -> dict[int, str]:
    """``BranchId -> circulax state variable name`` for every branch-current unknown.

    Covers both named branches (``Branch``) and unnamed substrate-network
    probes (``Unnamed``) that carry a ``branch_id`` from the JSON IR client.
    Prefers the user's ``.va`` branch name for named branches, falls back to
    ``i_br<id>``.
    """
    mapping: dict[int, str] = {}
    for interner in (cm.eval_interner, cm.init_interner, cm.setup_interner):
        for kind in interner.parameters.values():
            if not isinstance(kind, CurrentKind) or kind.branch_id is None:
                continue
            if kind.branch_id in mapping:
                continue
            if kind.kind == "Branch":
                mapping[kind.branch_id] = (
                    f"i_{kind.branch}" if kind.branch else f"i_br{kind.branch_id}"
                )
            elif kind.kind == "Unnamed":
                # Unnamed substrate-network probes get a synthetic state name
                # keyed on the stable BranchId assigned by the JSON IR client.
                hi = kind.hi or ""
                lo = kind.lo or ""
                mapping[kind.branch_id] = f"i_un_{hi}_{lo}_{kind.branch_id}"
    return mapping


# ---------------------------------------------------------------------------
# Helpers: constants, env initialisation, SSA walks.
# ---------------------------------------------------------------------------


def _build_const_table(cm: CompiledModule) -> dict[str, Constant]:
    """Gather all preamble constants from setup/init/eval into one table."""
    table: dict[str, Constant] = {}
    for fn in (cm.setup_fn, cm.init_fn, cm.eval_fn):
        for c in fn.constants:
            table.setdefault(c.name, c)
    return table


def _initial_env_for_function(
    fn: Function,
    interner: HirInterner,
    branch_state_name: dict[int, str],
    internal_name: dict[str, str],
    static_params: dict[str, int | float] | None = None,
    sentinel_params: set[str] | None = None,
) -> dict[str, Expr]:
    """Seed the SSA environment with each argument's semantic source expression.

    Argument positions in a MIR function track the HIR interner's printed
    order, one-to-one, for the leading N args; any trailing args correspond
    to cached-slot reads and are bound later by :func:`_bind_cslot_args`.

    Inputs whose :class:`InputKind` variant is not yet supported (e.g.
    ``Current(Unnamed)`` probes used only by output variables like ``gd`` /
    ``cd``) are silently skipped here — they'll surface as a clean
    "unresolved operand" error from :func:`_resolve_ssa` if anything
    reachable from a residual actually depends on them.
    """
    env: dict[str, Expr] = {}
    # Walk the interner in insertion order (preserved by ``dict`` since 3.7).
    for val, kind in interner.parameters.items():
        try:
            env[val] = _input_kind_expr(
                kind, branch_state_name, internal_name, static_params, sentinel_params
            )
        except LoweringError:
            continue
    _ = fn  # retained in the signature for future diagnostics
    return env


def _bind_cslot_args(
    cm: CompiledModule,
    eval_env: dict[str, Expr],
    init_env: dict[str, Expr],
) -> None:
    """Bind eval's trailing (non-interner) args to init's cached-slot outputs.

    OpenVAF emits the eval function with interner-bound args first, then
    one argument per entry in the ``Cached values during instance setup``
    table — **in declaration order**, not deduplicated by cslot. MOSFET-
    class models (BSIM4, PSP103) routinely have aggregate cslots where
    several distinct init SSAs share the same ``cslotN`` label (OpenVAF's
    way of packing a struct into one OSDI slot). Each of those init SSAs
    still gets its own eval arg.

    We therefore bind positionally against ``cached.mapping`` (Python
    dicts preserve insertion order since 3.7) rather than against the
    deduplicated cslot set.
    """
    interner_args = set(cm.eval_interner.parameters)
    remaining = [a for a in cm.eval_fn.args if a not in interner_args]
    init_vals = list(cm.cached.mapping)

    if len(remaining) != len(init_vals):
        msg = (
            f"eval has {len(remaining)} trailing args but "
            f"``cached.mapping`` has {len(init_vals)} entries — cannot bind cslot reads positionally"
        )
        raise LoweringError(msg)

    for init_val, eval_arg in zip(init_vals, remaining, strict=True):
        if init_val not in init_env:
            # The SCCP rewriter can eliminate the init-side block that
            # would have computed this cslot when ``static_params`` proves
            # it unreachable.  Skip the bridge — if eval has a live use of
            # ``eval_arg`` it'll surface as "unresolved operand" later;
            # otherwise XLA / DCE will drop the unused arg.
            continue
        eval_env[eval_arg] = init_env[init_val]


@dataclass
class _FunctionCfg:
    """Pre-computed control-flow info for a :class:`Function`.

    Holds the terminator per block and the predecessor list per block.
    Used by the phi-to-``jnp.where`` lifter to recognise diamond-shaped
    if/else merges.
    """

    terminators: dict[str, Inst]  # block label -> its terminator instruction
    preds: dict[str, list[str]]  # block label -> predecessor block labels


def _build_cfg(fn: Function) -> _FunctionCfg:
    terminators: dict[str, Inst] = {}
    preds: dict[str, list[str]] = {b.label: [] for b in fn.blocks}
    for b in fn.blocks:
        term: Inst | None = None
        for inst in b.insts:
            if inst.opcode in {"br", "jmp", "exit"}:
                term = inst
        if term is None:
            continue
        terminators[b.label] = term
        for tgt in term.targets:
            preds.setdefault(tgt, []).append(b.label)
    return _FunctionCfg(terminators=terminators, preds=preds)


def _defining_insts(fn: Function) -> dict[str, Inst]:
    """Return ``{result_ssa: inst}`` across all blocks of ``fn``."""
    out: dict[str, Inst] = {}
    for block in fn.blocks:
        for inst in block.insts:
            if inst.result is not None:
                out[inst.result] = inst
    return out


def _resolve_ssa(  # noqa: C901, PLR0912, PLR0915
    ssa: str,
    env: dict[str, Expr],
    defs: dict[str, Inst],
    const_table: dict[str, Constant],
    *,
    fn_name: str,
    visiting: set[str],
    call_kinds: dict[tuple[str, str], str] | None = None,
    used_simparams: set[str] | None = None,
    cfg: _FunctionCfg | None = None,
    cse: CseState | None = None,
    sccp: object | None = None,
) -> Expr:
    """Resolve ``ssa`` to an :class:`Expr`, caching the result into ``env``.

    Demand-driven: if ``ssa`` isn't in ``env``, looks up its defining
    instruction and recurses on its operands. Handles the case where
    OpenVAF emits blocks in non-topological order (common in multi-block
    init functions with phi-at-merge).
    """
    if ssa in env:
        return env[ssa]
    # A *local* instruction definition always takes precedence over the
    # cross-function constant pool — OpenVAF's SSA namespace is per-function
    # so a value like ``v26`` can be ``fconst 1e-12`` in the setup function
    # and ``optbarrier v33`` in the eval function, and these must not collide.
    if ssa in defs:
        inst = defs[ssa]
    elif ssa in const_table:
        env[ssa] = _const_expr(const_table[ssa])
        return env[ssa]
    else:
        msg = f"unresolved operand {ssa!r} in function {fn_name!r}"
        raise LoweringError(msg)

    if ssa in visiting:
        msg = f"cyclic SSA dependency at {ssa!r} in function {fn_name!r}"
        raise LoweringError(msg)
    op = inst.opcode
    call_kinds = call_kinds or {}
    used_simparams = used_simparams if used_simparams is not None else set()
    # Mutate ``visiting`` in place rather than cloning it at every recursion
    # level — PSP103 dataflow can dive 20 k deep, and ``visiting | {ssa}``
    # creates fresh sets of size O(depth) for each call, spending O(depth²)
    # on set allocations (observed OOM at > 50 GB RSS before this fix).
    # The ``try/finally`` ensures we always clean up on both normal return
    # and exception paths.
    visiting.add(ssa)
    try:

        def resolve(op_name: str) -> Expr:
            return _resolve_ssa(
                op_name,
                env,
                defs,
                const_table,
                fn_name=fn_name,
                visiting=visiting,
                call_kinds=call_kinds,
                used_simparams=used_simparams,
                cfg=cfg,
                cse=cse,
                sccp=sccp,
            )

        if op == "phi":
            expr = _lower_phi(
                inst,
                defs,
                const_table,
                env,
                fn_name,
                visiting,
                call_kinds,
                used_simparams,
                cfg,
                cse,
                sccp,
            )
        elif op in _PASSTHROUGH:
            expr = resolve(inst.operands[0])
        elif op in _BINOPS:
            py_op, prec = _BINOPS[op]
            left = resolve(inst.operands[0])
            right = resolve(inst.operands[1])
            # Identical-operand subtraction: X - X = 0 regardless of X's value.
            # Detects collapsed internal-node voltage differences (e.g. V(DI,S) and
            # V(D,S) both resolve to the same expression after node collapse) before
            # they reach downstream multiplications and produce 0 × ∞ = NaN.
            if py_op == "-" and left.text == right.text:
                zero = Expr("0.0")
                env[ssa] = zero
                return zero
            lv = _try_literal(left)
            rv = _try_literal(right)
            if lv is not None and rv is not None:
                fold_op = _BINOP_FOLDS.get(py_op)
                if fold_op is not None:
                    try:
                        folded = _literal_expr(fold_op(lv, rv))
                        env[ssa] = folded
                        return folded
                    except (ZeroDivisionError, OverflowError, TypeError):
                        pass
            # Half-literal zero folding for multiplication: when one operand is the
            # literal 0.0 (e.g. from X-X node-collapse subtraction), fold to "0.0".
            # The structural-zero path means the product is topologically zero;
            # emitting the Python literal "0.0" (not a JAX expression) is safe and
            # prevents the ``inf * 0 = NaN`` IEEE 754 artifact that would appear if
            # the non-zero operand overflows through a safe-divide-by-zero fallback.
            if py_op == "*":
                if lv == 0.0 and rv is None:
                    zero = Expr("0.0")
                    env[ssa] = zero
                    return zero
                if rv == 0.0 and lv is None:
                    zero = Expr("0.0")
                    env[ssa] = zero
                    return zero
            expr = Expr(
                f"{_paren(left, prec)} {py_op} {_paren_right(right, prec, py_op)}", prec
            )
        elif op in _UNARYOPS:
            py_op = _UNARYOPS[op]
            inner = resolve(inst.operands[0])
            expr = Expr(f"{py_op}{_paren(inner, 10)}", 10)
        elif op in _MATH1:
            inner = resolve(inst.operands[0])
            if op == "exp":
                # Clamp exp argument to float64 safe range before calling
                # jnp.exp. Verilog-A physics often guards exp() calls inside
                # jnp.where branches (e.g. PSP103's expl macro: only take the
                # exp branch when |x| < 230.26). JAX evaluates both branches
                # eagerly, so an unguarded exp(1e8) in the dead branch still
                # produces inf, which then propagates through subsequent ops
                # (e.g. sqrt(inf) = inf, sqrt(inf * negative) = NaN). Clamping
                # to [-709, 709] keeps both branches finite so jnp.where can
                # select the correct one cleanly.
                expr = Expr(f"jnp.exp(jnp.clip({inner.text}, -709.0, 709.0))")
            elif op in ("ln", "log"):
                # Floor log/log10 argument to 1e-300 to prevent log(0) = -inf and
                # the derivative 1/0 that turns into NaN via inf*0 in downstream
                # Jacobian expressions. The gradient of max(x, 1e-300) is 0 for
                # x < 1e-300, so the full chain gives 0 in dead branches.
                fn = "jnp.log" if op == "ln" else "jnp.log10"
                expr = Expr(f"{fn}(jnp.maximum({inner.text}, 1e-300))")
            elif op == "sqrt":
                # Floor sqrt argument to 1e-30 rather than 0. Using 0 causes
                # `0.5 / sqrt(0) = inf` in JAX's derivative path (the JVP of
                # sqrt is 0.5/sqrt(x), which diverges at x=0), and `inf * 0`
                # propagates as NaN through subsequent ops. A 1e-30 floor keeps
                # the derivative finite (0.5/sqrt(1e-30) ≈ 5e14) while the
                # gradient of jnp.maximum(x, 1e-30) w.r.t. x is 0 for x<1e-30,
                # so the chain rule gives 5e14 * 0 = 0 — no NaN. The primal
                # change (sqrt(1e-30) ≈ 3e-16 instead of 0) is negligible for
                # dead-branch values that would have been masked by jnp.where.
                # Verilog-A physics guards sqrt() calls with conditional branches
                # (e.g. JUNCAP200's `if (V < VMAX) ... else { zinv = sqrt(idmult) }`
                # — the else branch is only valid when V >= VMAX). The diamond-phi
                # lifter emits jnp.where for simple diamonds but falls back to
                # picking one edge when nested if/else defeats the single-
                # predecessor walk. JAX
                # evaluates all paths eagerly, so sqrt of a large negative from
                # the dead else-branch produces NaN that poisons downstream ops.
                # Flooring to 1e-30 (rather than 0) prevents inf gradients.
                expr = Expr(f"jnp.sqrt(jnp.maximum({inner.text}, 1e-30))")
            else:
                expr = Expr(f"{_MATH1[op]}({inner.text})")
        elif op in _MATH2:
            a = resolve(inst.operands[0])
            b = resolve(inst.operands[1])
            if op == "fdiv":
                # Safe divide: substitute 1e-300 for a zero denominator so
                # that 0/0 → 0 instead of NaN.  Arises when phi-resolution
                # fallback picks the wrong SSA edge for module-level loop
                # variables (e.g. JUNCAP200's ``vbi_minus_vjsrh``), placing
                # both operands at zero in a dead ``jnp.where`` branch.  The
                # NaN poisons the selected branch; a tiny denominator floors
                # the dead result at ~0 while keeping the live branch
                # accurate.  The choice of ``1e-300`` (rather than something
                # larger) matters: PSP103's ring-oscillator DC homotopy
                # needs the ``a/1e-300 ~ 1e+300`` magnitude to push Newton
                # AWAY from numerically-degenerate fixed points; a tighter
                # floor (``1e-30``) or an outer ``where`` mask to 0 changes
                # the residual landscape and Newton converges to a
                # nonphysical operating point at ~900 V.
                #
                # Compile-time fold: when the denominator is a literal we
                # know whether it's zero / non-finite at emit time, so skip
                # the wrapper entirely.  Emitting ``a / 1.6021918e-19``
                # instead of a 200-character ``jnp.where(...)`` is what
                # shrinks the all-static PSP103 source by ~45 % and lets
                # the rest of the binop folder collapse compound chains
                # when ``a`` is also a literal.
                _b = b.text
                _a = a.text
                bv = _try_literal(b)
                if bv is not None:
                    if bv == 0.0:
                        expr = Expr("0.0")
                    else:
                        # ``a / 1.602e-19`` — let the binop folder downstream
                        # handle compound static folding.
                        expr = Expr(f"jnp.divide({_a}, {_b})")
                elif cse is not None and _is_provably_nonzero(_b, cse.hoist_defs):
                    # Denominator is guaranteed non-zero (``jnp.exp(...)``,
                    # ``jnp.sqrt(jnp.maximum(_, K>0))``, or a product of
                    # such).  Skip the 4-op safe-divide wrapper — saves
                    # ``==``, ``~jnp.isfinite``, ``|``, ``jnp.where``
                    # per use site.  PSP103 N=9 has ~50 such patterns
                    # in the all-static emit, plus another ~300 ``a*b``
                    # squares whose operands trace back to these.
                    expr = Expr(f"jnp.divide({_a}, {_b})")
                else:
                    expr = Expr(
                        f"jnp.divide({_a}, jnp.where(({_b} == 0.0) | ~jnp.isfinite({_b}), 1e-300, {_b}))"
                    )
            elif op == "pow":
                # jnp.power(a, b) is computed via exp(b * log(a)) — returns
                # NaN for any negative base, even for integer-valued
                # exponents like 2.0 or 4.0 (because JAX cannot distinguish
                # float 4.0 from a genuine fractional exponent at trace
                # time).  JUNCAP's (Vbi−V)^p terms can have slightly
                # negative bases when V exceeds Vbi at strong forward bias;
                # clamp to 0 to keep the result finite.  This is physically
                # correct: the power term is only active in the depletion
                # region (V < Vbi) and the dead-branch value is masked away
                # by ``jnp.where``.
                _a = a.text
                expr = Expr(f"jnp.power(jnp.maximum({_a}, 0.0), {b.text})")
            else:
                expr = Expr(f"{_MATH2[op]}({a.text}, {b.text})")
        elif op in _CAST_OPS:
            # Numeric/boolean type casts lower to identity in a JAX world —
            # JAX arrays are dynamically typed through tracers.
            expr = resolve(inst.operands[0])
        elif op == "call":
            expr = _lower_call(
                inst, const_table, fn_name, call_kinds, used_simparams, resolve
            )
        else:
            msg = f"unhandled opcode {op!r} in function {fn_name!r}"
            raise LoweringError(msg)
    finally:
        visiting.discard(ssa)

    # CSE: hoist every instruction result that actually computes something
    # into a named local ``vN = <expr>``. Without this, a chain of phis /
    # ``jnp.where`` nodes whose conditions or edges each have refcount 1
    # inline each other's full text recursively, producing output that
    # grows exponentially in CFG depth (BSIM4 / PSP103 reach > 50 GB RSS
    # before the OOM killer steps in). Aggressive hoisting trades a few
    # extra named-local lines for an output size bounded by O(MIR ops).
    #
    # Passthroughs (``optbarrier``, numeric/boolean casts, ``ddt`` / ``ddx``
    # builtin calls lowered via ``_lower_call``) are excluded — they alias
    # their operand and emitting ``v26 = v38`` for them would just be
    # visual noise.
    skip_hoist = (
        op in _PASSTHROUGH or op in _CAST_OPS or bool(_TRIVIAL_EXPR_RE.match(expr.text))
    )
    if cse is not None and not skip_hoist:
        hoist_name = f"{cse.ssa_prefix}{ssa}" if cse.ssa_prefix else ssa
        if hoist_name not in cse.hoist_defs:
            cse.hoist_defs[hoist_name] = expr.text
            cse.hoist_order.append(hoist_name)
        env[ssa] = Expr(hoist_name, prec=100)
        return env[ssa]

    env[ssa] = expr
    return expr


def _lower_call(
    inst: Inst,
    const_table: dict[str, Constant],
    fn_name: str,
    call_kinds: dict[tuple[str, str], str],
    used_simparams: set[str],
    resolve: Callable[[str], Expr],
) -> Expr:
    """Lower a result-producing ``vN = call inst<k>(args)``.

    Dispatches on the pre-computed ``call_kinds`` table:

    - ``ddt`` / ``ddx`` are identity passthroughs: the DAE system already
      splits reactive contributions into the ``react`` residual slot, and
      ``ddx`` outputs only feed observer variables (``gd`` / ``cd``) that
      circulax would build via ``jax.jacfwd`` anyway.
    - ``simparam("name")`` becomes ``_simparam_<name>``, a new kwarg on
      the generated component (default from :data:`_SIMPARAM_DEFAULTS`).
    - ``user`` (Verilog-A ``analog function``) still raises — needs a
      function-lowering pass we haven't written.
    """
    target = inst.call_target
    if target is None:
        msg = f"call without target in function {fn_name!r}"
        raise LoweringError(msg)
    kind = call_kinds.get((fn_name, target), "unknown")

    if kind in {"ddt", "ddx"}:
        # Identity: the actual time-derivative / state-derivative behaviour
        # is handled by circulax's ``q_dict`` / ``jax.jacfwd`` machinery.
        return resolve(inst.operands[0])

    if kind == "simparam":
        # First (only) arg is an sconst holding the simparam name.
        arg_name = inst.operands[0]
        if arg_name not in const_table or const_table[arg_name].kind != "sconst":
            msg = f"simparam call with non-string arg {arg_name!r} in {fn_name!r}"
            raise LoweringError(msg)
        sim_name = const_table[arg_name].sconst or ""
        used_simparams.add(sim_name)
        return Expr(_simparam_kwarg_name(sim_name))

    if kind == "limit_passthrough":
        # ``StoreLimit(state)`` returns its argument verbatim (limiting is
        # disabled in circulax — see ``EnableLimInput``).  ``Print`` /
        # ``LimDiscontinuity`` are side-effect-only; their result is
        # nominally undefined, but the inst has a result slot so we feed
        # the first operand back through to keep the use-site happy.
        return resolve(inst.operands[0]) if inst.operands else Expr("0.0")

    if kind == "noise_zero":
        return Expr("0.0")

    if kind == "default_arg":
        # ``$simparam_opt(name, default)`` and ``$analysis(...)``: pick the
        # second positional argument when present (= the default), else 0.0.
        if len(inst.operands) >= 2:
            return resolve(inst.operands[1])
        return Expr("0.0")

    if kind == "user":
        msg = f"user-defined ``analog function`` lowering not implemented (target {target!r} in function {fn_name!r})"
        raise LoweringError(msg)

    msg = f"unsupported call kind {kind!r} for target {target!r} in {fn_name!r}"
    raise LoweringError(msg)


def _lower_phi(  # noqa: C901
    inst: Inst,
    defs: dict[str, Inst],
    const_table: dict[str, Constant],
    env: dict[str, Expr],
    fn_name: str,
    visiting: set[str],
    call_kinds: dict[tuple[str, str], str],
    used_simparams: set[str],
    cfg: _FunctionCfg | None,
    cse: CseState | None,
    sccp: object | None = None,
) -> Expr:
    """Lower a phi node.

    Two cases, tried in order:

    1. **Diamond if/else merge.** The phi has exactly two edges coming
       from blocks that share a common ``br``-terminating ancestor; we
       emit ``jnp.where(cond, true_expr, false_expr)`` with the correct
       sense. Covers the diode's ``Rs <= 0`` collapse, ``Vf < FC*Vjeff``
       junction-charge split, ``Rs > 1e-3`` exp-clamp split, and the
       ``lexp`` / ``spicepnjlim`` analog-function bodies once we teach
       their call sites.

    2. **Parameter-clamping fallback.** When diamond detection can't
       identify a single controlling condition — typical of OpenVAF's
       setup-function phis that fan in from multiple validator paths
       all producing the same value — we resolve every edge and pick
       the first non-zero result. This is correct for the clamping
       pattern because every non-invalid edge yields the same param.
    """
    assert inst.phi_edges is not None  # noqa: S101 — parser invariant

    def resolve_edge(ssa: str) -> Expr | None:
        try:
            return _resolve_ssa(
                ssa,
                env,
                defs,
                const_table,
                fn_name=fn_name,
                visiting=visiting,
                call_kinds=call_kinds,
                used_simparams=used_simparams,
                cfg=cfg,
                cse=cse,
                sccp=sccp,
            )
        except LoweringError:
            return None

    # NB: an earlier version of this function had a Case 0 SCCP-shortcut
    # that bypassed diamond detection when ``sccp.live_phi_value``
    # returned a single live edge.  That fired for legitimate
    # multi-edge PSP103 phis where SCCP's edge-executability analysis
    # was correct but choosing one edge dropped voltage-dependent
    # branches that the diamond / fallback rules would have kept via
    # ``jnp.where`` emission.  Removed: the existing diamond rule is
    # already SCCP-aware via the env-injected literal conditions, so
    # static-condition phis still fold; runtime-condition phis still
    # emit ``jnp.where``.

    # Case 1: try diamond.
    if cfg is not None and len(inst.phi_edges) == 2:
        diamond = _find_simple_diamond(inst.phi_edges, cfg)
        if diamond is not None:
            cond_ssa, true_edge_ssa, false_edge_ssa = diamond

            # SCCP fold: when the lattice has classified the diamond's
            # condition as CONSTANT, we can pick the live branch without
            # ever resolving the dead one — eliminating the "dead branch
            # is computed anyway" leak that turns sentinel-default chains
            # (BV=1e20, Cjo=0, etc.) into 1e+20-magnitude residuals under
            # aggressive static_params.  Bypasses the textual
            # ``_try_literal(cond)`` check, which fails when ``cond`` is
            # a CSE-hoisted SSA reference rather than a Python literal.
            from .sccp import SccpResult  # noqa: PLC0415

            if isinstance(sccp, SccpResult):
                cond_lat = sccp.lattice_value(cond_ssa)
                if cond_lat.is_constant and isinstance(
                    cond_lat.value, (bool, int, float)
                ):
                    live_ssa = true_edge_ssa if cond_lat.value else false_edge_ssa
                    live = resolve_edge(live_ssa)
                    if live is not None:
                        return live

            cond = resolve_edge(cond_ssa)
            if cond is not None:
                cond_val = _try_literal(cond)
                if cond_val is not None:
                    # Static condition — resolve only the live branch.
                    live_ssa = true_edge_ssa if cond_val else false_edge_ssa
                    live = resolve_edge(live_ssa)
                    if live is not None:
                        return live
            true_expr = resolve_edge(true_edge_ssa)
            false_expr = resolve_edge(false_edge_ssa)
            if cond is not None and true_expr is not None and false_expr is not None:
                return Expr(
                    f"jnp.where({cond.text}, {true_expr.text}, {false_expr.text})"
                )

    # Case 2: fallback — pick the most-informative edge.
    #
    # The original heuristic was "first resolvable non-zero edge", which
    # is fragile under literal substitution: when ``static_params`` baking
    # causes some edges to fold to literal ``"0.0"`` and others to fold
    # to a different sentinel constant (1.0, 8e22, etc.), the picked
    # edge can be a constant chain that has no voltage dependency, even
    # though the runtime-only counterpart would have selected a voltage-
    # using branch from the same phi.  PSP103 has dozens of such phis;
    # the picked edge differing between TYPE-only and all-static modes
    # was the structural cause of the all-static residual leak.
    #
    # Better priority order:
    #   1. Edge referencing voltages (``signals.X``) or state (``s.X``):
    #      these contribute physics and shouldn't be silently dropped.
    #   2. Edge that's a non-literal SSA / runtime expression: if no
    #      voltage edge, prefer expressions over literal sentinels.
    #   3. Non-zero literal edge: matches the old heuristic for the
    #      case where every edge is a constant.
    #   4. First resolved edge (zero or otherwise): last-resort default.
    resolved: list[Expr] = []
    for edge in inst.phi_edges:
        expr = resolve_edge(edge.value)
        if expr is not None:
            resolved.append(expr)
    if not resolved:
        msg = f"phi at result {inst.result!r} has no resolvable edges"
        raise LoweringError(msg)

    def _refs_voltage(text: str) -> bool:
        return "signals." in text or "s." in text

    def _is_literal(text: str) -> bool:
        try:
            float(text)
            return True
        except ValueError:
            return text in ("True", "False")

    # Priority 1: voltage-referencing edges.
    for e in resolved:
        if _refs_voltage(e.text):
            return e
    # Priority 2: non-literal expressions (runtime SSAs, jnp.* calls).
    for e in resolved:
        if not _is_literal(e.text):
            return e
    # Priority 3: non-zero literals (preserves previous behaviour for
    # all-literal phi groups).
    for e in resolved:
        if e.text != "0.0":
            return e
    # Priority 4: last-resort.
    return resolved[0]


def _find_simple_diamond(
    phi_edges: list, cfg: _FunctionCfg
) -> tuple[str, str, str] | None:  # noqa: C901
    """Detect a two-edge diamond merge and return its condition + branch SSAs.

    Matches ``phi [a, B1], [b, B2]`` where B1 / B2 share a
    ``br``-terminating ancestor D. Returns
    ``(cond_ssa, edge_ssa_for_true, edge_ssa_for_false)`` when it
    matches, else ``None``.

    Handles both the pure-diamond case (both edge-blocks ``jmp`` to the
    phi's block) and the asymmetric case where one edge-block itself is
    a ``br`` target of the decision (the other routing through a
    straight chain of ``jmp``-only blocks).
    """
    if len(phi_edges) != 2:
        return None

    def climb(start: str) -> tuple[str, str] | None:
        # Walk from ``start`` toward a br-terminated ancestor along
        # single-predecessor chains. Returns (decision_block, arrived_from).
        cur = start
        seen: set[str] = set()
        while True:
            if cur in seen:
                return None
            seen.add(cur)
            term = cfg.terminators.get(cur)
            if term is not None and term.opcode == "br":
                return (cur, cur)
            preds = cfg.preds.get(cur, [])
            if len(preds) != 1:
                return None
            # Step up; we're looking for the block that leads into ``cur``.
            p = preds[0]
            p_term = cfg.terminators.get(p)
            if p_term is None:
                return None
            if p_term.opcode == "br":
                return (p, cur)
            if p_term.opcode == "jmp":
                cur = p
                continue
            return None

    a = climb(phi_edges[0].block)
    b = climb(phi_edges[1].block)

    if a is not None and b is not None:
        # Both climbs succeeded — standard diamond.
        dec_a, from_a = a
        dec_b, from_b = b
        if dec_a != dec_b:
            return None
        term = cfg.terminators.get(dec_a)
        if term is None or term.opcode != "br" or len(term.targets) != 2:
            return None
        cond_ssa = term.operands[0]
        t_true, t_false = term.targets
        edges_by_from = {from_a: phi_edges[0].value, from_b: phi_edges[1].value}
        true_ssa = edges_by_from.get(t_true)
        false_ssa = edges_by_from.get(t_false)
        if true_ssa is None or false_ssa is None:
            return None
        return cond_ssa, true_ssa, false_ssa

    # One-sided climb: one edge's block has multiple predecessors (e.g. it is
    # the merge point of a nested if/else inside the true branch, as happens
    # when the Verilog-A `expl` macro is expanded inside `if (V < VMAX)`).
    # The successful climb still gives us the decision block; the failing
    # edge's block must be in the subtree of the other branch target.
    if a is None and b is None:
        return None
    succeeded_idx = 1 if a is None else 0
    failed_idx = 1 - succeeded_idx
    dec, from_ok = b if a is None else a
    term = cfg.terminators.get(dec)
    if term is None or term.opcode != "br" or len(term.targets) != 2:
        return None
    cond_ssa = term.operands[0]
    t_true, t_false = term.targets
    # Assign: the block we climbed to is one of t_true / t_false; the other
    # side contains the multi-predecessor merge block.
    ok_edge_ssa = phi_edges[succeeded_idx].value
    failed_edge_ssa = phi_edges[failed_idx].value
    if from_ok == t_true:
        return cond_ssa, ok_edge_ssa, failed_edge_ssa
    if from_ok == t_false:
        return cond_ssa, failed_edge_ssa, ok_edge_ssa
    return None


def _residual_ssa_names(cm: CompiledModule) -> list[str]:
    """SSA names referenced by DaeSystem residuals — the roots we must lower."""
    names: list[str] = []
    for residual in cm.dae.residual.values():
        names.extend([residual.resist, residual.react])
    return names


def _jacobian_ssa_names(cm: CompiledModule) -> list[str]:
    """SSA names referenced by DaeSystem Jacobian entries.

    Each ``MatrixEntry`` in the dump contributes two SSAs: ``resist`` for
    ``∂f/∂V`` and ``react`` for ``∂q/∂V``. OpenVAF's ``mir_autodiff`` has
    already run the chain rule, so these are fully-computed expressions
    in the eval function that share intermediates with the residuals.
    """
    names: list[str] = []
    for entry in cm.dae.jacobian.values():
        names.extend([entry.resist, entry.react])
    return names


# ---------------------------------------------------------------------------
# Operand resolution (SSA lookup or constant).
# ---------------------------------------------------------------------------


def _resolve_operand(
    name: str, env: dict[str, Expr], const_table: dict[str, Constant]
) -> Expr:
    if name in env:
        return env[name]
    if name in const_table:
        return _const_expr(const_table[name])
    msg = f"unresolved operand {name!r} — not a local SSA, not a known constant"
    raise LoweringError(msg)


def _const_expr(c: Constant) -> Expr:
    if c.kind == "fconst":
        assert c.fconst is not None  # noqa: S101 — parser invariant
        return Expr(_float_literal(c.fconst))
    if c.kind == "iconst":
        # Emit as float when the value is too large for safe integer tracing
        # in JAX (e.g. 1e20 from BSIM3v3's internal formulas).
        v = c.iconst
        if v is not None and abs(v) > 2**53:
            return Expr(_float_literal(float(v)))
        return Expr(str(v))
    if c.kind == "bconst":
        return Expr("True" if c.bconst else "False")
    if c.kind == "sconst":
        return Expr(repr(c.sconst))
    msg = f"unhandled constant kind {c.kind!r}"
    raise LoweringError(msg)


def _float_literal(v: float) -> str:
    if math.isinf(v):
        return "float('inf')" if v > 0 else "-float('inf')"
    if math.isnan(v):
        return "float('nan')"
    # Prefer short forms: integer-valued floats print as ``1.0``, not ``1``.
    if v == int(v) and abs(v) < 1e16:
        return f"{int(v)}.0"
    return repr(v)


def _paren(e: Expr, parent_prec: int) -> str:
    if e.prec < parent_prec:
        return f"({e.text})"
    return e.text


# Non-associative right-hand operators: ``a - (b - c)`` / ``a / (b / c)`` parse
# differently from their unparenthesised forms. Force parens on the right
# operand when its precedence equals the parent's for these.
_NONASSOC_RIGHT = {"-", "/", "%", "//"}


def _paren_right(e: Expr, parent_prec: int, op: str) -> str:
    if op in _NONASSOC_RIGHT:
        if e.prec <= parent_prec:
            return f"({e.text})"
        return e.text
    return _paren(e, parent_prec)


# ---------------------------------------------------------------------------
# HIR input-kind -> Python expression.
# ---------------------------------------------------------------------------


def _node_voltage_expr(name: str, node_id: str, internal_name: dict[str, str]) -> str:
    """Render a single-node voltage reference (``signals.<port>`` or ``s.v_<internal>``)."""
    if node_id in internal_name:
        return f"s.v_{internal_name[node_id]}"
    return f"signals.{name}"


def _input_kind_expr(  # noqa: C901, PLR0911
    kind: InputKind,
    branch_state_name: dict[int, str],
    internal_name: dict[str, str],
    static_params: dict[str, int | float] | None = None,
    sentinel_params: set[str] | None = None,
) -> Expr:
    if isinstance(kind, Voltage):
        if kind.hi is None:
            msg = f"Voltage input without a resolved hi-node name: {kind!r}"
            raise LoweringError(msg)
        hi_ref = _node_voltage_expr(kind.hi, kind.hi_node, internal_name)
        if kind.lo is None:
            # Single-ended probe ``V(port)`` or ``V(internal)``.
            return Expr(hi_ref)
        lo_ref = _node_voltage_expr(kind.lo, kind.lo_node or "", internal_name)
        return Expr(f"{hi_ref} - {lo_ref}", prec=6)
    if isinstance(kind, ParamRef):
        return Expr(kind.name)
    if isinstance(kind, ParamGivenRef):
        # $param_given(X) — was the param explicitly provided by the user?
        #   * In static_params (baked at compile time): True
        #   * In sentinel_params (auto-detected sentinel default): False
        #     The model's else-branch (computed default) is taken.
        #   * Otherwise (non-sentinel default, runtime-supplied): True
        #     The user gave a concrete value via the netlist.
        # The third case fixes PSP103: instance params W, L, AD, … are
        # always given via netlist settings; returning False there made
        # the model ignore them and use VA defaults instead.
        if static_params and kind.name in static_params:
            return Expr("True")
        if sentinel_params and kind.name in sentinel_params:
            return Expr("False")
        return Expr("True")
    if isinstance(kind, TemperatureInput):
        return Expr("_temperature")
    if isinstance(kind, ParamSysFunInput):
        return Expr(f"_{kind.name}")
    if isinstance(kind, HiddenStateInput):
        # HiddenState inputs come pre-computed into eval — the actual binding
        # is made by ``_bind_cslot_args`` (for cached values) or by pulling
        # the variable's definition from the init function. In circulax
        # semantics these are trace-time constants, so we name them by the
        # Verilog-A variable identifier.
        return Expr(kind.var)
    if isinstance(kind, CurrentKind):
        if kind.branch_id is not None and kind.kind in ("Branch", "Unnamed"):
            state = branch_state_name.get(kind.branch_id)
            if state is None:
                msg = f"branch-current input refers to BranchId({kind.branch_id}) with no state mapping"
                raise LoweringError(msg)
            return Expr(f"s.{state}")
        msg = f"Current kind {kind.kind!r} not supported in Stage 2"
        raise LoweringError(msg)
    if isinstance(kind, PortConnectedInput):
        # ``$port_connected(p)``: True iff the netlist wired ``p`` up.
        # In circulax all ports declared by the model are always
        # connected — there's no sub-circuit "leave it floating" mode —
        # so emit a compile-time True.  Models that gate physics on
        # ``$port_connected`` (BSIMBULK's bulk terminal, ASMHEMT's gate
        # field plate) get the connected-side branch; the alternative
        # branch is dead and SCCP will eliminate it.
        return Expr("True")
    if isinstance(kind, (PrevStateInput, NewStateInput)):
        # OpenVAF's ``$limit`` machinery (BSIM3v3, diodes with limiting):
        # ``prev_state_N`` is the previous Newton iterate, ``new_state_N``
        # the slot the model writes its damped update into.  Circulax's
        # solver manages convergence externally — we feed zero here so any
        # limiting expression in the model devolves into the unbounded
        # form, which matches what the OSDI / VACASK reference paths
        # produce when limiting is disabled.
        return Expr("0.0")
    if isinstance(kind, EnableLimInput):
        # Limiting is always off in circulax's pipeline — see PrevStateInput.
        return Expr("False")

    msg = f"no lowering for input kind {type(kind).__name__}"
    raise LoweringError(msg)


# ---------------------------------------------------------------------------
# Component surface: ports, states, parameter list.
# ---------------------------------------------------------------------------


def _plan_component_surface(  # noqa: C901, PLR0912
    cm: CompiledModule,
    branch_state_name: dict[int, str],
    internal_name: dict[str, str],
    va_defaults: dict[str, ParamSpec],
    used_simparams: set[str],
    static_params: dict[str, int | float] | None = None,
) -> tuple[list[str], list[str], list[tuple[str, str, str]]]:
    """Decide the circulax ``ports`` / ``states`` / ``params`` for the device.

    - Ports: the resolved port names from the module header (already in
      declaration order).
    - States: any DaeSystem unknown that's not a Kirchhoff-law node —
      currently ``Current(Branch(...))`` entries (inductor's ``i_L``) and
      internal nodes (diode's ``CI``).
    - Params: every distinct ``Param`` visible in the setup / init / eval
      interners, tagged with its Python type and default. Integer and
      string params land as ``eqx.field(static=True)`` in the emitter so
      JAX doesn't try to trace through them.  Params in ``static_params``
      are omitted entirely — their values are baked in as literals and must
      not appear in the emitted function signature.
    """
    _static = static_params or {}
    ports = list(cm.ports)

    # States = non-port DaeSystem unknowns (branch currents + internal nodes).
    states: list[str] = []
    for sim_id, node_repr in cm.dae.unknowns.items():
        if _is_branch_unknown(node_repr):
            states.append(_branch_state_from_unknown(node_repr, branch_state_name))
        elif node_repr in cm.internal_nodes:
            states.append(f"v_{internal_name.get(node_repr, node_repr)}")
        elif node_repr not in cm.port_nodes:
            # Implicit-equation unknowns would land here — defer.
            msg = f"unknown sim-id {sim_id!r} -> {node_repr!r}: non-port, non-internal, non-branch"
            raise LoweringError(msg)

    # Build the param list from eval-interner Param refs in declaration order,
    # then add any setup-only params missed. Each entry is
    # ``(name, python_type, default_literal)``.
    seen: set[str] = set()
    specs: list[tuple[str, str, str]] = []

    # Priority order: setup -> init -> eval, so earliest declaration wins.
    for interner in (cm.setup_interner, cm.init_interner, cm.eval_interner):
        for kind in interner.parameters.values():
            if isinstance(kind, ParamRef) and kind.name not in seen:
                seen.add(kind.name)
                if kind.name in _static:
                    continue  # baked in as literal; must not appear in the emitted signature
                spec = va_defaults.get(kind.name)
                if spec is None:
                    specs.append((kind.name, "float", "0.0"))
                else:
                    specs.append((kind.name, spec.type_, spec.default))

    # Append the simulator-supplied kwargs the eval references.
    eval_kinds = list(cm.eval_interner.parameters.values())
    if (
        any(isinstance(k, TemperatureInput) for k in eval_kinds)
        and "_temperature" not in seen
    ):
        seen.add("_temperature")
        specs.append(("_temperature", "float", "300.0"))
    if (
        any(isinstance(k, ParamSysFunInput) and k.name == "mfactor" for k in eval_kinds)
        and "_mfactor" not in seen
    ):
        seen.add("_mfactor")
        specs.append(("_mfactor", "float", "1.0"))

    # ``$simparam(...)`` references become ``_simparam_<name>`` kwargs with
    # sensible defaults. Sorted for deterministic output.
    for sim_name in sorted(used_simparams):
        kw = _simparam_kwarg_name(sim_name)
        if kw in seen:
            continue
        seen.add(kw)
        specs.append((kw, "float", _SIMPARAM_DEFAULTS.get(sim_name, "0.0")))

    return ports, states, specs


def _is_branch_unknown(node_repr: str) -> bool:
    return node_repr.startswith(("br[", "Branch(")) or "BranchId(" in node_repr


def _branch_state_from_unknown(
    node_repr: str, branch_state_name: dict[int, str]
) -> str:
    """Extract the BranchId from a ``br[Branch(BranchId(N))]`` DAE unknown and look up its state name."""
    m = re.search(r"BranchId\((\d+)\)", node_repr)
    if not m:
        msg = f"could not extract branch id from {node_repr!r}"
        raise LoweringError(msg)
    bid = int(m.group(1))
    name = branch_state_name.get(bid)
    if name is not None:
        return name
    # Fallback when no CurrentKind input lets us know the .va branch name.
    return f"i_br{bid}"


# ---------------------------------------------------------------------------
# Residual collection.
# ---------------------------------------------------------------------------


def _collect_residuals(
    cm: CompiledModule,
    eval_env: dict[str, Expr],
    const_table: dict[str, Constant],
    ports: list[str],
    states: list[str],
    branch_state_name: dict[int, str],
    internal_name: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build ``f`` / ``q`` expression dicts keyed by the circulax port/state name."""
    f_exprs: dict[str, str] = {}
    q_exprs: dict[str, str] = {}

    node_to_port = dict(zip(cm.port_nodes, ports, strict=True))

    for sim_id, residual in cm.dae.residual.items():
        node_repr = cm.dae.unknowns[sim_id]
        if _is_branch_unknown(node_repr):
            name = _branch_state_from_unknown(node_repr, branch_state_name)
        elif node_repr in node_to_port:
            name = node_to_port[node_repr]
        elif node_repr in cm.internal_nodes:
            name = f"v_{internal_name.get(node_repr, node_repr)}"
        else:
            # Fallback: surface with the printed name so a missing mapping is
            # visible in the emitted code rather than silently lost.
            name = node_repr

        _maybe_record(f_exprs, name, residual.resist, eval_env, const_table)
        _maybe_record(q_exprs, name, residual.react, eval_env, const_table)

    # Sanity: every state must appear in at least one of f / q.
    # Exception: when the DAE explicitly assigns the trivial zero SSA to
    # both ``resist`` and ``react`` for an internal-node residual, that
    # state is structurally undriven (BSIMBULK's spare noise-network slot
    # ``N2`` is the canonical example).  Pin it to a literal zero so the
    # solver still has a defined Kirchhoff equation but the state's value
    # has no effect on physics — it'll converge to whatever its initial
    # guess was.
    missing = [s for s in states if s not in f_exprs and s not in q_exprs]
    if missing:
        for s in list(missing):
            f_exprs[s] = "0.0"
            missing.remove(s)
    if missing:  # defensive — should never fire after the loop above
        msg = f"states with no residual contribution: {missing}"
        raise LoweringError(msg)

    return f_exprs, q_exprs


def _collect_jacobian(
    cm: CompiledModule,
    eval_env: dict[str, Expr],
    const_table: dict[str, Constant],
    ports: list[str],
    branch_state_name: dict[int, str],
    internal_name: dict[str, str],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """Turn each ``DaeSystem.jacobian`` entry into a pair of Python expressions.

    Returns two sparse dicts keyed by ``(row_name, col_name)`` —
    ``resist`` covers ``∂f/∂V`` contributions, ``react`` covers ``∂q/∂V``.
    Zero entries are omitted so the emitter can default missing positions
    to ``0.0`` in the dense matrix.
    """
    jac_resist: dict[tuple[str, str], str] = {}
    jac_react: dict[tuple[str, str], str] = {}
    node_to_port = dict(zip(cm.port_nodes, ports, strict=True))

    def unknown_name(sim_id: str) -> str:
        node_repr = cm.dae.unknowns[sim_id]
        if _is_branch_unknown(node_repr):
            return _branch_state_from_unknown(node_repr, branch_state_name)
        if node_repr in node_to_port:
            return node_to_port[node_repr]
        if node_repr in cm.internal_nodes:
            return f"v_{internal_name.get(node_repr, node_repr)}"
        return node_repr

    for entry in cm.dae.jacobian.values():
        row = unknown_name(entry.row)
        col = unknown_name(entry.col)
        _maybe_record(jac_resist, (row, col), entry.resist, eval_env, const_table)
        _maybe_record(jac_react, (row, col), entry.react, eval_env, const_table)

    return jac_resist, jac_react


def _maybe_record(
    out: dict,
    name: object,
    ssa: str,
    env: dict[str, Expr],
    const_table: dict[str, Constant],
) -> None:
    """Record ``out[name] += <python expr for ssa>``, skipping zero values.

    Accumulates (rather than overwrites) so that multiple Jacobian entries
    that collapse onto the same (row, col) pair are summed rather than
    silently dropped.  This arises after the node-collapse pass remaps
    formerly-distinct (v_DI, v_DI) and (D, v_DI) entries both onto (D, D).
    """
    if _is_zero(ssa, env, const_table):
        return
    expr = _resolve_operand(ssa, env, const_table)
    if name in out:
        out[name] = f"{out[name]} + {expr.text}"
    else:
        out[name] = expr.text


def _is_zero(ssa: str, env: dict[str, Expr], const_table: dict[str, Constant]) -> bool:
    """Check if ``ssa`` is a numeric-zero constant (either bound locally or via the const table)."""
    if ssa in const_table:
        c = const_table[ssa]
        if c.kind == "fconst" and c.fconst == 0.0:
            return True
        if c.kind == "iconst" and c.iconst == 0:
            return True
    if ssa in env:
        # An SSA bound to a zero-literal expression. Conservative match on the
        # emitted text: ``0.0`` is what _float_literal produces for zero.
        return env[ssa].text == "0.0"
    return False


# ---------------------------------------------------------------------------
# Identifier helpers.
# ---------------------------------------------------------------------------


def _camel_case(name: str) -> str:
    """``resistor_va`` → ``ResistorVa``, ``capacitor`` → ``Capacitor``."""
    parts = [p for p in re.split(r"[_\s]+", name) if p]
    return "".join(p[:1].upper() + p[1:] for p in parts)


__all__ = [
    "Expr",
    "LoweredDevice",
    "LoweringError",
    "lower",
]


# Re-export a few symbols for type-check readers, suppress unused-imports.
_ = (CachedValues, DaeResidual)
