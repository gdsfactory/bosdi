"""MIR interpreter — evaluates ``Function`` bodies with concrete inputs.

Used as an OSDI-equivalent reference for SSA-by-SSA debugging of the
lowering. Given the same ``DumpFile`` that the lowering ingests, this
module walks the ``init_fn`` / ``eval_fn`` block-by-block with concrete
parameter values and signal voltages, returning a ``dict[ssa_name, value]``
covering every value computed during the evaluation.

The intended use:

    >>> from bosdi.va.ir_client import compile_va  # JSON path; has init/eval split
    >>> from bosdi.va.interpret import interpret
    >>> dump = compile_va("juncap200.va")
    >>> m = dump.modules[0]
    >>> # Step 1: run init_fn to build the cache.
    >>> init_env = interpret(m.init_fn, m.init_interner,
    ...                       params={"AB": 1e-12, "TYPE": 1.0, ...},
    ...                       signals={}, temperature=300.15)
    >>> cache = {slot: init_env[ssa] for ssa, slot in m.cached.mapping.items()}
    >>> # Step 2: run eval_fn with the cache + concrete signals.
    >>> eval_env = interpret(m.eval_fn, m.eval_interner,
    ...                       params={"AB": 1e-12, ...},
    ...                       signals={"A": 0.5, "K": 0.0},
    ...                       temperature=300.15, cslot_args=cache)

This is a *reference* interpreter — straightforward, possibly slow, no
JIT, no AD. Used to compare against the lowering's emitted output at
the SSA level when debugging numerical divergence.

Limited to opcodes the simple two-terminal models (juncap200, diode,
capacitor) need; will raise for unhandled opcodes so we can iterate.
"""

from __future__ import annotations

import math
from typing import Any

from .mir import (
    AbstimeInput,
    Block,
    CurrentKind,
    EnableLimInput,
    Function,
    HiddenStateInput,
    HirInterner,
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


class InterpretError(RuntimeError):
    """Raised when the interpreter hits an unsupported opcode or an unbound SSA."""


# Maximum number of basic-block transitions before we declare a runaway loop.
_MAX_STEPS = 1_000_000


def interpret(  # noqa: C901, PLR0912, PLR0915
    fn: Function,
    interner: HirInterner,
    *,
    params: dict[str, float],
    signals: dict[str, float] | None = None,
    temperature: float = 300.15,
    cslot_args: dict[str, float] | None = None,
    extra_inputs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Walk ``fn`` block-by-block, return ``{ssa_name: computed_value}``.

    Args:
        fn: The MIR function to evaluate.
        interner: Maps function-arg SSA names to ``InputKind`` semantic tags
            (parameter, voltage, temperature, etc.).
        params: Parameter values keyed by Verilog-A name (matches
            ``ParamRef.name`` in the interner).
        signals: Branch-voltage values keyed by ``hi`` port name (matches
            ``Voltage.hi``). Values are the high-side voltage; the
            interpreter reconstructs ``V(hi, lo)`` automatically.
        temperature: Numeric temperature for ``TemperatureInput`` args.
        cslot_args: Cached slot values for trailing eval_fn args (matches
            ``CachedValues.mapping`` outputs from the init_fn run).
        extra_inputs: Escape hatch for unmodelled inputs (e.g. opaque
            limits, mfactor) — keyed by SSA name directly.

    Returns:
        ``env``: dict mapping every SSA name encountered to its scalar value.
        Constants are pre-loaded; instructions populate as they execute;
        phi reads from the predecessor block's contribution.
    """
    signals = signals or {}
    cslot_args = cslot_args or {}
    extra_inputs = extra_inputs or {}

    env: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Constants table from the function's preamble.
    # ------------------------------------------------------------------
    for c in fn.constants:
        if c.kind == "fconst":
            env[c.name] = c.fconst
        elif c.kind == "iconst":
            env[c.name] = c.iconst
        elif c.kind == "bconst":
            env[c.name] = c.bconst
        elif c.kind == "sconst":
            env[c.name] = c.sconst
        else:
            raise InterpretError(f"unknown constant kind {c.kind!r} for {c.name}")

    # ------------------------------------------------------------------
    # Bind function args from interner-described inputs.
    # ------------------------------------------------------------------
    bound_args: list[str] = []
    for arg in fn.args:
        kind = interner.parameters.get(arg)
        if kind is None:
            # Trailing cslot args (eval_fn only): consumed below by name match.
            continue
        if isinstance(kind, ParamRef):
            if kind.name not in params:
                raise InterpretError(f"missing param value for {kind.name!r}")
            env[arg] = float(params[kind.name])
        elif isinstance(kind, ParamGivenRef):
            env[arg] = bool(kind.name in params)
        elif isinstance(kind, Voltage):
            hi_v = float(signals.get(kind.hi, 0.0)) if kind.hi else 0.0
            lo_v = float(signals.get(kind.lo, 0.0)) if kind.lo else 0.0
            env[arg] = hi_v - lo_v
        elif isinstance(kind, TemperatureInput):
            env[arg] = float(temperature)
        elif isinstance(kind, ParamSysFunInput):
            env[arg] = float(extra_inputs.get(kind.name, 1.0))
        elif isinstance(kind, HiddenStateInput):
            env[arg] = float(extra_inputs.get(kind.var, 0.0))
        elif isinstance(kind, PortConnectedInput):
            env[arg] = bool(extra_inputs.get(f"port_connected_{kind.port}", True))
        elif isinstance(kind, AbstimeInput):
            env[arg] = float(extra_inputs.get("abstime", 0.0))
        elif isinstance(kind, (PrevStateInput, NewStateInput, EnableLimInput)):
            env[arg] = 0.0
        elif isinstance(kind, CurrentKind):
            # Branch / port current input — at residual evaluation time the
            # current isn't yet known (it's the unknown the residual constrains).
            # Use the supplied value if any, else 0; the residual output is
            # what matters, not the input.
            key = (
                f"current_{kind.branch}"
                if kind.branch
                else f"port_current_{kind.hi or 'anon'}"
            )
            env[arg] = float(extra_inputs.get(key, 0.0))
        else:
            raise InterpretError(
                f"unknown InputKind {type(kind).__name__} for arg {arg}"
            )
        bound_args.append(arg)

    # Trailing cslot args (eval_fn): bind any arg not already bound from a
    # cslot-named source. Order-preserving: the lowering binds positionally
    # against ``cached.mapping`` keys, so the caller passes ``cslot_args``
    # as ``{init_ssa_name: value}`` and we line them up here.
    cslot_names = list(cslot_args.keys())
    cslot_idx = 0
    for arg in fn.args:
        if arg in env:
            continue
        if cslot_idx < len(cslot_names):
            env[arg] = float(cslot_args[cslot_names[cslot_idx]])
            cslot_idx += 1
        else:
            # Genuinely unbound — leave for late binding via extra_inputs.
            if arg in extra_inputs:
                env[arg] = float(extra_inputs[arg])

    # ------------------------------------------------------------------
    # CFG walk.
    # ------------------------------------------------------------------
    blocks_by_label = {b.label: b for b in fn.blocks}
    if not fn.blocks:
        return env

    cur_block = fn.blocks[0].label
    prev_block: str | None = None
    steps = 0

    while True:
        steps += 1
        if steps > _MAX_STEPS:
            raise InterpretError(f"runaway: exceeded {_MAX_STEPS} block transitions")

        block: Block = blocks_by_label[cur_block]

        # Inside a block, evaluate each instruction in order.
        next_block: str | None = None
        for inst in block.insts:
            if inst.opcode == "br":
                cond_v = _resolve(inst.operands[0], env)
                next_block = inst.targets[0] if cond_v else inst.targets[1]
                break
            if inst.opcode == "jmp":
                next_block = inst.targets[0]
                break
            if inst.opcode == "exit":
                return env
            _eval_inst(inst, env, prev_block, fn)

        if next_block is None:
            # Block had no terminator? Treat as exit.
            return env

        prev_block = cur_block
        cur_block = next_block


def _resolve(name: str, env: dict[str, Any]) -> Any:
    if name not in env:
        raise InterpretError(f"unbound SSA: {name!r}")
    return env[name]


def _eval_inst(
    inst: Inst, env: dict[str, Any], prev_block: str | None, fn: Function
) -> None:  # noqa: C901, PLR0911, PLR0912, PLR0915
    op = inst.opcode

    # Phi: pick the operand whose edge.block matches prev_block.
    if op == "phi":
        if prev_block is None:
            raise InterpretError(f"phi {inst.result} at entry block (no prev)")
        for edge in inst.phi_edges or []:
            if edge.block == prev_block:
                env[inst.result] = _resolve(edge.value, env)
                return
        raise InterpretError(
            f"phi {inst.result} has no edge from prev_block {prev_block!r}"
        )

    # Pure binary / unary ops on SSA values.
    if op in {"fadd", "iadd"}:
        env[inst.result] = _resolve(inst.operands[0], env) + _resolve(
            inst.operands[1], env
        )
        return
    if op in {"fsub", "isub"}:
        env[inst.result] = _resolve(inst.operands[0], env) - _resolve(
            inst.operands[1], env
        )
        return
    if op in {"fmul", "imul"}:
        env[inst.result] = _resolve(inst.operands[0], env) * _resolve(
            inst.operands[1], env
        )
        return
    if op == "fdiv":
        a, b = _resolve(inst.operands[0], env), _resolve(inst.operands[1], env)
        env[inst.result] = (
            a / b if b != 0 else float("inf") * (1 if a > 0 else (-1 if a < 0 else 0))
        )
        return
    if op == "idiv":
        a, b = _resolve(inst.operands[0], env), _resolve(inst.operands[1], env)
        env[inst.result] = a // b if b != 0 else 0
        return
    if op in {"fneg", "ineg"}:
        env[inst.result] = -_resolve(inst.operands[0], env)
        return
    if op in {"flt", "ilt"}:
        env[inst.result] = _resolve(inst.operands[0], env) < _resolve(
            inst.operands[1], env
        )
        return
    if op in {"fgt", "igt"}:
        env[inst.result] = _resolve(inst.operands[0], env) > _resolve(
            inst.operands[1], env
        )
        return
    if op in {"fle", "ile"}:
        env[inst.result] = _resolve(inst.operands[0], env) <= _resolve(
            inst.operands[1], env
        )
        return
    if op in {"fge", "ige"}:
        env[inst.result] = _resolve(inst.operands[0], env) >= _resolve(
            inst.operands[1], env
        )
        return
    if op in {"feq", "ieq"}:
        env[inst.result] = _resolve(inst.operands[0], env) == _resolve(
            inst.operands[1], env
        )
        return
    if op in {"fne", "ine"}:
        env[inst.result] = _resolve(inst.operands[0], env) != _resolve(
            inst.operands[1], env
        )
        return
    if op in {"and", "band"}:
        env[inst.result] = bool(_resolve(inst.operands[0], env)) and bool(
            _resolve(inst.operands[1], env)
        )
        return
    if op in {"or", "bor"}:
        env[inst.result] = bool(_resolve(inst.operands[0], env)) or bool(
            _resolve(inst.operands[1], env)
        )
        return
    if op in {"not", "bnot"}:
        env[inst.result] = not bool(_resolve(inst.operands[0], env))
        return
    if op == "sqrt":
        x = _resolve(inst.operands[0], env)
        env[inst.result] = math.sqrt(max(x, 0.0))
        return
    if op == "exp":
        x = _resolve(inst.operands[0], env)
        env[inst.result] = math.exp(min(max(x, -709.0), 709.0))
        return
    if op == "log" or op == "ln":
        x = _resolve(inst.operands[0], env)
        env[inst.result] = math.log(max(x, 1e-300))
        return
    if op == "abs" or op == "fabs":
        env[inst.result] = abs(_resolve(inst.operands[0], env))
        return
    if op == "pow":
        a, b = _resolve(inst.operands[0], env), _resolve(inst.operands[1], env)
        try:
            env[inst.result] = math.pow(max(a, 0.0), b)
        except (ValueError, OverflowError):
            env[inst.result] = 0.0
        return
    if op == "optbarrier":
        # Identity passthrough — preserves an SSA boundary for the optimizer
        # but the actual value is whatever flows in.
        env[inst.result] = _resolve(inst.operands[0], env)
        return
    if op in {"cast", "ifcast", "ficast", "bcast", "icast", "fcast"}:
        # Type casts — the interpreter doesn't track types, so passthrough.
        env[inst.result] = _resolve(inst.operands[0], env)
        return
    if op in {"shl", "shr", "lshr", "ashr"}:
        a = int(_resolve(inst.operands[0], env))
        b = int(_resolve(inst.operands[1], env))
        if op == "shl":
            env[inst.result] = a << b
        elif op == "shr" or op == "lshr":
            env[inst.result] = (a & 0xFFFFFFFFFFFFFFFF) >> b
        else:  # ashr
            env[inst.result] = a >> b
        return
    if op in {"xor", "bxor"}:
        env[inst.result] = bool(_resolve(inst.operands[0], env)) != bool(
            _resolve(inst.operands[1], env)
        )
        return
    if op == "select":
        # select cond, if_true, if_false
        cond = bool(_resolve(inst.operands[0], env))
        env[inst.result] = (
            _resolve(inst.operands[1], env) if cond else _resolve(inst.operands[2], env)
        )
        return

    # call inst<n>(args)  — dispatch on the call_decl's raw text.
    if op == "call":
        # Most calls in DC-relevant chains: ddt → 0, white_noise → 0,
        # set_Invalid → 0 (validation no-op).
        target = inst.call_target or ""
        decl = next((d for d in fn.call_decls if d.name == target), None)
        raw = (decl.raw if decl else "").lower()
        if (
            "ddt" in raw
            or "white_noise" in raw
            or "set_invalid" in raw
            or "limexp" in raw
        ):
            if inst.result is not None:
                env[inst.result] = 0.0
            return
        if "%sqrt" in raw:
            env[inst.result] = math.sqrt(max(_resolve(inst.operands[0], env), 0.0))
            return
        if "%exp" in raw:
            env[inst.result] = math.exp(
                min(max(_resolve(inst.operands[0], env), -709.0), 709.0)
            )
            return
        if "%ln" in raw or "%log" in raw:
            env[inst.result] = math.log(max(_resolve(inst.operands[0], env), 1e-300))
            return
        if inst.result is not None:
            env[inst.result] = 0.0
        return

    # Boolean constant inline (rare; usually folded into constants table).
    if op == "bconst":
        env[inst.result] = inst.operands[0] in ("true", "True", "1")
        return

    raise InterpretError(f"unhandled opcode {op!r} (inst {inst.result})")


__all__ = ["interpret", "InterpretError"]
