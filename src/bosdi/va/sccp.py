"""Sparse Conditional Constant Propagation for circulax MIR.

Implements the Wegman-Zadeck SCCP algorithm (TOPLAS 1991) to propagate
known parameter values through the MIR before the lowering walk runs.
With proper lattice tracking, dead branches get marked unreachable, PHI
nodes whose dead predecessors get pruned simplify to trivial assignments,
and conditional terminators with constant conditions resolve to a single
successor — eliminating the existing heuristic diamond rule's
fold-vs-runtime divergence (the "1e+300 vs 0.375" bug that surfaced
when ``static_params`` covered every uniform-across-instance parameter).

The algorithm is the textbook one but specialised to circulax's MIR:

- Lattice values are ``TOP`` (unknown / not yet computed), ``CONSTANT``
  (with a Python value and a type tag), or ``BOTTOM`` (varies at
  runtime, so unfoldable).
- ``meet`` is glb on the standard lattice — TOP is identity, BOTTOM
  absorbs, equal constants stay constant, unequal constants drop to
  BOTTOM.
- Two worklists: one for newly-reachable blocks, one for SSAs whose
  lattice value just changed.  Process blocks first (so PHIs see their
  executable-edge set before evaluating); process SSAs second
  (propagating through use-sites).
- A terminator with a CONSTANT condition adds only the live successor
  to the executable-edge set.  A BOTTOM condition adds both.  TOP
  defers — we'll re-process when the condition's lattice value
  improves.
- PHIs only consider operands from edges that have been marked
  executable; this is the "conditional" part of conditional constant
  propagation and is what stops dead branches' values from polluting
  the meet result.

Public surface:

- ``run_sccp(fn, initial_constants)`` runs the analysis and returns
  an ``SccpResult`` with the lattice and dead-block set.
- ``SccpResult.lattice_value(ssa)`` returns the per-SSA lattice value;
  ``SccpResult.is_block_dead(label)`` is the convenience check.
- ``SccpResult.live_phi_value(inst, block_label)`` returns the single
  live phi-edge value when only one predecessor is executable, else
  ``None`` — used by the lowering's PHI handler to short-circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .mir import Block, Constant, Function, Inst


class LatticeState(Enum):
    """Position in the SCCP lattice."""

    TOP = auto()  # unknown / not yet computed
    CONSTANT = auto()  # known value (may be None for the constant ``None``)
    BOTTOM = auto()  # runtime / not constant


@dataclass(frozen=True)
class LatticeValue:
    """A point in the lattice.

    Stored separately from raw Python values so the ``CONSTANT`` state
    can carry typing info (mostly for emit-time formatting — bools want
    ``True``/``False``, ints want no decimal point, floats want ``repr``).
    """

    state: LatticeState
    value: Any = None
    py_type: str | None = None  # "bool" | "int" | "float" | "str"

    @classmethod
    def top(cls) -> "LatticeValue":
        return cls(LatticeState.TOP)

    @classmethod
    def bottom(cls) -> "LatticeValue":
        return cls(LatticeState.BOTTOM)

    @classmethod
    def from_python(cls, v: Any) -> "LatticeValue":
        if isinstance(v, bool):
            return cls(LatticeState.CONSTANT, v, "bool")
        if isinstance(v, int):
            return cls(LatticeState.CONSTANT, v, "int")
        if isinstance(v, float):
            return cls(LatticeState.CONSTANT, v, "float")
        if isinstance(v, str):
            return cls(LatticeState.CONSTANT, v, "str")
        return cls.bottom()

    @property
    def is_top(self) -> bool:
        return self.state is LatticeState.TOP

    @property
    def is_constant(self) -> bool:
        return self.state is LatticeState.CONSTANT

    @property
    def is_bottom(self) -> bool:
        return self.state is LatticeState.BOTTOM


_TOP = LatticeValue.top()
_BOTTOM = LatticeValue.bottom()


def _meet(a: LatticeValue, b: LatticeValue) -> LatticeValue:
    """Greatest lower bound on the lattice."""
    if a.is_top:
        return b
    if b.is_top:
        return a
    if a.is_bottom or b.is_bottom:
        return _BOTTOM
    if a.value == b.value and a.py_type == b.py_type:
        return a
    return _BOTTOM


def _meet_all(values: list[LatticeValue]) -> LatticeValue:
    if not values:
        return _TOP
    out = values[0]
    for v in values[1:]:
        out = _meet(out, v)
        if out.is_bottom:
            return out
    return out


# ---------------------------------------------------------------------------
# Opcode evaluation table.
# ---------------------------------------------------------------------------


def _eval_opcode(opcode: str, operands: list[LatticeValue]) -> LatticeValue:
    """Fold a primitive opcode whose every operand is a CONSTANT lattice value.

    Operands are guaranteed CONSTANT by the caller — never TOP/BOTTOM.
    Returns BOTTOM for opcodes the lattice can't usefully evaluate (math
    intrinsics where the result might lose precision, branches, calls).

    The set of folded opcodes intentionally mirrors what circulax's
    existing ``_BINOP_FOLDS`` covers, plus the comparisons and casts
    SCCP needs to make conditional propagation actually fire on the
    parameter-driven guards in PSP103 (``RDE > 0``, ``BV != 1e20``,
    etc.).
    """
    op = opcode.lower()

    if len(operands) == 2:
        a, b = operands[0].value, operands[1].value
        if op in ("ieq", "feq", "beq"):
            return LatticeValue(LatticeState.CONSTANT, a == b, "bool")
        if op in ("ine", "fne", "bne"):
            return LatticeValue(LatticeState.CONSTANT, a != b, "bool")
        if op in ("ilt", "flt"):
            return LatticeValue(LatticeState.CONSTANT, a < b, "bool")
        if op in ("ile", "fle"):
            return LatticeValue(LatticeState.CONSTANT, a <= b, "bool")
        if op in ("igt", "fgt"):
            return LatticeValue(LatticeState.CONSTANT, a > b, "bool")
        if op in ("ige", "fge"):
            return LatticeValue(LatticeState.CONSTANT, a >= b, "bool")
        if op == "iadd":
            return LatticeValue(LatticeState.CONSTANT, a + b, "int")
        if op == "isub":
            return LatticeValue(LatticeState.CONSTANT, a - b, "int")
        if op == "imul":
            return LatticeValue(LatticeState.CONSTANT, a * b, "int")
        if op == "idiv":
            if b == 0:
                return _BOTTOM
            return LatticeValue(LatticeState.CONSTANT, a // b, "int")
        if op == "irem":
            if b == 0:
                return _BOTTOM
            return LatticeValue(LatticeState.CONSTANT, a % b, "int")
        if op == "fadd":
            return LatticeValue(LatticeState.CONSTANT, a + b, "float")
        if op == "fsub":
            return LatticeValue(LatticeState.CONSTANT, a - b, "float")
        if op == "fmul":
            return LatticeValue(LatticeState.CONSTANT, a * b, "float")
        if op == "frem":
            if b == 0:
                return _BOTTOM
            return LatticeValue(LatticeState.CONSTANT, a % b, "float")
        # ``fdiv`` deliberately not folded — ``1.0 / 0.0`` semantics differ
        # between Python (raises ZeroDivisionError) and JAX (yields ``inf``
        # / ``nan``), and the existing ``jnp.divide(a, jnp.where(b == 0,
        # 1e-300, b))`` safe-divide pattern depends on staying at runtime
        # to preserve the guard.

    if len(operands) == 1:
        a = operands[0].value
        # Passthrough opcodes (``optbarrier`` is OpenVAF's no-op marker;
        # cast opcodes are JAX-traceable identities for typed integers
        # and booleans).  The MIR keeps these around for type-tracking
        # and as optimization barriers — for SCCP they're identity:
        # the result's lattice value is exactly the operand's.
        if op in (
            "optbarrier",
            "ifcast",
            "ficast",
            "fbcast",
            "bfcast",
            "bicast",
            "ibfcast",
            "sibitcast",
            "fibcast",
        ):
            return operands[0]
        if op == "bnot":
            return LatticeValue(LatticeState.CONSTANT, not a, "bool")
        if op == "fneg":
            return LatticeValue(LatticeState.CONSTANT, -a, "float")
        if op == "ineg":
            return LatticeValue(LatticeState.CONSTANT, -a, "int")

    return _BOTTOM


# ---------------------------------------------------------------------------
# Result.
# ---------------------------------------------------------------------------


@dataclass
class SccpResult:
    """Output of a single ``run_sccp`` invocation."""

    lattice: dict[str, LatticeValue] = field(default_factory=dict)
    executable_edges: set[tuple[str, str]] = field(default_factory=set)
    visited_blocks: set[str] = field(default_factory=set)
    # ``ssa_block[ssa]`` is the label of the block whose ``insts`` list
    # contains the def producing ``ssa``.  Populated by ``run_sccp`` so
    # callers (the lowering walk's PHI handler in particular) can ask
    # ``live_phi_value`` without rebuilding the index themselves.
    ssa_block: dict[str, str] = field(default_factory=dict)
    # ``voltage_tainted`` contains every SSA whose computation chain
    # involved at least one runtime input (Voltage / CurrentKind /
    # HiddenStateInput / etc. — any function arg seeded as BOTTOM).
    # An SSA can be both ``is_constant`` and voltage-tainted: this means
    # the value happens to be constant at SCCP analysis time (e.g. a
    # multiplication with a structural-zero) but only because of how
    # the runtime inputs cancel.  Such "structural-zero" constants are
    # NOT safe to bake into a setup function: at runtime the underlying
    # voltage chain still needs to produce intermediate non-zero values
    # for downstream computations to be correct.
    voltage_tainted: set[str] = field(default_factory=set)

    def lattice_value(self, ssa: str) -> LatticeValue:
        return self.lattice.get(ssa, _TOP)

    def is_voltage_tainted(self, ssa: str) -> bool:
        """True if SSA's computation involved any runtime (BOTTOM-seeded) input."""
        return ssa in self.voltage_tainted

    def is_block_dead(self, label: str) -> bool:
        return label not in self.visited_blocks

    def is_edge_executable(self, src: str, dst: str) -> bool:
        return (src, dst) in self.executable_edges

    def live_phi_value(self, phi_edges: list, block_label: str) -> str | None:
        """If exactly one predecessor edge is executable, return that
        edge's source SSA name; else None.

        Used by the lowering walk to short-circuit PHI resolution when
        SCCP has already eliminated all but one predecessor.
        """
        live: list[str] = []
        seen: set[str] = set()
        for pe in phi_edges:
            if (pe.block, block_label) in self.executable_edges:
                if pe.value not in seen:
                    seen.add(pe.value)
                    live.append(pe.value)
        if len(live) == 1:
            return live[0]
        return None

    def block_of(self, ssa: str) -> str | None:
        """Return the block label for the def of ``ssa``, or None."""
        return self.ssa_block.get(ssa)

    def summary(self) -> str:
        n_const = sum(1 for v in self.lattice.values() if v.is_constant)
        return (
            f"SccpResult: {n_const} constants, "
            f"{len(self.visited_blocks)} reachable / "
            f"{len(self.executable_edges)} edges"
        )


# ---------------------------------------------------------------------------
# The analysis driver.
# ---------------------------------------------------------------------------


@dataclass
class _Sccp:
    """Internal worklist driver for one ``Function``."""

    fn: Function
    initial_constants: dict[str, Any]
    voltage_arg_names: set[str] = field(default_factory=set)

    lattice: dict[str, LatticeValue] = field(default_factory=dict)
    executable_edges: set[tuple[str, str]] = field(default_factory=set)
    visited_blocks: set[str] = field(default_factory=set)

    cfg_worklist: list[str] = field(default_factory=list)
    ssa_worklist: list[str] = field(default_factory=list)

    blocks_by_label: dict[str, Block] = field(default_factory=dict)
    inst_block: dict[str, str] = field(default_factory=dict)  # ssa name -> block label
    uses: dict[str, list[tuple[str, Inst]]] = field(default_factory=dict)

    def run(self) -> SccpResult:
        self._build_index()
        self._initialize()
        while self.cfg_worklist or self.ssa_worklist:
            while self.cfg_worklist:
                self._process_block(self.cfg_worklist.pop())
            while self.ssa_worklist:
                self._propagate(self.ssa_worklist.pop())
        return SccpResult(
            lattice=self.lattice,
            executable_edges=self.executable_edges,
            visited_blocks=self.visited_blocks,
            ssa_block=self.inst_block,
            voltage_tainted=self._compute_voltage_taint(),
        )

    def _compute_voltage_taint(self) -> set[str]:
        """Forward-propagate voltage-input taint through the def-use graph.

        An SSA is voltage-tainted iff at least one of its dependencies is a
        Voltage / CurrentKind / HiddenStateInput (or other runtime-only
        input).  Param args that happen to be BOTTOM at SCCP analysis time
        (because they're runtime-supplied) are NOT taint sources — they're
        legitimate setup-function inputs.

        The caller distinguishes these via ``voltage_arg_names`` (passed
        through ``run_sccp``); only those args seed the taint set.

        Use case: lowering must NOT bake a voltage-tainted constant into a
        setup function — even if SCCP proves it constant via cancellation
        like ``V(D,D) * coeff = 0``, the runtime computation chain still
        needs to flow through the voltage operands for downstream
        intermediates to be correct.
        """
        # Seed: only explicitly-tagged voltage args.
        tainted: set[str] = set(self.voltage_arg_names)

        # Iterate to fixpoint: any inst whose result depends on a tainted
        # operand becomes tainted.
        changed = True
        while changed:
            changed = False
            for block in self.fn.blocks:
                if block.label not in self.visited_blocks:
                    continue
                for inst in block.insts:
                    if inst.result is None or inst.result in tainted:
                        continue
                    if inst.opcode == "phi":
                        operands = [
                            edge.value
                            for edge in (inst.phi_edges or [])
                            if (edge.block, block.label) in self.executable_edges
                        ]
                    else:
                        operands = list(inst.operands or [])
                    if any(op in tainted for op in operands):
                        tainted.add(inst.result)
                        changed = True
        return tainted

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        for block in self.fn.blocks:
            self.blocks_by_label[block.label] = block
            for inst in block.insts:
                if inst.result is not None:
                    self.inst_block[inst.result] = block.label
                # Build def-use chains.
                for op in inst.operands:
                    self.uses.setdefault(op, []).append((block.label, inst))
                if inst.phi_edges:
                    for edge in inst.phi_edges:
                        self.uses.setdefault(edge.value, []).append((block.label, inst))

    def _initialize(self) -> None:
        # Preamble constants: each ``v3 = fconst 0.0`` etc.
        for c in self.fn.constants:
            self.lattice[c.name] = _const_to_lattice(c)
        # Function args: BOTTOM unless overridden by initial_constants.
        for arg in self.fn.args:
            if arg in self.initial_constants:
                self.lattice[arg] = LatticeValue.from_python(
                    self.initial_constants[arg]
                )
            else:
                self.lattice[arg] = _BOTTOM
        # All other defs start at TOP.
        for block in self.fn.blocks:
            for inst in block.insts:
                if inst.result is not None and inst.result not in self.lattice:
                    self.lattice[inst.result] = _TOP
        # Caller may also supply initial constants for non-arg SSAs (init-cache
        # bridges feed eval-fn args this way).  Override here.
        for ssa, val in self.initial_constants.items():
            self.lattice[ssa] = LatticeValue.from_python(val)
        # Seed the entry block.
        if self.fn.blocks:
            entry = self.fn.blocks[0].label
            self.visited_blocks.add(entry)
            self.cfg_worklist.append(entry)

    # ------------------------------------------------------------------
    # Block / instruction processing
    # ------------------------------------------------------------------

    def _process_block(self, label: str) -> None:
        block = self.blocks_by_label.get(label)
        if block is None:
            return
        terminator: Inst | None = None
        for inst in block.insts:
            if inst.opcode in ("br", "jmp", "exit"):
                terminator = inst
                continue
            self._evaluate(inst, label)
        if terminator is not None:
            self._process_terminator(terminator, label)

    def _evaluate(self, inst: Inst, block_label: str) -> None:
        if inst.result is None:
            return
        old = self.lattice.get(inst.result, _TOP)
        if old.is_bottom:
            return  # can't go any lower
        if inst.phi_edges is not None:
            new = self._evaluate_phi(inst, block_label)
        else:
            new = self._evaluate_regular(inst)
        if new != old:
            self.lattice[inst.result] = new
            self.ssa_worklist.append(inst.result)

    def _evaluate_phi(self, phi: Inst, block_label: str) -> LatticeValue:
        if not phi.phi_edges:
            return _BOTTOM
        values: list[LatticeValue] = []
        for edge in phi.phi_edges:
            if (edge.block, block_label) in self.executable_edges:
                values.append(self.lattice.get(edge.value, _TOP))
        if not values:
            return _TOP  # no executable preds yet
        return _meet_all(values)

    def _evaluate_regular(self, inst: Inst) -> LatticeValue:
        op_vals: list[LatticeValue] = []
        for op in inst.operands:
            v = self.lattice.get(op, _TOP)
            if v.is_top:
                return _TOP  # operand not yet known — defer
            if v.is_bottom:
                return _BOTTOM  # runtime operand → runtime result
            op_vals.append(v)
        return _eval_opcode(inst.opcode, op_vals)

    def _process_terminator(self, term: Inst, block_label: str) -> None:
        if term.opcode == "jmp":
            for tgt in term.targets:
                self._add_edge(block_label, tgt)
            return
        if term.opcode == "exit":
            return
        # ``br <cond> [true_target, false_target]`` per circulax MIR.
        cond_ssa = term.operands[0] if term.operands else None
        if cond_ssa is None or len(term.targets) != 2:
            # Defensive: some malformed branch.  Treat both as live.
            for tgt in term.targets:
                self._add_edge(block_label, tgt)
            return
        cond_val = self.lattice.get(cond_ssa, _TOP)
        true_tgt, false_tgt = term.targets
        if cond_val.is_top:
            return  # defer until condition's lattice value firms up
        if cond_val.is_bottom:
            self._add_edge(block_label, true_tgt)
            self._add_edge(block_label, false_tgt)
            return
        # Constant condition — only the live side is executable.
        if cond_val.value:
            self._add_edge(block_label, true_tgt)
        else:
            self._add_edge(block_label, false_tgt)

    def _add_edge(self, src: str, dst: str) -> None:
        edge = (src, dst)
        if edge in self.executable_edges:
            return
        self.executable_edges.add(edge)
        if dst not in self.visited_blocks:
            self.visited_blocks.add(dst)
            self.cfg_worklist.append(dst)
        else:
            # Already visited — re-evaluate any PHIs at the head of the
            # destination block (they'll see the new edge).
            block = self.blocks_by_label.get(dst)
            if block is None:
                return
            for inst in block.insts:
                if inst.phi_edges is None:
                    continue
                if inst.result is None:
                    continue
                old = self.lattice.get(inst.result, _TOP)
                new = self._evaluate_phi(inst, dst)
                if new != old:
                    self.lattice[inst.result] = new
                    self.ssa_worklist.append(inst.result)

    def _propagate(self, ssa: str) -> None:
        for block_label, inst in self.uses.get(ssa, ()):
            if block_label not in self.visited_blocks:
                continue
            self._evaluate(inst, block_label)
            # If the changed SSA feeds a terminator's condition, the
            # branch direction may have firmed up — re-process so the
            # newly-determined live edge gets added.
            if inst.opcode in ("br", "jmp", "exit"):
                self._process_terminator(inst, block_label)


def _const_to_lattice(c: Constant) -> LatticeValue:
    if c.kind == "fconst" and c.fconst is not None:
        return LatticeValue(LatticeState.CONSTANT, c.fconst, "float")
    if c.kind == "iconst" and c.iconst is not None:
        return LatticeValue(LatticeState.CONSTANT, c.iconst, "int")
    if c.kind == "bconst" and c.bconst is not None:
        return LatticeValue(LatticeState.CONSTANT, c.bconst, "bool")
    if c.kind == "sconst" and c.sconst is not None:
        return LatticeValue(LatticeState.CONSTANT, c.sconst, "str")
    return _BOTTOM


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def run_sccp(
    fn: Function,
    initial_constants: dict[str, Any] | None = None,
    *,
    voltage_arg_names: set[str] | None = None,
) -> SccpResult:
    """Run SCCP on a single MIR :class:`Function`.

    Args:
        fn: The function to analyse.  Its blocks, preamble constants, and
            ``args`` list are inspected; the function is not modified.
        initial_constants: ``{ssa_name: python_value}`` for any SSA whose
            value is known at the start of analysis.  Typically this is
            the function arguments that correspond to ``static_params``
            (the lowering's caller resolves the argument-position-to-name
            mapping via the HirInterner before calling).  The map can
            also contain init-cache bridge SSAs when the eval function
            inherits constants from the init function.

    Returns:
        An :class:`SccpResult` carrying the final lattice and dead-block
        information.  The lowering walk consumes this result to substitute
        constants, prune dead blocks, and short-circuit PHIs without
        having to reproduce the analysis itself.
    """
    driver = _Sccp(
        fn=fn,
        initial_constants=dict(initial_constants or {}),
        voltage_arg_names=set(voltage_arg_names or ()),
    )
    return driver.run()


# ---------------------------------------------------------------------------
# MIR rewriter.
# ---------------------------------------------------------------------------


def rewrite_function(fn: Function, sccp: SccpResult) -> Function:
    """Return a structurally-simplified copy of ``fn`` informed by ``sccp``.

    Three transformations apply, all conservative — each one only fires
    when SCCP has *proven* the simplification is safe:

    1. **Dead-block elimination.**  Blocks SCCP didn't reach (no executable
       incoming edge) are dropped wholesale.  Their defs no longer exist
       in the rewritten function, so any later use-site walk that crosses
       a dead block is guaranteed to surface the dead operand as
       "unresolved" rather than silently picking up a sentinel constant.

    2. **Phi-edge pruning.**  Phi nodes only keep edges from blocks that
       remain in the rewritten function *and* whose ``(src, dst)`` pair
       is in the executable-edge set.  After pruning:

       - Zero edges left → drop the phi entirely (its result becomes
         unreachable anyway; if anything still uses it, that's a real
         bug worth surfacing).
       - One edge → replace with ``optbarrier <edge.value>``.  The
         lowering walk treats ``optbarrier`` as a passthrough, so the
         phi's result becomes a direct alias for the surviving edge's
         source — this is the simplification the heuristic phi-fallback
         used to chase by guesswork.
       - Two+ edges with all-equal source SSAs → same optbarrier
         shortcut.  This catches the case where SCCP keeps multiple
         predecessors live but all of them feed the same value.
       - Otherwise → keep as a phi with the pruned edge list.

    3. **Branch folding.**  ``br <cond>, [t, f]`` whose condition lattice
       is CONSTANT becomes ``jmp [<live_target>]``.  This is what eliminates
       the dead-side phi inputs that the lowering's old diamond rule had
       to identify heuristically; with the rewriter, the phi-input list
       is *literally shorter*.

    The function's preamble (``constants``, ``call_decls``, ``args``,
    ``name``) is preserved verbatim — SCCP doesn't touch those, and the
    lowering walks them independently of the block list.

    The rewriter is pure: ``fn`` is not mutated.  Callers who need the
    original (for debugging, alternative lowerings, etc.) can keep their
    handle.
    """
    visited = sccp.visited_blocks
    edges = sccp.executable_edges
    # ``original_predecessors`` for each block tracks *every* incoming
    # edge in the original CFG (not the executable subset).  We need it
    # to recognise blocks that lose all their predecessors after pruning
    # — those are unreachable in the rewritten CFG even if SCCP marked
    # them visited at some point during its analysis (defensive: SCCP
    # already drops these from ``visited_blocks``, but the check
    # double-checks rather than trusting the flag).
    original_predecessors: dict[str, list[str]] = {b.label: [] for b in fn.blocks}
    for block in fn.blocks:
        term = _terminator(block)
        if term is None:
            continue
        for tgt in term.targets:
            original_predecessors.setdefault(tgt, []).append(block.label)

    new_blocks: list[Block] = []
    for block in fn.blocks:
        if block.label not in visited:
            continue  # dead block — drop wholesale.
        new_insts: list[Inst] = []
        for inst in block.insts:
            rewritten = _rewrite_inst(inst, block.label, edges, visited)
            if rewritten is None:
                continue
            new_insts.append(rewritten)
        new_blocks.append(Block(label=block.label, insts=new_insts))

    return Function(
        name=fn.name,
        args=list(fn.args),
        constants=list(fn.constants),
        call_decls=list(fn.call_decls),
        blocks=new_blocks,
    )


def _terminator(block: Block) -> Inst | None:
    for inst in reversed(block.insts):
        if inst.opcode in ("br", "jmp", "exit"):
            return inst
    return None


def _rewrite_inst(
    inst: Inst,
    block_label: str,
    edges: set[tuple[str, str]],
    visited: set[str],
) -> Inst | None:
    """Return the rewritten instruction, or ``None`` to drop it.

    All instructions other than phis and branches pass through unchanged.
    """
    if inst.phi_edges is not None:
        return _rewrite_phi(inst, block_label, edges, visited)
    if inst.opcode == "br":
        return _rewrite_branch(inst, block_label, edges)
    return inst


def _rewrite_phi(
    inst: Inst,
    block_label: str,
    edges: set[tuple[str, str]],
    visited: set[str],
) -> Inst | None:
    """Prune dead phi edges and collapse single-source phis to optbarrier."""
    pruned: list = []
    for edge in inst.phi_edges or ():
        if edge.block not in visited:
            continue
        if (edge.block, block_label) not in edges:
            continue
        pruned.append(edge)

    if not pruned:
        # No live source.  The phi result is unreachable in the rewritten
        # CFG; drop it.  If something still uses it the lowering will
        # raise "unresolved operand" — which is the *correct* behaviour
        # because our pruning has eliminated the sentinel-default fallback
        # path that used to silently inject a wrong constant.
        return None

    # Dedupe by source value: if every live edge feeds the same SSA, the
    # phi is a no-op identity and we can rewrite to optbarrier.
    distinct = {edge.value for edge in pruned}
    if len(distinct) == 1:
        only = pruned[0].value
        return Inst(
            result=inst.result,
            opcode="optbarrier",
            operands=[only],
            phi_edges=None,
            targets=[],
            call_target=None,
            source_loc=inst.source_loc,
        )

    return Inst(
        result=inst.result,
        opcode=inst.opcode,
        operands=list(inst.operands),
        phi_edges=pruned,
        targets=list(inst.targets),
        call_target=inst.call_target,
        source_loc=inst.source_loc,
    )


def _rewrite_branch(
    inst: Inst,
    block_label: str,
    edges: set[tuple[str, str]],
) -> Inst:
    """Fold ``br cond, [t, f]`` whose condition is a single live edge into ``jmp [target]``."""
    if len(inst.targets) != 2:
        return inst
    true_tgt, false_tgt = inst.targets
    true_live = (block_label, true_tgt) in edges
    false_live = (block_label, false_tgt) in edges
    if true_live and not false_live:
        return Inst(
            result=None,
            opcode="jmp",
            operands=[],
            phi_edges=None,
            targets=[true_tgt],
            call_target=None,
            source_loc=inst.source_loc,
        )
    if false_live and not true_live:
        return Inst(
            result=None,
            opcode="jmp",
            operands=[],
            phi_edges=None,
            targets=[false_tgt],
            call_target=None,
            source_loc=inst.source_loc,
        )
    return inst


__all__ = [
    "LatticeState",
    "LatticeValue",
    "SccpResult",
    "rewrite_function",
    "run_sccp",
]
