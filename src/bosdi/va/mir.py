"""Dataclasses mirroring the structured content of an ``openvaf-r --dump-mir`` output.

These types are intentionally small and dumb: they are a faithful in-memory
representation of what the textual dump encodes, with no extra logic or
opinion. Any transformation, lowering, or emission lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Value:
    """An operand reference in MIR.

    Constants (fconst/iconst/bconst/sconst) are declared in a function's
    preamble with their own SSA name (e.g. ``v3 = fconst 0.0``); within an
    instruction's operand list they are referenced by that name only. So a
    bare ``Value(name="v3", kind="ssa")`` is enough — the const payload is
    looked up via the function's preamble.
    """

    name: str
    kind: str = "ssa"


@dataclass
class PhiEdge:
    """One incoming edge of a phi node: the value selected when control arrives from ``block``."""

    value: str
    block: str


@dataclass
class Inst:
    """One MIR instruction.

    - ``result`` is the SSA name the instruction defines (e.g. ``"v26"``) or
      ``None`` for control-flow (``br``/``jmp``/``call``/``exit``).
    - ``opcode`` is the printed opcode (``"fdiv"``, ``"phi"``, ``"br"``, ...).
    - ``operands`` is the positional list of value names. For ``phi`` this
      field is empty and ``phi_edges`` carries the data.
    - ``targets`` is the list of branch targets for control flow.
    - ``call_target`` is set for ``call inst<n>()`` — the inst name is an
      index into the declaring function's ``call_decls``.
    - ``source_loc`` preserves the ``@<hex>`` prefix if present.
    """

    result: str | None
    opcode: str
    operands: list[str] = field(default_factory=list)
    phi_edges: list[PhiEdge] | None = None
    targets: list[str] = field(default_factory=list)
    call_target: str | None = None
    source_loc: str | None = None


@dataclass
class Block:
    """A basic block in a MIR function: a labelled straight-line sequence of instructions."""

    label: str
    insts: list[Inst] = field(default_factory=list)


@dataclass
class Constant:
    """A preamble-declared constant (``v3 = fconst 0.0`` etc.).

    ``kind`` is one of ``"fconst"``, ``"iconst"``, ``"bconst"``, ``"sconst"``.
    Exactly one of the payload fields is populated. Bool constants are
    printed commented (``// v1 = bconst false``) in the dump because their
    only uses are phi arguments; we treat them uniformly.
    """

    name: str
    kind: str
    fconst: float | None = None
    iconst: int | None = None
    bconst: bool | None = None
    sconst: str | None = None


@dataclass
class CallDecl:
    """A callback-function declaration in the preamble.

    Example: ``inst0 = fn %set_Invalid(Parameter { id: ParamId(0) })(0) -> 0``

    We keep the whole right-hand side as a raw string for now; the emitter
    decides what (if anything) to do with each callback kind.
    """

    name: str  # "inst0"
    raw: str  # "fn %set_Invalid(Parameter { id: ParamId(0) })(0) -> 0"


@dataclass
class Function:
    """One of the three MIR functions emitted per module (eval / init / setup)."""

    name: str  # empty string for anonymous "function %(..)", else e.g. "_init"
    args: list[str]
    constants: list[Constant] = field(default_factory=list)
    call_decls: list[CallDecl] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HIR interner input kinds (what each function argument means semantically).
# ---------------------------------------------------------------------------


@dataclass
class InputKind:
    """Base marker for input-binding variants."""


@dataclass
class Voltage(InputKind):
    """A ``V(hi, lo)`` branch-voltage probe input."""

    hi: str | None  # resolved node name if available (e.g. "A")
    lo: str | None
    hi_node: str  # internal node id (e.g. "node0")
    lo_node: str | None


@dataclass
class CurrentKind(InputKind):
    """An ``I(...)`` branch-current input — named, unnamed, or a port current."""

    kind: str  # "Branch" | "Unnamed" | "Port"
    hi: str | None = None
    lo: str | None = None
    branch: str | None = None  # user-level branch name (from ``.va``), when printed
    branch_id: int | None = None  # OpenVAF ``BranchId(N)`` index, when available


@dataclass
class ParamRef(InputKind):
    """A ``Param(Parameter { id: ParamId(N) }) .. "name"`` input."""

    name: str
    param_id: int


@dataclass
class ParamGivenRef(InputKind):
    """A ``ParamGiven { param: ... } .. "name"`` input."""

    name: str
    param_id: int


@dataclass
class TemperatureInput(InputKind):
    """The ``$temperature`` simulator input."""


@dataclass
class ParamSysFunInput(InputKind):
    """A Verilog-A system-function input like ``$mfactor``."""

    name: str  # "mfactor"


@dataclass
class HiddenStateInput(InputKind):
    """A hidden-state variable fed into a MIR function as an input."""

    var: str  # semantic variable name, e.g. "res"
    var_id: int


@dataclass
class PortConnectedInput(InputKind):
    """A ``$port_connected(port)`` probe input."""

    port: str


@dataclass
class AbstimeInput(InputKind):
    """The ``$abstime`` simulator input."""


@dataclass
class PrevStateInput(InputKind):
    """A ``prev_state_N`` limiting input.

    OpenVAF emits these alongside ``new_state_N`` for models (BSIM3v3,
    diodes with ``$limit``) that use Verilog-A's ``$limit`` mechanism to
    aid Newton-Raphson convergence: the model receives the previous
    iteration's voltage/current and returns a damped update.  Circulax's
    solver already manages convergence externally, so the lowering treats
    ``prev_state`` as an opaque zero — the limiting expression devolves
    into the unbounded form which is what tests already exercise.
    """

    index: int


@dataclass
class NewStateInput(InputKind):
    """A ``new_state_N`` limiting output handle.

    Paired with :class:`PrevStateInput`; appears as a function arg the
    eval body writes into when limiting fires.  Treated as opaque zero
    in circulax's lowering — the value isn't consumed downstream.
    """

    index: int


@dataclass
class EnableLimInput(InputKind):
    """The ``enable_lim`` flag — True when the simulator wants limiting active.

    Circulax always emits this as ``False`` so any conditional limiting
    branch in the model collapses to the no-limit fallback.
    """


@dataclass
class HirInterner:
    """Inputs (value-name -> InputKind) and outputs (label -> value-name) for one function."""

    parameters: dict[str, InputKind] = field(default_factory=dict)
    outputs: dict[str, str | None] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DaeSystem block.
# ---------------------------------------------------------------------------


@dataclass
class DaeResidual:
    """Per-unknown residual fields: SSA names of the resistive/reactive/... contributions."""

    resist: str
    react: str
    resist_small_signal: str
    react_small_signal: str
    resist_lim_rhs: str
    react_lim_rhs: str
    nature_kind: str  # "Flow" | "Potential"


@dataclass
class DaeMatrixEntry:
    """A single ``(row, col)`` Jacobian entry with its resistive and reactive SSA values."""

    row: str  # "sim_node0"
    col: str
    resist: str
    react: str


@dataclass
class DaeInfo:
    """Parsed ``DaeSystem { ... }`` block: unknowns, residuals, Jacobian, and counters."""

    unknowns: dict[str, str] = field(default_factory=dict)  # "sim_node0" -> "node0"
    residual: dict[str, DaeResidual] = field(default_factory=dict)
    jacobian: dict[str, DaeMatrixEntry] = field(default_factory=dict)
    small_signal_parameters: dict = field(default_factory=dict)
    noise_sources: list = field(default_factory=list)
    model_inputs: list[tuple[int, int]] = field(default_factory=list)
    num_resistive: int = 0
    num_reactive: int = 0


@dataclass
class CachedValues:
    """``Cached values during instance setup`` section: bridges init -> eval.

    ``mapping`` maps an SSA value name in the init function to a slot name
    (``"v34" -> "cslot0"``). ``slots`` maps each slot to its class metadata
    (left as a raw string for now; not needed for Stage 1).
    """

    mapping: dict[str, str] = field(default_factory=dict)
    slots: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level module and dump structures.
# ---------------------------------------------------------------------------


@dataclass
class CompiledModule:
    """All parsed information for one Verilog-A module as emitted by ``--dump-mir``."""

    name: str  # "resistor_va"
    ports: list[str]  # resolved names ["A", "B"] when known, else the node ids
    port_nodes: list[str]  # ["node0", "node1"]
    internal_nodes: list[str]
    dae: DaeInfo
    eval_fn: Function
    init_fn: Function
    setup_fn: Function
    eval_interner: HirInterner
    init_interner: HirInterner
    setup_interner: HirInterner
    cached: CachedValues


@dataclass
class DumpFile:
    """Top-level parsed dump: the compilation unit, literals table, and one or more modules."""

    compilation_unit: str
    literals: dict[int, str]
    modules: list[CompiledModule]
