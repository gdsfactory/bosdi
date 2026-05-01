"""PyO3-binding adapter — converts an ``openvaf_py.VaModule`` to circulax's
``CompiledModule`` / ``DumpFile`` dataclasses.

This replaces the role of ``circulax/va/dump_parser.py`` (907 lines of
regex-based text parsing of ``openvaf-r --dump-mir`` output) with a thin
marshalling layer (~250 lines) over the typed PyO3 binding at
``vajax/openvaf_jax/openvaf_py``.  The downstream ``circulax/va/lowering.py``
walker is unchanged — it consumes the same dataclasses produced either way.

Why bother:
- The text dump is a stringly-typed format that the compiler can only emit;
  recovering the structured data needs a hand-written parser per InputKind /
  rendering. Three of the four big MOSFET models in this tree currently
  fail to parse because of unhandled renderings (``PrevState``,
  ``PortConnected { ... } .. "..."``, etc.).
- The PyO3 binding gives us the compiler's typed objects directly. Every
  model openvaf-r can compile becomes available with no new parser work.
- Fewer subprocess invocations of ``openvaf-r``; faster front-end too.

Usage:

    from bosdi.va import compile_va

    dump = compile_va("path/to/diode.va")
    # ``dump`` is a ``DumpFile``; downstream APIs (``lower``,
    # ``parse_va_defaults``, ...) consume it identically to ``parse_dump``.
"""

from __future__ import annotations

import re
from typing import Any

from .mir import (
    AbstimeInput,
    Block,
    CachedValues,
    CallDecl,
    CompiledModule,
    Constant,
    CurrentKind,
    DaeInfo,
    DaeMatrixEntry,
    DaeResidual,
    DumpFile,
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
    PhiEdge,
    PortConnectedInput,
    PrevStateInput,
    TemperatureInput,
    Voltage,
)


# ---------------------------------------------------------------------------
# Param-name parsing — translates the textual names openvaf_py emits for
# Voltage / Current probes back into structured InputKind dataclasses.
# ---------------------------------------------------------------------------

_VOLTAGE_RE = re.compile(r"^V\(([^,)]+)(?:,\s*([^)]+))?\)$")
_CURRENT_RE = re.compile(r"^I\(([^,)]+)(?:,\s*([^)]+))?\)$")


def _parse_voltage_name(name: str) -> tuple[str, str | None] | None:
    """Parse ``V(hi, lo)`` or single-ended ``V(hi)``; return ``(hi, lo)`` or
    ``None`` if the name doesn't match the expected shape.
    """
    m = _VOLTAGE_RE.match(name)
    if not m:
        return None
    hi = m.group(1).strip()
    lo_raw = m.group(2)
    lo = lo_raw.strip() if lo_raw is not None else None
    return hi, lo


def _parse_current_name(name: str) -> tuple[str, str | None] | None:
    m = _CURRENT_RE.match(name)
    if not m:
        return None
    hi = m.group(1).strip()
    lo_raw = m.group(2)
    lo = lo_raw.strip() if lo_raw is not None else None
    return hi, lo


def _input_kind_from_param(
    name: str,
    kind: str,
    param_id: int,
    branch_registry: dict[tuple[str, str | None], int] | None = None,
) -> InputKind:
    """Build the matching ``InputKind`` dataclass from the binding's
    ``(name, kind)`` strings.

    Every kind ``openvaf_py`` reports is mapped here.  Unknown kinds get a
    bare ``InputKind`` so the lowering walk can still skip them gracefully
    rather than raising at marshal-time.

    ``branch_registry`` (when provided) gives named-branch ``CurrentKind``
    inputs stable, dense ``branch_id`` values shared across the eval and
    init interners.  ``openvaf_py`` doesn't expose OpenVAF's internal
    ``BranchId``, so we synthesise one keyed on the branch's ``(hi, lo)``
    node-name pair — the lowering only needs the id to be consistent, not
    to match the compiler's internal numbering.
    """
    if kind == "temperature":
        return TemperatureInput()
    if kind == "abstime":
        return AbstimeInput()
    if kind == "sysfun":
        # ``$mfactor`` etc.  The leading ``$`` is sometimes absent in the
        # binding's reported name; circulax's expected ``ParamSysFunInput``
        # carries the bare identifier.
        return ParamSysFunInput(name=name.lstrip("$"))
    if kind == "param":
        return ParamRef(name=name, param_id=param_id)
    if kind == "param_given":
        return ParamGivenRef(name=name, param_id=param_id)
    if kind == "voltage":
        parsed = _parse_voltage_name(name)
        if parsed is not None:
            hi, lo = parsed
            # Circulax's ``Voltage`` carries both the resolved node name and
            # the internal node id.  The binding only gives us the name as
            # reported by the compiler, which is the resolved name when the
            # source spelled it that way.  Use the same string for both
            # fields — the lowering walker doesn't distinguish them in any
            # path that fires for ``openvaf_py``-marshalled models.
            return Voltage(hi=hi, lo=lo, hi_node=hi, lo_node=lo)
        return Voltage(hi=name, lo=None, hi_node=name, lo_node=None)
    if kind == "current":
        parsed = _parse_current_name(name)
        if parsed is not None:
            hi, lo = parsed
            if "Unnamed" in name:
                return CurrentKind(kind="Unnamed", hi=hi, lo=lo)
            # Named branch I(name) or I(hi, lo).  Synthesise a stable
            # ``branch_id`` (keyed on ``(hi, lo)``) and use ``hi`` as the
            # ``branch`` field — that's the user-facing branch identifier
            # the lowering uses to derive the state name ``s.i_<branch>``.
            if branch_registry is not None:
                key = (hi, lo)
                bid = branch_registry.setdefault(key, len(branch_registry))
            else:
                bid = param_id
            return CurrentKind(kind="Branch", branch=hi, branch_id=bid, hi=hi, lo=lo)
        return CurrentKind(kind="Unnamed")
    if kind == "hidden_state":
        return HiddenStateInput(var=name, var_id=param_id)
    if kind == "port_connected":
        return PortConnectedInput(port=name)
    if kind == "prev_state":
        # OpenVAF appends an integer suffix (``prev_state_3``) — extract
        # it so the lowering handler can pair the input with its
        # corresponding ``new_state`` slot if useful.
        idx = _trailing_int(name)
        return PrevStateInput(index=idx)
    if kind == "new_state":
        return NewStateInput(index=_trailing_int(name))
    if kind == "enable_lim":
        return EnableLimInput()
    return InputKind()


_TRAILING_INT_RE = re.compile(r"_(\d+)$")


def _trailing_int(name: str) -> int:
    m = _TRAILING_INT_RE.search(name)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Function / Block / Inst marshalling.
# ---------------------------------------------------------------------------


def _build_constants(mir: dict) -> list[Constant]:
    """Rebuild circulax's preamble constant list from the binding's
    per-type constant dicts.

    Order matters for the lowering walk's CSE-by-name lookup: all four
    types are interleaved into one list, sorted by SSA name to keep the
    behaviour deterministic.
    """
    out: list[Constant] = []
    for name, val in (mir.get("constants") or {}).items():
        out.append(Constant(name=name, kind="fconst", fconst=float(val)))
    for name, val in (mir.get("int_constants") or {}).items():
        out.append(Constant(name=name, kind="iconst", iconst=int(val)))
    for name, val in (mir.get("bool_constants") or {}).items():
        out.append(Constant(name=name, kind="bconst", bconst=bool(val)))
    for name, val in (mir.get("str_constants") or {}).items():
        # str_constants in the binding map ssa-name → literal-table-index.
        # Circulax treats them as opaque ``sconst`` payloads; carry the
        # value through as a string.
        out.append(Constant(name=name, kind="sconst", sconst=str(val)))
    out.sort(key=lambda c: c.name)
    return out


def _normalize_callback_name(
    cb_name: str, node_names: dict[str, str] | None = None
) -> str:
    """Map ``openvaf_py``'s CamelCase callback names to the snake_case
    form circulax's ``_classify_callbacks`` (lowering.py) recognises.

    Translations (binding → circulax):
      ``TimeDerivative``         → ``ddt``
      ``NodeDerivative(nodeN)``  → ``ddx_nodeN``
      ``SimParam``               → ``simparam``
      ``CollapseHint(a, b)``     → ``collapse_a_b``  (any naming with
                                   ``collapse_`` prefix is recognised)
      ``Invalid`` / ``SetInvalid`` → ``set_Invalid``

    Anything else passes through unchanged so the lowering's "user-defined
    analog function not implemented" path still surfaces unknown
    callbacks rather than silently masking them.
    """
    if cb_name == "TimeDerivative":
        return "ddt"
    if cb_name.startswith("NodeDerivative"):
        # NodeDerivative(node0) → ddx_node0 (lowering.py treats anything
        # starting with ``ddx_`` as ddx).
        inner = cb_name[len("NodeDerivative") :].strip("()")
        return f"ddx_{inner}" if inner else "ddx_"
    if cb_name == "SimParam":
        return "simparam"
    if cb_name.startswith("CollapseHint"):
        # Lowering's ``_collapse_trivial_nodes`` greps for
        # ``%collapse_(<repr>)_Some\((<repr>)\)`` and then looks the
        # ``<repr>`` strings up in ``cm.dae.unknowns`` (binding side:
        # semantic node names like ``"GP"`` / ``"D"``).  ``openvaf_py``
        # reports ``CollapseHint(nodeN, Some(nodeM))`` using its internal
        # ``node{N}`` ids, so we translate via ``node_names`` (built
        # from ``dae["nodes"]``'s ``idx`` / ``name`` pairs) before
        # emitting.  Without the translation the lowering can't find the
        # nodes in its DAE map and silently skips the collapse — which
        # leaves PSP103's sys_size at 176 instead of the post-collapse 50.
        m = re.match(
            r"CollapseHint\((node\d+),\s*Some\((node\d+)\)\)$",
            cb_name,
        )
        if m:
            a, b = m.group(1), m.group(2)
            if node_names:
                a = node_names.get(a, a)
                b = node_names.get(b, b)
            return f"collapse_{a}_Some({b})"
        # Fallback for shapes we haven't seen yet — preserve a "collapse_"
        # prefix so the callback is still classified as one, even if the
        # node-pair regex won't match.
        inner = cb_name[len("CollapseHint") :]
        if inner.startswith("(") and inner.endswith(")"):
            inner = inner[1:-1]
        return "collapse_" + inner.replace(", ", "_").replace(" ", "_")
    if cb_name in ("SetInvalid", "Invalid"):
        return "set_Invalid"
    return cb_name


def _build_call_decls(
    mir: dict, node_names: dict[str, str] | None = None
) -> list[CallDecl]:
    """Translate ``function_decls`` (dict keyed by ``inst{N}``) into circulax's
    flat ``CallDecl`` list, preserving the inst-name → callback-name mapping.

    ``node_names`` (when provided) is a ``{"node{N}": semantic_name}``
    mapping used by the CollapseHint translator to emit collapse decls
    against the same node identifiers ``cm.dae.unknowns`` carries.
    """
    decls = mir.get("function_decls") or {}
    out: list[CallDecl] = []
    for inst_name, info in decls.items():
        # ``info`` is ``{'name': 'NodeDerivative(node0)', 'num_args': 1, 'num_returns': 1}``.
        # circulax's ``_classify_callbacks`` greps for ``fn %name`` and
        # then compares against snake_case identifiers — translate first.
        cb_name = info["name"] if isinstance(info, dict) else str(info)
        cb_name = _normalize_callback_name(cb_name, node_names=node_names)
        out.append(CallDecl(name=inst_name, raw=f"fn %{cb_name}"))
    return out


def _build_inst(raw: dict[str, Any]) -> Inst:
    """One instruction dict from the binding → one circulax ``Inst``.

    The binding represents control-flow opcodes with named slots
    (``condition``, ``true_block``, ``false_block``, ``destination``,
    ``phi_operands``); circulax's ``Inst`` uses ``operands`` + ``targets``
    + ``phi_edges`` for the same content.  This translates between them.
    """
    op = raw["opcode"]
    result = raw.get("result")
    if op == "br":
        # br <cond> [true, false]  →  operands=[cond], targets=[true, false]
        return Inst(
            result=None,
            opcode="br",
            operands=[raw["condition"]],
            targets=[raw["true_block"], raw["false_block"]],
        )
    if op == "jmp":
        return Inst(
            result=None,
            opcode="jmp",
            operands=[],
            targets=[raw["destination"]],
        )
    if op == "exit":
        return Inst(result=None, opcode="exit", operands=[], targets=[])
    if op == "phi":
        edges = [
            PhiEdge(value=pe["value"], block=pe["block"])
            for pe in raw.get("phi_operands", [])
        ]
        return Inst(
            result=result,
            opcode="phi",
            operands=[],
            phi_edges=edges,
        )
    if op == "call":
        # The binding's call carries the target callback in ``func_ref``
        # (matches the ``inst{N}`` keys in ``function_decls``); circulax's
        # ``Inst.call_target`` carries the same string.
        return Inst(
            result=result,
            opcode="call",
            operands=list(raw.get("operands", [])),
            call_target=raw.get("func_ref")
            or raw.get("target")
            or raw.get("call_target"),
        )
    # Regular SSA-defining instruction (fadd, fmul, fdiv, exp, …).
    return Inst(
        result=result,
        opcode=op,
        operands=list(raw.get("operands", [])),
    )


def _build_function(
    name: str, params: list[str], blocks_meta: dict, instructions: list[dict]
) -> Function:
    """Group the binding's flat instruction list back into circulax's
    per-block ``Block`` dataclasses, preserving order.

    ``blocks_meta`` (the binding's per-block CFG metadata) is iterated in
    insertion order so the resulting ``fn.blocks`` list keeps a stable
    ordering.  ``instructions`` is a flat list with each entry tagged by
    its ``block`` label; we bucket by that label and respect within-bucket
    list order (which matches the OpenVAF MIR's intra-block ordering).
    """
    by_block: dict[str, list[Inst]] = {label: [] for label in blocks_meta}
    for raw in instructions:
        label = raw["block"]
        by_block.setdefault(label, []).append(_build_inst(raw))
    blocks = [Block(label=label, insts=insts) for label, insts in by_block.items()]
    return Function(
        name=name,
        args=list(params),
        blocks=blocks,
    )


def _build_interner(
    ssa_names: list[str],
    user_names: list[str],
    user_kinds: list[str],
    branch_registry: dict[tuple[str, str | None], int] | None = None,
) -> HirInterner:
    """``HirInterner.parameters`` maps SSA-name → InputKind for every
    function input.  ``openvaf_py`` exposes three position-parallel lists:
    ``params`` (SSA names like ``"v21"``), ``param_names`` /
    ``init_param_names`` (user-facing names like ``"$temperature"``,
    ``"Is"``, ``"V(A,CI)"``), and ``param_kinds`` (the ``"voltage"`` /
    ``"param"`` / ... classification).  Circulax's downstream consumers
    key on the SSA name; the user name is what the InputKind dataclass
    carries internally.
    """
    # Only the first ``len(user_names)`` SSAs are interner-described
    # parameters; any trailing SSAs in ``ssa_names`` are cslot bridges
    # populated later by ``_bind_cslot_args`` in the lowering walk.
    interner = HirInterner()
    n_interner = min(len(ssa_names), len(user_names), len(user_kinds))
    for idx in range(n_interner):
        ssa, uname, kind = ssa_names[idx], user_names[idx], user_kinds[idx]
        interner.parameters[ssa] = _input_kind_from_param(
            uname, kind, param_id=idx, branch_registry=branch_registry
        )
    return interner


_MIR_REF = re.compile(r"^mir_(\d+)$")


def _normalize_var(ref: str | None) -> str:
    """Translate the binding's ``mir_NNN`` SSA references into the ``vNNN``
    prefix the lowering walk's instruction list uses.

    The DAE descriptor uses the OpenVAF global MIR value index (rendered
    as ``mir_465``); the same index inside the per-function instruction
    list is rendered as ``v465``.  Circulax's downstream code (the
    ``_residual_ssa_names`` and ``_jacobian_ssa_names`` helpers) expects
    the ``v``-prefixed form because it looks them up via the function's
    instruction defs.  Normalising here keeps both namespaces lined up.
    """
    if ref is None or not ref:
        return ""
    m = _MIR_REF.match(ref)
    if m is None:
        return ref
    return f"v{m.group(1)}"


_FLOW_RE = re.compile(r"^flow\(([^)]+)\)$")


def _rewrite_branch_unknown(
    node_name: str,
    branch_registry: dict[tuple[str, str | None], int],
) -> str:
    """Rewrite ``flow(<name>)`` DAE unknowns to ``br[Branch(BranchId(N))]``.

    ``openvaf_py``'s DAE view labels branch-current unknowns as
    ``flow(<branch_name>)`` while the lowering walker (and the legacy text
    parser) recognise the OpenVAF-internal form
    ``br[Branch(BranchId(N))]``.  Translate via the registry so the
    ``BranchId`` matches whatever ``_input_kind_from_param`` synthesised.
    """
    m = _FLOW_RE.match(node_name)
    if m is None:
        return node_name
    branch_name = m.group(1).strip()
    key = (branch_name, None)
    bid = branch_registry.setdefault(key, len(branch_registry))
    return f"br[Branch(BranchId({bid}))]"


def _build_dae(
    va_module: Any,
    branch_registry: dict[tuple[str, str | None], int] | None = None,
) -> DaeInfo:
    """Translate ``VaModule.get_dae_system()`` into circulax's ``DaeInfo``.

    The two views of the DAE are structurally similar — both express a
    list of residuals indexed by node and a Jacobian as
    ``(row, col, resist_var, react_var)`` entries.
    """
    if branch_registry is None:
        branch_registry = {}
    dae = va_module.get_dae_system()
    info = DaeInfo()
    info.num_resistive = sum(1 for r in dae["residuals"] if r.get("resist_var"))
    info.num_reactive = sum(1 for r in dae["residuals"] if r.get("react_var"))
    # ``unknowns`` maps simulator-internal ``sim_node{N}`` to user-visible
    # node names.  Reconstruct from the residual entries.
    for r in dae["residuals"]:
        sim_name = f"sim_node{r['equation_idx']}"
        info.unknowns[sim_name] = _rewrite_branch_unknown(
            r["node_name"], branch_registry
        )
        info.residual[sim_name] = DaeResidual(
            resist=_normalize_var(r.get("resist_var")),
            react=_normalize_var(r.get("react_var")),
            resist_small_signal=_normalize_var(r.get("resist_small_signal_var")),
            react_small_signal=_normalize_var(r.get("react_small_signal_var")),
            resist_lim_rhs=_normalize_var(r.get("resist_lim_rhs_var")),
            react_lim_rhs=_normalize_var(r.get("react_lim_rhs_var")),
            nature_kind="Flow",  # binding doesn't currently distinguish; default
        )
    for j in dae["jacobian"]:
        # ``row`` / ``col`` carry the simulator-internal ``sim_node{N}``
        # ids (lookup-keyable in ``info.unknowns``) — not the user-facing
        # node names.  The text parser used the same convention so the
        # downstream ``_collect_jacobian`` walk (lowering.py) treats them
        # uniformly with ``info.residual`` keys.
        row = f"sim_node{j['row_node_idx']}"
        col = f"sim_node{j['col_node_idx']}"
        key = f"{row},{col}"
        info.jacobian[key] = DaeMatrixEntry(
            row=row,
            col=col,
            resist=_normalize_var(j.get("resist_var")),
            react=_normalize_var(j.get("react_var")),
        )
    return info


def _build_cached(va_module: Any) -> CachedValues:
    """Translate the binding's ``cache_mapping`` (init-SSA → eval-arg-index)
    into circulax's ``CachedValues``.

    The text parser produces ``mapping: dict[init_ssa, cslot_name]`` and
    ``slots: dict[cslot_name, raw_string]``.  We synthesise cslot names of
    the form ``"cslotN"`` based on the eval-arg-index so downstream code
    that iterates ``cached.mapping`` and ``cached.slots`` stays oblivious.
    """
    cached = CachedValues()
    init_mir = va_module.get_init_mir_instructions()
    for entry in init_mir.get("cache_mapping") or []:
        init_ssa = entry["init_value"]
        eval_param_idx = entry["eval_param"]
        cslot = f"cslot{eval_param_idx - va_module.func_num_params + va_module.num_cached_values}"
        # The cslot index isn't literally important; we just need stable
        # unique names that line up positionally with cm.eval_fn.args after
        # the interner-param prefix.  Use the eval_param index directly:
        cslot = f"cslot{eval_param_idx}"
        cached.mapping[init_ssa] = cslot
        cached.slots[cslot] = "<openvaf_py-bound>"
    return cached


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def from_va_module(va_module: Any) -> CompiledModule:
    """Convert a single ``openvaf_py.VaModule`` to circulax's ``CompiledModule``.

    The setup function is left empty: ``openvaf_py`` doesn't expose a
    distinct ``setup`` MIR function, only ``init`` and ``eval``.
    Downstream code that iterates ``(setup, init, eval)`` (e.g.
    ``_classify_callbacks``, ``_inject_static_params``) treats an empty
    Function as a no-op, so this works without any changes.
    """
    eval_mir = va_module.get_mir_instructions()
    init_mir = va_module.get_init_mir_instructions()

    # Resolve port + internal node names from the DAE view.
    dae_view = va_module.get_dae_system()
    terminals = list(dae_view["terminals"])
    # Drop ``flow(<name>)`` entries — those are branch-current unknowns,
    # which circulax's lowering picks up from ``dae.unknowns`` via the
    # ``br[...]`` rewrite below.  Leaving them in ``internal_nodes``
    # would confuse ``_plan_component_surface`` into emitting a node-
    # voltage state for what is structurally a branch current.
    internals = [n for n in dae_view["internal_nodes"] if not _FLOW_RE.match(n)]
    # ``port_nodes`` mirrors how the text parser surfaces the resolved
    # internal node ids; ``openvaf_py`` already gives us the user-facing
    # names so we use those as both port and node ids (the lowering walk
    # only consults these for diagnostic / debug purposes).
    port_nodes = list(terminals)

    eval_fn = _build_function(
        name="",
        params=eval_mir.get("params", []),
        blocks_meta=eval_mir.get("blocks", {}),
        instructions=eval_mir.get("instructions", []),
    )
    init_fn = _build_function(
        name="_init",
        params=init_mir.get("params", []),
        blocks_meta=init_mir.get("blocks", {}),
        instructions=init_mir.get("instructions", []),
    )
    # Setup function: empty per above.
    setup_fn = Function(name="_setup", args=[], blocks=[])

    branch_registry: dict[tuple[str, str | None], int] = {}
    eval_interner = _build_interner(
        ssa_names=eval_mir.get("params", []),
        user_names=va_module.param_names,
        user_kinds=va_module.param_kinds,
        branch_registry=branch_registry,
    )
    init_interner = _build_interner(
        ssa_names=init_mir.get("params", []),
        user_names=va_module.init_param_names,
        user_kinds=va_module.init_param_kinds,
        branch_registry=branch_registry,
    )
    setup_interner = HirInterner()

    # Build a ``node{idx}`` → semantic-name map from the DAE node table
    # so collapse decls (``CollapseHint(node1, Some(node5))``) translate
    # to the same identifiers ``cm.dae.unknowns`` uses (``"G"`` / ``"GP"``
    # rather than the raw OpenVAF ``node{N}`` id).  Without this the
    # lowering's collapse pass silently skips PSP103's seven trivial
    # voltage-source pairs and the solver runs at sys_size = 176 instead
    # of 50 (3.5× per-step penalty observed).
    node_names: dict[str, str] = {}
    for entry in dae_view.get("nodes") or []:
        if isinstance(entry, dict) and "idx" in entry and "name" in entry:
            node_names[f"node{entry['idx']}"] = entry["name"]

    # Carry preamble constants + callback decls onto each function the
    # lowering walker iterates.  The text parser put these on every fn;
    # mirror that so existing code that consults ``fn.constants`` /
    # ``fn.call_decls`` works for both init and eval.
    eval_fn.constants = _build_constants(eval_mir)
    eval_fn.call_decls = _build_call_decls(eval_mir, node_names=node_names)
    init_fn.constants = _build_constants(init_mir)
    init_fn.call_decls = _build_call_decls(init_mir, node_names=node_names)

    return CompiledModule(
        name=va_module.name,
        ports=terminals,
        port_nodes=port_nodes,
        internal_nodes=internals,
        dae=_build_dae(va_module, branch_registry=branch_registry),
        eval_fn=eval_fn,
        init_fn=init_fn,
        setup_fn=setup_fn,
        eval_interner=eval_interner,
        init_interner=init_interner,
        setup_interner=setup_interner,
        cached=_build_cached(va_module),
    )


def compile_va(
    va_path: str,
    *,
    allow_analog_in_cond: bool = False,
    allow_builtin_primitives: bool = False,
) -> DumpFile:
    """Compile a ``.va`` source file and return a circulax ``DumpFile``.

    Drop-in replacement for ``circulax.va.parse_dump(text)`` for the
    common case where the caller already has a path.  When the caller
    needs to pre-process the source (e.g. macro expansion) or already
    has a MIR text dump, ``parse_dump`` from ``dump_parser.py`` remains
    the path to use.

    Args:
        va_path: Path to the root Verilog-A file.
        allow_analog_in_cond: Pass-through to ``openvaf_py.compile_va``.
            Foundry models (e.g. GF130 PDK) sometimes use analog operators
            (``limexp``, ``ddt``, ``idt``) inside conditionals; set True
            to tolerate.
        allow_builtin_primitives: Pass-through to ``openvaf_py.compile_va``.

    Returns:
        A ``DumpFile`` with one ``CompiledModule`` per Verilog-A module
        in the source.  ``compilation_unit`` and ``literals`` carry stub
        values — the binding doesn't surface these and the lowering walk
        doesn't need them.
    """
    try:
        import openvaf_py  # noqa: PLC0415 — keep optional at module-load time
    except ImportError as exc:  # pragma: no cover — install-path guidance
        msg = (
            "openvaf_py PyO3 binding not installed. Build it from "
            "`vajax/openvaf_jax/openvaf_py/` with `maturin develop` (after "
            "running `git submodule update --init vendor/OpenVAF` in the "
            "vajax checkout), or fall back to the legacy text parser by "
            "calling `circulax.va.parse_dump` on `openvaf-r --dump-mir` "
            "output instead."
        )
        raise ImportError(msg) from exc
    modules = openvaf_py.compile_va(
        va_path,
        allow_analog_in_cond=allow_analog_in_cond,
        allow_builtin_primitives=allow_builtin_primitives,
    )
    return DumpFile(
        compilation_unit=va_path,
        literals={},
        modules=[from_va_module(m) for m in modules],
    )


__all__ = ["compile_va", "from_va_module"]
