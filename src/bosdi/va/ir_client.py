"""IR client — calls ``openvaf-r`` (GPL, separate process) and parses output.

Two ingestion modes are supported:

- :func:`compile_va` — calls ``openvaf-r --dump-json`` and parses the
  structured JSON output. The post-MIR-optimization form: clean,
  setup/init/eval-split, but the optimizer collapses N-level nested
  if-blocks into single N-edge phis. That collapse defeats the
  lowering's 2-edge diamond detection on patterns like Verilog-A's
  ``analog initial`` conditional assignments, so on heavily nested
  models (juncap200, BSIM4) some phis are mishandled.

- :func:`compile_va_unopt` — calls ``openvaf-r --dump-unopt-mir`` and
  parses the text output via :mod:`bosdi.va.dump_parser`. The
  partially-optimized-with-DAE form: chain of 2-edge phis preserved,
  no setup/init/eval split (the function combines them; the lowering
  re-derives the split via signal-dependency analysis).

No GPL code is imported at any point — only the binary is shelled out.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .dump_parser import DumpParseError
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

_ENV_VAR = "OPENVAF_IR"
_DEFAULT_BINARY = "openvaf-r"

_INF_SENTINEL = "__inf__"
_NEGINF_SENTINEL = "__neginf__"
_NAN_SENTINEL = "__nan__"


def _find_binary() -> str:
    """Locate the ``openvaf-r`` binary or raise with install guidance."""
    path = os.environ.get(_ENV_VAR) or shutil.which(_DEFAULT_BINARY)
    if path:
        return path
    raise RuntimeError(
        f"'{_DEFAULT_BINARY}' not found in PATH and ${_ENV_VAR} is not set.\n"
        "Install OpenVAF: https://openvaf.semimod.de  or set "
        f"${_ENV_VAR}=/path/to/openvaf-r"
    )


def compile_va(
    va_path: str | Path,
    *,
    allow_analog_in_cond: bool = False,
    allow_builtin_primitives: bool = False,
) -> DumpFile:
    """Compile a Verilog-A source file and return a ``DumpFile``.

    Drop-in replacement for ``binding.compile_va``.  The flags
    ``allow_analog_in_cond`` and ``allow_builtin_primitives`` are accepted
    for API compatibility but are currently ignored (openvaf-r applies its
    own defaults).
    """
    binary = _find_binary()
    result = subprocess.run(
        [binary, "--dump-json", str(va_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise DumpParseError(
            f"openvaf-r exited {result.returncode} for {va_path}:\n{result.stderr}"
        )
    try:
        docs = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise DumpParseError(f"Invalid JSON from openvaf-r: {exc}") from exc

    if not isinstance(docs, list):
        raise DumpParseError("Expected JSON array at top level")

    schema_ver = docs[0].get("schema_version", 0) if docs else 0
    if schema_ver != 1:
        raise DumpParseError(
            f"Unsupported --dump-json schema version {schema_ver}; expected 1"
        )

    return DumpFile(
        compilation_unit=str(va_path),
        literals={},
        modules=[_parse_module(doc) for doc in docs],
    )


def compile_va_unopt(va_path: str | Path) -> DumpFile:
    """Compile via ``openvaf-r --dump-unopt-mir`` and parse the text output.

    Returns a ``DumpFile`` whose modules carry the **partially-optimized
    (with DAE)** function in :attr:`CompiledModule.eval_fn`. The
    pre-MIR-optimization form preserves the chain of 2-edge phis from
    nested ``if (cond) { x = expr; }`` blocks that the JSON path's
    optimizer collapses into single N-edge phis.

    Trade-off: ``init_fn`` and ``setup_fn`` are left empty — the
    partially-optimized text dump is one combined function. The
    lowering re-derives any per-instance setup hoists from
    signal-dependency analysis at emit time. For models with substantial
    init-time-only computation, this produces a slower (but correct)
    eval body than the JSON path. The bug fix is structural, not
    performance: nested conditional inits will round-trip correctly
    here where the JSON path silently produces wrong values.

    Use :func:`compile_va` (JSON path) when the model is shallow enough
    that phi collapse doesn't bite — the JSON output is the canonical
    setup/init/eval split with full optimization applied.
    """
    from .dump_parser import parse_dump  # local import: optional dep

    binary = _find_binary()
    result = subprocess.run(
        [binary, "--dump-unopt-mir", str(va_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise DumpParseError(
            f"openvaf-r exited {result.returncode} for {va_path}:\n{result.stderr}"
        )
    dumped = parse_dump(result.stdout)
    # In unopt mode the partially-optimized function combines init+eval —
    # no init→eval cslot bridge exists. Clear ``cached.mapping`` on each
    # module so the lowering's ``_bind_cslot_args`` skips the bridging
    # step entirely.
    from .mir import CachedValues  # local import to keep top-level lean

    for module in dumped.modules:
        module.cached = CachedValues()
    return dumped


def compile_va_opt_mir(va_path: str | Path) -> DumpFile:
    """Compile via ``openvaf-r --dump-mir`` (optimised text MIR) and parse.

    Unlike ``compile_va`` (``--dump-json``), this uses the textual MIR dump
    format that has been part of openvaf-r since its initial release — no
    upstream format changes needed.

    The optimised MIR **has** the init/eval split and ``CachedValues``
    metadata intact, making it useful as a **validation reference** for the
    unopt-MIR dependency-analysis partition: compare the number of cache
    slots openvaf reports here against the number of ``i_``-prefixed hoists
    the unopt partition produces.

    The eval_fn physics from this path may still have collapsed N-edge phis
    for complex models (PSP103 ``expll`` macro) — use ``compile_va_unopt``
    for the actual lowering. This function is for structural metadata only.
    """
    from .dump_parser import parse_dump  # local import: optional dep

    binary = _find_binary()
    result = subprocess.run(
        [binary, "--dump-mir", str(va_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise DumpParseError(
            f"openvaf-r exited {result.returncode} for {va_path}:\n{result.stderr}"
        )
    return parse_dump(result.stdout)


# ---------------------------------------------------------------------------
# Module-level parser.
# ---------------------------------------------------------------------------


def _parse_module(doc: dict) -> CompiledModule:
    ports = list(doc["ports"])
    port_nodes = list(doc.get("port_nodes", ports))
    internal_nodes = list(doc.get("internal_nodes", []))

    eval_fn, eval_interner, eval_cslot_args = _parse_function(doc["eval_fn"])
    init_fn, init_interner, _ = _parse_function(doc["init_fn"])
    setup_fn, setup_interner, _ = _parse_function(doc["setup_fn"])

    cached = _parse_cache_mapping(doc["init_fn"].get("cache_mapping", []))

    # Eval function needs the extra cslot args appended to its args list
    # so that the lowering's cslot-bridge pass can match them up.
    eval_fn.args = eval_fn.args + eval_cslot_args

    dae = _parse_dae(doc["dae"])

    return CompiledModule(
        name=doc["name"],
        ports=ports,
        port_nodes=port_nodes,
        internal_nodes=internal_nodes,
        dae=dae,
        eval_fn=eval_fn,
        init_fn=init_fn,
        setup_fn=setup_fn,
        eval_interner=eval_interner,
        init_interner=init_interner,
        setup_interner=setup_interner,
        cached=cached,
    )


# ---------------------------------------------------------------------------
# Function parser.
# Returns (Function, HirInterner, extra_cslot_arg_names).
# ---------------------------------------------------------------------------


def _parse_function(
    fn_doc: dict,
) -> tuple[Function, HirInterner, list[str]]:
    name = fn_doc.get("name", "")
    args: list[str] = list(fn_doc.get("args", []))
    params_doc: dict = fn_doc.get("params", {})
    constants_doc: dict = fn_doc.get("constants", {})
    call_decls_doc: list = fn_doc.get("call_decls", [])
    blocks_doc: list = fn_doc.get("blocks", [])
    cache_mapping_doc: list = fn_doc.get("cache_mapping", [])

    # Parse interner: SSA name → InputKind
    interner = HirInterner()
    for ssa, kind_doc in params_doc.items():
        interner.parameters[ssa] = _parse_input_kind(kind_doc)

    # Constants preamble
    constants: list[Constant] = _parse_constants(constants_doc)

    # Call decls
    call_decls: list[CallDecl] = [
        CallDecl(name=cd["name"], raw=cd["raw"]) for cd in call_decls_doc
    ]

    # Blocks
    blocks: list[Block] = [_parse_block(b) for b in blocks_doc]

    fn = Function(
        name=name,
        args=list(args),
        constants=constants,
        call_decls=call_decls,
        blocks=blocks,
    )

    # Cache mapping cslot args: the eval function gets extra args for each
    # cslot bridge after the interner params.  Collect cslot names in order.
    cslot_args: list[str] = []
    if cache_mapping_doc:
        # In the JSON, cache_mapping lives in init_fn but the cslot arg
        # names are used to extend eval_fn.args.  We collect them here
        # so the caller (for eval_fn) can append them.  For init_fn itself
        # this list is unused (init doesn't have cslot input args).
        pass  # handled in _parse_module via eval_cslot_args being empty

    return fn, interner, cslot_args


def _parse_constants(doc: dict) -> list[Constant]:
    consts: list[Constant] = []
    for name, val in doc.items():
        if isinstance(val, bool):
            consts.append(Constant(name=name, kind="bconst", bconst=val))
        elif isinstance(val, int):
            consts.append(Constant(name=name, kind="iconst", iconst=val))
        elif isinstance(val, float):
            consts.append(Constant(name=name, kind="fconst", fconst=val))
        elif isinstance(val, str):
            if val == _INF_SENTINEL:
                consts.append(Constant(name=name, kind="fconst", fconst=float("inf")))
            elif val == _NEGINF_SENTINEL:
                consts.append(Constant(name=name, kind="fconst", fconst=float("-inf")))
            elif val == _NAN_SENTINEL:
                consts.append(Constant(name=name, kind="fconst", fconst=float("nan")))
            else:
                consts.append(Constant(name=name, kind="sconst", sconst=val))
    return consts


def _parse_block(doc: dict) -> Block:
    label = doc["label"]
    insts = [_parse_inst(i) for i in doc.get("insts", [])]
    return Block(label=label, insts=insts)


def _parse_inst(doc: dict) -> Inst:
    opcode = doc["opcode"]
    result = doc.get("result")  # may be None

    if opcode == "br":
        return Inst(
            result=None,
            opcode="br",
            operands=[doc["condition"]],
            targets=[doc["true_block"], doc["false_block"]],
        )
    if opcode == "jmp":
        return Inst(
            result=None,
            opcode="jmp",
            operands=[],
            targets=[
                doc.get("targets", [doc.get("destination", "")])[0]
                if "targets" in doc
                else doc.get("destination", "")
            ],
        )
    if opcode == "exit":
        return Inst(result=None, opcode="exit", operands=[], targets=[])
    if opcode == "phi":
        edges = [
            PhiEdge(value=e["value"], block=e["block"])
            for e in doc.get("phi_edges", [])
        ]
        return Inst(result=result, opcode="phi", operands=[], phi_edges=edges)
    if opcode == "call":
        return Inst(
            result=result,
            opcode="call",
            operands=list(doc.get("operands", [])),
            call_target=doc.get("call_target"),
        )
    # Regular SSA instruction
    return Inst(
        result=result,
        opcode=opcode,
        operands=list(doc.get("operands", [])),
    )


# ---------------------------------------------------------------------------
# InputKind parser.
# ---------------------------------------------------------------------------


def _parse_input_kind(doc: dict) -> InputKind:
    tag = doc.get("tag", "")
    if tag == "Voltage":
        return Voltage(
            hi=doc.get("hi"),
            lo=doc.get("lo"),
            hi_node=doc.get("hi_node", doc.get("hi", "")),
            lo_node=doc.get("lo_node", doc.get("lo")),
        )
    if tag == "CurrentBranch":
        return CurrentKind(
            kind="Branch",
            branch=doc.get("branch"),
            branch_id=doc.get("branch_id"),
            hi=doc.get("hi"),
            lo=doc.get("lo"),
        )
    if tag == "CurrentUnnamed":
        return CurrentKind(
            kind="Unnamed",
            branch_id=doc.get("branch_id"),
            hi=doc.get("hi"),
            lo=doc.get("lo"),
        )
    if tag == "CurrentPort":
        return CurrentKind(kind="Port", branch=doc.get("port"), hi=doc.get("port"))
    if tag == "Param":
        return ParamRef(name=doc["name"], param_id=doc.get("param_id", 0))
    if tag == "ParamGiven":
        return ParamGivenRef(name=doc["name"], param_id=doc.get("param_id", 0))
    if tag == "Temperature":
        return TemperatureInput()
    if tag == "Abstime":
        return AbstimeInput()
    if tag == "ParamSysFun":
        return ParamSysFunInput(name=doc.get("name", ""))
    if tag == "HiddenState":
        return HiddenStateInput(var=doc.get("var", ""), var_id=doc.get("var_id", 0))
    if tag == "PortConnected":
        return PortConnectedInput(port=doc.get("port", ""))
    if tag == "PrevState":
        return PrevStateInput(index=doc.get("index", 0))
    if tag == "NewState":
        return NewStateInput(index=doc.get("index", 0))
    if tag == "EnableLim":
        return EnableLimInput()
    # EnableIntegration, ImplicitUnknown, unknown tags → base InputKind
    return InputKind()


# ---------------------------------------------------------------------------
# Cache mapping parser.
# ---------------------------------------------------------------------------


def _parse_cache_mapping(mapping_doc: list[dict]) -> CachedValues:
    cached = CachedValues()
    for entry in mapping_doc:
        init_ssa = entry["init_value"]
        cslot_name = entry["cslot"]
        cached.mapping[init_ssa] = cslot_name
        cached.slots[cslot_name] = "<json-bound>"
    return cached


# ---------------------------------------------------------------------------
# DAE parser.
# ---------------------------------------------------------------------------


def _parse_dae(doc: dict) -> DaeInfo:
    info = DaeInfo()
    info.unknowns = dict(doc.get("unknowns", {}))
    for sim_key, residual_doc in doc.get("residual", {}).items():
        info.residual[sim_key] = DaeResidual(
            resist=residual_doc.get("resist", ""),
            react=residual_doc.get("react", ""),
            resist_small_signal=residual_doc.get("resist_small_signal", ""),
            react_small_signal=residual_doc.get("react_small_signal", ""),
            resist_lim_rhs=residual_doc.get("resist_lim_rhs", ""),
            react_lim_rhs=residual_doc.get("react_lim_rhs", ""),
            nature_kind=residual_doc.get("nature_kind", "Flow"),
        )
    for jac_key, jac_doc in doc.get("jacobian", {}).items():
        row = jac_doc.get("row", "")
        col = jac_doc.get("col", "")
        info.jacobian[jac_key] = DaeMatrixEntry(
            row=row,
            col=col,
            resist=jac_doc.get("resist", ""),
            react=jac_doc.get("react", ""),
        )
    info.num_resistive = doc.get("num_resistive", 0)
    info.num_reactive = doc.get("num_reactive", 0)
    return info


__all__ = ["compile_va"]
