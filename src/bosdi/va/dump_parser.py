"""Parser for ``openvaf-r --dump-mir`` and ``--dump-unopt-mir`` textual output.

The dump format is stable (it round-trips through OpenVAF's own
``mir_reader`` crate) but is not a formal grammar — this parser is
regex-driven and pragmatic. It handles exactly the shapes observed in
``tests/data/va/resistor.mir.txt``; unrecognised input raises ``DumpParseError``
rather than silently dropping data.

Both optimized and unoptimized text dumps share the function-body
syntax. They differ in section headers:

- ``--dump-mir`` emits three split functions per module:
  ``Optimized model setup MIR of <NAME>``,
  ``Optimized instance setup MIR of <NAME>``, and
  ``Optimized evaluation MIR of <NAME>``.
- ``--dump-unopt-mir`` emits two combined functions:
  ``Unoptimized MIR (no DAE) of <NAME>`` (pre-DAE-construction analog
  body) and ``Partially optimized MIR (with DAE) of <NAME>`` (post-DAE
  but pre-MIR-optimization).

For unopt mode we route the partially-optimized-with-DAE function into
``eval_fn`` since it carries the DAE residual/jacobian outputs via
``optbarrier``. ``init_fn`` and ``setup_fn`` are left empty — the
combined function does init+eval together; the lowering's emitter then
re-derives the cache-vs-eval split from signal-dependency analysis.
This sacrifices some pre-instance hoisting but preserves the full
chain of 2-edge phis (the structural detail that the MIR optimizer
collapses into N-way phis on deeply-nested conditional inits, which
in turn breaks the lowering's diamond detection).

The output is a :class:`circulax.va.mir.DumpFile`, a faithful in-memory
representation. No lowering or opinion applied here.
"""

from __future__ import annotations

import re

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
    Function,
    HiddenStateInput,
    HirInterner,
    InputKind,
    Inst,
    ParamGivenRef,
    ParamRef,
    ParamSysFunInput,
    PhiEdge,
    PortConnectedInput,
    TemperatureInput,
    Voltage,
)


class DumpParseError(ValueError):
    """Raised when the dump text doesn't match an expected shape."""


# ---------------------------------------------------------------------------
# Line iterator helper.
# ---------------------------------------------------------------------------


class _Lines:
    """Cursor over a list of lines with peek / advance / expect helpers."""

    def __init__(self, text: str) -> None:
        # strip ANSI escape codes (the final "Finished ..." line is green-coded)
        ansi = re.compile(r"\x1b\[[0-9;]*m")
        self._lines = [ansi.sub("", ln) for ln in text.splitlines()]
        self._i = 0

    def eof(self) -> bool:
        return self._i >= len(self._lines)

    def peek(self) -> str | None:
        return None if self.eof() else self._lines[self._i]

    def advance(self) -> str:
        if self.eof():
            msg = "unexpected end of input"
            raise DumpParseError(msg)
        line = self._lines[self._i]
        self._i += 1
        return line

    def skip_blanks(self) -> None:
        while not self.eof() and self._lines[self._i].strip() == "":
            self._i += 1

    def lineno(self) -> int:
        return self._i + 1


# ---------------------------------------------------------------------------
# Regexes.
# ---------------------------------------------------------------------------

_FUNC_HEADER_RE = re.compile(r"^function %([A-Za-z_]\w*)?\((.*?)\)\s*\{\s*$")
_CALL_DECL_RE = re.compile(r"^\s*(\w+)\s*=\s*((?:const\s+)?fn\s+.+)$")
_FCONST_RE = re.compile(r"^\s*(\w+)\s*=\s*fconst\s+(.+?)\s*$")
_ICONST_RE = re.compile(r"^\s*(\w+)\s*=\s*iconst\s+(-?\d+)\s*$")
_BCONST_COMMENT_RE = re.compile(r"^\s*//\s*(\w+)\s*=\s*bconst\s+(true|false)\s*$")
_SCONST_RE = re.compile(r'^\s*(\w+)\s*=\s*sconst\s+"(.*)"\s*$')
_BLOCK_HEADER_RE = re.compile(r"^\s+(block\w+):\s*$")
# Instruction line may optionally start at col 0 with an @<hex> source loc.
_SRCLOC_RE = re.compile(r"^(@[0-9a-fA-F]+)\s+(\S.*)$")

# Instruction shapes, post-strip:
_RESULT_INST_RE = re.compile(r"^(\w+)\s*=\s*(\w+)(?:\s+(.*))?$")
_BR_RE = re.compile(r"^br\s+(\w+),\s*(\w+)(?:\[loop\])?(?:,\s*(\w+)(?:\[loop\])?)?\s*$")
_JMP_RE = re.compile(r"^jmp\s+(\w+)(?:\[loop\])?\s*$")
_CALL_RE = re.compile(r"^call\s+(\w+)\s*\(\s*(.*?)\s*\)\s*$")
_CALL_REST_RE = re.compile(
    r"^(\w+)\s*\(\s*(.*?)\s*\)\s*$"
)  # for "inst2(v338)" on RHS of `vN = call ...`
_EXIT_RE = re.compile(r"^exit\s*$")
_PHI_EDGE_RE = re.compile(r"\[(\w+),\s*(\w+)\]")


# ---------------------------------------------------------------------------
# Top-level dispatch.
# ---------------------------------------------------------------------------


def parse_dump(text: str) -> DumpFile:  # noqa: C901, PLR0912, PLR0915
    """Parse a full ``openvaf-r --dump-mir`` output into a :class:`DumpFile`.

    The top-level dispatch is intrinsically a long flat switch over section
    headers; complexity warnings are suppressed rather than fragmented into
    helpers that would obscure the section order.
    """
    lines = _Lines(text)

    setup_fns: dict[str, Function] = {}
    init_fns: dict[str, Function] = {}
    eval_fns: dict[str, Function] = {}
    literals: dict[int, str] = {}
    compilation_unit = ""
    cached = CachedValues()
    module_metas: list[dict] = []
    setup_interners: dict[str, HirInterner] = {}
    init_interners: dict[str, HirInterner] = {}
    eval_interners: dict[str, HirInterner] = {}

    while not lines.eof():
        line = lines.peek()
        if line is None:
            break
        stripped = line.strip()

        if line.startswith("Optimized model setup MIR of "):
            mod = line[len("Optimized model setup MIR of ") :].strip()
            lines.advance()
            fn = _parse_function(lines)
            setup_fns[mod] = fn
        elif line.startswith("Optimized instance setup MIR of "):
            mod = line[len("Optimized instance setup MIR of ") :].strip()
            lines.advance()
            fn = _parse_function(lines)
            init_fns[mod] = fn
        elif line.startswith("Optimized evaluation MIR of "):
            mod = line[len("Optimized evaluation MIR of ") :].strip()
            lines.advance()
            fn = _parse_function(lines)
            eval_fns[mod] = fn
        elif line.startswith("Partially optimized MIR (with DAE) of "):
            # ``--dump-unopt-mir`` mode: this function combines init+eval
            # logic with rich (uncollapsed) phi structure. Route it into
            # eval_fn; init_fn / setup_fn stay empty so the lowering's
            # signal-dependency analysis can re-derive the split.
            mod = line[len("Partially optimized MIR (with DAE) of ") :].strip()
            lines.advance()
            fn = _parse_function(lines)
            eval_fns[mod] = fn
        elif line.startswith("Unoptimized MIR (no DAE) of "):
            # ``--dump-unopt-mir`` mode also emits a pre-DAE form of the
            # same function; we don't need it (the with-DAE version has
            # everything plus the residual annotations), but we must
            # consume it so the parser doesn't choke on its body.
            lines.advance()
            _ = _parse_function(lines)
        elif line.startswith("Compilation unit:"):
            compilation_unit = line.split(":", 1)[1].strip()
            lines.advance()
        elif stripped == "Literals:":
            lines.advance()
            literals = _parse_literals(lines)
        elif line.startswith("  Module: "):
            meta = _parse_module_meta(lines)
            module_metas.append(meta)
        elif line.startswith("Cached values during instance setup"):
            lines.advance()
            cached = _parse_cached_values(lines)
        elif line.startswith("Model setup HIR interner of "):
            mod = line[len("Model setup HIR interner of ") :].strip()
            lines.advance()
            setup_interners[mod] = _parse_hir_interner(lines)
        elif line.startswith("Instance setup HIR interner of "):
            mod = line[len("Instance setup HIR interner of ") :].strip()
            lines.advance()
            init_interners[mod] = _parse_hir_interner(lines)
        elif line.startswith("Evaluation HIR interner of "):
            mod = line[len("Evaluation HIR interner of ") :].strip()
            lines.advance()
            eval_interners[mod] = _parse_hir_interner(lines)
        elif stripped.startswith("Finished "):
            # Success-banner line from the driver; ignore.
            lines.advance()
        elif stripped == "":
            lines.advance()
        else:
            msg = f"line {lines.lineno()}: unrecognised section header: {line!r}"
            raise DumpParseError(msg)

    # Assemble modules. Use setup_fns keys as the canonical module list.
    modules: list[CompiledModule] = []
    for meta in module_metas:
        name = meta["name"]
        eval_fn = eval_fns.get(name) or _empty_fn()
        eval_int = eval_interners.get(name, HirInterner())

        # Resolve port names from the eval interner if possible: any Voltage
        # input that references the module's port nodes carries their .va
        # names.
        port_name_map = _port_name_map(eval_int, meta["port_nodes"])
        resolved_ports = [port_name_map.get(nd, nd) for nd in meta["port_nodes"]]

        modules.append(
            CompiledModule(
                name=name,
                ports=resolved_ports,
                port_nodes=meta["port_nodes"],
                internal_nodes=meta["internal_nodes"],
                dae=meta["dae"],
                eval_fn=eval_fn,
                init_fn=init_fns.get(name) or _empty_fn(),
                setup_fn=setup_fns.get(name) or _empty_fn(),
                eval_interner=eval_int,
                init_interner=init_interners.get(name, HirInterner()),
                setup_interner=setup_interners.get(name, HirInterner()),
                cached=cached,
            )
        )

    return DumpFile(
        compilation_unit=compilation_unit, literals=literals, modules=modules
    )


def _empty_fn() -> Function:
    return Function(name="", args=[])


def _port_name_map(eval_int: HirInterner, port_nodes: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for kind in eval_int.parameters.values():
        if isinstance(kind, Voltage):
            if kind.hi_node in port_nodes and kind.hi is not None:
                mapping[kind.hi_node] = kind.hi
            if (
                kind.lo_node is not None
                and kind.lo_node in port_nodes
                and kind.lo is not None
            ):
                mapping[kind.lo_node] = kind.lo
    return mapping


# ---------------------------------------------------------------------------
# Function parser.
# ---------------------------------------------------------------------------


def _parse_function(lines: _Lines) -> Function:  # noqa: C901, PLR0912, PLR0915
    header = lines.advance()
    m = _FUNC_HEADER_RE.match(header)
    if not m:
        msg = f"line {lines.lineno()}: expected function header, got {header!r}"
        raise DumpParseError(msg)
    name = m.group(1) or ""
    args_raw = m.group(2).strip()
    args = [a.strip() for a in args_raw.split(",") if a.strip()] if args_raw else []
    fn = Function(name=name, args=args)

    # Preamble: call decls, constants, comments, until first block header or "}".
    while (line := lines.peek()) is not None:
        stripped_l = line.lstrip()
        if stripped_l.startswith("}"):
            lines.advance()
            return fn
        if _BLOCK_HEADER_RE.match(line):
            break
        if line.strip() == "":
            lines.advance()
            continue

        m_b = _BCONST_COMMENT_RE.match(line)
        if m_b:
            fn.constants.append(
                Constant(
                    name=m_b.group(1), kind="bconst", bconst=(m_b.group(2) == "true")
                )
            )
            lines.advance()
            continue

        m_f = _FCONST_RE.match(line)
        if m_f:
            fn.constants.append(
                Constant(
                    name=m_f.group(1), kind="fconst", fconst=_parse_float(m_f.group(2))
                )
            )
            lines.advance()
            continue

        m_i = _ICONST_RE.match(line)
        if m_i:
            fn.constants.append(
                Constant(name=m_i.group(1), kind="iconst", iconst=int(m_i.group(2)))
            )
            lines.advance()
            continue

        m_s = _SCONST_RE.match(line)
        if m_s:
            fn.constants.append(
                Constant(name=m_s.group(1), kind="sconst", sconst=m_s.group(2))
            )
            lines.advance()
            continue

        m_c = _CALL_DECL_RE.match(line)
        if m_c and m_c.group(1).startswith("inst"):
            fn.call_decls.append(CallDecl(name=m_c.group(1), raw=m_c.group(2).strip()))
            lines.advance()
            continue

        msg = f"line {lines.lineno()}: unrecognised preamble line: {line!r}"

        raise DumpParseError(msg)

    # Blocks.
    while (line := lines.peek()) is not None:
        if line.strip() == "":
            lines.advance()
            continue
        if line.lstrip().startswith("}"):
            lines.advance()
            return fn
        m_bh = _BLOCK_HEADER_RE.match(line)
        if not m_bh:
            msg = f"line {lines.lineno()}: expected block header, got {line!r}"
            raise DumpParseError(msg)
        block = Block(label=m_bh.group(1))
        fn.blocks.append(block)
        lines.advance()
        _parse_block_body(lines, block)

    msg = "function body missing closing '}'"

    raise DumpParseError(msg)


def _parse_block_body(lines: _Lines, block: Block) -> None:
    while (line := lines.peek()) is not None:
        if line.strip() == "":
            lines.advance()
            continue
        if line.lstrip().startswith("}"):
            # Closing brace belongs to the caller.
            return
        if _BLOCK_HEADER_RE.match(line):
            # Next block starts.
            return
        inst = _parse_instruction(lines)
        block.insts.append(inst)


def _parse_instruction(lines: _Lines) -> Inst:  # noqa: C901, PLR0911
    raw = lines.advance()
    source_loc: str | None = None
    m_src = _SRCLOC_RE.match(raw)
    if m_src:
        source_loc = m_src.group(1)
        body = m_src.group(2).strip()
    else:
        body = raw.strip()

    # Result-producing instructions first.
    m = _RESULT_INST_RE.match(body)
    if m and m.group(2) not in {"br", "jmp", "exit", "phi", "call"}:
        result = m.group(1)
        opcode = m.group(2)
        operands_raw = (m.group(3) or "").strip()
        operands = _parse_operand_list(operands_raw)
        return Inst(
            result=result, opcode=opcode, operands=operands, source_loc=source_loc
        )

    if m and m.group(2) == "phi":
        # Result-producing phi: "vN = phi [val, blk], ..."
        result = m.group(1)
        edges = [
            PhiEdge(value=v, block=b) for v, b in _PHI_EDGE_RE.findall(m.group(3) or "")
        ]
        if not edges:
            msg = f"line {lines.lineno()}: phi with no edges: {body!r}"
            raise DumpParseError(msg)
        return Inst(result=result, opcode="phi", phi_edges=edges, source_loc=source_loc)

    if m and m.group(2) == "call":
        # Result-producing call: "vN = call inst<k>(args)"
        result = m.group(1)
        m_rest = _CALL_REST_RE.match((m.group(3) or "").strip())
        if not m_rest:
            msg = f"line {lines.lineno()}: malformed call-with-result: {body!r}"
            raise DumpParseError(msg)
        return Inst(
            result=result,
            opcode="call",
            operands=_parse_operand_list(m_rest.group(2)),
            call_target=m_rest.group(1),
            source_loc=source_loc,
        )

    # Control-flow / no-result instructions.
    m_br = _BR_RE.match(body)
    if m_br:
        cond = m_br.group(1)
        t1 = m_br.group(2)
        t2 = m_br.group(3)
        targets = [t1] + ([t2] if t2 else [])
        return Inst(
            result=None,
            opcode="br",
            operands=[cond],
            targets=targets,
            source_loc=source_loc,
        )

    m_jmp = _JMP_RE.match(body)
    if m_jmp:
        return Inst(
            result=None, opcode="jmp", targets=[m_jmp.group(1)], source_loc=source_loc
        )

    m_call = _CALL_RE.match(body)
    if m_call:
        return Inst(
            result=None,
            opcode="call",
            operands=_parse_operand_list(m_call.group(2)),
            call_target=m_call.group(1),
            source_loc=source_loc,
        )

    if _EXIT_RE.match(body):
        return Inst(result=None, opcode="exit", source_loc=source_loc)

    msg = f"line {lines.lineno()}: unrecognised instruction: {body!r}"
    raise DumpParseError(msg)


def _parse_operand_list(s: str) -> list[str]:
    if not s:
        return []
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _parse_float(tok: str) -> float:
    tok = tok.strip()
    if tok in ("+Inf", "Inf"):
        return float("inf")
    if tok == "-Inf":
        return float("-inf")
    if tok in ("NaN", "+NaN"):
        return float("nan")
    if tok.startswith(("0x", "-0x", "+0x")):
        return float.fromhex(tok)
    return float(tok)


# ---------------------------------------------------------------------------
# Literals section.
# ---------------------------------------------------------------------------

_LITERAL_RE = re.compile(r"^\s*Spur\((\d+)\)\s*->\s*'(.*)'\s*$")
_LITERAL_START_RE = re.compile(r"^\s*Spur\((\d+)\)\s*->\s*'(.*)$")


def _parse_literals(lines: _Lines) -> dict[int, str]:
    r"""Parse the ``Literals:`` section, tolerating multi-line string values.

    OpenVAF prints strings via Rust's ``Debug`` formatter, which embeds
    literal newlines for strings that contain them (e.g. BSIM4 warning
    messages that end with ``.\n``). For those cases we accumulate
    subsequent lines until we see a closing single quote.
    """
    out: dict[int, str] = {}
    while not lines.eof():
        line = lines.peek()
        if line is None:
            break
        if line.strip() == "":
            # Blank line ends the Literals section when it's at the top level.
            # An interior blank inside a multi-line string is handled by the
            # continuation path below; this branch only fires between entries.
            lines.advance()
            return out
        m = _LITERAL_RE.match(line)
        if m:
            out[int(m.group(1))] = m.group(2)
            lines.advance()
            continue
        m = _LITERAL_START_RE.match(line)
        if m:
            # Multi-line string — keep reading until we see a line ending
            # with a single quote (allowing for trailing whitespace).
            pieces = [m.group(2)]
            lines.advance()
            while not lines.eof():
                nxt = lines.advance()
                stripped_end = nxt.rstrip()
                if stripped_end.endswith("'"):
                    pieces.append(stripped_end[:-1])
                    break
                pieces.append(nxt)
            out[int(m.group(1))] = "\n".join(pieces)
            continue
        # End of the Literals block (something else is starting).
        return out
    return out


# ---------------------------------------------------------------------------
# Module metadata section.
# ---------------------------------------------------------------------------

_MODULE_NAME_RE = re.compile(r'^\s*Module:\s*"([^"]+)"\s*$')
_PORTS_RE = re.compile(r"^\s*Ports:\s*\[(.*?)\]\s*$")
_INTERNAL_NODES_RE = re.compile(r"^\s*Internal nodes:\s*\[(.*?)\]\s*$")


def _parse_module_meta(lines: _Lines) -> dict:
    name_line = lines.advance()
    m = _MODULE_NAME_RE.match(name_line)
    if not m:
        msg = f"line {lines.lineno()}: expected Module: line, got {name_line!r}"
        raise DumpParseError(msg)
    name = m.group(1)

    ports_line = lines.advance()
    m = _PORTS_RE.match(ports_line)
    if not m:
        msg = f"line {lines.lineno()}: expected Ports: line, got {ports_line!r}"
        raise DumpParseError(msg)
    port_nodes = _split_list(m.group(1))

    internal_line = lines.advance()
    m = _INTERNAL_NODES_RE.match(internal_line)
    if not m:
        msg = f"line {lines.lineno()}: expected Internal nodes: line, got {internal_line!r}"
        raise DumpParseError(msg)
    internal_nodes = _split_list(m.group(1))

    # Next expected: DaeSystem { ... }
    dae_hdr = lines.advance()
    if not dae_hdr.strip().startswith("DaeSystem {"):
        msg = f"line {lines.lineno()}: expected DaeSystem, got {dae_hdr!r}"
        raise DumpParseError(msg)
    dae = _parse_dae_system(lines)

    return {
        "name": name,
        "port_nodes": port_nodes,
        "internal_nodes": internal_nodes,
        "dae": dae,
    }


def _split_list(s: str) -> list[str]:
    s = s.strip()
    if not s:
        return []
    return [tok.strip() for tok in s.split(",") if tok.strip()]


# ---------------------------------------------------------------------------
# DaeSystem parser. Rust Debug format is deterministic; we use regex per line.
# ---------------------------------------------------------------------------

_KV_RE = re.compile(r"^\s*(\w+)\s*:\s*(.*?),?\s*$")


def _parse_dae_system(lines: _Lines) -> DaeInfo:  # noqa: C901, PLR0912
    """Parse a pretty-printed ``DaeSystem { ... }`` block.

    We already consumed the opening ``DaeSystem {`` line. Consume lines until
    we see a closing ``}`` at top level, dispatching on recognised field
    names. Nested structures are parsed by dedicated helpers.
    """
    dae = DaeInfo()
    while not lines.eof():
        line = lines.advance()
        stripped = line.strip()
        if stripped.startswith("}"):
            return dae
        if stripped == "":
            continue

        if stripped.startswith("unknowns: {"):
            dae.unknowns = _parse_unknowns_block(lines)
        elif stripped.startswith("residual: {"):
            dae.residual = _parse_residual_block(lines)
        elif stripped.startswith("jacobian: {"):
            dae.jacobian = _parse_jacobian_block(lines)
        elif stripped.startswith("small_signal_parameters:"):
            # {} or a small inline struct; we only need to detect and skip on this device.
            # If the value isn't empty braces, swallow until we close it.
            if "{}" in stripped:
                continue
            _skip_balanced(lines, open_ch="{", close_ch="}")
        elif stripped.startswith("noise_sources:"):
            if "[]" in stripped:
                continue
            _skip_balanced(lines, open_ch="[", close_ch="]")
        elif stripped.startswith("model_inputs:"):
            # model_inputs: [ ( 0, 1, ), ]   (may span multiple lines)
            dae.model_inputs = _parse_model_inputs(lines, stripped)
        elif stripped.startswith("num_resistive:"):
            dae.num_resistive = int(stripped.split(":", 1)[1].rstrip(",").strip())
        elif stripped.startswith("num_reactive:"):
            dae.num_reactive = int(stripped.split(":", 1)[1].rstrip(",").strip())
        else:
            msg = f"line {lines.lineno()}: unrecognised DaeSystem field: {line!r}"
            raise DumpParseError(msg)
    msg = "DaeSystem missing closing '}'"
    raise DumpParseError(msg)


def _parse_unknowns_block(lines: _Lines) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _consume_until_close(lines):
        m = _KV_RE.match(line)
        if not m:
            msg = f"line {lines.lineno()}: bad unknowns entry: {line!r}"
            raise DumpParseError(msg)
        out[m.group(1)] = m.group(2)
    return out


def _parse_residual_block(lines: _Lines) -> dict[str, DaeResidual]:
    out: dict[str, DaeResidual] = {}
    while not lines.eof():
        line = lines.advance()
        stripped = line.strip()
        if stripped == "},":
            return out
        if stripped == "}":
            return out
        if stripped == "":
            continue
        # Header like "sim_node0: Residual {"
        m = re.match(r"^(\w+)\s*:\s*Residual\s*\{\s*$", stripped)
        if not m:
            msg = f"line {lines.lineno()}: bad residual entry: {line!r}"
            raise DumpParseError(msg)
        key = m.group(1)
        fields = _read_struct_fields(lines)
        out[key] = DaeResidual(
            resist=fields["resist"],
            react=fields["react"],
            resist_small_signal=fields["resist_small_signal"],
            react_small_signal=fields["react_small_signal"],
            resist_lim_rhs=fields["resist_lim_rhs"],
            react_lim_rhs=fields["react_lim_rhs"],
            nature_kind=fields["nature_kind"],
        )
    msg = "residual block missing closing '}'"
    raise DumpParseError(msg)


def _parse_jacobian_block(lines: _Lines) -> dict[str, DaeMatrixEntry]:
    out: dict[str, DaeMatrixEntry] = {}
    while not lines.eof():
        line = lines.advance()
        stripped = line.strip()
        if stripped in ("},", "}"):
            return out
        if stripped == "":
            continue
        m = re.match(r"^(\w+)\s*:\s*MatrixEntry\s*\{\s*$", stripped)
        if not m:
            msg = f"line {lines.lineno()}: bad jacobian entry: {line!r}"
            raise DumpParseError(msg)
        key = m.group(1)
        fields = _read_struct_fields(lines)
        out[key] = DaeMatrixEntry(
            row=fields["row"],
            col=fields["col"],
            resist=fields["resist"],
            react=fields["react"],
        )
    msg = "jacobian block missing closing '}'"
    raise DumpParseError(msg)


def _read_struct_fields(lines: _Lines) -> dict[str, str]:
    """Read ``key: value,`` lines until a closing ``},`` or ``}``."""
    out: dict[str, str] = {}
    while not lines.eof():
        line = lines.advance()
        stripped = line.strip()
        if stripped in ("},", "}"):
            return out
        if stripped == "":
            continue
        m = _KV_RE.match(line)
        if not m:
            msg = f"line {lines.lineno()}: bad struct field: {line!r}"
            raise DumpParseError(msg)
        out[m.group(1)] = m.group(2)
    msg = "struct missing closing '}'"
    raise DumpParseError(msg)


def _consume_until_close(lines: _Lines) -> list[str]:
    """Return raw lines until ``},`` / ``}`` is seen at depth 0; consume the closer."""
    out: list[str] = []
    while not lines.eof():
        line = lines.advance()
        s = line.strip()
        if s in ("},", "}"):
            return out
        if s == "":
            continue
        out.append(line)
    msg = "block missing closing '}'"
    raise DumpParseError(msg)


def _skip_balanced(lines: _Lines, *, open_ch: str, close_ch: str) -> None:
    depth = 1
    while depth > 0 and not lines.eof():
        line = lines.advance()
        depth += line.count(open_ch) - line.count(close_ch)


def _parse_model_inputs(lines: _Lines, first_stripped: str) -> list[tuple[int, int]]:
    """Parse a ``model_inputs: [ ( N, M, ), ... ],`` block spanning multiple lines."""
    if first_stripped.endswith(("[]", "[],")):
        return []
    collected_ints: list[int] = []
    depth = first_stripped.count("[") - first_stripped.count("]")
    while depth > 0 and not lines.eof():
        line = lines.advance()
        depth += line.count("[") - line.count("]")
        collected_ints.extend(int(tok) for tok in re.findall(r"-?\d+", line))
    # Pair them up — OSDI model_inputs are tuples.
    return [
        (collected_ints[i], collected_ints[i + 1])
        for i in range(0, len(collected_ints) - 1, 2)
    ]


# ---------------------------------------------------------------------------
# Cached values section.
# ---------------------------------------------------------------------------


_CACHE_VAL_RE = re.compile(r"^\s*(v\d+)\s*->\s*(cslot\w*)\s*$")
_CACHE_SLOT_RE = re.compile(r"^\s*(cslot\w*)\s*->\s*(.+?)\s*$")


def _parse_cached_values(lines: _Lines) -> CachedValues:
    cv = CachedValues()
    while not lines.eof():
        line = lines.peek()
        if line is None:
            break
        if line.strip() == "":
            lines.advance()
            return cv
        m = _CACHE_VAL_RE.match(line)
        if m:
            cv.mapping[m.group(1)] = m.group(2)
            lines.advance()
            continue
        m = _CACHE_SLOT_RE.match(line)
        if m:
            cv.slots[m.group(1)] = m.group(2)
            lines.advance()
            continue
        # End of block.
        return cv
    return cv


# ---------------------------------------------------------------------------
# HIR interner section.
# ---------------------------------------------------------------------------


_HEADING_PARAMS = re.compile(r"^\s{2}Parameters:\s*$")
_HEADING_OUTPUTS = re.compile(r"^\s{2}Outputs:\s*$")
_HEADING_TAGGED = re.compile(r"^\s{2}Tagged reads:\s*$")
_HEADING_IMPLICIT = re.compile(r"^\s{2}Implicit equations:\s*$")

# Individual input lines (4-space indent inside "Parameters:"):
_PARAM_LINE_RE = re.compile(r"^\s{4}(.+?)\s*->\s*(v\d+)\s*$")
_OUTPUT_LINE_RE = re.compile(r"^\s{4}(.+?)\s*->\s*(\S+)\s*$")


def _parse_hir_interner(lines: _Lines) -> HirInterner:  # noqa: C901, PLR0912, PLR0915
    intern = HirInterner()
    section = None  # one of: parameters, outputs, tagged, implicit
    while not lines.eof():
        line = lines.peek()
        if line is None:
            break
        # Stop if we hit a top-level section header.
        stripped = line.strip()
        if (
            line
            and not line.startswith(" ")
            and stripped
            and not stripped.startswith("//")
            and stripped.startswith(
                (
                    "Model setup HIR interner of ",
                    "Instance setup HIR interner of ",
                    "Evaluation HIR interner of ",
                    "Cached values during instance setup",
                    "Finished ",
                    "Optimized ",
                    "Compilation unit:",
                    "Literals:",
                )
            )
        ):
            return intern

        if _HEADING_PARAMS.match(line):
            section = "parameters"
            lines.advance()
            continue
        if _HEADING_OUTPUTS.match(line):
            section = "outputs"
            lines.advance()
            continue
        if _HEADING_TAGGED.match(line):
            section = "tagged"
            lines.advance()
            continue
        if _HEADING_IMPLICIT.match(line):
            section = "implicit"
            lines.advance()
            continue

        if stripped == "":
            lines.advance()
            continue

        if section == "parameters":
            m = _PARAM_LINE_RE.match(line)
            if not m:
                msg = f"line {lines.lineno()}: bad parameter line: {line!r}"
                raise DumpParseError(msg)
            kind_str = m.group(1).strip()
            val_name = m.group(2)
            intern.parameters[val_name] = _parse_input_kind(kind_str)
            lines.advance()
            continue
        if section == "outputs":
            m = _OUTPUT_LINE_RE.match(line)
            if not m:
                msg = f"line {lines.lineno()}: bad output line: {line!r}"
                raise DumpParseError(msg)
            key = m.group(1).strip()
            val = m.group(2).strip()
            intern.outputs[key] = None if val == "None" else val
            lines.advance()
            continue
        if section in ("tagged", "implicit"):
            # Tagged reads / Implicit equations subsections — empty for Stage 1 devices.
            lines.advance()
            continue

        msg = f"line {lines.lineno()}: HIR interner line outside a section: {line!r}"

        raise DumpParseError(msg)
    return intern


# ---------------------------------------------------------------------------
# Input-kind mini-parser.
# ---------------------------------------------------------------------------

_PARAM_ONLY_RE = re.compile(
    r"^Param\(Parameter\s*\{\s*id:\s*ParamId\((\d+)\)\s*\}\)\s*\.\.\s*\"([^\"]+)\"$"
)
_PARAMGIVEN_RE = re.compile(
    r"^ParamGiven\s*\{\s*param:\s*Parameter\s*\{\s*id:\s*ParamId\((\d+)\)\s*\}\s*\}\s*\.\.\s*\"([^\"]+)\"$"
)
_VOLTAGE_RE = re.compile(
    r"^Voltage\s*\{\s*hi:\s*(\w+)\s*,\s*lo:\s*(None|Some\(([^)]+)\))\s*\}"
    r"(?:\s*\.\.\s*V\(\s*(?:\"([^\"]+)\")?\s*(?:,\s*\"([^\"]+)\")?\s*\))?$"
)
_HIDDEN_RE = re.compile(r"^HiddenState\(VarId\((\d+)\)\)\s*\.\.\s*\"([^\"]+)\"$")
_SYSFUN_RE = re.compile(r"^ParamSysFun\((\w+)\)$")
_PORT_CONN_RE = re.compile(r"^PortConnected\s*\{\s*port:\s*(\w+)\s*\}$")
_CURRENT_BRANCH_RE = re.compile(
    r"^Current\(Branch\(BranchId\((\d+)\)\)\)\s*\.\.\s*\"([^\"]+)\"$"
)
_CURRENT_UNNAMED_RE = re.compile(
    r"^Current\(Unnamed\s*\{\s*hi:\s*(\w+)\s*,\s*lo:\s*(None|Some\(([^)]+)\))\s*\}\)"
    r"(?:\s*\.\.\s*I\(\s*(?:\"([^\"]+)\")?\s*(?:,\s*\"([^\"]+)\")?\s*\))?$"
)
_CURRENT_PORT_RE = re.compile(r"^Current\(Port\((\w+)\)\)\s*\.\.\s*\"([^\"]+)\"$")


def _parse_input_kind(s: str) -> InputKind:  # noqa: C901, PLR0911
    s = s.strip()
    if s == "Temperature":
        return TemperatureInput()
    if s == "Abstime":
        return AbstimeInput()

    m = _PARAM_ONLY_RE.match(s)
    if m:
        return ParamRef(name=m.group(2), param_id=int(m.group(1)))

    m = _PARAMGIVEN_RE.match(s)
    if m:
        return ParamGivenRef(name=m.group(2), param_id=int(m.group(1)))

    m = _VOLTAGE_RE.match(s)
    if m:
        hi_node = m.group(1)
        lo_node = m.group(3)  # content of Some(...), or None
        hi_name = m.group(4)
        lo_name = m.group(5)
        return Voltage(hi=hi_name, lo=lo_name, hi_node=hi_node, lo_node=lo_node)

    m = _HIDDEN_RE.match(s)
    if m:
        return HiddenStateInput(var=m.group(2), var_id=int(m.group(1)))

    m = _SYSFUN_RE.match(s)
    if m:
        return ParamSysFunInput(name=m.group(1))

    m = _PORT_CONN_RE.match(s)
    if m:
        return PortConnectedInput(port=m.group(1))

    m = _CURRENT_BRANCH_RE.match(s)
    if m:
        return CurrentKind(kind="Branch", branch=m.group(2), branch_id=int(m.group(1)))

    m = _CURRENT_UNNAMED_RE.match(s)
    if m:
        hi_name = m.group(4)
        lo_name = m.group(5)
        return CurrentKind(kind="Unnamed", hi=hi_name, lo=lo_name)

    m = _CURRENT_PORT_RE.match(s)
    if m:
        return CurrentKind(kind="Port", branch=m.group(2))

    # Unknown variant — surface it verbatim so the caller knows what's missing.
    msg = f"unknown InputKind rendering: {s!r}"
    raise DumpParseError(msg)
