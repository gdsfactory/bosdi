"""Verilog-A → circulax component translator.

Stage 1 scope: parse ``openvaf-r --dump-mir`` textual output into structured
MIR dataclasses. Lowering and code emission are follow-on stages.
"""

from .binding import compile_va
from .dump_parser import DumpParseError, parse_dump
from .emitter import emit_source, write_source
from .lowering import LoweredDevice, LoweringError, lower
from .uniform_params import uniform_static_params
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
    Value,
    Voltage,
)

__all__ = [
    "AbstimeInput",
    "Block",
    "CachedValues",
    "CallDecl",
    "CompiledModule",
    "Constant",
    "CurrentKind",
    "DaeInfo",
    "DaeMatrixEntry",
    "DaeResidual",
    "DumpFile",
    "DumpParseError",
    "Function",
    "HiddenStateInput",
    "HirInterner",
    "InputKind",
    "Inst",
    "LoweredDevice",
    "LoweringError",
    "ParamGivenRef",
    "ParamRef",
    "ParamSysFunInput",
    "PhiEdge",
    "PortConnectedInput",
    "TemperatureInput",
    "Value",
    "Voltage",
    "compile_va",
    "emit_source",
    "lower",
    "parse_dump",
    "uniform_static_params",
    "write_source",
]
