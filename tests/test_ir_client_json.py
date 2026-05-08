"""Tests for the JSON-based IR client paths.

Covers compile_va, compile_va_unopt_json, and compile_va_unopt_json_with_split.
The latter two require openvaf-r >= the feat/dump-json branch that added
--dump-unopt-json and --dump-unopt-json-with-split; tests are skipped when
those flags are absent.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

# ── path setup ──────────────────────────────────────────────────────────────
# When run via `pixi run pytest` the package is installed; when invoked
# directly from the repo root it may not be, so add src/ as a fallback.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from bosdi.va import DumpFile
from bosdi.va.ir_client import (
    compile_va,
    compile_va_unopt_json,
    compile_va_unopt_json_with_split,
)

# ── test fixtures ────────────────────────────────────────────────────────────

RESISTOR_VA = (
    pathlib.Path(__file__).parent.parent
    / "src"
    / "devices"
    / "verilog_a"
    / "resistor.va"
)
CAPACITOR_VA = pathlib.Path(__file__).parent / "capacitor_va.va"

OPENVAF_MISSING = pytest.mark.skipif(
    subprocess.run(["which", "openvaf-r"], capture_output=True).returncode != 0,
    reason="openvaf-r not in PATH",
)


def _has_flag(flag: str) -> bool:
    result = subprocess.run(["openvaf-r", "--help"], capture_output=True, text=True)
    return flag in result.stdout or flag in result.stderr


UNOPT_JSON_MISSING = pytest.mark.skipif(
    not _has_flag("--dump-unopt-json"),
    reason="openvaf-r does not support --dump-unopt-json (rebuild needed)",
)
UNOPT_JSON_SPLIT_MISSING = pytest.mark.skipif(
    not _has_flag("--dump-unopt-json-with-split"),
    reason="openvaf-r does not support --dump-unopt-json-with-split (rebuild needed)",
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _assert_valid_dumpfile(df: DumpFile, va_path: pathlib.Path) -> None:
    assert isinstance(df, DumpFile)
    assert len(df.modules) >= 1, "expected at least one module"
    for mod in df.modules:
        assert mod.name, "module name should not be empty"
        assert len(mod.ports) >= 2, f"{mod.name}: expected ≥2 ports"
        assert mod.eval_fn is not None
        assert len(mod.eval_fn.blocks) >= 1, f"{mod.name}: eval_fn has no blocks"
        assert mod.dae is not None


# ── compile_va (--dump-json, optimised) ─────────────────────────────────────


@OPENVAF_MISSING
def test_compile_va_resistor_returns_dumpfile():
    df = compile_va(RESISTOR_VA)
    _assert_valid_dumpfile(df, RESISTOR_VA)


@OPENVAF_MISSING
def test_compile_va_resistor_module_name():
    df = compile_va(RESISTOR_VA)
    assert df.modules[0].name == "resistor_va"


@OPENVAF_MISSING
def test_compile_va_resistor_ports():
    df = compile_va(RESISTOR_VA)
    mod = df.modules[0]
    assert set(mod.ports) == {"A", "B"}


@OPENVAF_MISSING
def test_compile_va_capacitor_returns_dumpfile():
    df = compile_va(CAPACITOR_VA)
    _assert_valid_dumpfile(df, CAPACITOR_VA)


# ── compile_va_unopt_json (--dump-unopt-json) ────────────────────────────────


@OPENVAF_MISSING
@UNOPT_JSON_MISSING
def test_compile_va_unopt_json_resistor_returns_dumpfile():
    df = compile_va_unopt_json(RESISTOR_VA)
    _assert_valid_dumpfile(df, RESISTOR_VA)


@OPENVAF_MISSING
@UNOPT_JSON_MISSING
def test_compile_va_unopt_json_resistor_module_name():
    df = compile_va_unopt_json(RESISTOR_VA)
    assert df.modules[0].name == "resistor_va"


@OPENVAF_MISSING
@UNOPT_JSON_MISSING
def test_compile_va_unopt_json_resistor_ports():
    df = compile_va_unopt_json(RESISTOR_VA)
    mod = df.modules[0]
    assert set(mod.ports) == {"A", "B"}


@OPENVAF_MISSING
@UNOPT_JSON_MISSING
def test_compile_va_unopt_json_has_more_cslots_than_opt():
    """Unoptimised path should produce ≥ as many cache slots as optimised."""
    opt = compile_va(RESISTOR_VA)
    unopt = compile_va_unopt_json(RESISTOR_VA)
    opt_slots = len(opt.modules[0].cached.slots)
    unopt_slots = len(unopt.modules[0].cached.slots)
    assert unopt_slots >= opt_slots, (
        f"expected unopt cslots ({unopt_slots}) ≥ opt cslots ({opt_slots})"
    )


# ── compile_va_unopt_json_with_split (--dump-unopt-json-with-split) ──────────


@OPENVAF_MISSING
@UNOPT_JSON_SPLIT_MISSING
def test_compile_va_unopt_json_with_split_resistor_returns_dumpfile():
    df = compile_va_unopt_json_with_split(RESISTOR_VA)
    _assert_valid_dumpfile(df, RESISTOR_VA)


@OPENVAF_MISSING
@UNOPT_JSON_SPLIT_MISSING
def test_compile_va_unopt_json_with_split_resistor_module_name():
    df = compile_va_unopt_json_with_split(RESISTOR_VA)
    assert df.modules[0].name == "resistor_va"


@OPENVAF_MISSING
@UNOPT_JSON_SPLIT_MISSING
def test_compile_va_unopt_json_with_split_has_init_and_eval():
    """with-split path should have non-empty init_fn and eval_fn."""
    df = compile_va_unopt_json_with_split(RESISTOR_VA)
    mod = df.modules[0]
    assert mod.eval_fn is not None and len(mod.eval_fn.blocks) >= 1
    assert mod.init_fn is not None and len(mod.init_fn.blocks) >= 1


@OPENVAF_MISSING
@UNOPT_JSON_SPLIT_MISSING
def test_compile_va_unopt_json_with_split_has_cache_slots():
    """with-split path should populate the cslot bridge."""
    df = compile_va_unopt_json_with_split(RESISTOR_VA)
    mod = df.modules[0]
    assert len(mod.cached.slots) >= 1, "expected ≥1 cache slot in with-split path"


# ── cross-path consistency ───────────────────────────────────────────────────


@OPENVAF_MISSING
@UNOPT_JSON_MISSING
@UNOPT_JSON_SPLIT_MISSING
def test_three_json_paths_agree_on_ports():
    """All three JSON paths should report the same ports for the resistor."""
    opt = compile_va(RESISTOR_VA)
    unopt = compile_va_unopt_json(RESISTOR_VA)
    split = compile_va_unopt_json_with_split(RESISTOR_VA)
    assert (
        set(opt.modules[0].ports)
        == set(unopt.modules[0].ports)
        == set(split.modules[0].ports)
    )


@OPENVAF_MISSING
@UNOPT_JSON_MISSING
@UNOPT_JSON_SPLIT_MISSING
def test_three_json_paths_agree_on_dae_unknowns():
    """All three JSON paths should report the same DAE unknowns."""
    opt = compile_va(RESISTOR_VA)
    unopt = compile_va_unopt_json(RESISTOR_VA)
    split = compile_va_unopt_json_with_split(RESISTOR_VA)
    assert (
        set(opt.modules[0].dae.unknowns)
        == set(unopt.modules[0].dae.unknowns)
        == set(split.modules[0].dae.unknowns)
    )
