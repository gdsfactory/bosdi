"""Tests for safe_divide_mode='overflow' vs 'mask' fdiv guard forms."""

import ast
import pathlib
import re

import pytest

from bosdi.va import compile_va, emit_source, lower


DIODE_VA = pathlib.Path(__file__).parent / "devices" / "diode.va"

OVERFLOW_GUARD = re.compile(r"1e-300,")
MASK_GUARD = re.compile(r"jnp\.where\(\(.*?, 0\.0, jnp\.divide")


def _emit(mode):
    dump = compile_va(str(DIODE_VA), allow_analog_in_cond=True)
    dev = lower(dump.modules[0], class_name="Diode", safe_divide_mode=mode)
    return emit_source([dev])


def test_default_keeps_overflow_guards():
    src = _emit("overflow")
    ast.parse(src)
    assert OVERFLOW_GUARD.search(src)
    assert not MASK_GUARD.search(src)


def test_mask_mode_replaces_every_overflow_guard():
    src = _emit("mask")
    ast.parse(src)
    assert not OVERFLOW_GUARD.search(src), (
        "mask mode still emits n / where(bad, 1e-300, d) guards"
    )
    assert MASK_GUARD.search(src)


def test_invalid_mode_raises():
    dump = compile_va(str(DIODE_VA), allow_analog_in_cond=True)
    with pytest.raises(ValueError, match="safe_divide_mode"):
        lower(dump.modules[0], class_name="Diode", safe_divide_mode="nope")
