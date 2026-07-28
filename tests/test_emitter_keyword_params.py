"""Regression: VA params named after Python keywords must emit valid code."""

from __future__ import annotations

import ast
import keyword
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from bosdi.va.emitter import emit_source
from bosdi.va.lowering import lower, py_param_name
from bosdi.va.va_defaults import parse_va_defaults_expanded

BSIM4_VA = pathlib.Path(__file__).parent / "devices" / "bsim4v8.va"

openvaf_py = pytest.importorskip("openvaf_py")

from bosdi.va.binding import compile_va  # noqa: E402 — after importorskip


def test_py_param_name_keywords_get_trailing_underscore():
    assert py_param_name("as") == "as_"
    assert py_param_name("lambda") == "lambda_"


def test_py_param_name_leaves_ordinary_names_alone():
    for name in ("vth0", "toxe", "min", "type_ish", "_mfactor"):
        assert py_param_name(name) == name


def test_py_param_name_covers_every_python_keyword():
    for kw in keyword.kwlist:
        assert py_param_name(kw) == f"{kw}_"


@pytest.fixture(scope="module")
def bsim4_device():
    df = compile_va(str(BSIM4_VA), allow_analog_in_cond=True)
    defaults = parse_va_defaults_expanded(BSIM4_VA)
    return lower(df.modules[0], va_defaults=defaults)


def test_bsim4_keyword_params_are_renamed_in_surface(bsim4_device):
    names = {name for name, _ty, _default in bsim4_device.params}
    assert "as_" in names
    assert "lambda_" in names
    assert "as" not in names
    assert "lambda" not in names


def test_bsim4_emitted_source_parses(bsim4_device):
    src = emit_source([bsim4_device])
    ast.parse(src)


def test_bsim4_body_references_use_the_alias(bsim4_device):
    src = emit_source([bsim4_device])
    names = {node.id for node in ast.walk(ast.parse(src)) if isinstance(node, ast.Name)}
    args = {
        arg.arg
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.FunctionDef)
        for arg in node.args.args
    }
    assert "as_" in names | args
    assert "lambda_" in names | args
