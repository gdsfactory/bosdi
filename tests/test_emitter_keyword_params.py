"""Regression test: VA params named after Python keywords must emit valid code.

BSIM4 declares ``parameter real as`` (source area) and ``parameter real
lambda`` (velocity overshoot) — both are Python *keywords*, so putting the
names verbatim into the generated ``def`` signature produces a module that
``ast.parse`` rejects (``def Bsim4(..., as: float = 0.0, ...)``).

The fix (``py_param_name`` in lowering.py) suffixes keyword-colliding
names with an underscore — ``as_`` / ``lambda_``, PEP 8's convention — in
the emitted signature AND at every body reference, while everything keyed
by the VA name (``static_params``, ``va_defaults``, ``$param_given``)
keeps using the original spelling.

The test intentionally lowers bsim4v8.va WITHOUT baking ``as`` / ``lambda``
via ``static_params`` — baking removes them from the signature entirely,
which is how the collision went unnoticed before this regression test.
"""

from __future__ import annotations

import ast
import keyword
import pathlib
import sys

import pytest

# ── path setup ──────────────────────────────────────────────────────────────
# When run via `pixi run pytest` the package is installed; when invoked
# directly from the repo root it may not be, so add src/ as a fallback.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from bosdi.va.emitter import emit_source
from bosdi.va.lowering import lower, py_param_name
from bosdi.va.va_defaults import parse_va_defaults_expanded

BSIM4_VA = pathlib.Path(__file__).parent / "devices" / "bsim4v8.va"

# The binding-based compile path needs the openvaf_py PyO3 module; skip
# (rather than error) where it isn't built, matching the other VA tests.
openvaf_py = pytest.importorskip("openvaf_py", reason="openvaf_py binding not built")

from bosdi.va.binding import compile_va  # noqa: E402 — after importorskip


# ── py_param_name unit behaviour ─────────────────────────────────────────────


def test_py_param_name_keywords_get_trailing_underscore():
    assert py_param_name("as") == "as_"
    assert py_param_name("lambda") == "lambda_"


def test_py_param_name_leaves_ordinary_names_alone():
    # Ordinary VA names — including builtins like ``min``, which are legal
    # (if shadowing) Python identifiers — pass through untouched.
    for name in ("vth0", "toxe", "min", "type_ish", "_mfactor"):
        assert py_param_name(name) == name


def test_py_param_name_covers_every_python_keyword():
    for kw in keyword.kwlist:
        assert py_param_name(kw) == f"{kw}_"


# ── end-to-end: BSIM4 emits parseable source without baking as/lambda ────────


@pytest.fixture(scope="module")
def bsim4_device():
    """Lower bsim4v8.va with NO static_params, so ``as`` / ``lambda`` stay
    in the emitted signature instead of being baked away as literals."""
    df = compile_va(str(BSIM4_VA), allow_analog_in_cond=True)
    defaults = parse_va_defaults_expanded(BSIM4_VA)
    return lower(df.modules[0], va_defaults=defaults)


def test_bsim4_keyword_params_are_renamed_in_surface(bsim4_device):
    names = {name for name, _ty, _default in bsim4_device.params}
    # The keyword-colliding params surface under their aliased names…
    assert "as_" in names
    assert "lambda_" in names
    # …and never under the raw keyword spelling.
    assert "as" not in names
    assert "lambda" not in names


def test_bsim4_emitted_source_parses(bsim4_device):
    src = emit_source([bsim4_device])
    # This is the regression: with the raw names in the signature this
    # raised ``SyntaxError: invalid syntax`` on ``as: float = 0.0``.
    ast.parse(src)


def test_bsim4_body_references_use_the_alias(bsim4_device):
    """Body references must match the renamed kwarg, not the VA name.

    ``ast.parse`` alone already proves no bare ``as`` / ``lambda``
    identifier survives anywhere (either would be a syntax error in an
    expression), so it's enough to check the aliases are actually *used*
    — a rename that dropped the params entirely would also parse.
    """
    src = emit_source([bsim4_device])
    names = {
        node.id for node in ast.walk(ast.parse(src)) if isinstance(node, ast.Name)
    }
    args = {
        arg.arg
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.FunctionDef)
        for arg in node.args.args
    }
    assert "as_" in names | args
    assert "lambda_" in names | args
