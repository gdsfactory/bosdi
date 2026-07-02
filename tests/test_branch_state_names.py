"""Branch-current state names must be consistent between the DAE states
tuple and the probe reads inside the emitted eval body.

Regression test for the BSIM4 rbodymod=1 substrate network: its five
branch-current unknowns surface as ``flow(hi,lo)`` DAE nodes, and three of
them share the hi node ``sbulk``.  Two historical bugs made the emitted
Python reference states that don't exist:

1. ``_rewrite_branch_unknown`` keyed the branch registry on the raw
   ``"hi,lo"`` string instead of the ``(hi, lo)`` tuple the probe inputs
   registered, minting fresh BranchIds for every DAE branch unknown.
2. ``_build_branch_state_name_map`` derived ``i_<branch>`` names from the
   hi node, which is not unique across branches.
"""

import ast
import pathlib

import pytest

openvaf_py = pytest.importorskip("openvaf_py")

from bosdi.va import compile_va, emit_source, lower

BSIM4_VA = pathlib.Path(__file__).parent / "devices" / "bsim4v8.va"


@pytest.fixture(scope="module")
def bsim4_rbody_source():
    dump = compile_va(str(BSIM4_VA), allow_analog_in_cond=True)
    dev = lower(
        dump.modules[0],
        class_name="Bsim4Rbody",
        # rbodymod=1 keeps the substrate resistor network live.  "as" and
        # "lambda" are Python keywords, so they must be baked static or the
        # emitted def signature would not parse.
        static_params={"rbodymod": 1, "as": 0.0, "lambda": 0.0},
    )
    return emit_source([dev])


def _declared_states(tree: ast.Module) -> set[str]:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "va_component"
        ):
            for kw in node.keywords:
                if kw.arg == "states" and isinstance(kw.value, ast.Tuple):
                    return {ast.literal_eval(e) for e in kw.value.elts}
    return set()


def test_every_state_ref_is_declared(bsim4_rbody_source):
    tree = ast.parse(bsim4_rbody_source)
    states = _declared_states(tree)
    assert states, "emitted component declares no states"
    refs = {
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "s"
    }
    undefined = refs - states
    assert not undefined, (
        f"emitted body references undeclared states {sorted(undefined)}; "
        f"declared: {sorted(states)}"
    )


def test_branch_state_names_are_unique(bsim4_rbody_source):
    tree = ast.parse(bsim4_rbody_source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "va_component"
        ):
            for kw in node.keywords:
                if kw.arg == "states" and isinstance(kw.value, ast.Tuple):
                    names = [ast.literal_eval(e) for e in kw.value.elts]
                    assert len(names) == len(set(names)), (
                        f"duplicate state names in states tuple: {names}"
                    )


def test_emitted_module_executes(bsim4_rbody_source):
    pytest.importorskip("circulax")
    ns: dict = {}
    exec(compile(bsim4_rbody_source, "<bsim4_rbody>", "exec"), ns)  # noqa: S102
    assert "Bsim4Rbody" in ns
