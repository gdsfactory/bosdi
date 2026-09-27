"""Branch-current state names: DAE states tuple vs eval body references."""

import ast
import pathlib

import pytest


from bosdi.va import compile_va, emit_source, lower

BSIM4_VA = pathlib.Path(__file__).parent / "devices" / "bsim4v8.va"


@pytest.fixture(scope="module")
def bsim4_rbody_source():
    dump = compile_va(str(BSIM4_VA), allow_analog_in_cond=True)
    dev = lower(
        dump.modules[0],
        class_name="Bsim4Rbody",
        # "as" and "lambda" are Python keywords — must be baked static.
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
