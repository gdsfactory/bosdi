"""``collapse_nodes`` must be restrictable to the hints that are actually live.

OpenVAF emits a ``CollapseHint`` for every conditional ``V(a,b) <+ 0`` in the
source; whether each hint fires depends on parameter values resolved at setup
(BSIM4's ``rdsmod``/``rgatemod``/``rbodymod``).  ``collapse_nodes=True``
applies every hint unconditionally, which shorts any resistance network the
card keeps live — BSIM4 with ``rbodymod=1`` loses its whole substrate mesh.
Passing an allow-list of node pairs applies only the collapses the mode flags
enable, matching what OSDI/ngspice do at setup.
"""

import pathlib

import pytest

openvaf_py = pytest.importorskip("openvaf_py")

from bosdi.va import compile_va, lower

BSIM4_VA = pathlib.Path(__file__).parent / "devices" / "bsim4v8.va"

# rbodymod=1: substrate resistor network live; rdsmod=0 / rgatemod=0: those
# node merges match ngspice.  "as"/"lambda" are Python keywords -> baked.
STATIC = {"rbodymod": 1, "rdsmod": 0, "rgatemod": 0, "as": 0.0, "lambda": 0.0}
SUBSTRATE_NODES = {"v_bi", "v_dbulk", "v_sbulk"}


def _states(collapse_nodes):
    dump = compile_va(str(BSIM4_VA), allow_analog_in_cond=True)
    dev = lower(
        dump.modules[0],
        class_name="Bsim4Collapse",
        static_params=STATIC,
        collapse_nodes=collapse_nodes,
    )
    return set(dev.states)


def test_allowlist_keeps_live_substrate_network():
    states = _states([("d", "di"), ("s", "si"), ("g", "gm"), ("gm", "gi")])
    # rdsmod=0 / rgatemod=0 collapses applied ...
    assert not {"v_di", "v_si", "v_gi", "v_gm"} & states
    # ... but the rbodymod=1 substrate mesh survives.
    assert SUBSTRATE_NODES <= states, (
        f"live substrate nodes were collapsed away; states = {sorted(states)}"
    )


def test_pair_order_is_irrelevant():
    a = _states([("d", "di"), ("s", "si")])
    b = _states([("di", "d"), ("si", "s")])
    assert a == b


def test_true_still_applies_every_hint():
    states = _states(True)
    assert not SUBSTRATE_NODES & states
