"""
Structural tests for tests/compiled_osdi/diode.osdi.

Cross-checks the OSDI descriptor data bosdi parses out of the binary against
the OpenVAF IR visualisation at
    https://robtaylor.github.io/OpenVAF/pr-preview/pr-20/DIODE/diode.html
(captured from the site's window.MODULE_DATA).

The viz page was generated from diode_va compiled **with** `SELFHEATING=1`
(3 ports A/C/dT, 1 internal CI, 4 unknowns, 14 resistive + 6 reactive
Jacobian entries). The .osdi binary in this repo was compiled **without**
self-heating, so its shape differs: 2 terminals + 1 internal = 3 raw nodes,
collapsed to 2 unknowns. The assertions below target this specific variant.

The set of 13 model parameters (is, tnom, zetais, n, ea, rs, zetars, zetarth,
vj, m, cj0, rth, minr) and the two instance parameters ($mfactor + area/temp
offset) are common to both variants.
"""

import pathlib
import pytest

from osdi_loader import load_osdi_model


DIODE_PATH = pathlib.Path(__file__).parent / "compiled_osdi" / "diode.osdi"


@pytest.fixture(scope="module")
def diode():
    return load_osdi_model(str(DIODE_PATH))


def test_diode_terminal_count(diode):
    assert diode.num_pins == 2, (
        "diode.osdi was compiled without SELFHEATING, so there are 2 terminals "
        "(A, C). The viz's 3-terminal variant is a different compile-time build."
    )


def test_diode_unknown_count(diode):
    # Raw descriptor reports 3 nodes (2 terminals + 1 internal CI). After the
    # single collapsible pair (2→1) both remaining unknowns are terminals.
    assert diode.num_nodes == 2, (
        f"Expected num_nodes=2 after collapse (2 terminals + 1 internal→cathode); "
        f"got {diode.num_nodes}"
    )


def test_diode_param_counts(diode):
    kinds = diode.param_kinds()
    tally = {k: kinds.count(k) for k in set(kinds)}
    # From the viz: 13 `param` entries (is, tnom, zetais, n, ea, rs, zetars,
    # zetarth, vj, m, cj0, rth, minr) + 2 instance (e.g. $mfactor + area).
    assert diode.num_params == 15, (
        f"Expected 15 OSDI params (13 MODEL + 2 INST); got {diode.num_params}"
    )
    assert tally.get("MODEL", 0) == 13, f"Expected 13 MODEL params; got {tally}"
    assert tally.get("INST", 0) == 2, f"Expected 2 INST params; got {tally}"
    assert tally.get("OPVAR", 0) == 0, f"Expected 0 OPVARs; got {tally}"


def test_diode_param_types_all_real(diode):
    # Per the viz, every `param` is a REAL scalar. No ints, no strings.
    types = diode.param_types()
    assert all(t == "REAL" for t in types), f"Non-REAL param types present: {types}"


def test_diode_resist_jacobian_pattern(diode):
    # Raw OSDI node indices: 0=A (anode), 1=C (cathode), 2=CI (internal).
    # The 7 resistive Jacobian entries the binary reports are the non-zeros of
    # a 3×3 stamp with row 0 partially zero (no (0,1) entry because the
    # A→C conductance flows only through the internal node CI):
    #   (0,0)  (0,2)
    #   (1,1)  (1,2)
    #   (2,0)  (2,1)  (2,2)
    expected = {
        (0, 0),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    }
    assert diode.num_resist_jac == 7, (
        f"Expected 7 resistive Jacobian entries; got {diode.num_resist_jac}"
    )
    assert set(diode.resist_jac_pairs) == expected, (
        f"Resistive Jacobian sparsity pattern does not match expected.\n"
        f"  got:      {sorted(diode.resist_jac_pairs)}\n"
        f"  expected: {sorted(expected)}"
    )


def test_diode_collapsible_pair(diode):
    # The viz and the binary agree on exactly one collapsible pair: the
    # internal node CI (raw idx 2) collapses onto the cathode C (raw idx 1)
    # when the series resistance `rs` is zero. bosdi always applies the
    # collapse so internal voltages track the cathode terminal.
    assert len(diode.collapsible_pairs) == 1, (
        f"Expected 1 collapsible pair; got {diode.collapsible_pairs}"
    )
    assert diode.collapsible_pairs[0] == (2, 1), (
        f"Collapsible pair should be (2, 1) — internal CI → terminal C; "
        f"got {diode.collapsible_pairs[0]}"
    )


def test_diode_resistive_mask_after_collapse(diode):
    # After collapse there are 2 output slots (both terminals). Both must be
    # reachable as Jacobian rows — neither row should be structurally zero.
    assert len(diode.resistive_mask) == diode.num_nodes
    assert all(diode.resistive_mask), (
        f"resistive_mask after collapse should mark all unknowns live; "
        f"got {diode.resistive_mask}"
    )


def test_diode_react_jacobian_pattern(diode):
    # Junction capacitance stamp lives between anode (raw 0 = A) and internal
    # (raw 2 = CI) — both diagonal and both off-diagonal entries. The other
    # three resistive entries involving the cathode are flagged RESIST_CONST
    # but not REACT, so they do not appear in the per-iteration reactive list
    # written by write_jacobian_array_react.
    expected = {(0, 0), (0, 2), (2, 0), (2, 2)}
    assert diode.num_react_jac == 4
    assert set(diode.react_jac_pairs) == expected, (
        f"Reactive Jacobian sparsity pattern does not match expected.\n"
        f"  got:      {sorted(diode.react_jac_pairs)}\n"
        f"  expected: {sorted(expected)}"
    )
