"""Debug/analysis helpers for OSDI devices loaded through bosdi.

These are post-processing tools; nothing here runs on bosdi's hot
evaluation path. Offered for host-simulator authors who need to compare
bosdi's per-device stamp against their own assembly, or to experiment
with internal-node elimination without carrying it into production code.

Two pieces:

* ``schur_reduce`` — eliminate internal-node rows/cols from a device's
  ``(cur, cond, chg, cap)`` stamp, returning a terminal-only equivalent
  Jacobian and residual at a chosen integrator coefficient ``alpha``.
  Pass ``alpha=0`` for a pure-DC Schur reduction on ``G`` (with residual
  ``cur``); pass the host's per-stage coefficient for the transient
  reduction suitable as an implicit-step Newton stamp.

* ``dump_jacobian`` — produce a per-entry listing of a single device's
  post-collapse ``(cond, cap)`` with a heuristic flag marking
  Lagrange-style constraint rows (``±1`` identity with zero reactive
  row). Useful for auditing how a host solver has consumed the stamp.

``schur_reduce`` is JAX-compatible and vmap-able across a leading batch
axis. ``dump_jacobian`` is plain Python and expects numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Schur-complement reduction
# ---------------------------------------------------------------------------


@dataclass
class SchurResult:
    """Terminal-only equivalent of a single device's stamp at one ``alpha``.

    Attributes:
        j_eff: ``(..., num_pins, num_pins)`` — the α-reduced Jacobian
            ``A_TT − A_TI · A_II⁻¹ · A_IT`` where ``A = cond + α·cap``.
            For ``alpha=0`` this is the DC Schur complement of ``cond``.
        r_eff: ``(..., num_pins)`` — the α-reduced residual
            ``r_T − A_TI · A_II⁻¹ · r_I`` where ``r = cur + α·chg``.
            For ``alpha=0`` this is the DC Schur complement of ``cur``.
        singular: True iff ``|det(A_II)| < singularity_threshold`` *before*
            the ``gmin`` regularisation was added. Host should treat the
            reduction as suspect when this is set.
    """

    j_eff: Any
    r_eff: Any
    singular: bool


def schur_reduce(
    cur,
    cond,
    chg,
    cap,
    num_pins: int,
    alpha: float = 0.0,
    gmin: float = 1e-12,
    singularity_threshold: float = 1e-30,
) -> SchurResult:
    """Eliminate internal-node rows/columns from a device's OSDI stamp.

    Given bosdi's per-device ``(cur, cond, chg, cap)`` outputs (each shaped
    for ``num_nodes`` post-collapse unknowns, terminals in the first
    ``num_pins`` slots) and an integrator coefficient ``alpha``, compute:

    .. code-block::

        A      = cond + alpha * cap                                (num_nodes × num_nodes)
        r      = cur  + alpha * chg                                (num_nodes,)
        A_II_reg = A_II + gmin * I
        j_eff  = A_TT − A_TI · A_II_reg⁻¹ · A_IT                   (num_pins × num_pins)
        r_eff  = r_T  − A_TI · A_II_reg⁻¹ · r_I                    (num_pins,)

    For DC analysis pass ``alpha=0``; ``j_eff`` is then the Schur-reduced
    ``G`` and ``r_eff`` the Schur-reduced resistive residual.

    For a transient implicit step, pass the host's per-stage integrator
    coefficient (e.g. ``2/h`` for trapezoidal, ``1/h`` for backward Euler,
    the current SDIRK stage coefficient, etc.) and Newton-solve against
    ``j_eff`` and ``r_eff`` directly.

    Args:
        cur:   ``(..., num_nodes)`` — resistive residual.
        cond:  ``(..., num_nodes, num_nodes)`` — ``G = ∂cur/∂V``.
        chg:   ``(..., num_nodes)`` — reactive residual (charges).
        cap:   ``(..., num_nodes, num_nodes)`` — ``C = ∂chg/∂V``.
        num_pins: number of terminal unknowns; first ``num_pins`` rows
            and cols of ``cond``/``cap`` (and entries of ``cur``/``chg``)
            are terminals, the rest are internal nodes and branch-current
            auxiliaries.
        alpha: integrator coefficient at which to evaluate the reduction.
        gmin:  diagonal regularisation added to ``A_II`` before the solve.
            Matches the standard SPICE gmin convention.
        singularity_threshold: flag ``singular=True`` when the raw
            ``|det(A_II)|`` falls below this.

    Returns:
        ``SchurResult`` with ``j_eff``, ``r_eff``, ``singular``.

    Notes:
        No attempt is made to decompose ``j_eff`` back into a
        ``(G_eff, C_eff)`` pair. In general the Schur complement of
        ``G + α·C`` is nonlinear in ``α`` (except when ``G_II`` and
        ``C_II`` commute), so a clean post-hoc split doesn't exist. If
        the host needs to see ``C_eff`` specifically for debugging, call
        this twice at two different α and finite-difference.
    """
    cond = jnp.asarray(cond)
    cap = jnp.asarray(cap)
    cur = jnp.asarray(cur)
    chg = jnp.asarray(chg)

    num_nodes = cond.shape[-1]
    if not (0 < num_pins <= num_nodes):
        raise ValueError(f"num_pins must be in (0, {num_nodes}]; got {num_pins}")
    if cond.shape[-2] != num_nodes or cap.shape[-1] != num_nodes:
        raise ValueError(
            f"cond/cap must be square ({num_nodes}×{num_nodes}); "
            f"got cond={cond.shape}, cap={cap.shape}"
        )

    if num_pins == num_nodes:
        # Nothing to eliminate — device has no internal nodes.
        j_eff = cond + alpha * cap
        r_eff = cur + alpha * chg
        return SchurResult(j_eff=j_eff, r_eff=r_eff, singular=False)

    T = num_pins
    A = cond + alpha * cap
    r = cur + alpha * chg

    A_TT = A[..., :T, :T]
    A_TI = A[..., :T, T:]
    A_IT = A[..., T:, :T]
    A_II = A[..., T:, T:]
    r_T = r[..., :T]
    r_I = r[..., T:]

    I_inner = A_II.shape[-1]
    eye = jnp.eye(I_inner, dtype=A_II.dtype)
    A_II_reg = A_II + gmin * eye

    # One LU factorisation, two right-hand sides: solve for (A_IT, r_I).
    rhs = jnp.concatenate([A_IT, r_I[..., None]], axis=-1)
    sol = jnp.linalg.solve(A_II_reg, rhs)
    X = sol[..., :T]
    y = sol[..., T:].squeeze(-1)

    j_eff = A_TT - A_TI @ X
    r_eff = r_T - (A_TI @ y[..., None]).squeeze(-1)

    det = jnp.linalg.det(A_II)
    singular = bool(jnp.any(jnp.abs(det) < singularity_threshold))

    return SchurResult(j_eff=j_eff, r_eff=r_eff, singular=singular)


# ---------------------------------------------------------------------------
# Per-entry Jacobian dump
# ---------------------------------------------------------------------------


@dataclass
class JacobianEntry:
    """One non-zero ``(row, col)`` cell in a post-collapse device stamp."""

    row: int
    col: int
    cond: float
    cap: float
    is_likely_constraint: bool


@dataclass
class RowClassification:
    """Heuristic classification of a single row of the stamp."""

    row: int
    kind: str  # "physics" | "constraint" | "reactive_only" | "empty"
    detail: str


_ZERO_TOL = 1e-30  # absolute floor for "actually non-zero" in cond/cap cells
_UNIT_MAGNITUDE_TOL = 1e-6  # how close a constraint entry must be to |1|


def _classify_row(
    cond_row: np.ndarray, cap_row: np.ndarray, row: int
) -> RowClassification:
    """Heuristic: is this row a Lagrange-style identity constraint?

    A constraint row looks like:
      * exactly one or two non-zero entries in ``cond_row``, all with
        magnitude very close to 1 (``±1.0`` within ``_UNIT_MAGNITUDE_TOL``);
      * corresponding ``cap_row`` is all zero (identity constraints carry
        no reactive contribution).
    """
    cond_nz_idx = np.where(np.abs(cond_row) > _ZERO_TOL)[0]
    cap_nz_idx = np.where(np.abs(cap_row) > _ZERO_TOL)[0]

    if len(cond_nz_idx) == 0 and len(cap_nz_idx) == 0:
        return RowClassification(row=row, kind="empty", detail="all zeros")

    if len(cond_nz_idx) == 0 and len(cap_nz_idx) > 0:
        return RowClassification(
            row=row,
            kind="reactive_only",
            detail=f"{len(cap_nz_idx)} cap entries, no conductance",
        )

    # Constraint test: 1–2 non-zero cond entries, all ±1, cap row entirely zero.
    if len(cap_nz_idx) == 0 and 1 <= len(cond_nz_idx) <= 2:
        vals = cond_row[cond_nz_idx]
        if np.all(
            np.isclose(
                np.abs(vals), 1.0, rtol=_UNIT_MAGNITUDE_TOL, atol=_UNIT_MAGNITUDE_TOL
            )
        ):
            return RowClassification(
                row=row,
                kind="constraint",
                detail=(
                    f"cond nonzeros at cols={cond_nz_idx.tolist()} "
                    f"with values={vals.tolist()} — Lagrange identity row"
                ),
            )

    return RowClassification(
        row=row,
        kind="physics",
        detail=f"{len(cond_nz_idx)} cond + {len(cap_nz_idx)} cap non-zeros",
    )


def dump_jacobian(
    cond,
    cap,
    *,
    threshold: float = 1e-30,
) -> list[JacobianEntry]:
    """Per-entry listing of non-zero cells in a single device's stamp.

    Walks a ``(num_nodes, num_nodes)`` pair of ``cond`` / ``cap`` and
    returns every ``(row, col)`` cell whose conductance or capacitance
    exceeds ``threshold``. The per-cell ``is_likely_constraint`` flag is
    True iff the whole row classifies as a Lagrange identity row (see
    ``classify_rows``); callers can use it to audit whether their host
    solver is treating those rows as equality constraints rather than as
    1 S conductances.

    Args:
        cond: ``(num_nodes, num_nodes)`` dI/dV Jacobian.
        cap:  ``(num_nodes, num_nodes)`` dQ/dV Jacobian.
        threshold: absolute cutoff for "non-zero".

    Returns:
        list of ``JacobianEntry`` in ``(row, col)`` order.
    """
    cond = np.asarray(cond)
    cap = np.asarray(cap)
    if cond.shape != cap.shape:
        raise ValueError(
            f"cond and cap must share shape; got {cond.shape} vs {cap.shape}"
        )
    if cond.ndim != 2 or cond.shape[0] != cond.shape[1]:
        raise ValueError(
            f"cond must be 2D square (single device); got shape {cond.shape}"
        )

    classifications = classify_rows(cond, cap)
    by_row = {c.row: c for c in classifications}
    entries = []
    n = cond.shape[0]
    for r in range(n):
        for c in range(n):
            cval = float(cond[r, c])
            kval = float(cap[r, c])
            if abs(cval) <= threshold and abs(kval) <= threshold:
                continue
            entries.append(
                JacobianEntry(
                    row=r,
                    col=c,
                    cond=cval,
                    cap=kval,
                    is_likely_constraint=(by_row[r].kind == "constraint"),
                )
            )
    return entries


def classify_rows(cond, cap) -> list[RowClassification]:
    """Return per-row classification for a single device's ``(cond, cap)``.

    Rows are classified as:
      * ``physics``        — ordinary conductance + capacitance row
      * ``constraint``     — Lagrange-style ``±1`` identity, zero cap row
      * ``reactive_only``  — cap-only row (pure capacitor-like contribution)
      * ``empty``          — all-zero row (would be singular if it remained
        in a DC solve; typical for collapsed-out internal nodes)
    """
    cond = np.asarray(cond)
    cap = np.asarray(cap)
    return [_classify_row(cond[r], cap[r], r) for r in range(cond.shape[0])]


def format_jacobian_table(
    cond,
    cap,
    *,
    threshold: float = 1e-30,
) -> str:
    """Format a single device's non-zero Jacobian entries as a readable table.

    Intended for eyeballing a per-device dump during debugging. For
    programmatic use prefer ``dump_jacobian``.

    Returns:
        Multi-line string with a header, one row per non-zero cell, and
        a final row classification block.
    """
    entries = dump_jacobian(cond, cap, threshold=threshold)
    classifications = classify_rows(cond, cap)

    lines = [
        f"{'row':>4} {'col':>4} {'cond':>14} {'cap':>14}  flag",
        "-" * 54,
    ]
    for e in entries:
        flag = "  ← constraint" if e.is_likely_constraint else ""
        lines.append(f"{e.row:>4} {e.col:>4} {e.cond:>14.6g} {e.cap:>14.6g}{flag}")
    lines.append("")
    lines.append("Row classifications:")
    for c in classifications:
        lines.append(f"  row {c.row:>2}: {c.kind:<14} — {c.detail}")
    return "\n".join(lines)
