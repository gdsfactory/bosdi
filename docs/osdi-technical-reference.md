# OSDI Technical Reference

Detailed reference for the OSDI binary evaluation path — parameter handling, model introspection, output layout,
host-simulator integration, and debug utilities.

## Parameters

OSDI models order parameters as listed in the `param_opvar` table compiled into the binary. bosdi uses the OSDI
`access()` function to write each parameter to the correct slot in the model/instance struct, so parameter ordering is
handled automatically.

Pass `jnp.nan` for any parameter you want to leave at its Verilog-A default — bosdi skips writing NaN values and lets
`setup_model`/`setup_instance` apply the compiled-in defaults. This is the recommended approach for complex models
(BSIM, PSP) where many parameters have safe defaults.

### Addressing parameters by name

The `OsdiModel` dataclass exposes every parameter's canonical name (alias 0), kind (`MODEL`/`INST`/`OPVAR`), and type
(`REAL`/`INT`/`STR`) read directly from the binary's `OsdiParamOpvar` table. That makes it safe to drive a model card
without knowing the OSDI index layout up-front:

```python
m = load_osdi_model("bsim4v8.osdi")
name_to_idx = {n.lower(): i for i, n in enumerate(m.param_names) if n}

p = jnp.full((1, m.num_params), jnp.nan, dtype=jnp.float64)
p = p.at[0, 0].set(1.0)                               # $mfactor
p = p.at[0, name_to_idx["toxe"]].set(1.85e-9)         # oxide thickness
p = p.at[0, name_to_idx["ndep"]].set(2.54e18)         # channel doping
```

Bosdi can write REAL and INT params through the f64 array (INT is rounded from the float); STR parameters (like BSIM4's
`version="4.8.2"`) are skipped — the model must provide a default for them or be compiled without them as required.

`tests/fixtures/bsim4v82_nmos.json` ships a 213-param BSIM4 NMOS 60 nm card ported from VACASK's
`demo/spice/bsim4v82.inc`, and `tests/test_bsim4_model_card.py` demonstrates loading it by name.

## Introspecting a model's structure

`OsdiModel` carries the structural data bosdi decoded from the OSDI descriptor, useful for building Jacobian stamps or
validating a model before wiring it into a solver:

| Field                              | Meaning                                                                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `num_pins`                         | Number of external terminals                                                                         |
| `num_nodes`                        | Total unknowns = terminals + internal Kirchhoff nodes + branch-current aux                           |
| `num_states`                       | Stateful-model limiter state count (non-zero → bosdi skips eval, for now)                            |
| `num_resist_jac` / `num_react_jac` | Count of Jacobian entries with the variable RESIST / REACT flag                                      |
| `resist_jac_pairs`                 | `List[(node_1, node_2)]` — raw OSDI indices of each resistive Jacobian entry                         |
| `react_jac_pairs`                  | `List[(node_1, node_2)]` — same for reactive (dQ/dV)                                                 |
| `collapsible_pairs`                | `List[(node_1, node_2)]` — internal nodes that collapse onto terminals when a coupling param is zero |
| `resistive_mask`                   | `List[bool]` of length `num_nodes` — `True` iff that unknown's row can be non-zero at DC             |
| `param_names`                      | Canonical alias-0 name per param, in OSDI index order                                                |
| `param_flags`                      | Raw `OsdiParamOpvar.flags` per param (kind/type bits)                                                |
| `.param_kinds()`                   | Decoded `["INST", "MODEL", "OPVAR", ...]`                                                            |
| `.param_types()`                   | Decoded `["REAL", "INT", "STR", ...]`                                                                |

Indices in `*_jac_pairs` are **raw OSDI node indices** (pre-collapse, 0..num_raw_nodes). After bosdi applies
`collapsible_pairs`, the resulting `num_nodes`-wide output rows are mapped through an internal `node_map` before the
scatter. Structural tests should assert against `resist_jac_pairs`/`react_jac_pairs`; callers that want to stamp the
Jacobian into their host system's matrix should use the `cond`/`cap` outputs, which already run through the collapse and
are shaped `num_nodes × num_nodes`.

## Outputs

Outputs are one row per *unknown* — terminals first, then internal Kirchhoff nodes and branch-current auxiliaries.
`model.num_nodes` is the width; `model.num_pins` is the terminal count (`num_pins ≤ num_nodes`).

| Output | Shape             | Description                                  |
| ------ | ----------------- | -------------------------------------------- |
| `cur`  | `[N, num_nodes]`  | Resistive current residual at each unknown   |
| `cond` | `[N, num_nodes²]` | `G = ∂cur/∂V` Jacobian (flattened row-major) |
| `chg`  | `[N, num_nodes]`  | Charge residual at each unknown              |
| `cap`  | `[N, num_nodes²]` | `C = ∂chg/∂V` Jacobian (flattened row-major) |

Reactive entries are produced by `write_jacobian_array_react` in the binary; bosdi scatters them into `cap` using the
per-entry RESIST/REACT flag bits from each `OsdiJacobianEntry`. Entries that are dual-flagged (a frequent case —
junction capacitance stamps on `(A,CI)`/`(CI,A)` in diodes, gate/drain capacitance in BSIM4) appear in both `cond` and
`cap`.

## Using the outputs in a host simulator

`cur`, `cond`, `chg`, `cap` are the raw Verilog-A contributions. How you assemble them into a time-stepping Newton
iteration depends on which DAE formulation the host uses.

### Companion-method hosts (classic SPICE-style — e.g. Circulax)

Replace each reactive element with its Norton-equivalent companion at every timestep. For trapezoidal integration with
step `h`:

```
α        = 2 / h                          # or 1 / h for Backward Euler, or the BDF-k coefficient
G_total  = cond + α · cap                 # effective conductance stamped into the Newton matrix
i_eq     = cur + α · chg  − i_prev         # Norton-equivalent current into the Kirchhoff rows
```

`i_prev` is the companion source from the previous accepted step (trapezoidal carries an explicit history term; BDF
multistep methods carry the last *k* charge residuals). The Newton iteration solves one purely-resistive system per
timestep — no auxiliary reactive rows, no explicit DAE. bosdi's `chg`/`cap` outputs feed the *α-scaled* contribution;
the integrator and history bookkeeping stay on the host side.

### MNA / direct-DAE hosts (e.g. VACASK)

Stamp resistive and reactive contributions into separate rows of the global Jacobian; the simulator's DAE/BDF solver
handles the time derivative implicitly:

```
J  +=  G  +  α · C                 # sum of resistive and α-weighted reactive Jacobians
f  +=  cur  +  dchg/dt             # dchg/dt is computed by the host's integrator from chg history
```

Branch-current auxiliary unknowns (the ones that make `num_nodes > num_pins + internal_voltage_count`) map onto extra
MNA rows/columns. The host is responsible for placing those unknowns in its global solution vector and for treating the
corresponding Kirchhoff rows as *branch-defining equations* rather than current-balance sums — a plain KCL sum into a
branch-current row is incorrect and will give a singular Jacobian. See `OsdiModel.resistive_mask` to detect rows whose
`G[i, :]` is structurally zero at DC; those rows need regularisation (e.g. an explicit branch equation) before Newton
will converge.

### Comparison

|                            | Companion method                                 | MNA / DAE                                        |
| -------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| Newton matrix per step     | single resistive stamp `G + α·C`                 | `G` and `α·C` stamped into separate contribution |
| History bookkeeping        | host keeps previous `i_eq` / charge residual     | host integrator keeps `chg` history internally   |
| Branch-current auxiliaries | eliminated inside each element's companion model | live in the MNA system as extra unknowns         |
| bosdi's responsibility     | return `cur`, `cond`, `chg`, `cap` per step      | identical                                        |
| host's extra work          | pick α, add companion-source history term        | pick α, manage aux rows, handle branch equations |

Either way, bosdi's outputs are the same. Only the assembly recipe on the host side changes.

## Debug helpers

`src/osdi_debug.py` ships two post-processing utilities for host-simulator authors who need to audit how bosdi's stamp
translates into their own assembly. Neither is on the hot path.

```python
from osdi_debug import schur_reduce, dump_jacobian, format_jacobian_table, classify_rows

# 1. Reduce a device's (num_nodes × num_nodes) stamp onto its terminals.
#    Pass alpha=0 for a DC Schur reduction on G; pass the host's per-step
#    integrator coefficient for the transient Newton stamp.
result = schur_reduce(cur[0], G, chg[0], C, num_pins=model.num_pins, alpha=0.0)
print(result.j_eff)      # (num_pins, num_pins) reduced Jacobian
print(result.r_eff)      # (num_pins,)          reduced residual
print(result.singular)   # True iff A_II was near-singular before gmin

# 2. Dump all non-zero (row, col) entries of a single device's stamp, with
#    a heuristic flag for Lagrange-style ±1 identity constraint rows
#    (useful for catching hosts that misread a constraint as a 1 S conductance).
print(format_jacobian_table(G, C))
rows = classify_rows(G, C)  # one of: physics / constraint / reactive_only / empty
```

`schur_reduce` works on both single-device and batched `(N, num_nodes, num_nodes)` arrays and is JAX-compatible, so you
can `jax.vmap` or `jax.jit` it over a batch of devices. The α-reduced Jacobian is not decomposed back into
`(G_eff, C_eff)` — the Schur complement of `G + α·C` is nonlinear in α in general, so there's no clean split. Call at
two α values and finite-difference if you really need a separated capacitance.
