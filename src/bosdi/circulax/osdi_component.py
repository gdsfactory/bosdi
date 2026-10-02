"""OSDI device model integration for circulax.

Provides :func:`osdi_component` to load OpenVAF-compiled ``.osdi`` binaries and
use them as circuit components inside :func:`~circulax.compiler.compile_netlist`.

This module lives in ``bosdi`` so circulax users can get OSDI + VA support
via a single optional install (``pip install circulax[verilog-a]``) without
requiring ``bosdi`` as a mandatory circulax dependency.

Requires the ``bosdi`` package to be installed (``osdi_loader`` must be importable).
"""

import difflib
from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp

try:
    from osdi_loader import OsdiModel, load_osdi_model

    _BOSDI_AVAILABLE = True
    _BOSDI_ERR = None
except ImportError as _bosdi_err:
    _BOSDI_AVAILABLE = False
    _BOSDI_ERR = _bosdi_err
    OsdiModel = None  # type: ignore[assignment]
    load_osdi_model = None  # type: ignore[assignment]


class OsdiComponentGroup(eqx.Module):
    """Batch of N identical OSDI device instances, evaluated via bosdi.

    Unlike :class:`~circulax.compiler.ComponentGroup`, this group calls
    ``osdi_eval`` with all N instances at once (leveraging Rayon parallelism)
    and uses the analytical Jacobians (conductances/capacitances) that the
    OSDI model returns directly — no ``jax.jacfwd`` required.

    All raw internal OSDI nodes (e.g. PSP103's ``di``, ``si``) are allocated
    as extra unknowns in the global state vector. Setup-selected collapses
    use equality constraints in these slots,
    exactly like VoltageSource's ``i_src``.  ``num_nodes`` covers all of them;
    ``num_pins`` is the external terminal count only.

    ``var_indices`` has shape ``(N, num_nodes)``: the first ``num_pins`` columns
    index terminal node voltages in the circuit node block; the remaining
    ``num_nodes - num_pins`` columns index the internal-node slots appended at
    the end of the global state vector.  ``eq_indices`` is identical — each
    node's KCL equation lives at the same global index as its voltage unknown.
    """

    name: str = eqx.field(static=True)
    model_id: int = eqx.field(static=True)  # bosdi registry ID — not differentiable
    num_pins: int = eqx.field(static=True)  # external terminals only
    num_nodes: int = eqx.field(
        static=True
    )  # terminals + all raw internal/auxiliary nodes
    num_params: int = eqx.field(static=True)
    num_states: int = eqx.field(static=True)

    params: jnp.ndarray  # (N, num_params) float64 — batched device parameters
    states: jnp.ndarray  # (N, num_states) float64 — zeros for stateless models

    var_indices: (
        jnp.ndarray
    )  # (N, num_nodes) int32 — terminal + internal indices into global y
    eq_indices: (
        jnp.ndarray
    )  # (N, num_nodes) int32 — same as var_indices (eq lives at its own y slot)
    jac_rows: jnp.ndarray  # (N*num_nodes*num_nodes,) int32 — COO row indices
    jac_cols: jnp.ndarray  # (N*num_nodes*num_nodes,) int32 — COO col indices

    # Precomputed diagonal regularisation matrix (num_nodes, num_nodes).
    # Diagonal entry i is 1.0 if node i is reactive-only (G[i,:]=0 always, F[i]=0),
    # and 0.0 otherwise.  Added to j_eff in assembly so DC Newton stays non-singular.
    reg_diag: jnp.ndarray  # (num_nodes, num_nodes) float64

    index_map: dict | None = eqx.field(static=True, default=None)
    is_fdomain: bool = eqx.field(static=True, default=False)
    amplitude_param: str = eqx.field(static=True, default="")

    # Experimental: use bosdi.osdi_debug.schur_reduce to eliminate internal
    # nodes from the per-device stamp before handing it to global Newton.
    # When True, the assembly pads the reduced 4x4 stamp back to num_nodes with
    # identity rows on internal slots so the compiler-allocated internal
    # unknowns stay self-consistent.
    use_schur_reduction: bool = eqx.field(static=True, default=False)

    # Optional bosdi Tier-3 handle (``osdi_jax.OsdiBatchHandle``) pre-baked
    # with this group's params.  Stored as a static Equinox field so JAX
    # tracing treats it as a closure constant (handle is a Python object, not
    # a JAX array, and its ``handle_id`` is baked into the compiled XLA graph).
    # When non-None, assembly uses ``osdi_eval_with_handle`` /
    # ``osdi_residual_eval_with_handle`` which skip per-call param upload
    # (~20–40 % faster for PSP103).  None falls back to the model_id + params path.
    handle: object | None = eqx.field(static=True, default=None)

    def without_handle(self) -> "OsdiComponentGroup":
        """Return a copy without the Tier-3 handle.

        The returned group uses the ``model_id + params`` evaluation path
        instead of the pre-baked handle path.  This is necessary when
        ``params`` will be swapped via ``eqx.tree_at`` inside JIT/vmap/scan,
        since the handle is a static field that cannot be re-created during
        tracing.
        """
        return OsdiComponentGroup(
            name=self.name,
            model_id=self.model_id,
            num_pins=self.num_pins,
            num_nodes=self.num_nodes,
            num_params=self.num_params,
            num_states=self.num_states,
            params=self.params,
            states=self.states,
            var_indices=self.var_indices,
            eq_indices=self.eq_indices,
            jac_rows=self.jac_rows,
            jac_cols=self.jac_cols,
            reg_diag=self.reg_diag,
            index_map=self.index_map,
            is_fdomain=self.is_fdomain,
            amplitude_param=self.amplitude_param,
            use_schur_reduction=self.use_schur_reduction,
            handle=None,
        )

    def with_params(self, new_params: jnp.ndarray) -> "OsdiComponentGroup":
        """Return a copy of this group with updated params and a fresh handle.

        Use this in parameter-optimisation loops — topology is fixed so
        the expensive parts of ``compile_netlist`` are reused unchanged;
        only the params array and the Tier-3 handle are swapped.

        Args:
            new_params: array of shape ``(N, num_params)`` in the same
                column order as ``self.params``.

        Returns:
            A new :class:`OsdiComponentGroup` with ``params`` = ``new_params``
            and ``handle`` rebuilt from ``new_params``.
        """
        import numpy as _np

        new_params = jnp.asarray(new_params, dtype=jnp.float64)
        new_handle = None
        try:
            from osdi_jax import osdi_setup_batch

            # Traced updates use the uncached FFI path; setup cannot consume
            # a tracer as a host NumPy array. Eager updates retain cached setup.
            if not isinstance(new_params, jax.core.Tracer):
                new_handle = osdi_setup_batch(self.model_id, _np.asarray(new_params))
        except ImportError:
            pass  # older bosdi without Tier-3; legacy path still works
        return OsdiComponentGroup(
            name=self.name,
            model_id=self.model_id,
            num_pins=self.num_pins,
            num_nodes=self.num_nodes,
            num_params=self.num_params,
            num_states=self.num_states,
            params=new_params,
            states=self.states,
            var_indices=self.var_indices,
            eq_indices=self.eq_indices,
            jac_rows=self.jac_rows,
            jac_cols=self.jac_cols,
            reg_diag=self.reg_diag,
            index_map=self.index_map,
            is_fdomain=self.is_fdomain,
            amplitude_param=self.amplitude_param,
            use_schur_reduction=self.use_schur_reduction,
            handle=new_handle,
        )


class OsdiModelDescriptor:
    """Descriptor returned by :func:`osdi_component`, consumed by ``compile_netlist``.

    Behaves like a component class from ``models_map``'s perspective but carries
    OSDI-specific metadata instead of Equinox fields.

    Two construction modes:

    1. **Canonical (recommended)** — ``param_names`` is ``None``, so the
       descriptor uses ``model.param_names`` directly.  Settings dicts may
       reference any canonical parameter name case-insensitively.

    2. **Legacy positional** — ``param_names`` is an explicit tuple whose
       *order* must match the OSDI model's internal parameter ordering.
       Retained for backward compatibility.
    """

    _is_osdi_descriptor: bool = True

    def __init__(
        self,
        model: OsdiModel,
        ports: tuple,
        param_names: tuple | None,
        default_params: dict,
        use_schur_reduction: bool = False,
        *,
        state_policy: str = "reject",
    ) -> None:
        self.model = model
        self.ports = ports
        self.states: tuple = ()
        self.use_schur_reduction = use_schur_reduction
        self.state_policy = state_policy
        self._analysis_variants = {}

        if param_names is None:
            self.param_names = tuple(model.param_names)
            self.is_canonical = True
            self._name_to_idx = {
                n.lower(): i for i, n in enumerate(model.param_names) if n
            }
        else:
            self.param_names = param_names
            self.is_canonical = False
            self._name_to_idx = {n.lower(): i for i, n in enumerate(param_names)}

        self.default_params = self._canonicalise(
            default_params, source="default_params"
        )

    def with_analysis(self, analysis: str) -> "OsdiModelDescriptor":
        """Return a cached immutable registration, preserving all model defaults."""
        if analysis == self.model.analysis:
            return self
        if analysis not in self._analysis_variants:
            if not self.model.path:
                raise ValueError(
                    "Changing OSDI analysis requires the binary source path"
                )
            self._analysis_variants[analysis] = osdi_component(
                self.model.path,
                self.ports,
                param_names=None if self.is_canonical else self.param_names,
                default_params=self.default_params.copy(),
                use_schur_reduction=self.use_schur_reduction,
                temperature=self.model.temperature,
                analysis=analysis,
                state_policy=self.state_policy,
                simparams=dict(self.model.simparams),
            )
        return self._analysis_variants[analysis]

    def with_simparams(self, simparams: Mapping[str, float]) -> "OsdiModelDescriptor":
        """Return a registration with simulator overrides; never mutate this descriptor."""
        settings = dict(self.model.simparams)
        settings.update(simparams)
        return osdi_component(
            self.model.path,
            self.ports,
            param_names=None if self.is_canonical else self.param_names,
            default_params=self.default_params.copy(),
            use_schur_reduction=self.use_schur_reduction,
            temperature=self.model.temperature,
            analysis=self.model.analysis,
            state_policy=self.state_policy,
            simparams=settings,
        )

    def _canonicalise(self, d: dict, *, source: str) -> dict:
        """Case-insensitive: rewrite ``d``'s keys to match ``self.param_names``.

        Absent parameters are filled with NaN (the OSDI "not given" marker)
        so the model's own Verilog-A defaults apply.
        """
        out = dict.fromkeys(self.param_names, float("nan"))
        for k, v in d.items():
            idx = self._name_to_idx.get(k.lower())
            if idx is None:
                candidates = list(self._name_to_idx.keys())
                close = difflib.get_close_matches(k.lower(), candidates, n=5)
                msg = (
                    f"Unknown OSDI parameter {k!r} in {source}. "
                    f"Did you mean one of: {close}?"
                )
                raise ValueError(msg)
            out[self.param_names[idx]] = v
        return out

    def make_instance(self, settings: dict) -> dict:
        """Merge per-instance ``settings`` into ``default_params`` by canonical name."""
        merged = dict(self.default_params)
        for k, v in settings.items():
            idx = self._name_to_idx.get(k.lower())
            if idx is None:
                candidates = list(self._name_to_idx.keys())
                close = difflib.get_close_matches(k.lower(), candidates, n=5)
                msg = (
                    f"Unknown OSDI parameter {k!r} in instance settings. "
                    f"Did you mean one of: {close}?"
                )
                raise ValueError(msg)
            merged[self.param_names[idx]] = v
        return {k: merged[k] for k in self.param_names}


def osdi_component(
    osdi_path: str,
    ports: tuple,
    param_names: tuple | None = None,
    default_params: dict | None = None,
    use_schur_reduction: bool = False,
    *,
    temperature: float = 300.0,
    analysis: str = "ac",
    state_policy: str = "reject",
    simparams: Mapping[str, float] | None = None,
) -> OsdiModelDescriptor:
    """Load a compiled ``.osdi`` binary and return a descriptor for ``compile_netlist``.

    Args:
        osdi_path:      Absolute path to the OpenVAF-compiled ``.osdi`` file.
        ports:          Ordered tuple of port names matching the Verilog-A terminals.
        param_names:    *(Optional, legacy)* Ordered tuple of parameter names.
                        If ``None`` (recommended), canonical names are read from
                        the OSDI binary and resolved case-insensitively.
        default_params: Default values for selected parameters.  Keys may be
                        any subset of canonical parameter names.  Unspecified
                        parameters are filled with NaN (OSDI "not given") so
                        the model's own Verilog-A defaults apply.
        use_schur_reduction: Eliminate OSDI internal nodes via Schur complement
                        before global Newton (experimental).

        analysis: Immutable native evaluation mode (dc, ac, or tran). Default
                  ac preserves the full current/charge stamp API.
        temperature: Setup temperature in kelvin, retained across parameter
                     updates and both cached/uncached evaluation paths.
        simparams: Numeric simulator settings queried through $simparam, e.g.
                   {"scale": 2.0, "tnom": 27.0}. These are separate from device
                   parameters, copied at registration, and retained by DC/AC/tran
                   variants and parameter updates. Names are case-sensitive.
                   This does not configure the circuit solver's own tolerances.
        state_policy: "reject" (default) guards models declaring state slots.
                      "limiting_only" explicitly asserts the binary's slots are
                      OpenVAF voltage-limiting buffers. ENABLE_LIM stays disabled;
                      these buffers are not physical history or circuit unknowns.

    Returns:
        :class:`OsdiModelDescriptor` — pass this as a value in the
        ``models_map`` argument of :func:`~circulax.compiler.compile_netlist`.

    Raises:
        ImportError: If ``bosdi`` runtime (``osdi_loader``) is not available.
        ValueError:  If port/param counts don't match the OSDI binary.
        NotImplementedError: If the model declares ABI state slots and the policy is reject.

    Example::

        OsdiPSP103N = osdi_component(
            osdi_path="psp103v4_psp103.osdi",
            ports=("D", "G", "S", "B"),
            default_params={"TYPE": 1.0, "L": 1e-6, "W": 10e-6},
        )
    """
    if not _BOSDI_AVAILABLE:
        raise ImportError(
            "OSDI support requires bosdi's native extension (osdi_loader), which could "
            "not be imported. Install circulax[verilog-a] to get OSDI support."
        ) from _BOSDI_ERR

    if state_policy not in {"reject", "limiting_only"}:
        raise ValueError("state_policy must be reject or limiting_only")
    model = load_osdi_model(
        osdi_path, temperature=temperature, analysis=analysis, simparams=simparams
    )

    if model.num_pins != len(ports):
        msg = f"OSDI model has {model.num_pins} pins but {len(ports)} port names given"
        raise ValueError(msg)
    if param_names is not None and model.num_params != len(param_names):
        msg = f"OSDI model has {model.num_params} params but {len(param_names)} param names given"
        raise ValueError(msg)
    if model.num_states > 0 and state_policy == "reject":
        msg = "Stateful OSDI models (num_states > 0) are not yet supported"
        raise NotImplementedError(msg)

    return OsdiModelDescriptor(
        model=model,
        ports=ports,
        param_names=param_names,
        default_params=default_params or {},
        use_schur_reduction=use_schur_reduction,
        state_policy=state_policy,
    )


__all__ = [
    "OsdiComponentGroup",
    "OsdiModelDescriptor",
    "osdi_component",
    "_BOSDI_AVAILABLE",
    "_BOSDI_ERR",
]
