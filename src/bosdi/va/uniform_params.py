"""Auto-detect parameters that are uniform across all netlist instances of one model.

For a circuit like the PSP103 ring oscillator where every NMOS instance
shares the same model card (same ``TOXE``, ``VTH0``, …), every value
that is identical across instances can be promoted to ``static_params``
in :func:`circulax.va.lower`.  The lowering then substitutes the literal
during the MIR walk, the constprop pass folds chains, dead branches
disappear, and the emitted body shrinks meaningfully.

This is a topology-aware helper that runs on a SAX netlist (the same
``net_dict`` that ``compile_netlist`` consumes) plus a mapping of
component name → VA model name.  Returns ``{model_name: {param: value}}``
so the caller can pass each model's set into a separate
``lower(static_params=...)`` call when emitting a per-model class.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def uniform_static_params(
    net_dict: dict[str, Any],
    model_for_component: dict[str, str],
) -> dict[str, dict[str, int | float]]:
    """Return per-model static_params dicts holding only uniform values.

    Args:
        net_dict: SAX-format netlist with an ``"instances"`` key.  Each
            instance is ``{"component": <component-name>, "settings":
            {<param>: <value>, ...}}``.
        model_for_component: Maps each component name (the key under
            ``"instances"[*]["component"]``) to the underlying VA model
            name so several component variants can share one model
            (e.g. NMOS and PMOS sharing PSP103 with different TYPE).

    Returns:
        ``{model_name: {param_name: value}}`` containing only those
        parameters that have **identical** numeric values across every
        instance whose component maps to that model.  String parameters
        are skipped — they are already handled via ``eqx.field(static=
        True)`` and don't participate in the MIR-walk substitution.

    Example::

        >>> uniform_static_params(
        ...     {"instances": {
        ...         "mn1": {"component": "nmos", "settings": {"W": 10e-6, "VTH0": 0.5}},
        ...         "mn2": {"component": "nmos", "settings": {"W": 20e-6, "VTH0": 0.5}},
        ...     }},
        ...     {"nmos": "PSP103"},
        ... )
        {'PSP103': {'VTH0': 0.5}}  # W differs across instances → not uniform
    """
    # Group settings dicts by the model they map to.
    settings_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for inst_name, inst in net_dict.get("instances", {}).items():
        component = inst.get("component")
        if component is None:
            continue
        model = model_for_component.get(component)
        if model is None:
            continue
        settings_by_model[model].append(inst.get("settings", {}))

    out: dict[str, dict[str, int | float]] = {}
    for model, settings_list in settings_by_model.items():
        if not settings_list:
            continue
        # Intersection of keys.
        common_keys = set(settings_list[0].keys())
        for s in settings_list[1:]:
            common_keys &= set(s.keys())
        uniform: dict[str, int | float] = {}
        for k in common_keys:
            values = [s[k] for s in settings_list]
            first = values[0]
            if not isinstance(first, (int, float)) or isinstance(first, bool):
                continue
            if all(v == first for v in values):
                uniform[k] = first
        if uniform:
            out[model] = uniform
    return out
