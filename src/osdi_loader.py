import os
import math
from copy import deepcopy
from functools import lru_cache
from threading import RLock
from dataclasses import dataclass, field
import jax.numpy as jnp
import osdi_shim_nb

_VERSION_MAP = {"0.4": 4, "0.5": 5}
_REGISTRATION_LOCK = RLock()

# OsdiParamOpvar flag bit decoding (from OSDI 0.4 header).
_PARA_KIND_MASK = 0xC0000000  # bits 30..31
_PARA_KIND_MODEL = 0x00000000
_PARA_KIND_INST = 0x40000000
_PARA_KIND_OPVAR = 0x80000000
_PARA_TY_MASK = 0x3  # bits 0..1: 0=REAL, 1=INT, 2=STR


def _decode_param_kind(flag: int) -> str:
    k = flag & _PARA_KIND_MASK
    if k == _PARA_KIND_MODEL:
        return "MODEL"
    if k == _PARA_KIND_INST:
        return "INST"
    if k == _PARA_KIND_OPVAR:
        return "OPVAR"
    return "UNKNOWN"


def _decode_param_type(flag: int) -> str:
    return {0: "REAL", 1: "INT", 2: "STR"}.get(flag & _PARA_TY_MASK, "UNKNOWN")


@dataclass
class OsdiModel:
    """A Python representation of a loaded Verilog-A device model."""

    id: int
    num_pins: int  # = num_terminals (external pins only)
    num_nodes: int  # = all raw OSDI nodes, including collapse equality slots
    num_params: int
    num_states: int
    osdi_version: str
    resistive_mask: list  # len == num_nodes; True iff G[i,:] can be non-zero at DC
    # Raw OSDI node-index pairs (0..num_nodes, pre-collapse). Use the collapsible
    # pairs to compute which internal slots merge onto terminals.
    resist_jac_pairs: list = field(default_factory=list)
    react_jac_pairs: list = field(default_factory=list)
    collapsible_pairs: list = field(default_factory=list)
    # Per-param flags from OsdiParamOpvar (see _decode_param_kind/type).
    param_flags: list = field(default_factory=list)
    # Per-param canonical (alias 0) names in OSDI order. Length == num_params.
    param_names: list = field(default_factory=list)
    temperature: float = 300.0
    analysis: str = "ac"
    path: str = ""

    @property
    def num_resist_jac(self) -> int:
        return len(self.resist_jac_pairs)

    @property
    def num_react_jac(self) -> int:
        return len(self.react_jac_pairs)

    def param_kinds(self) -> list:
        """List of per-param kind strings: MODEL / INST / OPVAR."""
        return [_decode_param_kind(f) for f in self.param_flags]

    def param_types(self) -> list:
        """List of per-param type strings: REAL / INT / STR."""
        return [_decode_param_type(f) for f in self.param_flags]

    def allocate_jax_buffers(self, num_devices: int):
        return {
            "voltages": jnp.zeros((num_devices, self.num_nodes), dtype=jnp.float64),
            "params": jnp.zeros((num_devices, self.num_params), dtype=jnp.float64),
            "states": jnp.zeros((num_devices, self.num_states), dtype=jnp.float64),
        }


def load_osdi_model(
    osdi_filepath: str,
    version: str = "0.4",
    *,
    temperature: float = 300.0,
    analysis: str = "ac",
) -> OsdiModel:
    """
    Load an OpenVAF-compiled .osdi binary and register it for JAX evaluation.

    Equivalent calls share a process-local native ID through a bounded LRU.
    Returned metadata is independent. Canonical path, file identity, ABI,
    temperature and analysis mode define the cache key. Native IDs remain
    registered for the process lifetime, including after Python cache eviction.

    Args:
        osdi_filepath: Path to the .osdi ELF binary.
        version:       OSDI standard version to use ("0.4" or "0.5").
        analysis:      Immutable evaluation mode: dc, ac, or tran. The default
                       ac retains the full current/charge stamp API. dc disables
                       integration to impose idt initial-value equations.
        temperature:   Immutable setup temperature in kelvin for this model id.
                       Changing temperature selects a separate model id; parameter
                       updates and cached handles retain this value.
    """
    modes = {"dc": 0, "ac": 1, "tran": 2}
    if analysis not in modes:
        raise ValueError("analysis must be dc, ac, or tran")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive (kelvin)")
    version_int = _VERSION_MAP.get(version)
    if version_int is None:
        raise ValueError(
            f"Unknown OSDI version '{version}'. Supported: {list(_VERSION_MAP)}"
        )

    if not os.path.exists(osdi_filepath):
        raise FileNotFoundError(f"OSDI binary not found at {osdi_filepath}")

    path = os.path.realpath(osdi_filepath)
    stat = os.stat(path)
    identity = (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    # Serialize misses as well as hits: lru_cache alone permits duplicate work
    # when two threads concurrently request the same uncached registration.
    with _REGISTRATION_LOCK:
        model = _load_registration(path, identity, version, temperature, analysis)
        # Metadata is mutable for compatibility. Never expose the cached object.
        return deepcopy(model)


@lru_cache(maxsize=128)
def _load_registration(
    path: str,
    identity: tuple[int, ...],
    version: str,
    temperature: float,
    analysis: str,
) -> OsdiModel:
    """Cache native registration IDs across descriptor and circuit lifetimes."""
    version_int = _VERSION_MAP[version]
    modes = {"dc": 0, "ac": 1, "tran": 2}
    meta = osdi_shim_nb.load_osdi_library(
        path, version_int, temperature, modes[analysis]
    )

    if not meta.success:
        detail = osdi_shim_nb.get_last_error()
        raise RuntimeError(
            f"Failed to load OSDI binary '{path}' as OSDI {version}. "
            f"{detail or 'Ensure it is a valid OpenVAF compiled .osdi file.'}"
        )

    mid = meta.model_id
    return OsdiModel(
        id=mid,
        temperature=temperature,
        analysis=analysis,
        path=path,
        num_pins=meta.num_pins,
        num_nodes=meta.num_nodes,
        num_params=meta.num_params,
        num_states=meta.num_states,
        osdi_version=version,
        resistive_mask=list(osdi_shim_nb.get_resistive_mask(mid)),
        resist_jac_pairs=list(osdi_shim_nb.get_resist_jac_pairs(mid)),
        react_jac_pairs=list(osdi_shim_nb.get_react_jac_pairs(mid)),
        collapsible_pairs=list(osdi_shim_nb.get_collapsible_pairs(mid)),
        param_flags=list(osdi_shim_nb.get_param_flags(mid)),
        param_names=list(osdi_shim_nb.get_param_names(mid)),
    )
