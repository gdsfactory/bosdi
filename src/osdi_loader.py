import os
from dataclasses import dataclass, field
import jax.numpy as jnp
import osdi_shim_nb

_VERSION_MAP = {"0.4": 4, "0.5": 5}

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
    num_nodes: (
        int  # = num_terminals + num_non_collapsed_internal + branch-current auxiliaries
    )
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


def load_osdi_model(osdi_filepath: str, version: str = "0.4") -> OsdiModel:
    """
    Load an OpenVAF-compiled .osdi binary and register it for JAX evaluation.

    Args:
        osdi_filepath: Path to the .osdi ELF binary.
        version:       OSDI standard version to use ("0.4" or "0.5").
    """
    version_int = _VERSION_MAP.get(version)
    if version_int is None:
        raise ValueError(
            f"Unknown OSDI version '{version}'. Supported: {list(_VERSION_MAP)}"
        )

    if not os.path.exists(osdi_filepath):
        raise FileNotFoundError(f"OSDI binary not found at {osdi_filepath}")

    meta = osdi_shim_nb.load_osdi_library(osdi_filepath, version_int)

    if not meta.success:
        raise RuntimeError(
            f"Failed to load OSDI binary '{osdi_filepath}' as OSDI {version}. "
            "Ensure it is a valid OpenVAF compiled .osdi file."
        )

    mid = meta.model_id
    return OsdiModel(
        id=mid,
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
    )
