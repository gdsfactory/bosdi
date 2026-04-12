import os
from dataclasses import dataclass
import jax.numpy as jnp
import osdi_shim_nb

_VERSION_MAP = {"0.4": 4, "0.5": 5}


@dataclass
class OsdiModel:
    """A Python representation of a loaded Verilog-A device model."""

    id: int
    num_pins: int  # = num_terminals (external pins only)
    num_nodes: int  # = num_terminals + num_non_collapsed_internal (voltage array width)
    num_params: int
    num_states: int
    osdi_version: str
    resistive_mask: list  # len == num_nodes; True iff G[i,:] can be non-zero at DC

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

    return OsdiModel(
        id=meta.model_id,
        num_pins=meta.num_pins,
        num_nodes=meta.num_nodes,
        num_params=meta.num_params,
        num_states=meta.num_states,
        osdi_version=version,
        resistive_mask=list(osdi_shim_nb.get_resistive_mask(meta.model_id)),
    )
