import os
from dataclasses import dataclass
import jax.numpy as jnp
import osdi_shim_nb  # Your statically-linked monolithic extension!

@dataclass
class OsdiModel:
    """A clean Python representation of the loaded Verilog-A model."""
    id: int
    num_pins: int
    num_params: int
    num_states: int
    
    def allocate_jax_buffers(self, num_devices: int):
        return {
            "voltages": jnp.zeros((num_devices, self.num_pins), dtype=jnp.float64),
            "params": jnp.zeros((num_devices, self.num_params), dtype=jnp.float64),
            "states": jnp.zeros((num_devices, self.num_states), dtype=jnp.float64)
        }

def load_osdi_model(osdi_filepath: str) -> OsdiModel:
    """
    Calls the C++ Nanobind module (which statically calls Rust) to open 
    a foundry .osdi file, cache its pointers, and return the array bounds.
    """
    if not os.path.exists(osdi_filepath):
        raise FileNotFoundError(f"OSDI binary not found at {osdi_filepath}")

    # Call the newly exposed Nanobind wrapper!
    meta = osdi_shim_nb.load_osdi_library(osdi_filepath)

    if not meta.success:
        raise RuntimeError(
            f"Failed to load OSDI binary or find evaluation symbols: {osdi_filepath}. "
            "Ensure it is a valid OpenVAF compiled .osdi file."
        )
    
    return OsdiModel(
        id=meta.model_id,
        num_pins=meta.num_pins,
        num_params=meta.num_params,
        num_states=meta.num_states
    )