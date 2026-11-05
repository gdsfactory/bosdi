"""Check installed native wheels, optionally in a fresh virtual environment."""

import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-from", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    model = args.model.resolve() if args.model else None
    if args.install_from:
        wheels = list(args.install_from.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"Expected one wheel, found {len(wheels)}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(root)
            python = root / (
                "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
            )
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-index",
                    "--ignore-installed",
                    str(wheels[0].resolve()),
                ],
                check=True,
            )
            command = [str(python), str(Path(__file__).resolve())]
            if model:
                command += ["--model", str(model)]
            subprocess.run(command, cwd=root, check=True)
        return

    import bosdi
    from bosdi._version import __version__
    import osdi_jax
    from osdi_loader import load_osdi_model
    import osdi_shim_nb

    if __version__ != importlib.metadata.version("bosdi"):
        raise AssertionError("Installed package and generated versions differ")
    for module in (bosdi, osdi_jax, osdi_shim_nb):
        if (
            not Path(module.__file__)
            .resolve()
            .is_relative_to(Path(sys.prefix).resolve())
        ):
            raise AssertionError(
                f"Module came from outside the wheel environment: {module.__file__}"
            )
    if "circulax" in sys.modules:
        raise AssertionError("Native imports unexpectedly imported Circulax")
    # Exercise Rust registration and its error boundary even without a compiler.
    with tempfile.TemporaryDirectory() as directory:
        try:
            binary = Path(directory) / "invalid.osdi"
            binary.write_bytes(b"invalid shared library")
            load_osdi_model(str(binary))
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("Invalid model registration should fail")
    if model:
        import jax
        import jax.numpy as jnp
        import numpy as np

        jax.config.update("jax_enable_x64", True)
        for analysis in ("dc", "ac", "tran"):
            device = load_osdi_model(
                str(model),
                analysis=analysis,
                temperature=310,
                simparams={"gmin": 1e-12},
            )
            params = jnp.full((1, device.num_params), jnp.nan)
            params = params.at[0, device.param_names.index("R")].set(50)
            voltage = jnp.array([[1.0, 0.0]])
            states = jnp.empty((1, device.num_states))
            handle = osdi_jax.osdi_setup_batch(device.id, params)
            try:
                for evaluate in (
                    lambda v: osdi_jax.osdi_eval(device.id, v, params, states),
                    lambda v: osdi_jax.osdi_eval_with_handle(handle, v, states),
                ):
                    current, conductance, charge, capacitance, _ = jax.jit(evaluate)(
                        voltage
                    )
                    np.testing.assert_allclose(current, [[0.02, -0.02]])
                    np.testing.assert_allclose(
                        conductance.reshape(2, 2), [[0.02, -0.02], [-0.02, 0.02]]
                    )
                    np.testing.assert_allclose(charge, 0)
                    np.testing.assert_allclose(capacitance, 0)
            finally:
                handle.free()
    print(f"Installed bosdi {__version__} native wheel smoke passed")


if __name__ == "__main__":
    main()
