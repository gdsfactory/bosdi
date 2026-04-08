import os
import sys
import subprocess
import tomllib
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import nanobind

with open(os.path.join(os.path.dirname(__file__), "pyproject.toml"), "rb") as _f:
    _version = tomllib.load(_f)["project"]["version"]

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_include_dirs():
    dirs = []
    # ... (Keep the exact same get_include_dirs() function from before) ...
    try:
        import jax.ffi

        dirs.append(jax.ffi.include_dir())
    except ImportError:
        pass
    dirs.append(nanobind.include_dir())
    return dirs


class BuildExt(build_ext):
    def build_extension(self, ext):
        if ext.name == "osdi_shim_nb":
            manifest_path = os.path.join(_ROOT_DIR, "Cargo.toml")

            # 1. Build the Rust static library
            subprocess.check_call(
                ["cargo", "build", "--release", "--manifest-path", manifest_path]
            )

            # 2. Locate the output static library
            rust_target_dir = os.path.join(_ROOT_DIR, "target", "release")
            static_lib_name = "bosdi.lib" if sys.platform == "win32" else "libbosdi.a"
            static_lib_path = os.path.join(rust_target_dir, static_lib_name)

            # 3. Link the Rust library
            ext.extra_objects = [static_lib_path]
            if sys.platform != "win32":
                ext.extra_link_args = list(ext.extra_link_args or []) + [
                    "-lm",
                    "-ldl",
                    "-pthread",
                ]
                if sys.platform == "darwin":
                    ext.extra_link_args += ["-framework", "CoreFoundation"]

            # Touch the C++ source in the src/ directory
            cpp_src = os.path.join(_ROOT_DIR, "src", "osdi_shim.cpp")
            if os.path.exists(cpp_src):
                os.utime(cpp_src, None)

        super().build_extension(ext)


# Combine your shim with the Nanobind core logic
nanobind_src = os.path.join(
    os.path.dirname(nanobind.__file__), "src", "nb_combined.cpp"
)

osdi_extension = Extension(
    "osdi_shim_nb",
    sources=["src/osdi_shim.cpp", nanobind_src],  # Points to the src/ folder
    include_dirs=get_include_dirs(),
    extra_compile_args=["-std=c++17"] if sys.platform != "win32" else ["/std:c++17"],
)

setup(
    name="bosdi",
    version=_version,
    # --- THIS IS THE KEY CHANGE ---
    package_dir={"": "src"},
    py_modules=["osdi_loader", "osdi_jax"],
    ext_modules=[osdi_extension],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
