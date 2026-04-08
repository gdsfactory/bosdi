import os
import sys
import shutil
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

    # JAX/XLA headers — try jax.ffi.include_dir() first, fall back to jaxlib path
    found_xla = False
    try:
        import jax.ffi

        jax_inc = jax.ffi.include_dir()
        if os.path.exists(os.path.join(jax_inc, "xla", "ffi", "api", "ffi.h")):
            dirs.append(jax_inc)
            found_xla = True
    except (ImportError, AttributeError):
        pass

    if not found_xla:
        candidates = []
        try:
            import jaxlib

            jaxlib_dir = os.path.dirname(os.path.abspath(jaxlib.__file__))
            candidates += [os.path.join(jaxlib_dir, "include"), jaxlib_dir]
        except ImportError:
            pass
        candidates += [os.path.join(sys.prefix, "include"), os.getcwd()]
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "xla", "ffi", "api", "ffi.h")):
                dirs.append(cand)
                found_xla = True
                break

    if not found_xla:
        print("WARNING: could not find xla/ffi/api/ffi.h — build will likely fail")

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


# Copy the entire nanobind src/ directory into the build tree.
# nb_combined.cpp #includes sibling .cpp files, so the whole directory is needed.
# A relative path is also required — absolute paths are rejected when building from an sdist.
_nb_src_dir = os.path.join(_ROOT_DIR, "_nanobind_src")
if os.path.exists(_nb_src_dir):
    shutil.rmtree(_nb_src_dir)
shutil.copytree(
    os.path.join(os.path.dirname(nanobind.__file__), "src"),
    _nb_src_dir,
)
nanobind_src = "_nanobind_src/nb_combined.cpp"

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
