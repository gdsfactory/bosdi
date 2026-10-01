"""Native descriptors must remain usable without the host circulax package."""

import os
from pathlib import Path
import subprocess
import sys


def test_native_import_does_not_require_circulax():
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys

class BlockCirculax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "circulax" or fullname.startswith("circulax."):
            raise ModuleNotFoundError("circulax deliberately unavailable", name=fullname)

sys.meta_path.insert(0, BlockCirculax())
from bosdi.circulax import osdi_component, OsdiModelDescriptor, OsdiComponentGroup
assert callable(osdi_component)
assert "bosdi.circulax.va_component" not in sys.modules
try:
    from bosdi.circulax import va_component
except ModuleNotFoundError as exc:
    assert exc.name.startswith("circulax")
else:
    raise AssertionError("The VA decorator should require circulax when requested")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(source), env.get("PYTHONPATH", "")])
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
