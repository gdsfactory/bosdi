"""Shared openvaf-r capability probes for the Verilog-A test modules.

``--dump-json`` and the ``--dump-unopt-json*`` variants only exist on the
custom OpenVAF fork.  CI installs a stock ``openvaf-reloaded`` build on
Windows, where those flags are absent and the binary writes non-JSON to
stdout, so every test that lowers a ``.va`` has to be gated on the flag
actually working rather than merely being installed.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

CAPACITOR_VA = pathlib.Path(__file__).parent / "devices" / "capacitor_va.va"

_openvaf_available = shutil.which("openvaf-r") is not None

OPENVAF_MISSING = pytest.mark.skipif(
    not _openvaf_available,
    reason="openvaf-r not in PATH",
)


def _has_flag(flag: str) -> bool:
    if not _openvaf_available:
        return False
    result = subprocess.run(["openvaf-r", "--help"], capture_output=True, text=True)
    return flag in result.stdout or flag in result.stderr


def _dump_json_works() -> bool:
    """Check that --dump-json actually produces output (not just listed in help)."""
    if not _has_flag("--dump-json"):
        return False
    result = subprocess.run(
        ["openvaf-r", "--dump-json", str(CAPACITOR_VA)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and len(result.stdout.strip()) > 2


DUMP_JSON_MISSING = pytest.mark.skipif(
    not _dump_json_works(),
    reason="openvaf-r --dump-json not functional (custom fork needed)",
)
UNOPT_JSON_MISSING = pytest.mark.skipif(
    not _has_flag("--dump-unopt-json"),
    reason="openvaf-r does not support --dump-unopt-json (rebuild needed)",
)
UNOPT_JSON_SPLIT_MISSING = pytest.mark.skipif(
    not _has_flag("--dump-unopt-json-with-split"),
    reason="openvaf-r does not support --dump-unopt-json-with-split (rebuild needed)",
)
