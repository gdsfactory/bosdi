"""Compile Verilog-A sources to .osdi before test collection."""

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEVICES_DIR = Path(__file__).parent / "devices"
COMPILED_DIR = Path(__file__).parent / "compiled_osdi"

# (va_path relative to DEVICES_DIR, output .osdi name, extra_flags)
COMPILE_MANIFEST = [
    # Simple test models (for test_osdi.py)
    ("resistor_va.va", "resistor_va.osdi", ()),
    ("capacitor_va.va", "capacitor_va.osdi", ()),
    # Root-level vacask models
    ("resistor.va", "resistor.osdi", ()),
    ("capacitor.va", "capacitor.osdi", ()),
    ("inductor.va", "inductor.osdi", ()),
    ("diode.va", "diode.osdi", ()),
    ("opamp.va", "opamp.osdi", ()),
    ("bsim3v3.va", "bsim3v3.osdi", ()),
    ("bsim4v8.va", "bsim4v8.osdi", ()),
    ("bsimbulk106.va", "bsimbulk106.osdi", ()),
    # VBIC
    ("vbic/vbic_1p3.va", "vbic_vbic_1p3.osdi", ("-D__NGSPICE__",)),
    ("vbic/vbic_4T_et_cf.va", "vbic_vbic_4T_et_cf.osdi", ()),
    ("vbic/cmcGeneralMacrosAndDefines.va", "vbic_cmcGeneralMacrosAndDefines.osdi", ()),
    ("vbic/cmcStandardModelMacros.va", "vbic_cmcStandardModelMacros.osdi", ()),
    # PSP103
    ("psp103v4/psp103.va", "psp103v4_psp103.osdi", ()),
    ("psp103v4/psp103t.va", "psp103v4_psp103t.osdi", ()),
    ("psp103v4/psp103_nqs.va", "psp103v4_psp103_nqs.osdi", ()),
    ("psp103v4/juncap200.va", "psp103v4_juncap200.osdi", ()),
    # SPICE wrappers
    ("spice/resistor.va", "spice_resistor.osdi", ()),
    ("spice/capacitor.va", "spice_capacitor.osdi", ()),
    ("spice/inductor.va", "spice_inductor.osdi", ()),
    ("spice/diode.va", "spice_diode.osdi", ()),
    ("spice/bjt.va", "spice_bjt.osdi", ()),
    ("spice/jfet1.va", "spice_jfet1.osdi", ()),
    ("spice/jfet2.va", "spice_jfet2.osdi", ()),
    ("spice/mes1.va", "spice_mes1.osdi", ()),
    ("spice/mos1.va", "spice_mos1.osdi", ()),
    ("spice/mos2.va", "spice_mos2.osdi", ()),
    ("spice/mos3.va", "spice_mos3.osdi", ()),
    ("spice/mos6.va", "spice_mos6.osdi", ()),
    ("spice/mos9.va", "spice_mos9.osdi", ()),
    ("spice/vdmos.va", "spice_vdmos.osdi", ()),
    ("spice/bsim3v3.va", "spice_bsim3v3.osdi", ()),
    ("spice/bsim4v8.va", "spice_bsim4v8.osdi", ()),
    # SPICE signal-node variants
    ("spice/sn/bjt.va", "spice_sn_bjt.osdi", ()),
    ("spice/sn/diode.va", "spice_sn_diode.osdi", ()),
    ("spice/sn/jfet1.va", "spice_sn_jfet1.osdi", ()),
    ("spice/sn/jfet2.va", "spice_sn_jfet2.osdi", ()),
    ("spice/sn/mes1.va", "spice_sn_mes1.osdi", ()),
    ("spice/sn/mos1.va", "spice_sn_mos1.osdi", ()),
    ("spice/sn/mos2.va", "spice_sn_mos2.osdi", ()),
    ("spice/sn/mos3.va", "spice_sn_mos3.osdi", ()),
    ("spice/sn/mos6.va", "spice_sn_mos6.osdi", ()),
    ("spice/sn/mos9.va", "spice_sn_mos9.osdi", ()),
    # SPICE full variants
    ("spice/full/bjt.va", "spice_full_bjt.osdi", ()),
    ("spice/full/bsim3v3.va", "spice_full_bsim3v3.osdi", ()),
    ("spice/full/capacitor.va", "spice_full_capacitor.osdi", ()),
    ("spice/full/diode.va", "spice_full_diode.osdi", ()),
    ("spice/full/inductor.va", "spice_full_inductor.osdi", ()),
    ("spice/full/jfet1.va", "spice_full_jfet1.osdi", ()),
    ("spice/full/jfet2.va", "spice_full_jfet2.osdi", ()),
    ("spice/full/mes1.va", "spice_full_mes1.osdi", ()),
    ("spice/full/mos1.va", "spice_full_mos1.osdi", ()),
    ("spice/full/mos2.va", "spice_full_mos2.osdi", ()),
    ("spice/full/mos3.va", "spice_full_mos3.osdi", ()),
    ("spice/full/mos6.va", "spice_full_mos6.osdi", ()),
    ("spice/full/mos9.va", "spice_full_mos9.osdi", ()),
    ("spice/full/vdmos.va", "spice_full_vdmos.osdi", ()),
]


def _compile_one(openvaf: str, va_rel: str, osdi_name: str, extra_flags: tuple) -> bool:
    va_file = DEVICES_DIR / va_rel
    osdi_file = COMPILED_DIR / osdi_name

    if osdi_file.exists() and osdi_file.stat().st_mtime > va_file.stat().st_mtime:
        return True

    cmd = [
        openvaf,
        "--allow",
        "variant_const_simparam",
        f"-I{DEVICES_DIR}",
        f"-I{va_file.parent}",
        *extra_flags,
        str(va_file),
        "-o",
        str(osdi_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        logger.warning("openvaf-r failed for %s: %s", va_rel, result.stderr[:500])
        return False
    return True


def pytest_configure(config):
    openvaf = shutil.which("openvaf-r")
    if openvaf is None:
        logger.warning("openvaf-r not found — .osdi files will not be compiled")
        return

    COMPILED_DIR.mkdir(exist_ok=True)

    failed = []
    for va_rel, osdi_name, extra_flags in COMPILE_MANIFEST:
        if not _compile_one(openvaf, va_rel, osdi_name, extra_flags):
            failed.append(osdi_name)

    if failed:
        logger.warning("Failed to compile %d models: %s", len(failed), failed)
