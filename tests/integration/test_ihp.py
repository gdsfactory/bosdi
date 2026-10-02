"""Execute every shipped IHP SG13G2 device card through Circulax and bosdi."""

from dataclasses import replace
import json
from pathlib import Path
import re
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("circulax")
vacask_bin = pytest.importorskip("vacask_bin")
from circulax.netlist_io import Library  # noqa: E402 — optional test dependencies
from circulax.netlist_io.osdi import compile_va  # noqa: E402

PDK = Path(__file__).parents[1] / "pdks" / "ihp"
MODELS = PDK / "ihp" / "models" / "vacask" / "models"

# Every subcircuit in the pinned PDK has an explicit, nonzero bias and sizing.
# The catalogue test prevents a newly shipped model from silently going untested.
FAMILIES = [
    (
        "cornerMOSlv.lib",
        "mos_tt",
        "sg13g2_moslv_mod.lib",
        {
            "sg13_lv_nmos": ((0.8, 0.7, 0, 0), "w=1u l=0.13u"),
            "sg13_lv_pmos": ((-0.8, -0.7, 0, 0), "w=2u l=0.13u"),
            "nmoscl_2": ((0.2, 0), ""),
            "nmoscl_4": ((0.2, 0), ""),
        },
    ),
    (
        "cornerMOShv.lib",
        "mos_tt",
        "sg13g2_moshv_mod.lib",
        {
            "sg13_hv_nmos": ((2, 2, 0, 0), "w=2u l=0.45u"),
            "sg13_hv_pmos": ((-2, -2, 0, 0), "w=2u l=0.45u"),
        },
    ),
    (
        "cornerHBT.lib",
        "hbt_typ",
        "sg13g2_hbt_mod.lib",
        {
            **{
                name: ((1.2, 0.7, 0, 0), "nx=1 selft=0")
                for name in ("npn13G2", "npn13G2l", "npn13G2v")
            },
            **{
                name: ((1.2, 0.7, 0, 0, 0), "nx=1 selft=0")
                for name in ("npn13G2_5t", "npn13G2l_5t", "npn13G2v_5t")
            },
            "pnpMPA": ((-0.8, -0.6, 0), ""),
        },
    ),
    (
        "cornerRES.lib",
        "res_typ",
        "resistors_mod.lib",
        {
            "ptap1": ((0.2, 0), "r=100"),
            "ntap1": ((0.2, 0), "r=100"),
            "Rparasitic": ((0.2, 0), "r=100"),
            **{name: ((0.2, 0, 0), "w=1u l=5u") for name in ("rsil", "rhigh", "rppd")},
        },
    ),
    (
        "cornerCAP.lib",
        "cap_typ",
        "capacitors_mod.lib",
        {
            "cparasitic": ((0.2, 0), "c=1p"),
            "cap_cmim": ((0.2, 0), "w=10u l=10u"),
            "cap_rfcmim": ((0.2, 0, 0), "w=10u l=10u"),
        },
    ),
    (
        "cornerDIO.lib",
        "dio_tt",
        "diodes.lib",
        {
            "dantenna": ((0.2, 0), ""),
            "dpantenna": ((0.2, 0), ""),
            "isolbox": ((0.2, 0, 0), "xd=1u"),
            "dpwdnw": ((0.2, 0), ""),
            "ddnwpsub": ((0.2, 0), "xd=1u"),
        },
    ),
    (
        "cornerDIO.lib",
        "dio_tt",
        "sg13g2_esd.lib",
        {
            name: ((1.2, 0.6, 0), "")
            for name in ("diodevdd_2kv", "diodevdd_4kv", "diodevss_2kv", "diodevss_4kv")
        },
    ),
    (
        "cornerDIO.lib",
        "dio_tt",
        "sg13g2_dschottky_nbl1_mod.lib",
        {
            "schottky_nbl1": ((0.2, 0, 0), ""),
        },
    ),
    (
        "cornerMOShv.lib",
        "mos_tt",
        "sg13g2_svaricaphv_mod.lib",
        {
            "sg13_hv_svaricap": ((0.3, 0, 0.3, 0), "w=4u l=600n"),
        },
    ),
    (
        "sg13g2_bondpad.lib",
        None,
        "sg13g2_bondpad.lib",
        {
            "bondpad": ((0.2,), ""),
        },
    ),
]
CASES = [
    pytest.param(library, section, name, bias, settings, id=name)
    for library, section, _, devices in FAMILIES
    for name, (bias, settings) in devices.items()
]


@pytest.fixture(scope="module")
def compiler():
    assert MODELS.is_dir(), "Run git submodule update --init tests/pdks/ihp"
    executable = shutil.which("openvaf-r")
    assert executable, "IHP integration tests require openvaf-r on PATH"
    return executable


def test_every_shipped_device_has_a_case(compiler):
    expected = {
        (path.name, name)
        for path in MODELS.glob("*.lib")
        for name in re.findall(r"^subckt\s+(\w+)\s*\(", path.read_text(), re.MULTILINE)
    }
    covered = {(source, name) for _, _, source, devices in FAMILIES for name in devices}
    assert covered == expected


@pytest.fixture(scope="module")
def libraries(compiler, tmp_path_factory):
    """Materialize include-once libraries without touching the pinned checkout.

    Upstream converted cards each include the same common parameter declaration;
    VACASK 0.3.3 rejects the repeated definition. Hoist that common include once.
    Fold the varactor's three zero voltage coefficients: neither
    pinned card parser supports runtime v() in resistor parameters. The fixture
    checks that this simplification preserves the selected model coefficients.
    isolbox forwards derived parameters that VACASK considers private; its child
    cards already compute those identical values from l, w, xd, mf and scaling.
    """
    root = tmp_path_factory.mktemp("ihp-libraries")
    for source in MODELS.glob("*.lib"):
        text = source.read_text().replace('include "sg13g2_vacask_common.lib"', "")
        if source.name == "sg13g2_vacask_common.lib":

            def absolute_load(match):
                reference = match[1]
                path = (
                    MODELS / reference
                    if reference.endswith(".va")
                    else Path(vacask_bin.MOD_DIR) / reference
                )
                return f'load "{path.resolve().as_posix()}"'

            text = re.sub(r'load "([^"]+)"', absolute_load, text)
        if source.name == "diodes.lib":
            # Keep public geometry overrides. The children recompute these
            # private derived values with the same formulas as isolbox.
            for forwarding in (
                "l=l w=w aw=aw pw=pw xd=xd mf=mf scaling=scaling xds=xds pws=pws aws=aws",
                "l=l w=w ab=ab pb=pb xd=xd mf=mf scaling=scaling xds=xds pbs=pbs abus=abus",
            ):
                assert text.count(forwarding) == 1
                text = text.replace(forwarding, "l=l w=w xd=xd mf=mf scaling=scaling")
        if source.name == "sg13g2_svaricaphv_mod.lib":
            for coefficient in ("rwellvw", "rwellwvw", "rwellnxvw"):
                assert re.search(rf"^parameters {coefficient}=0$", text, re.MULTILINE)
            text = text.replace("v(2,4)", "0")
        (root / source.name).write_text(text)
    return root


def read_ascii_raw(path):
    """Read the simulator's real/complex ASCII vectors without another dependency."""
    header, values = path.read_text().split("Values:\n", 1)
    names = [
        line.split()[1]
        for line in header.split("Variables:\n", 1)[1].splitlines()
        if line.strip()
    ]
    tokens = values.split()
    width = len(names) + 1  # Each sample starts with its integer index.
    assert len(tokens) % width == 0
    samples = np.asarray(
        [
            [
                complex(*map(float, token.split(",")))
                if "," in token
                else complex(token)
                for token in tokens[start + 1 : start + width]
            ]
            for start in range(0, len(tokens), width)
        ]
    )
    return dict(zip(names, samples.T, strict=True))


def native_reference(deck, resolved, compiler, directory):
    """Let VACASK independently elaborate and solve the shared model cards."""
    for reference, base in resolved.loads:
        source = (base / reference).resolve()
        if source.suffix == ".va":
            shutil.copyfile(
                compile_va(source, compiler=compiler),
                directory / (source.stem + ".osdi"),
            )
    config = "[Paths]\nmodule_path_prefix=" + json.dumps([vacask_bin.MOD_DIR])
    config += "\n[Binaries]\nopenvaf=" + json.dumps(compiler) + "\n"
    (directory / ".vacaskrc.toml").write_text(config)
    # Inject -1 A into the output: V(out) is its small-signal driving-point
    # impedance. Convert it to the same 50-ohm reflection measured by sp().
    reference = "IHP device regression\nground 0\n" + deck
    reference += "\nmodel probe isource\niprobe (out 0) probe dc=0 mag=-1\n"
    reference += '\ncontrol\nabort always\noptions temp=27 rawfile="ascii" reltol=1e-8 vntol=1e-10 abstol=1e-14\n'
    reference += (
        'analysis op1 op\nanalysis ac1 ac from=1e6 to=1e9 mode="lin" points=1\nendc\n'
    )
    (directory / "reference.sim").write_text(reference)
    result = subprocess.run(
        [vacask_bin.VACASK_CMD, "reference.sim"],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    dc = read_ascii_raw(directory / "op1.raw")["out"].real[0]
    ac = read_ascii_raw(directory / "ac1.raw")
    np.testing.assert_allclose(ac["frequency"].real, [1e6, 1e9])
    impedance = ac["out"]
    return dc, (impedance - 50) / (impedance + 50)


@pytest.mark.parametrize("library,section,name,bias,settings", CASES)
def test_device_dc_and_ac(
    library, section, name, bias, settings, compiler, libraries, tmp_path
):
    include = f'include "{(libraries / library).as_posix()}"'
    if section:
        include += f" section={section}"
    # A finite drive resistance leaves the device port free for DC and AC.
    # Other terminals receive fixed biases; thermal/substrate ports are grounded.
    deck = f'include "{(libraries / "sg13g2_vacask_common.lib").as_posix()}"\n'
    deck += include + "\nmodel vs vsource\nmodel drive_r sp_resistor\n"
    deck += f"drive (input 0) vs dc={bias[0]}\nrdrive (input out) drive_r r=1000\n"
    nodes = ["out"]
    for index, voltage in enumerate(bias[1:], 1):
        if voltage == 0:
            nodes.append("0")
        else:
            node = f"bias{index}"
            nodes.append(node)
            deck += f"v{index} ({node} 0) vs dc={voltage}\n"
    deck += f"dut ({' '.join(nodes)}) {name} {settings}\n"
    path = tmp_path / "device.lib"
    path.write_text(deck)
    resolved = Library.from_file(path).resolve()
    # SPICE permits a three-terminal BJT with an implicit grounded substrate.
    # The pinned Circulax adapter requires explicit terminal counts. Make that
    # simulator convention explicit in the native harness, without rewriting
    # upstream libraries or altering the independent VACASK reference deck.
    resolved.instances = [
        replace(leaf, nodes=(*leaf.nodes, "0"))
        if leaf.module == "sp_bjt" and len(leaf.nodes) == 3
        else leaf
        for leaf in resolved.instances
    ]
    if name == "bondpad":
        # The upstream pad is explicitly an empty placeholder, not a capacitor.
        assert all(
            leaf.module in {"vsource", "sp_resistor"} for leaf in resolved.instances
        )
    else:
        assert any(leaf.name.startswith("dut/") for leaf in resolved.instances)
    circuit = resolved.compile(
        compiler=compiler,
        module_paths=(Path(vacask_bin.MOD_DIR),),
        state_policy="limiting_only",
    )
    dc = jax.jit(circuit.dc)()
    assert np.all(np.isfinite(dc)), name
    voltage = float(circuit.port(dc, "out"))
    if name in {"ptap1", "ntap1", "Rparasitic"}:
        assert voltage == pytest.approx(0.2 * 100 / 1100, rel=1e-7)
    if name in {"cparasitic", "cap_cmim", "cap_rfcmim", "bondpad"}:
        assert voltage == pytest.approx(bias[0], rel=1e-6)
    frequencies = jnp.array([1e6, 1e9])
    response = jax.jit(lambda f: circuit.sp(ports="out", freqs=f))(frequencies)
    assert response.shape == (2, 1, 1)
    assert np.all(np.isfinite(response)), name
    reference_dc, reference_sp = native_reference(deck, resolved, compiler, tmp_path)
    np.testing.assert_allclose(voltage, reference_dc, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(response[:, 0, 0], reference_sp, rtol=1e-5, atol=1e-8)
    if name in {"cparasitic", "cap_cmim", "cap_rfcmim"}:
        assert abs(complex(response[1, 0, 0]) - complex(response[0, 0, 0])) > 1e-4
