"""CLI entry point: ``python -m bosdi.va <path/to/device.va>``.

Compiles the Verilog-A source through ``openvaf_py`` (or text MIR via
``openvaf-r --dump-mir``), lowers each module into a circulax-compatible
component, and writes a ``.py`` file next to the input (or to
``--out PATH`` if given).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .binding import compile_va as _binding_compile_va
from .dump_parser import parse_dump
from .emitter import write_source
from .lowering import lower
from .va_defaults import ParamSpec, parse_va_defaults, parse_va_defaults_expanded


def _run_dump_mir(va_path: Path) -> str:
    """Shell out to ``openvaf-r --dump-mir``; stream its output back as a string.

    OpenVAF also emits a ``.osdi`` binary as a side effect of ``--dump-mir``;
    we don't clean it up here since the caller may have other reasons to
    want it. The generator itself doesn't need it.
    """
    try:
        completed = subprocess.run(  # noqa: S603
            ["openvaf-r", "--dump-mir", str(va_path)],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        msg = "openvaf-r not found on PATH — install it or capture the dump manually and pipe it in"
        raise SystemExit(msg) from None
    except subprocess.CalledProcessError as exc:
        msg = f"openvaf-r failed:\n{exc.stderr}"
        raise SystemExit(msg) from exc
    return completed.stdout


def main(argv: list[str] | None = None) -> int:
    """Run ``python -m circulax.va`` against ``argv`` (defaults to ``sys.argv``)."""
    parser = argparse.ArgumentParser(prog="python -m circulax.va", description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Path to the .va file (or a captured .mir.txt dump)"
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output .py path. Defaults to ``<source>.py`` next to the input.",
    )
    parser.add_argument(
        "--no-format",
        action="store_true",
        help="Skip the ``ruff format`` post-pass on the emitted file.",
    )
    parser.add_argument(
        "--use-text-parser",
        action="store_true",
        help="Force the legacy ``--dump-mir`` text parser even for ``.va`` "
        "inputs.  By default we go through the ``openvaf_py`` PyO3 binding, "
        "which skips the subprocess and handles models (e.g. BSIM4) that "
        "the text parser can't lower.",
    )
    args = parser.parse_args(argv)

    source: Path = args.source
    va_defaults: dict[str, ParamSpec] = {}
    if source.suffix == ".va":
        # Run through the preprocessor so macro-referenced defaults
        # (e.g. BSIM4's ``parameter integer verbose = `INT_NOT_GIVEN``)
        # resolve before we regex them out.
        va_defaults = parse_va_defaults_expanded(source)
        out = args.out or source.with_suffix(".py")
        if args.use_text_parser:
            dump = parse_dump(_run_dump_mir(source))
        else:
            dump = _binding_compile_va(str(source))
    else:
        # Allow feeding a pre-captured dump (useful for tests). If there's a
        # sibling ``.va`` next to the dump, harvest defaults from it.
        text = source.read_text()
        sibling_va = source.with_suffix("").with_suffix(".va")
        if sibling_va.exists():
            va_defaults = parse_va_defaults(sibling_va.read_text())
        out = args.out or source.with_suffix(".py")
        dump = parse_dump(text)

    devices = [lower(m, va_defaults=va_defaults) for m in dump.modules]
    if not devices:
        sys.stderr.write(f"No modules found in {source}\n")
        return 1

    written = write_source(devices, out, run_ruff_format=not args.no_format)
    sys.stderr.write(f"wrote {written}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
