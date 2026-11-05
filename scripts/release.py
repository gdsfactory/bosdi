"""Validate release tags and distribution metadata before publishing."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import re
import tarfile
import zipfile

from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


def release_version(tag: str) -> Version:
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:a\d+|b\d+|rc\d+|\.dev\d+)?", tag):
        raise ValueError(
            f"Invalid release tag: {tag!r}; use vX.Y.Z, optionally aN/bN/rcN/.devN"
        )
    version = Version(tag[1:])
    if str(version) != tag[1:]:
        raise ValueError(f"Release tag must use the canonical version: v{version}")
    return version


def validate_dist(directory: Path, version: Version) -> None:
    files = sorted(directory.iterdir())
    if not any(p.suffix == ".whl" for p in files) or not any(
        p.name.endswith(".tar.gz") for p in files
    ):
        raise ValueError("Release must contain both wheels and an sdist")
    for path in files:
        if path.suffix == ".whl":
            name, file_version, _, _ = parse_wheel_filename(path.name)
            with zipfile.ZipFile(path) as archive:
                entries = [
                    n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
                ]
                if len(entries) != 1:
                    raise ValueError(f"Expected one METADATA file in {path.name}")
                metadata = archive.read(entries[0])
        elif path.name.endswith(".tar.gz"):
            name, file_version = parse_sdist_filename(path.name)
            with tarfile.open(path) as archive:
                for member in archive.getmembers():
                    relative = member.name.partition("/")[2]
                    if relative.startswith(("tests/compiled_osdi/", "tests/pdks/")):
                        raise ValueError(
                            f"Generated or external test artifact in {path.name}: {relative}"
                        )
                entries = [
                    m
                    for m in archive.getmembers()
                    if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")
                ]
                if len(entries) != 1:
                    raise ValueError(f"Expected one root PKG-INFO file in {path.name}")
                with archive.extractfile(entries[0]) as stream:
                    metadata = stream.read()
        else:
            raise ValueError(f"Unexpected release artifact: {path.name}")
        headers = BytesParser().parsebytes(metadata)
        if name != "bosdi" or headers["Name"] != "bosdi":
            raise ValueError(f"Unexpected package name in {path.name}")
        if file_version != version or Version(headers["Version"]) != version:
            raise ValueError(f"{path.name} does not match release version {version}")
        print(f"Validated {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--dist", type=Path)
    args = parser.parse_args()
    version = release_version(args.tag)
    if args.dist:
        validate_dist(args.dist, version)
    else:
        # Suitable for appending to GITHUB_OUTPUT in the tag validation job.
        print(f"version={version}")
        print(
            f"prerelease={str(version.is_prerelease or version.is_devrelease).lower()}"
        )


if __name__ == "__main__":
    main()
