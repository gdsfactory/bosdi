"""Guard against publishing stale versions under a new release tag."""

import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest
from packaging.version import Version

spec = importlib.util.spec_from_file_location(
    "release", Path(__file__).parents[1] / "scripts" / "release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.mark.parametrize("suffix", ["", "a1", "b2", "rc3", ".dev4"])
def test_release_tags(suffix):
    version = release.release_version(f"v0.1.8{suffix}")
    assert version == Version(f"0.1.8{suffix}")
    assert version.is_prerelease == bool(suffix)


@pytest.mark.parametrize(
    "tag", ["0.1.8", "v0.1", "v0.1.8rc", "v0.1.8alpha1", "v0.01.8", "v0.1.8+local"]
)
def test_invalid_release_tags(tag):
    with pytest.raises(ValueError):
        release.release_version(tag)


def distributions(directory, version="0.1.8", metadata_version=None):
    metadata = f"Metadata-Version: 2.1\nName: bosdi\nVersion: {metadata_version or version}\n".encode()
    with zipfile.ZipFile(
        directory / f"bosdi-{version}-cp313-cp313-linux_x86_64.whl", "w"
    ) as archive:
        archive.writestr(f"bosdi-{version}.dist-info/METADATA", metadata)
    with tarfile.open(directory / f"bosdi-{version}.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo(f"bosdi-{version}/PKG-INFO")
        member.size = len(metadata)
        archive.addfile(member, io.BytesIO(metadata))


def test_matching_distributions(tmp_path):
    distributions(tmp_path)
    release.validate_dist(tmp_path, Version("0.1.8"))


def test_previous_release_distributions_rejected(tmp_path):
    distributions(tmp_path, version="0.1.6")
    with pytest.raises(ValueError, match="does not match"):
        release.validate_dist(tmp_path, Version("0.1.7"))


def test_renamed_stale_distributions_rejected(tmp_path):
    distributions(tmp_path, metadata_version="0.1.6")
    with pytest.raises(ValueError, match="does not match"):
        release.validate_dist(tmp_path, Version("0.1.8"))


def test_missing_sdist_rejected(tmp_path):
    distributions(tmp_path)
    next(tmp_path.glob("*.tar.gz")).unlink()
    with pytest.raises(ValueError, match="both wheels and an sdist"):
        release.validate_dist(tmp_path, Version("0.1.8"))
