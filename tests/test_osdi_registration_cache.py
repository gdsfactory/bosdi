"""Native registrations are reused across instances without sharing mutable metadata."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

import osdi_loader


@pytest.fixture
def registration(tmp_path, monkeypatch):
    path = tmp_path / "model.osdi"
    path.write_bytes(b"fake binary")
    calls = []

    def load(*args):
        calls.append(args)
        return SimpleNamespace(
            success=True,
            model_id=len(calls),
            num_pins=2,
            num_nodes=2,
            num_params=1,
            num_states=0,
        )

    osdi_loader._load_registration.cache_clear()
    monkeypatch.setattr(osdi_loader.osdi_shim_nb, "load_osdi_library", load)
    for name in [
        "get_resistive_mask",
        "get_resist_jac_pairs",
        "get_react_jac_pairs",
        "get_collapsible_pairs",
        "get_param_flags",
    ]:
        monkeypatch.setattr(osdi_loader.osdi_shim_nb, name, lambda mid: [0])
    monkeypatch.setattr(osdi_loader.osdi_shim_nb, "get_param_names", lambda mid: ["r"])
    yield path, calls
    osdi_loader._load_registration.cache_clear()


def test_cross_instance_reuse_and_metadata_isolation(registration):
    path, calls = registration
    first = osdi_loader.load_osdi_model(str(path))
    first.param_names.append("edited")
    first.resist_jac_pairs.append((9, 9))
    second = osdi_loader.load_osdi_model(str(path.parent / "." / path.name))
    assert first is not second
    assert first.id == second.id
    assert second.param_names == ["r"]
    assert second.resist_jac_pairs == [0]
    assert len(calls) == 1


def test_contexts_and_changed_binaries_are_separate(registration):
    path, calls = registration
    original = osdi_loader.load_osdi_model(str(path))
    variants = [
        osdi_loader.load_osdi_model(str(path), **kwargs)
        for kwargs in [
            {"temperature": 310},
            {"analysis": "dc"},
            {"analysis": "tran"},
            {"version": "0.5"},
        ]
    ]
    assert len({original.id, *(model.id for model in variants)}) == 5
    path.write_bytes(b"changed fake binary")
    changed = osdi_loader.load_osdi_model(str(path))
    assert changed.id != original.id
    assert len(calls) == 6
    path.unlink()
    with pytest.raises(FileNotFoundError):
        osdi_loader.load_osdi_model(str(path))


def test_registration_lru_is_bounded(registration):
    path, calls = registration
    original = osdi_loader.load_osdi_model(str(path))
    for temperature in range(301, 429):
        osdi_loader.load_osdi_model(str(path), temperature=temperature)
    assert osdi_loader._load_registration.cache_info().currsize == 128
    assert osdi_loader.load_osdi_model(str(path)).id != original.id
    assert len(calls) == 130
    # Python eviction cannot invalidate IDs already held by callers/JIT graphs.
    assert original.id == 1


def test_concurrent_instances_share_registration(registration):
    path, calls = registration
    barrier = Barrier(8)

    def load(index):
        barrier.wait()
        return osdi_loader.load_osdi_model(str(path)).id

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(load, range(8)))
    assert len(set(ids)) == 1
    assert len(calls) == 1


def test_failed_registration_is_not_cached(registration, monkeypatch):
    path, calls = registration
    load = osdi_loader.osdi_shim_nb.load_osdi_library

    def fail_once(*args):
        result = load(*args)
        result.success = len(calls) != 1
        return result

    monkeypatch.setattr(osdi_loader.osdi_shim_nb, "load_osdi_library", fail_once)
    monkeypatch.setattr(
        osdi_loader.osdi_shim_nb, "get_last_error", lambda: "test failure"
    )
    with pytest.raises(RuntimeError, match="test failure"):
        osdi_loader.load_osdi_model(str(path))
    assert osdi_loader.load_osdi_model(str(path)).id == 2
