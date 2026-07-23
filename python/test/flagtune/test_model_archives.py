"""Validate version selection and the single-archive security contract."""

from __future__ import annotations

import io
import tarfile

import pytest

from triton.flagtune.artifacts import (
    ModelArchiveError,
    parse_model_version,
    read_model_archive,
    read_model_archive_bytes,
    validate_model_version,
    write_model_archive,
)
from triton.flagtune.model_manager import FlagTuneModelManager, IncompatibleModelError

IDENTITY_PATH = ("nvidia-h800-sm90", "vendor", "mm", "general", "bf16-bf16-f32")


def _members(marker=b"model"):
    return {
        "xgboost_ranker.json": marker,
        "flagtune_config.yaml": b"{}",
        "training_summary.json": b"{}",
    }


def _raw_archive(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload, kind in entries:
            member = tarfile.TarInfo(name)
            if kind == "file":
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            elif kind == "dir":
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            elif kind == "link":
                member.type = tarfile.SYMTYPE
                member.linkname = "xgboost_ranker.json"
                archive.addfile(member)
    return buffer.getvalue()


def test_strict_semver_precedence_and_build_tie_break():
    versions = [
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-beta.2",
        "1.0.0-beta.11",
        "1.0.0-rc.1",
        "1.0.0",
        "1.0.0+build.2",
        "1.0.0+build.10",
    ]
    ordered = sorted(versions, key=lambda value: parse_model_version(value).selection_key)
    assert ordered[:6] == versions[:6]
    assert ordered[-1] == "1.0.0+build.2"
    assert parse_model_version("1.0.0").precedence_key == parse_model_version("1.0.0+abc").precedence_key


@pytest.mark.parametrize("value", ["1", "1.0", "01.0.0", "1.0.0-01", "v1.0.0", "1.0.0 "])
def test_invalid_semver_is_rejected(value):
    with pytest.raises(ValueError, match="SemVer"):
        validate_model_version(value)


def test_archive_is_reproducible_atomic_and_has_fixed_metadata(tmp_path):
    path = tmp_path / "model.tar.gz"
    first = write_model_archive(path, _members(b"first"))
    first_bytes = first.read_bytes()
    write_model_archive(path, _members(b"first"))
    assert path.read_bytes() == first_bytes
    write_model_archive(path, _members(b"second"))
    assert read_model_archive(path)["xgboost_ranker.json"] == b"second"
    assert not list(tmp_path.glob("*.tmp"))
    with tarfile.open(path, mode="r:gz") as archive:
        assert [member.name for member in archive.getmembers()] == list(_members())
        for member in archive.getmembers():
            assert member.isfile()
            assert (member.mtime, member.uid, member.gid, member.uname, member.gname) == (0, 0, 0, "", "")


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([('../escape', b'x', 'file')], "root-level"),
        ([('nested/file', b'x', 'file')], "root-level"),
        ([('extra', b'', 'dir')], "regular file"),
        ([('extra', b'', 'link')], "regular file"),
        ([('xgboost_ranker.json', b'again', 'file')], "duplicate"),
    ],
)
def test_archive_rejects_unsafe_and_duplicate_members(extra, message):
    base = [(name, payload, "file") for name, payload in _members().items()]
    with pytest.raises(ModelArchiveError, match=message):
        read_model_archive_bytes(_raw_archive(base + extra))


def test_archive_rejects_missing_member_and_corrupt_gzip():
    entries = [(name, payload, "file") for name, payload in _members().items() if name != "training_summary.json"]
    with pytest.raises(ModelArchiveError, match="missing required"):
        read_model_archive_bytes(_raw_archive(entries))
    with pytest.raises(ModelArchiveError, match="invalid gzip tar"):
        read_model_archive_bytes(b"not a gzip tar")


def test_manager_selects_highest_semver_and_supports_explicit_pin(tmp_path, monkeypatch, caplog):
    identity_root = tmp_path.joinpath(*IDENTITY_PATH)
    for version in ("1.9.0", "2.0.0-rc.1", "2.0.0", "2.0.0+build.2"):
        write_model_archive(identity_root / version / "model.tar.gz", _members(version.encode()))
    (identity_root / "invalid").mkdir()
    (identity_root / "3.0.0").mkdir()
    monkeypatch.setenv("TRITON_FLAGTUNE_MODEL_DIR", str(tmp_path))
    manager = FlagTuneModelManager()

    selected = manager.resolve("vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1])
    assert selected.parent.name == "2.0.0+build.2"
    pinned = manager.resolve(
        "vendor/mm",
        "general",
        gpu_key=IDENTITY_PATH[0],
        dtype_key=IDENTITY_PATH[-1],
        model_version="1.9.0",
    )
    assert pinned.parent.name == "1.9.0"
    assert "invalid FlagTune version directory" in caplog.text
    assert "without model.tar.gz" in caplog.text


def test_manager_does_not_fall_back_to_legacy_loose_bundle(tmp_path, monkeypatch):
    identity_root = tmp_path.joinpath(*IDENTITY_PATH)
    identity_root.mkdir(parents=True)
    for name, payload in _members().items():
        (identity_root / name).write_bytes(payload)
    monkeypatch.setenv("TRITON_FLAGTUNE_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")

    with pytest.raises(FileNotFoundError):
        FlagTuneModelManager().resolve("vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1])


def test_selected_corrupt_highest_version_does_not_fall_back(tmp_path, monkeypatch):
    identity_root = tmp_path.joinpath(*IDENTITY_PATH)
    write_model_archive(identity_root / "1.0.0" / "model.tar.gz", _members())
    corrupt = identity_root / "2.0.0" / "model.tar.gz"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"corrupt")
    monkeypatch.setenv("TRITON_FLAGTUNE_MODEL_DIR", str(tmp_path))
    manager = FlagTuneModelManager()
    assert manager.resolve("vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1]) == corrupt
    with pytest.raises(IncompatibleModelError, match="invalid FlagTune model archive"):
        manager.load("vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1])
