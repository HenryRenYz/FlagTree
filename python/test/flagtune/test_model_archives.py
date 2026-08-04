"""Validate version selection and the single-archive security contract."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.request import BaseHandler
from urllib.response import addinfourl

import pytest
import yaml
import numpy as np

from triton.flagtune.contract.archive import (
    ModelArchiveError,
    PACKAGE_MANIFEST_NAME,
    parse_model_version,
    platform_package_name,
    read_model_archive,
    read_model_archive_bytes,
    read_platform_package,
    read_platform_package_bytes,
    validate_model_version,
    write_model_archive,
    write_platform_package,
)
from triton.flagtune.contract.identity import ModelIdentity
from triton.flagtune.contract.operator_schema import model_config_sha256
from triton.flagtune.runtime import model_loader, model_sources
from triton.flagtune.runtime.model_loader import FlagTuneModelManager, IncompatibleModelError

IDENTITY_PATH = ("nvidia-h800", "vendor", "mm", "general", "bf16-bf16-f32")


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
            elif kind == "hardlink":
                member.type = tarfile.LNKTYPE
                member.linkname = "xgboost_ranker.json"
                archive.addfile(member)
            elif kind == "fifo":
                member.type = tarfile.FIFOTYPE
                archive.addfile(member)
            elif kind == "char":
                member.type = tarfile.CHRTYPE
                archive.addfile(member)
            elif kind == "block":
                member.type = tarfile.BLKTYPE
                archive.addfile(member)
    return buffer.getvalue()


def _platform_model_archive(tmp_path, variant, version="1.0.0", marker=b""):
    identity = ModelIdentity("nvidia-h20", "flaggems/mm", variant, "bf16-bf16-bf16")
    config = {
        "format_version": 5,
        "model_version": version,
        "platform_key": "nvidia-h20",
        "op_id": "flaggems/mm",
        "variant": variant,
        "dtype_key": "bf16-bf16-bf16",
        "dtypes": ["bfloat16", "bfloat16", "bfloat16"],
        "gpu": {
            "backend": "cuda",
            "vendor": "NVIDIA",
            "device_name": "NVIDIA H20-3e",
            "architecture": "sm90",
            "platform_key": "nvidia-h20",
        },
        "inputs": {"M": {"min": 1}},
        "params": {"BLOCK_M": {"values": [16, 32]}},
        "features": ["M"],
    }
    path = tmp_path / f"{variant}-{version}.tar.gz"
    write_model_archive(
        path, {
            "xgboost_ranker.json":
            variant.encode() + marker,
            "flagtune_config.yaml":
            yaml.safe_dump(config, sort_keys=True).encode(),
            "training_summary.json":
            json.dumps({
                "feature_cols": ["M"],
                "feature_count": 1,
                "model_config_sha256": model_config_sha256(config),
                "model_version": version,
                "op_id": "flaggems/mm",
                "variant": variant,
            }).encode(),
        })
    return identity, path


def _platform_manifest(path="flaggems/mm/gemv/bf16-bf16-bf16/model.tar.gz"):
    return {
        "schema_version": 1,
        "platform_key": "nvidia-h20",
        "package_version": "1.0.0",
        "models": {
            "nvidia-h20/flaggems/mm/gemv/bf16-bf16-bf16": {"path": path},
        },
    }


def _raw_platform_package(manifest, entries=()):
    manifest_bytes = manifest if isinstance(manifest, bytes) else json.dumps(manifest).encode()
    return _raw_archive([(PACKAGE_MANIFEST_NAME, manifest_bytes, "file"), *entries])


def _install_platform_package(root, version="1.0.0", *, cache=False, marker=b""):
    children = root / f"children-{version}-{marker.hex()}"
    children.mkdir(parents=True, exist_ok=True)
    archives = dict(
        _platform_model_archive(children, variant, version, marker) for variant in (
            "gemv",
            "general_tma",
            "splitk",
        ))
    filename = platform_package_name("nvidia-h20", version)
    path = root / "packages" / "nvidia-h20" / version / filename if cache else root / filename
    return write_platform_package(
        path,
        platform_key="nvidia-h20",
        package_version=version,
        model_archives=archives,
        required_identities=tuple(archives),
    )


def _configure_platform_remote(monkeypatch, version, payload, *, skip_runtime_validation=True):
    requests = []

    class Response:

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return payload

    def urlopen(request, **_kwargs):
        requests.append(request)
        return Response()

    filename = platform_package_name("nvidia-h20", version)
    monkeypatch.delenv("FLAGTUNE_MODEL_DIR", raising=False)
    monkeypatch.delenv("FLAGTUNE_DISABLE_REMOTE", raising=False)
    monkeypatch.setenv(
        "FLAGTUNE_MODEL_URLS",
        json.dumps({
            "schema_version": 1,
            "packages": {
                "nvidia-h20": {
                    "versions": {
                        version: {
                            "url": f"https://example.invalid/{filename}",
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        },
                    },
                },
            },
        }))
    monkeypatch.setattr(model_sources, "_open_https", urlopen)
    if skip_runtime_validation:
        monkeypatch.setattr(
            FlagTuneModelManager,
            "_validate_bundle_members",
            lambda *_args, **_kwargs: None,
            raising=False,
        )
    return requests


def _install_archive(root, version, marker=b"model"):
    return write_model_archive(
        root.joinpath(*IDENTITY_PATH, version, "model.tar.gz"),
        _members(marker),
    )


def _configure_remote(monkeypatch, version, payload):
    requests = []

    class Response:

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return payload

    def urlopen(request, **_kwargs):
        requests.append(request)
        return Response()

    monkeypatch.delenv("FLAGTUNE_MODEL_DIR", raising=False)
    monkeypatch.delenv("FLAGTUNE_DISABLE_REMOTE", raising=False)
    monkeypatch.setenv(
        "FLAGTUNE_MODEL_URLS",
        json.dumps({
            "models": {
                "/".join(IDENTITY_PATH): {
                    "versions": {
                        version: {
                            "url": "https://example.invalid/model.tar.gz",
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        },
                    },
                },
            },
        }))
    monkeypatch.setattr(model_sources, "_open_https", urlopen)
    monkeypatch.setattr(FlagTuneModelManager, "_load_bundle_members", lambda *_args, **_kwargs: None)
    return requests


class _RedirectingTransport(BaseHandler):
    handler_order = 100

    def __init__(self, payload, redirected_url, opened):
        self._payload = payload
        self._redirected_url = redirected_url
        self._opened = opened

    @staticmethod
    def _response(payload, headers, url, code, message):
        response = addinfourl(io.BytesIO(payload), headers, url, code=code)
        response.msg = message
        return response

    def https_open(self, request):
        self._opened.append(request.full_url)
        headers = Message()
        headers["Location"] = self._redirected_url
        return self._response(b"", headers, request.full_url, 302, "Found")

    def http_open(self, request):
        self._opened.append(request.full_url)
        return self._response(self._payload, Message(), request.full_url, 200, "OK")


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


def test_platform_package_is_deterministic_and_indexes_literal_model_paths(tmp_path):
    children = dict(_platform_model_archive(tmp_path, variant) for variant in ("gemv", "general_tma", "splitk"))
    first_path = tmp_path / "first" / "nvidia-h20_v1.0.0.tar.gz"
    second_path = tmp_path / "second" / "nvidia-h20_v1.0.0.tar.gz"

    write_platform_package(
        first_path,
        platform_key="nvidia-h20",
        package_version="1.0.0",
        model_archives=children,
    )
    write_platform_package(
        second_path,
        platform_key="nvidia-h20",
        package_version="1.0.0",
        model_archives=children,
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    expected_paths = {
        "flaggems/mm/gemv/bf16-bf16-bf16/model.tar.gz",
        "flaggems/mm/general_tma/bf16-bf16-bf16/model.tar.gz",
        "flaggems/mm/splitk/bf16-bf16-bf16/model.tar.gz",
    }
    with tarfile.open(first_path, mode="r:gz") as archive:
        assert {member.name for member in archive.getmembers()} == {PACKAGE_MANIFEST_NAME, *expected_paths}
        manifest = json.load(archive.extractfile(PACKAGE_MANIFEST_NAME))
        for member in archive.getmembers():
            assert member.isfile()
            assert (member.mtime, member.uid, member.gid, member.uname, member.gname) == (0, 0, 0, "", "")
    assert manifest == {
        "schema_version": 1,
        "platform_key": "nvidia-h20",
        "package_version": "1.0.0",
        "models":
        {f"nvidia-h20/{path.removesuffix('/model.tar.gz')}": {"path": path}
         for path in sorted(expected_paths)},
    }

    package = read_platform_package(
        first_path,
        expected_platform_key="nvidia-h20",
        expected_version="1.0.0",
    )
    assert package.platform_key == "nvidia-h20"
    assert package.package_version == "1.0.0"
    assert set(package.models) == set(manifest["models"])
    assert set(package.archives) == set(manifest["models"])


def test_platform_package_name_is_canonical():
    assert platform_package_name("nvidia-h20", "1.0.0") == "nvidia-h20_v1.0.0.tar.gz"


@pytest.mark.parametrize(
    ("name", "kind", "message"),
    [
        ("", "file", "safe relative path"),
        ("../escape", "file", "safe relative path"),
        ("/absolute", "file", "safe relative path"),
        (".", "file", "safe relative path"),
        ("..", "file", "safe relative path"),
        ("./child", "file", "safe relative path"),
        ("child/../model.tar.gz", "file", "safe relative path"),
        (r"flaggems\mm\model.tar.gz", "file", "safe relative path"),
        ("unexpected-link", "link", "regular file"),
        ("unexpected-hardlink", "hardlink", "regular file"),
        ("unexpected-directory", "dir", "regular file"),
        ("unexpected-device", "fifo", "regular file"),
        ("unexpected-char-device", "char", "regular file"),
        ("unexpected-block-device", "block", "regular file"),
    ],
)
def test_platform_package_rejects_unsafe_nested_members(tmp_path, name, kind, message):
    _, child = _platform_model_archive(tmp_path, "gemv")
    valid_path = "flaggems/mm/gemv/bf16-bf16-bf16/model.tar.gz"
    entries = [(valid_path, child.read_bytes(), "file"), (name, b"bad", kind)]
    with pytest.raises(ModelArchiveError, match=message):
        read_platform_package_bytes(
            _raw_platform_package(_platform_manifest(), entries),
            expected_platform_key="nvidia-h20",
            expected_version="1.0.0",
            source="unsafe fixture",
        )


def test_platform_package_rejects_duplicate_members(tmp_path):
    _, child = _platform_model_archive(tmp_path, "gemv")
    path = "flaggems/mm/gemv/bf16-bf16-bf16/model.tar.gz"
    with pytest.raises(ModelArchiveError, match="duplicate"):
        read_platform_package_bytes(
            _raw_platform_package(_platform_manifest(), [
                (path, child.read_bytes(), "file"),
                (path, child.read_bytes(), "file"),
            ]),
            expected_platform_key="nvidia-h20",
            expected_version="1.0.0",
            source="duplicate fixture",
        )


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([], "missing indexed"),
        ([
            ("flaggems/mm/gemv/bf16-bf16-bf16/model.tar.gz", b"child", "file"),
            ("unindexed/model.tar.gz", b"child", "file"),
        ], "unindexed"),
    ],
)
def test_platform_package_requires_exactly_indexed_children(entries, message):
    with pytest.raises(ModelArchiveError, match=message):
        read_platform_package_bytes(
            _raw_platform_package(_platform_manifest(), entries),
            expected_platform_key="nvidia-h20",
            expected_version="1.0.0",
            source="index fixture",
        )


def test_platform_package_rejects_duplicate_model_paths():
    manifest = _platform_manifest()
    manifest["models"]["nvidia-h20/flaggems/mm/splitk/bf16-bf16-bf16"] = {
        "path": "flaggems/mm/gemv/bf16-bf16-bf16/model.tar.gz",
    }
    with pytest.raises(ModelArchiveError, match="duplicate model path"):
        read_platform_package_bytes(
            _raw_platform_package(manifest),
            expected_platform_key="nvidia-h20",
            expected_version="1.0.0",
            source="duplicate path fixture",
        )


@pytest.mark.parametrize(
    ("manifest_update", "expected_platform", "expected_version", "message"),
    [
        ({"platform_key": "amd-mi300x"}, "nvidia-h20", "1.0.0", "platform mismatch"),
        ({"package_version": "2.0.0"}, "nvidia-h20", "1.0.0", "version mismatch"),
    ],
)
def test_platform_package_rejects_wrong_platform_or_version(
    manifest_update,
    expected_platform,
    expected_version,
    message,
):
    manifest = _platform_manifest()
    manifest.update(manifest_update)
    with pytest.raises(ModelArchiveError, match=message):
        read_platform_package_bytes(
            _raw_platform_package(manifest),
            expected_platform_key=expected_platform,
            expected_version=expected_version,
            source="mismatch fixture",
        )


def test_platform_package_rejects_noncanonical_manifest_platform():
    manifest = _platform_manifest()
    manifest["platform_key"] = "NVIDIA-H20"
    with pytest.raises(ModelArchiveError, match="canonical"):
        read_platform_package_bytes(
            _raw_platform_package(manifest),
            expected_platform_key="NVIDIA-H20",
            expected_version="1.0.0",
            source="noncanonical platform fixture",
        )


def test_platform_package_rejects_noncanonical_artifact_key(tmp_path):
    _, child = _platform_model_archive(tmp_path, "gemv")
    path = "flaggems/mm/gemv/bf16-bf16-bf16/model.tar.gz"
    manifest = _platform_manifest()
    entry = manifest["models"].pop("nvidia-h20/flaggems/mm/gemv/bf16-bf16-bf16")
    manifest["models"]["nvidia-h20/FlagGems/mm/gemv/bf16-bf16-bf16"] = entry
    with pytest.raises(ModelArchiveError, match="canonical"):
        read_platform_package_bytes(
            _raw_platform_package(manifest, [(path, child.read_bytes(), "file")]),
            expected_platform_key="nvidia-h20",
            expected_version="1.0.0",
            source="noncanonical artifact fixture",
        )


def test_platform_package_rejects_wrong_filename(tmp_path):
    _, child = _platform_model_archive(tmp_path, "gemv")
    wrong = tmp_path / "wrong-name.tar.gz"
    write_platform_package(
        wrong,
        platform_key="nvidia-h20",
        package_version="1.0.0",
        model_archives={ModelIdentity("nvidia-h20", "flaggems/mm", "gemv", "bf16-bf16-bf16"): child},
    )
    with pytest.raises(ModelArchiveError, match="filename"):
        read_platform_package(wrong, expected_platform_key="nvidia-h20", expected_version="1.0.0")


@pytest.mark.parametrize(
    ("config_update", "message"),
    [
        ({"variant": "splitk"}, "identity mismatch"),
        ({"model_version": "2.0.0"}, "version mismatch"),
    ],
)
def test_platform_package_rejects_wrong_child_identity_or_version(tmp_path, config_update, message):
    _, child = _platform_model_archive(tmp_path, "gemv")
    members = read_model_archive(child)
    config = yaml.safe_load(members["flagtune_config.yaml"])
    config.update(config_update)
    members["flagtune_config.yaml"] = yaml.safe_dump(config, sort_keys=True).encode()
    write_model_archive(child, members)
    path = "flaggems/mm/gemv/bf16-bf16-bf16/model.tar.gz"
    with pytest.raises(ModelArchiveError, match=message):
        read_platform_package_bytes(
            _raw_platform_package(_platform_manifest(), [(path, child.read_bytes(), "file")]),
            expected_platform_key="nvidia-h20",
            expected_version="1.0.0",
            source="child mismatch fixture",
        )


@pytest.mark.parametrize("validation_mode", ["reader", "writer"])
@pytest.mark.parametrize(
    ("config_update", "message"),
    [
        ({"extra": True}, "unknown keys"),
        ({"format_version": 4}, "format_version must be 5"),
    ],
)
def test_platform_package_rejects_child_config_outside_full_contract(
    tmp_path,
    validation_mode,
    config_update,
    message,
):
    identity, child = _platform_model_archive(tmp_path, "gemv")
    members = read_model_archive(child)
    config = yaml.safe_load(members["flagtune_config.yaml"])
    config.update(config_update)
    members["flagtune_config.yaml"] = yaml.safe_dump(config, sort_keys=True).encode()
    write_model_archive(child, members)

    with pytest.raises(ModelArchiveError, match=message):
        if validation_mode == "writer":
            write_platform_package(
                tmp_path / "nvidia-h20_v1.0.0.tar.gz",
                platform_key="nvidia-h20",
                package_version="1.0.0",
                model_archives={identity: child},
            )
        else:
            path = "flaggems/mm/gemv/bf16-bf16-bf16/model.tar.gz"
            read_platform_package_bytes(
                _raw_platform_package(_platform_manifest(), [(path, child.read_bytes(), "file")]),
                expected_platform_key="nvidia-h20",
                expected_version="1.0.0",
                source="invalid child contract fixture",
            )


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (b"{not-json", "Manifest"),
        ({
            "schema_version": 1,
            "platform": "nvidia-h20",
            "package_version": "1.0.0",
            "models": {},
        }, "root keys"),
        ({**_platform_manifest(), "extra": True}, "root keys"),
        ({
            **_platform_manifest(),
            "models": {
                "nvidia-h20/flaggems/mm/gemv/bf16-bf16-bf16": {
                    "path": "flaggems/mm/gemv/bf16-bf16-bf16/model.tar.gz",
                    "sha256": "old-field",
                },
            },
        }, "entry keys"),
    ],
)
def test_platform_package_rejects_malformed_manifest(manifest, message):
    with pytest.raises(ModelArchiveError, match=message):
        read_platform_package_bytes(
            _raw_platform_package(manifest),
            expected_platform_key="nvidia-h20",
            expected_version="1.0.0",
            source="manifest fixture",
        )


def test_platform_package_writer_requires_supplied_identities(tmp_path):
    identity, child = _platform_model_archive(tmp_path, "gemv")
    missing = ModelIdentity("nvidia-h20", "flaggems/mm", "splitk", "bf16-bf16-bf16")
    with pytest.raises(ModelArchiveError, match="missing required identities"):
        write_platform_package(
            tmp_path / "nvidia-h20_v1.0.0.tar.gz",
            platform_key="nvidia-h20",
            package_version="1.0.0",
            model_archives={identity: child},
            required_identities=(identity, missing),
        )


def test_user_flat_platform_package_precedes_cache_and_remote(tmp_path, monkeypatch):
    user = _install_platform_package(tmp_path / "user", "1.0.0")
    _install_platform_package(tmp_path / "cache", "2.0.0", cache=True)
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", "1")
    monkeypatch.setattr(
        model_sources,
        "resolve_package_info",
        lambda *_args, **_kwargs: pytest.fail("remote resolution called"),
    )

    selected = FlagTuneModelManager().resolve(
        "flaggems/mm",
        "gemv",
        platform_key="nvidia-h20",
        dtype_key="bf16-bf16-bf16",
    )

    assert selected == user


def test_old_nested_single_model_layout_is_ignored(tmp_path, monkeypatch):
    old = tmp_path / "nvidia-h20" / "flaggems" / "mm" / "gemv" / "bf16-bf16-bf16" / "1.0.0"
    write_model_archive(old / "model.tar.gz", _members())
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")

    with pytest.raises(FileNotFoundError):
        FlagTuneModelManager().resolve(
            "flaggems/mm",
            "gemv",
            platform_key="nvidia-h20",
            dtype_key="bf16-bf16-bf16",
        )


def test_platform_package_cache_uses_canonical_path_and_exact_pin(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    first = _install_platform_package(cache_root, "1.0.0", cache=True)
    latest = _install_platform_package(cache_root, "2.0.0", cache=True)
    monkeypatch.delenv("FLAGTUNE_MODEL_DIR", raising=False)
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")
    manager = FlagTuneModelManager()

    assert manager.resolve(
        "flaggems/mm",
        "general_tma",
        platform_key="nvidia-h20",
        dtype_key="bf16-bf16-bf16",
    ) == latest
    assert manager.resolve(
        "flaggems/mm",
        "general_tma",
        platform_key="nvidia-h20",
        dtype_key="bf16-bf16-bf16",
        model_version="1.0.0",
    ) == first
    assert latest == (cache_root / "packages" / "nvidia-h20" / "2.0.0" / "nvidia-h20_v2.0.0.tar.gz")


def test_one_remote_platform_download_serves_all_variants(tmp_path, monkeypatch):
    remote = _install_platform_package(tmp_path / "remote", "1.0.0")
    requests = _configure_platform_remote(monkeypatch, "1.0.0", remote.read_bytes())
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    manager = FlagTuneModelManager()

    selected = [
        manager.resolve(
            "flaggems/mm",
            variant,
            platform_key="nvidia-h20",
            dtype_key="bf16-bf16-bf16",
        ) for variant in ("gemv", "general_tma", "splitk")
    ]

    expected = cache_root / "packages" / "nvidia-h20" / "1.0.0" / "nvidia-h20_v1.0.0.tar.gz"
    assert selected == [expected, expected, expected]
    assert expected.read_bytes() == remote.read_bytes()
    assert len(requests) == 1
    assert not list(cache_root.rglob("*.tmp"))


def test_remote_h20_package_requires_all_three_models_before_cache(tmp_path, monkeypatch, caplog):
    children = tmp_path / "children"
    children.mkdir()
    identity, child = _platform_model_archive(children, "gemv")
    remote = write_platform_package(
        tmp_path / "remote" / "nvidia-h20_v1.0.0.tar.gz",
        platform_key="nvidia-h20",
        package_version="1.0.0",
        model_archives={identity: child},
    )
    _configure_platform_remote(monkeypatch, "1.0.0", remote.read_bytes(), skip_runtime_validation=False)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))

    with caplog.at_level("WARNING"), pytest.raises(IncompatibleModelError, match="missing required H20 models"):
        FlagTuneModelManager()._download_package(
            "nvidia-h20",
            model_sources.RemotePackage(
                "1.0.0",
                "https://example.invalid/nvidia-h20_v1.0.0.tar.gz",
                hashlib.sha256(remote.read_bytes()).hexdigest(),
            ),
        )

    assert not list(cache_root.rglob("nvidia-h20_v1.0.0.tar.gz"))


def test_remote_package_validates_unselected_child_before_cache(tmp_path, monkeypatch):
    xgboost = pytest.importorskip("xgboost")
    model = xgboost.XGBRanker(n_estimators=0)
    model.fit(np.zeros((2, 1)), np.zeros(2), group=[2])
    children = tmp_path / "children"
    children.mkdir()
    archives = {}
    for variant in ("gemv", "general_tma", "splitk"):
        identity, child = _platform_model_archive(children, variant)
        members = read_model_archive(child)
        config = yaml.safe_load(members["flagtune_config.yaml"])
        booster = model.get_booster()
        booster.feature_names = ["M"]
        booster.set_attr(flagtune_config_sha256=("0" * 64 if variant == "splitk" else model_config_sha256(config)))
        loose = children / f"{variant}.json"
        model.save_model(str(loose))
        write_model_archive(
            child,
            {**members, "xgboost_ranker.json": loose.read_bytes()},
        )
        archives[identity] = child
    remote = write_platform_package(
        tmp_path / "remote" / "nvidia-h20_v1.0.0.tar.gz",
        platform_key="nvidia-h20",
        package_version="1.0.0",
        model_archives=archives,
        required_identities=tuple(archives),
    )
    _configure_platform_remote(monkeypatch, "1.0.0", remote.read_bytes(), skip_runtime_validation=False)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))

    with pytest.raises(IncompatibleModelError, match="config digest mismatch"):
        FlagTuneModelManager().resolve(
            "flaggems/mm",
            "gemv",
            platform_key="nvidia-h20",
            dtype_key="bf16-bf16-bf16",
        )

    assert not list(cache_root.rglob("nvidia-h20_v1.0.0.tar.gz"))


def test_remote_package_rejects_malformed_training_summary_before_cache(tmp_path, monkeypatch, caplog):
    children = tmp_path / "children"
    children.mkdir()
    archives = {}
    for variant in ("gemv", "general_tma", "splitk"):
        identity, child = _platform_model_archive(children, variant)
        if variant == "splitk":
            members = read_model_archive(child)
            write_model_archive(
                child,
                {**members, "training_summary.json": b"{not-json"},
            )
        archives[identity] = child
    remote = write_platform_package(
        tmp_path / "remote" / "nvidia-h20_v1.0.0.tar.gz",
        platform_key="nvidia-h20",
        package_version="1.0.0",
        model_archives=archives,
        required_identities=tuple(archives),
    )
    _configure_platform_remote(monkeypatch, "1.0.0", remote.read_bytes(), skip_runtime_validation=False)
    monkeypatch.setattr(model_loader, "_XGBoostPredictorCompat", lambda *_args: object())
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path / "cache"))

    with caplog.at_level("WARNING"), pytest.raises(IncompatibleModelError, match="training summary"):
        FlagTuneModelManager()._download_package(
            "nvidia-h20",
            model_sources.RemotePackage(
                "1.0.0",
                "https://example.invalid/nvidia-h20_v1.0.0.tar.gz",
                hashlib.sha256(remote.read_bytes()).hexdigest(),
            ),
        )

    assert not list((tmp_path / "cache").rglob("nvidia-h20_v1.0.0.tar.gz"))


def test_concurrent_package_publication_never_overwrites_race_winner(tmp_path, monkeypatch):
    remote = _install_platform_package(tmp_path / "remote", "1.0.0", marker=b"remote")
    remote_payload = remote.read_bytes()
    rival = _install_platform_package(tmp_path / "rival", "1.0.0", marker=b"rival")
    rival_payload = rival.read_bytes()
    _configure_platform_remote(monkeypatch, "1.0.0", remote_payload)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    destination = (cache_root / "packages" / "nvidia-h20" / "1.0.0" / "nvidia-h20_v1.0.0.tar.gz")

    def racing_link(_source, target):
        Path(target).write_bytes(rival_payload)
        raise FileExistsError(target)

    monkeypatch.setattr(model_loader.os, "link", racing_link)

    with pytest.raises(IncompatibleModelError, match="different SHA-256"):
        FlagTuneModelManager()._download_package(
            "nvidia-h20",
            model_sources.RemotePackage(
                "1.0.0",
                "https://example.invalid/nvidia-h20_v1.0.0.tar.gz",
                hashlib.sha256(remote_payload).hexdigest(),
            ),
        )

    assert destination.read_bytes() == rival_payload
    assert not list(destination.parent.glob("*.tmp"))
    assert not list(destination.parent.glob(".*.tmp"))


def test_refresh_rejects_changed_digest_for_immutable_package_version(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cached = _install_platform_package(cache_root, "1.0.0", cache=True, marker=b"cached")
    remote = _install_platform_package(tmp_path / "remote", "1.0.0", marker=b"changed")
    _configure_platform_remote(monkeypatch, "1.0.0", remote.read_bytes())
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", "1")

    with pytest.raises(IncompatibleModelError, match="immutable"):
        FlagTuneModelManager().resolve(
            "flaggems/mm",
            "splitk",
            platform_key="nvidia-h20",
            dtype_key="bf16-bf16-bf16",
        )

    assert cached.read_bytes() != remote.read_bytes()


def test_three_model_loads_reuse_one_parsed_platform_package(tmp_path, monkeypatch):
    package_path = _install_platform_package(tmp_path / "models", "1.0.0")
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path / "models"))
    parsed = []
    real_read = read_platform_package

    def counted_read(*args, **kwargs):
        parsed.append(args[0])
        return real_read(*args, **kwargs)

    def load_members(_manager, identity, version, _members, outer_path, member):
        return SimpleNamespace(
            identity=identity,
            variant=SimpleNamespace(),
            predictor=SimpleNamespace(),
            package_path=outer_path,
            model_member=member,
            model_version=version,
        )

    monkeypatch.setattr(model_loader, "read_platform_package", counted_read, raising=False)
    monkeypatch.setattr(FlagTuneModelManager, "_load_bundle_members", load_members)
    manager = FlagTuneModelManager()

    loaded = [
        manager.load(
            "flaggems/mm",
            variant,
            platform_key="nvidia-h20",
            dtype_key="bf16-bf16-bf16",
        ) for variant in ("gemv", "general_tma", "splitk")
    ]

    assert [item.package_path for item in loaded] == [package_path] * 3
    assert [item.model_member for item in loaded] == [
        f"flaggems/mm/{variant}/bf16-bf16-bf16/model.tar.gz" for variant in ("gemv", "general_tma", "splitk")
    ]
    assert parsed == [package_path]
