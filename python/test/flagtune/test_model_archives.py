"""Validate version selection and the single-archive security contract."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from email.message import Message
from urllib.request import BaseHandler
from urllib.response import addinfourl

import pytest

from triton.flagtune.contract.archive import (
    ModelArchiveError,
    parse_model_version,
    read_model_archive,
    read_model_archive_bytes,
    validate_model_version,
    write_model_archive,
)
from triton.flagtune.runtime import model_sources
from triton.flagtune.runtime.model_loader import FlagTuneModelManager, IncompatibleModelError

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
    monkeypatch.setenv("FLAGTUNE_MODEL_URLS", json.dumps({
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


def test_manager_selects_highest_semver_and_supports_explicit_pin(tmp_path, monkeypatch, caplog):
    identity_root = tmp_path.joinpath(*IDENTITY_PATH)
    for version in ("1.9.0", "2.0.0-rc.1", "2.0.0", "2.0.0+build.2"):
        write_model_archive(identity_root / version / "model.tar.gz", _members(version.encode()))
    (identity_root / "invalid").mkdir()
    (identity_root / "3.0.0").mkdir()
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))
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


def test_environment_version_pin_selects_exact_version(tmp_path, monkeypatch):
    identity_root = tmp_path.joinpath(*IDENTITY_PATH)
    for version in ("1.0.0", "2.0.0"):
        write_model_archive(identity_root / version / "model.tar.gz", _members(version.encode()))
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("FLAGTUNE_MODEL_VERSION", "1.0.0")

    selected = FlagTuneModelManager().resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )

    assert selected.parent.name == "1.0.0"


def test_environment_version_pin_rejects_invalid_semver(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    monkeypatch.setenv("FLAGTUNE_MODEL_VERSION", "v1")
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")

    with pytest.raises(ValueError, match="SemVer"):
        FlagTuneModelManager().resolve(
            "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
        )


def test_explicit_version_overrides_environment_pin(tmp_path, monkeypatch):
    identity_root = tmp_path.joinpath(*IDENTITY_PATH)
    for version in ("1.0.0", "2.0.0"):
        write_model_archive(identity_root / version / "model.tar.gz", _members(version.encode()))
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("FLAGTUNE_MODEL_VERSION", "1.0.0")

    selected = FlagTuneModelManager().resolve(
        "vendor/mm",
        "general",
        gpu_key=IDENTITY_PATH[0],
        dtype_key=IDENTITY_PATH[-1],
        model_version="2.0.0",
    )

    assert selected.parent.name == "2.0.0"


def test_default_cache_hit_does_not_resolve_remote(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cached = _install_archive(cache_root, "1.0.0")
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.delenv("FLAGTUNE_MODEL_REFRESH", raising=False)
    monkeypatch.setattr(
        model_sources,
        "resolve_artifact_info",
        lambda *_args, **_kwargs: pytest.fail("remote resolution called"),
    )

    selected = FlagTuneModelManager().resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )

    assert selected == cached


def test_user_model_root_precedes_refresh_cache_and_remote(tmp_path, monkeypatch):
    user_root = tmp_path / "user"
    user = _install_archive(user_root, "1.0.0")
    _install_archive(tmp_path / "cache", "2.0.0")
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(user_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", "1")
    monkeypatch.setattr(
        model_sources,
        "resolve_artifact_info",
        lambda *_args, **_kwargs: pytest.fail("remote resolution called"),
    )

    selected = FlagTuneModelManager().resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )

    assert selected == user


def test_refresh_downloads_higher_remote_version_and_keeps_old_cache(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cached = _install_archive(cache_root, "1.0.0", b"cached")
    remote = _install_archive(tmp_path / "remote", "2.0.0", b"remote")
    requests = _configure_remote(monkeypatch, "2.0.0", remote.read_bytes())
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", " 1 ")

    selected = FlagTuneModelManager().resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )

    assert selected.parent.name == "2.0.0"
    assert selected.read_bytes() == remote.read_bytes()
    assert cached.is_file()
    assert len(requests) == 1


def test_refresh_keeps_higher_valid_cache(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cached = _install_archive(cache_root, "2.0.0")
    remote = _install_archive(tmp_path / "remote", "1.0.0")
    requests = _configure_remote(monkeypatch, "1.0.0", remote.read_bytes())
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", "1")

    selected = FlagTuneModelManager().resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )

    assert selected == cached
    assert requests == []


def test_refresh_reuses_equal_digest_valid_cache(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    remote = _install_archive(tmp_path / "remote", "2.0.0", b"same")
    cached = cache_root.joinpath(*IDENTITY_PATH, "2.0.0", "model.tar.gz")
    cached.parent.mkdir(parents=True)
    cached.write_bytes(remote.read_bytes())
    requests = _configure_remote(monkeypatch, "2.0.0", remote.read_bytes())
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", "1")

    selected = FlagTuneModelManager().resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )

    assert selected == cached
    assert requests == []


def test_refresh_preserves_valid_equal_version_cache_with_changed_digest(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cached = _install_archive(cache_root, "2.0.0", b"published")
    published_bytes = cached.read_bytes()
    remote = _install_archive(tmp_path / "remote", "2.0.0", b"replacement")
    requests = _configure_remote(monkeypatch, "2.0.0", remote.read_bytes())
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", "1")

    selected = FlagTuneModelManager().resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )

    assert selected == cached
    assert cached.read_bytes() == published_bytes
    assert requests == []


def test_download_preserves_concurrently_published_valid_version(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    remote = _install_archive(tmp_path / "remote", "2.0.0", b"remote")
    concurrent = _install_archive(tmp_path / "concurrent", "2.0.0", b"concurrent")
    concurrent_bytes = concurrent.read_bytes()
    requests = _configure_remote(monkeypatch, "2.0.0", remote.read_bytes())
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))

    def publish_concurrently(_manager, _identity, _version, _members, model_path):
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(concurrent_bytes)

    monkeypatch.setattr(FlagTuneModelManager, "_load_bundle_members", publish_concurrently)

    selected = FlagTuneModelManager().resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )

    assert selected.read_bytes() == concurrent_bytes
    assert len(requests) == 1


def test_refresh_redownloads_equal_corrupt_cache(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cached = cache_root.joinpath(*IDENTITY_PATH, "2.0.0", "model.tar.gz")
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"corrupt")
    remote = _install_archive(tmp_path / "remote", "2.0.0", b"replacement")
    requests = _configure_remote(monkeypatch, "2.0.0", remote.read_bytes())
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", "1")

    selected = FlagTuneModelManager().resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )

    assert selected == cached
    assert selected.read_bytes() == remote.read_bytes()
    assert len(requests) == 1


def test_refresh_download_failure_falls_back_to_valid_cache(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cached = _install_archive(cache_root, "1.0.0")
    remote = _install_archive(tmp_path / "remote", "2.0.0")
    _configure_remote(monkeypatch, "2.0.0", remote.read_bytes())
    monkeypatch.setattr(
        model_sources,
        "_open_https",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", "1")

    selected = FlagTuneModelManager().resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )

    assert selected == cached


def test_refresh_skips_corrupt_highest_cache_and_falls_back_to_valid_lower(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cached = _install_archive(cache_root, "1.0.0")
    corrupt = cache_root.joinpath(*IDENTITY_PATH, "2.0.0", "model.tar.gz")
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"corrupt")
    remote = _install_archive(tmp_path / "remote", "3.0.0")
    _configure_remote(monkeypatch, "3.0.0", remote.read_bytes())
    monkeypatch.setattr(
        model_sources,
        "_open_https",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", "1")

    selected = FlagTuneModelManager().resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )

    assert selected == cached


def test_artifact_https_redirect_to_http_is_rejected(tmp_path, monkeypatch):
    import urllib.request

    cache_root = tmp_path / "cache"
    remote = _install_archive(tmp_path / "remote", "2.0.0")
    real_open_https = model_sources._open_https
    _configure_remote(monkeypatch, "2.0.0", remote.read_bytes())
    monkeypatch.setattr(model_sources, "_open_https", real_open_https)
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    opened = []
    transport = _RedirectingTransport(
        remote.read_bytes(),
        "http://example.invalid/model.tar.gz",
        opened,
    )
    real_build_opener = urllib.request.build_opener
    monkeypatch.setattr("urllib.request.urlopen", real_build_opener(transport).open)
    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *handlers: real_build_opener(*handlers, transport),
    )

    with pytest.raises(FileNotFoundError):
        FlagTuneModelManager().resolve(
            "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
        )

    assert opened == ["https://example.invalid/model.tar.gz"]
    assert not list(cache_root.rglob("model.tar.gz"))


def test_remote_disable_takes_precedence_over_refresh(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cached = _install_archive(cache_root, "1.0.0")
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", "1")
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "0")
    monkeypatch.setattr(
        model_sources,
        "resolve_artifact_info",
        lambda *_args, **_kwargs: pytest.fail("remote resolution called"),
    )

    selected = FlagTuneModelManager().resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )

    assert selected == cached


def test_remote_disable_prevents_download_without_cache(tmp_path, monkeypatch):
    remote = _install_archive(tmp_path / "remote", "2.0.0")
    requests = _configure_remote(monkeypatch, "2.0.0", remote.read_bytes())
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "0")

    with pytest.raises(FileNotFoundError):
        FlagTuneModelManager().resolve(
            "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
        )
    assert requests == []


@pytest.mark.parametrize("value", ["", "0", "true", "2"])
def test_refresh_requires_exact_value_one(tmp_path, monkeypatch, value):
    cache_root = tmp_path / "cache"
    cached = _install_archive(cache_root, "1.0.0")
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", value)
    monkeypatch.setattr(
        model_sources,
        "resolve_artifact_info",
        lambda *_args, **_kwargs: pytest.fail("remote resolution called"),
    )

    selected = FlagTuneModelManager().resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )

    assert selected == cached


def test_missing_environment_pin_never_substitutes_cached_highest(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    _install_archive(cache_root, "2.0.0")
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_VERSION", "1.0.0")
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")

    with pytest.raises(FileNotFoundError, match="1.0.0"):
        FlagTuneModelManager().resolve(
            "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
        )


def test_missing_environment_pin_never_substitutes_remote_highest(tmp_path, monkeypatch):
    remote = _install_archive(tmp_path / "remote", "2.0.0")
    requests = _configure_remote(monkeypatch, "2.0.0", remote.read_bytes())
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FLAGTUNE_MODEL_VERSION", "1.0.0")

    with pytest.raises(FileNotFoundError, match="1.0.0"):
        FlagTuneModelManager().resolve(
            "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
        )
    assert requests == []


def test_pinned_refresh_uses_cache_before_exact_remote_download(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cached = _install_archive(cache_root, "1.0.0")
    remote = _install_archive(tmp_path / "remote", "2.0.0")
    requests = _configure_remote(monkeypatch, "2.0.0", remote.read_bytes())
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(cache_root))
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", "1")
    monkeypatch.setenv("FLAGTUNE_MODEL_VERSION", "1.0.0")
    manager = FlagTuneModelManager()

    assert manager.resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    ) == cached
    assert requests == []

    monkeypatch.setenv("FLAGTUNE_MODEL_VERSION", "2.0.0")
    selected = manager.resolve(
        "vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1],
    )
    assert selected.parent.name == "2.0.0"
    assert len(requests) == 1


def test_manager_does_not_fall_back_to_legacy_loose_bundle(tmp_path, monkeypatch):
    identity_root = tmp_path.joinpath(*IDENTITY_PATH)
    identity_root.mkdir(parents=True)
    for name, payload in _members().items():
        (identity_root / name).write_bytes(payload)
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")

    with pytest.raises(FileNotFoundError):
        FlagTuneModelManager().resolve("vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1])


def test_selected_corrupt_highest_version_does_not_fall_back(tmp_path, monkeypatch):
    identity_root = tmp_path.joinpath(*IDENTITY_PATH)
    write_model_archive(identity_root / "1.0.0" / "model.tar.gz", _members())
    corrupt = identity_root / "2.0.0" / "model.tar.gz"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"corrupt")
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))
    manager = FlagTuneModelManager()
    assert manager.resolve("vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1]) == corrupt
    with pytest.raises(IncompatibleModelError, match="invalid FlagTune model archive"):
        manager.load("vendor/mm", "general", gpu_key=IDENTITY_PATH[0], dtype_key=IDENTITY_PATH[-1])
