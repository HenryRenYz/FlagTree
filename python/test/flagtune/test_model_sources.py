import json
import time
from email.message import Message
from io import BytesIO
from urllib.request import BaseHandler
from urllib.response import addinfourl

import pytest

from triton.flagtune.contract.identity import ModelIdentityError
from triton.flagtune.runtime import model_sources

PLATFORM_KEY = "nvidia-h20"
ENTRY_1 = {
    "url": "https://example.invalid/nvidia-h20_v1.0.0.tar.gz",
    "sha256": "1" * 64,
}
ENTRY_2 = {
    "url": "https://example.invalid/nvidia-h20_v2.0.0.tar.gz",
    "sha256": "2" * 64,
}


@pytest.fixture(autouse=True)
def clean_remote_environment(monkeypatch):
    for name in (
            "FLAGTUNE_DISABLE_REMOTE",
            "FLAGTUNE_MANIFEST_URL",
            "FLAGTUNE_MODEL_CACHE",
            "FLAGTUNE_MODEL_URLS",
    ):
        monkeypatch.delenv(name, raising=False)


def manifest_with(entry, *, platform_key=PLATFORM_KEY):
    return {"schema_version": 1, "packages": {platform_key: entry}}


def install_override(monkeypatch, entry, *, platform_key=PLATFORM_KEY):
    monkeypatch.setenv("FLAGTUNE_MODEL_URLS", json.dumps(manifest_with(entry, platform_key=platform_key)))
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")


def response_for(manifest, calls):
    payload = json.dumps(manifest).encode("utf-8")

    class Response:

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return payload

    def urlopen(request, **_kwargs):
        calls.append(request)
        return Response()

    return urlopen


class _RedirectingTransport(BaseHandler):
    handler_order = 100

    def __init__(self, payload, redirected_url, opened):
        self._payload = payload
        self._redirected_url = redirected_url
        self._opened = opened

    @staticmethod
    def _response(payload, headers, url, code, message):
        response = addinfourl(BytesIO(payload), headers, url, code=code)
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


def test_manifest_url_defaults_and_allows_environment_override(monkeypatch):
    assert model_sources._manifest_url() == "https://models.example.com/flagtune/manifest.json"
    monkeypatch.setenv("FLAGTUNE_MANIFEST_URL", " https://mirror.example.com/packages.json ")
    assert model_sources._manifest_url() == "https://mirror.example.com/packages.json"


def test_exact_pin_and_highest_semver_selection_ignore_latest(monkeypatch):
    install_override(monkeypatch, {
        "latest": "1.0.0",
        "versions": {"1.0.0": ENTRY_1, "2.0.0": ENTRY_2},
    })

    highest = model_sources.resolve_package_info(PLATFORM_KEY)
    exact = model_sources.resolve_package_info(PLATFORM_KEY, version="1.0.0")

    assert highest == model_sources.RemotePackage("2.0.0", ENTRY_2["url"], "2" * 64)
    assert exact == model_sources.RemotePackage("1.0.0", ENTRY_1["url"], "1" * 64)
    assert model_sources.resolve_package_info(PLATFORM_KEY, version="3.0.0") is None


def test_platform_key_is_normalized_and_validated(monkeypatch):
    install_override(monkeypatch, {"versions": {"1.0.0": ENTRY_1}})

    assert model_sources.resolve_package_info(" NVIDIA-H20 ") == model_sources.RemotePackage(
        "1.0.0",
        ENTRY_1["url"],
        "1" * 64,
    )
    with pytest.raises(ModelIdentityError):
        model_sources.resolve_package_info("../nvidia-h20")


def test_force_refresh_bypasses_fresh_cached_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    cached = manifest_with({"versions": {"1.0.0": ENTRY_1}})
    cached["_fetched_at"] = time.time()
    (tmp_path / "manifest.json").write_text(json.dumps(cached))
    calls = []
    monkeypatch.setattr(
        model_sources,
        "_open_https",
        response_for(manifest_with({"versions": {"2.0.0": ENTRY_2}}), calls),
    )

    package = model_sources.resolve_package_info(PLATFORM_KEY, force_refresh=True)

    assert package == model_sources.RemotePackage("2.0.0", ENTRY_2["url"], "2" * 64)
    assert len(calls) == 1
    stored = json.loads((tmp_path / "manifest.json").read_text())
    assert stored["packages"][PLATFORM_KEY]["versions"] == {"2.0.0": ENTRY_2}
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_failed_refresh_preserves_valid_cached_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    cached = manifest_with({"versions": {"1.0.0": ENTRY_1}})
    cached["_fetched_at"] = time.time()
    cached_path = tmp_path / "manifest.json"
    cached_path.write_text(json.dumps(cached, sort_keys=True))
    cached_bytes = cached_path.read_bytes()
    calls = []
    monkeypatch.setattr(model_sources, "_open_https", response_for({"schema_version": 1, "packages": []}, calls))

    package = model_sources.resolve_package_info(PLATFORM_KEY, force_refresh=True)

    assert package == model_sources.RemotePackage("1.0.0", ENTRY_1["url"], "1" * 64)
    assert cached_path.read_bytes() == cached_bytes
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("metadata", "accepted"),
    [
        ({"url": "https://example.invalid/nvidia-h20_v1.0.0.tar.gz", "sha256": "a" * 64}, True),
        ({"url": "http://example.invalid/nvidia-h20_v1.0.0.tar.gz", "sha256": "a" * 64}, False),
        ({"url": "https:///nvidia-h20_v1.0.0.tar.gz", "sha256": "a" * 64}, False),
        ({"url": "https://example.invalid/nvidia-h20_v1.0.0.tar.gz"}, False),
        ({"url": "https://example.invalid/nvidia-h20_v1.0.0.tar.gz", "sha256": "A" * 64}, False),
        ({"url": "https://example.invalid/nvidia-h20_v1.0.0.tar.gz", "sha256": "a" * 63}, False),
        ({"url": "https://example.invalid/nvidia-h20_v2.0.0.tar.gz", "sha256": "a" * 64}, False),
        ({"url": "https://example.invalid/nvidia-h20_v1.0.0.tgz", "sha256": "a" * 64}, False),
        ({"url": "https://example.invalid/other_v1.0.0.tar.gz", "sha256": "a" * 64}, False),
    ],
)
def test_package_metadata_requires_https_lowercase_sha_and_canonical_basename(monkeypatch, metadata, accepted):
    install_override(monkeypatch, {"versions": {"1.0.0": metadata}})

    if accepted:
        assert model_sources.resolve_package_info(PLATFORM_KEY) is not None
    else:
        with pytest.raises(model_sources.ManifestContractError):
            model_sources.resolve_package_info(PLATFORM_KEY)


def test_manifest_validation_checks_every_package_url_basename(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    invalid_entry = {
        "versions": {
            "1.0.0": ENTRY_1,
            "2.0.0": {**ENTRY_2, "url": "https://example.invalid/wrong-name.tar.gz"},
        }
    }
    calls = []
    monkeypatch.setattr(model_sources, "_open_https", response_for(manifest_with(invalid_entry), calls))

    assert model_sources.resolve_package_info(PLATFORM_KEY, version="1.0.0") is None
    assert len(calls) == 1
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize(
    "location",
    ("root", "package", "metadata"),
)
def test_manifest_rejects_unknown_fields(monkeypatch, location):
    manifest = manifest_with({"versions": {"1.0.0": dict(ENTRY_1)}})
    if location == "root":
        manifest["extra"] = True
    elif location == "package":
        manifest["packages"][PLATFORM_KEY]["extra"] = True
    else:
        manifest["packages"][PLATFORM_KEY]["versions"]["1.0.0"]["extra"] = True
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")
    monkeypatch.setenv("FLAGTUNE_MODEL_URLS", json.dumps(manifest))

    with pytest.raises(model_sources.ManifestContractError, match="FLAGTUNE_MODEL_URLS"):
        model_sources.resolve_package_info(PLATFORM_KEY)


def test_invalid_override_does_not_fall_through_to_builtin(monkeypatch):
    """A present malformed override is authoritative and fails closed."""
    monkeypatch.setenv("FLAGTUNE_MODEL_URLS", "{not-json")
    monkeypatch.setattr(
        model_sources,
        "_BUILTIN_TABLE",
        {PLATFORM_KEY: {"versions": {"1.0.0": ENTRY_1}}},
    )

    with pytest.raises(model_sources.ManifestContractError, match="FLAGTUNE_MODEL_URLS"):
        model_sources.resolve_package_info(PLATFORM_KEY)


def test_valid_override_precedes_cache_remote_and_builtin(tmp_path, monkeypatch):
    """A valid explicit override remains the only consulted source."""
    install_override(monkeypatch, {"versions": {"2.0.0": ENTRY_2}})
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    monkeypatch.setattr(
        model_sources,
        "_load_cached_manifest",
        lambda: pytest.fail("cache consulted after valid override"),
    )
    monkeypatch.setattr(
        model_sources,
        "_download_manifest",
        lambda: pytest.fail("remote consulted after valid override"),
    )

    assert model_sources.resolve_package_info(PLATFORM_KEY) == model_sources.RemotePackage(
        "2.0.0", ENTRY_2["url"], ENTRY_2["sha256"])


def test_non_https_manifest_url_never_opens_network(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    monkeypatch.setenv("FLAGTUNE_MANIFEST_URL", "http://example.invalid/manifest.json")
    monkeypatch.setattr(model_sources, "_open_https", lambda *_args, **_kwargs: pytest.fail("network opened"))

    assert model_sources.resolve_package_info(PLATFORM_KEY) is None


def test_manifest_https_redirect_to_http_is_rejected(tmp_path, monkeypatch):
    import urllib.request

    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    payload = json.dumps(manifest_with({"versions": {"2.0.0": ENTRY_2}})).encode()
    opened = []
    transport = _RedirectingTransport(payload, "http://example.invalid/manifest.json", opened)
    real_build_opener = urllib.request.build_opener
    monkeypatch.setattr("urllib.request.urlopen", real_build_opener(transport).open)
    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *handlers: real_build_opener(*handlers, transport),
    )

    package = model_sources.resolve_package_info(PLATFORM_KEY)

    assert package is None
    assert opened == ["https://models.example.com/flagtune/manifest.json"]
    assert not (tmp_path / "manifest.json").exists()
