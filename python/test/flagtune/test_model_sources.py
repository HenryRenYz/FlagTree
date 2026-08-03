import json
import time
from email.message import Message
from io import BytesIO
from urllib.request import BaseHandler
from urllib.response import addinfourl

import pytest

from triton.flagtune.runtime import model_sources


GPU_KEY = "nvidia-h800-sm90"
DTYPE_KEY = "bf16-bf16-f32"
IDENTITY = f"{GPU_KEY}/vendor/mm/general/{DTYPE_KEY}"
ENTRY_1 = {"url": "https://example.com/v1/model.tar.gz", "sha256": "1" * 64}
ENTRY_2 = {"url": "https://example.com/v2/model.tar.gz", "sha256": "2" * 64}


@pytest.fixture(autouse=True)
def clean_remote_environment(monkeypatch):
    for name in (
        "FLAGTUNE_DISABLE_REMOTE",
        "FLAGTUNE_MANIFEST_URL",
        "FLAGTUNE_MODEL_CACHE",
        "FLAGTUNE_MODEL_URLS",
    ):
        monkeypatch.delenv(name, raising=False)


def install_override(monkeypatch, entry):
    monkeypatch.setenv(
        "FLAGTUNE_MODEL_URLS",
        json.dumps({"models": {IDENTITY: entry}}),
    )


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
    monkeypatch.delenv("FLAGTUNE_MANIFEST_URL", raising=False)
    assert model_sources._manifest_url() == "https://models.example.com/flagtune/manifest.json"
    monkeypatch.setenv("FLAGTUNE_MANIFEST_URL", " https://mirror.example.com/models.json ")
    assert model_sources._manifest_url() == "https://mirror.example.com/models.json"


def test_force_refresh_bypasses_fresh_cached_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "_fetched_at": time.time(),
        "models": {IDENTITY: {"versions": {"1.0.0": ENTRY_1}}},
    }))
    calls = []
    monkeypatch.setattr(model_sources, "_open_https", response_for(
        {"schema_version": 1, "models": {IDENTITY: {"versions": {"2.0.0": ENTRY_2}}}},
        calls,
    ))
    artifact = model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
        force_refresh=True,
    )
    assert artifact.version == "2.0.0"
    assert len(calls) == 1
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


@pytest.mark.parametrize(
    "refreshed",
    [
        [],
        {"models": {}},
        {"schema_version": 2, "models": {}},
        {"schema_version": 1, "models": []},
        {"schema_version": 1, "models": {IDENTITY: []}},
        {"schema_version": 1, "models": {IDENTITY: {"versions": []}}},
        {"schema_version": 1, "models": {IDENTITY: {"versions": {"2.0.0": []}}}},
        {"schema_version": 1, "models": {IDENTITY: {"versions": {"2.0.0": {}}}}},
    ],
)
def test_malformed_refreshed_manifest_preserves_valid_cache(tmp_path, monkeypatch, refreshed):
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    cached = {
        "schema_version": 1,
        "_fetched_at": time.time(),
        "models": {IDENTITY: {"versions": {"1.0.0": ENTRY_1}}},
    }
    cached_path = tmp_path / "manifest.json"
    cached_path.write_text(json.dumps(cached, sort_keys=True))
    cached_bytes = cached_path.read_bytes()
    calls = []
    monkeypatch.setattr(model_sources, "_open_https", response_for(refreshed, calls))

    refreshed_artifact = model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY, force_refresh=True,
    )
    cached_artifact = model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
    )

    expected = model_sources.RemoteArtifact("1.0.0", ENTRY_1["url"], "1" * 64)
    assert refreshed_artifact == cached_artifact == expected
    assert cached_path.read_bytes() == cached_bytes
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("metadata", "accepted"),
    [
        ({"url": "https://example.com/model.tar.gz", "sha256": "a" * 64}, True),
        ({"url": "http://example.com/model.tar.gz", "sha256": "a" * 64}, False),
        ({"url": "https:///model.tar.gz", "sha256": "a" * 64}, False),
        ({"url": "https://example.com/model.tar.gz"}, False),
        ({"url": "https://example.com/model.tar.gz", "sha256": "A" * 64}, False),
        ({"url": "https://example.com/model.tar.gz", "sha256": "a" * 63}, False),
    ],
)
def test_artifact_metadata_requires_https_and_lowercase_sha256(monkeypatch, metadata, accepted):
    install_override(monkeypatch, {"versions": {"1.0.0": metadata}})
    artifact = model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
    )
    assert (artifact is not None) is accepted


def test_exact_and_highest_semver_ignore_latest(monkeypatch):
    install_override(monkeypatch, {
        "latest": "1.0.0",
        "versions": {"1.0.0": ENTRY_1, "2.0.0": ENTRY_2, "newest": ENTRY_1},
    })
    highest = model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
    )
    exact = model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
        version="1.0.0",
    )
    assert highest == model_sources.RemoteArtifact("2.0.0", ENTRY_2["url"], "2" * 64)
    assert exact == model_sources.RemoteArtifact("1.0.0", ENTRY_1["url"], "1" * 64)


def test_legacy_wrappers_keep_their_return_shapes(monkeypatch):
    install_override(monkeypatch, {"versions": {"1.0.0": ENTRY_1}})
    args = ("vendor/mm", "general")
    kwargs = {"gpu_key": GPU_KEY, "dtype_key": DTYPE_KEY}
    assert model_sources.resolve_artifact(*args, **kwargs) == ("1.0.0", ENTRY_1["url"])
    assert model_sources.resolve_url(*args, **kwargs) == ENTRY_1["url"]


def test_unmatched_override_falls_through_to_fresh_cached_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    monkeypatch.setenv("FLAGTUNE_MODEL_URLS", json.dumps({"models": {"other": {}}}))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "_fetched_at": time.time(),
        "models": {IDENTITY: {"versions": {"2.0.0": ENTRY_2}}},
    }))
    artifact = model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
    )
    assert artifact.version == "2.0.0"


@pytest.mark.parametrize("fallback", ["nested", "cache", "builtin"])
def test_empty_exact_override_falls_through_to_lower_priority_entry(tmp_path, monkeypatch, fallback):
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    override = {IDENTITY: {}}
    expected = model_sources.RemoteArtifact("2.0.0", ENTRY_2["url"], "2" * 64)
    if fallback == "nested":
        override["models"] = {IDENTITY: {"versions": {"2.0.0": ENTRY_2}}}
    elif fallback == "cache":
        (tmp_path / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "_fetched_at": time.time(),
            "models": {IDENTITY: {"versions": {"2.0.0": ENTRY_2}}},
        }))
    else:
        monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")
        monkeypatch.setitem(model_sources._BUILTIN_TABLE, IDENTITY, {"versions": {"1.0.0": ENTRY_1}})
        expected = model_sources.RemoteArtifact("1.0.0", ENTRY_1["url"], "1" * 64)
    monkeypatch.setenv("FLAGTUNE_MODEL_URLS", json.dumps(override))

    artifact = model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
    )

    assert artifact == expected


@pytest.mark.parametrize(
    "override",
    [
        ["not-a-mapping"],
        {"models": ["not-a-mapping"]},
        {IDENTITY: ["not-an-entry"]},
        {"models": {IDENTITY: ["not-an-entry"]}},
    ],
)
def test_malformed_override_mappings_fall_through_to_cached_manifest(tmp_path, monkeypatch, override):
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    monkeypatch.setenv("FLAGTUNE_MODEL_URLS", json.dumps(override))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "_fetched_at": time.time(),
        "models": {IDENTITY: {"versions": {"2.0.0": ENTRY_2}}},
    }))

    artifact = model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
    )

    assert artifact == model_sources.RemoteArtifact("2.0.0", ENTRY_2["url"], "2" * 64)


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {"schema_version": 1, "_fetched_at": time.time(), "models": ["not-a-mapping"]},
        {"schema_version": 1, "_fetched_at": time.time(), "models": {IDENTITY: ["not-an-entry"]}},
    ],
)
def test_malformed_manifest_mappings_fall_through_to_builtin(tmp_path, monkeypatch, manifest):
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setitem(model_sources._BUILTIN_TABLE, IDENTITY, {"versions": {"1.0.0": ENTRY_1}})

    artifact = model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
    )

    assert artifact == model_sources.RemoteArtifact("1.0.0", ENTRY_1["url"], "1" * 64)


@pytest.mark.parametrize(
    "cached_payload",
    [
        "{not-json",
        json.dumps([]),
        json.dumps({"schema_version": 1, "_fetched_at": "not-a-timestamp", "models": {}}),
        json.dumps({"schema_version": 1, "_fetched_at": 0, "models": {}}),
        json.dumps({"_fetched_at": time.time(), "models": {}}),
        json.dumps({"schema_version": 2, "_fetched_at": time.time(), "models": {}}),
        json.dumps({"schema_version": 1, "_fetched_at": time.time(), "models": []}),
        json.dumps({"schema_version": 1, "_fetched_at": time.time(), "models": {IDENTITY: []}}),
        json.dumps({
            "schema_version": 1,
            "_fetched_at": time.time(),
            "models": {IDENTITY: {"versions": []}},
        }),
        json.dumps({
            "schema_version": 1,
            "_fetched_at": time.time(),
            "models": {IDENTITY: {"versions": {"1.0.0": []}}},
        }),
        json.dumps({
            "schema_version": 1,
            "_fetched_at": time.time(),
            "models": {IDENTITY: {"versions": {"1.0.0": {}}}},
        }),
    ],
)
def test_malformed_or_stale_cached_manifest_is_replaced(tmp_path, monkeypatch, cached_payload):
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    (tmp_path / "manifest.json").write_text(cached_payload)
    calls = []
    monkeypatch.setattr(model_sources, "_open_https", response_for(
        {"schema_version": 1, "models": {IDENTITY: {"versions": {"2.0.0": ENTRY_2}}}},
        calls,
    ))
    artifact = model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
    )
    assert artifact.version == "2.0.0"
    assert len(calls) == 1
    assert json.loads((tmp_path / "manifest.json").read_text())["models"][IDENTITY]


def test_non_https_manifest_url_never_opens_network(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    monkeypatch.setenv("FLAGTUNE_MANIFEST_URL", "http://example.com/manifest.json")
    monkeypatch.setattr(model_sources, "_open_https", lambda *_args, **_kwargs: pytest.fail("network opened"))
    assert model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
    ) is None


def test_manifest_https_redirect_to_http_is_rejected(tmp_path, monkeypatch):
    import urllib.request

    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    payload = json.dumps({
        "schema_version": 1,
        "models": {IDENTITY: {"versions": {"2.0.0": ENTRY_2}}},
    }).encode()
    opened = []
    transport = _RedirectingTransport(payload, "http://example.com/manifest.json", opened)
    real_build_opener = urllib.request.build_opener
    monkeypatch.setattr("urllib.request.urlopen", real_build_opener(transport).open)
    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *handlers: real_build_opener(*handlers, transport),
    )

    artifact = model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
    )

    assert artifact is None
    assert opened == ["https://models.example.com/flagtune/manifest.json"]
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("disabled_value", ["0", " ", "false"])
def test_any_nonempty_disable_remote_value_never_opens_network(tmp_path, monkeypatch, disabled_value):
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", disabled_value)
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    monkeypatch.setattr(model_sources, "_open_https", lambda *_args, **_kwargs: pytest.fail("network opened"))

    assert model_sources.resolve_artifact_info(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
    ) is None
