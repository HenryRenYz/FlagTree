from __future__ import annotations

import hashlib
import json
import math
import numpy as np
import pytest

from triton.flagtune.runtime import proposer as predict
from triton.flagtune.contract.archive import read_model_archive, write_model_archive
from triton.flagtune.contract.identity import (
    ModelIdentity,
    ModelIdentityError,
    artifact_key,
    gpu_metadata,
    make_dtype_key,
    make_gpu_key,
    normalize_dtype_name,
)
from triton.flagtune.runtime.model_loader import FlagTuneModelManager, IncompatibleModelError
from triton.flagtune.contract.operator_schema import (
    BUILTIN_OPS,
    FlagTuneConfigError,
    load_model_config,
    load_operator_config,
    model_config_sha256,
    parse_model_config,
    parse_operator_config,
    variant_to_model_config,
)

GPU = dict(gpu_metadata(
    backend="cuda",
    vendor="nvidia",
    device_name="NVIDIA H800 80GB HBM3",
    architecture="sm90",
))
H20_GPU = dict(gpu_metadata(
    backend="cuda",
    vendor="nvidia",
    device_name="NVIDIA H20-3e",
    architecture="sm90",
))
GPU_KEY = GPU["gpu_key"]
DTYPES = ["bfloat16", "bfloat16", "float32"]
DTYPE_KEY = "bf16-bf16-f32"
MODEL_VERSION = "1.2.3"


def _identity(op_id="vendor/mm", variant="general"):
    return ModelIdentity(GPU_KEY, op_id, variant, DTYPE_KEY)


def _archive(path, config=b"{}", model=b"{}"):
    return write_model_archive(
        path,
        {
            "xgboost_ranker.json": model,
            "flagtune_config.yaml": config,
            "training_summary.json": b"{}",
        },
    )


def _configure_remote_download(monkeypatch, tmp_path, payload, *, url="https://example.invalid/model.tar.gz", sha256=None):
    requests = []
    digest = sha256 or hashlib.sha256(payload).hexdigest()

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
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(
        "FLAGTUNE_MODEL_URLS",
        json.dumps({
            "models": {
                f"{GPU_KEY}/vendor/mm/general/{DTYPE_KEY}": {
                    "versions": {
                        "2.0.0": {"url": url, "sha256": digest},
                    },
                },
            },
        }),
    )
    monkeypatch.setattr("triton.flagtune.runtime.model_sources._open_https", urlopen)
    return requests


def _config():
    return {
        "op_id": "vendor/mm",
        "variants": {
            "general": {
                "inputs": {
                    "M": {"min": 1},
                    "N": {},
                    "K": {},
                    "stride_am": {"default": "K"},
                    "stride_bk": {"default": "N"},
                },
                "when": {"op": "gt", "args": ["N", 1]},
                "params": {
                    "BLOCK_M": {"values": [16, 32]},
                    "num_warps": {"values": [4, 8]},
                },
                "features": [
                    "M",
                    {"name": "N", "op": "ident", "args": ["N"]},
                    {"name": "tile", "op": "mul", "args": ["M", "BLOCK_M"]},
                    {"name": "grid", "op": "cdiv", "args": ["M", "BLOCK_M"]},
                    {"name": "ratio", "op": "fdiv", "args": ["BLOCK_M", "M"]},
                    {"name": "aligned", "op": "alignup", "args": ["M", 16]},
                    {"name": "power", "op": "pow", "args": [2, 3]},
                    {"name": "log_tile", "op": "log2", "args": ["tile"]},
                ],
            }
        },
    }


def _export_model_archive(root, model_version):
    xgboost = pytest.importorskip("xgboost")
    from triton.flagtune.training.ranker import export_ranker_model

    variant = parse_operator_config(_config()).get_variant("general")
    model = xgboost.XGBRanker(n_estimators=0)
    model.fit(np.zeros((2, len(variant.feature_names))), np.zeros(2), group=[2])
    return export_ranker_model(
        model,
        variant,
        root,
        {},
        identity=_identity(),
        dtypes=DTYPES,
        gpu=GPU,
        model_version=model_version,
    ).model_path


@pytest.fixture(autouse=True)
def clean_model_manager(monkeypatch):
    """Reset lazy runtime caches and environment overrides around each test."""
    predict._MODEL_MANAGER = None
    predict._TOP_K_CACHE = None
    monkeypatch.delenv("FLAGTUNE_DISABLE_OPS", raising=False)
    monkeypatch.delenv("FLAGTUNE_TOP_K", raising=False)
    yield
    predict._MODEL_MANAGER = None
    predict._TOP_K_CACHE = None


def test_parse_operator_builds_inputs_params_and_ordered_features():
    """Compile an operator without changing process-global state."""
    info = parse_operator_config(_config())
    variant = info.get_variant("general")
    inputs = variant.normalize_inputs({"M": 33, "N": 8, "K": 64})

    assert inputs == {"M": 33, "N": 8, "K": 64, "stride_am": 64, "stride_bk": 8}
    assert variant.matches(inputs)
    assert list(variant.iter_configs()) == [
        {"BLOCK_M": 16, "num_warps": 4},
        {"BLOCK_M": 16, "num_warps": 8},
        {"BLOCK_M": 32, "num_warps": 4},
        {"BLOCK_M": 32, "num_warps": 8},
    ]

    rows = variant.build_feature_rows(inputs, [{"BLOCK_M": 16, "num_warps": 4}])
    assert list(rows[0]) == ["M", "N", "tile", "grid", "ratio", "aligned", "power", "log_tile"]
    assert rows[0] == pytest.approx({
        "M": 33,
        "N": 8,
        "tile": 528,
        "grid": 3,
        "ratio": 16 / 33,
        "aligned": 48,
        "power": 8,
        "log_tile": math.log2(528),
    })


def test_when_rejects_wrong_variant_shape():
    variant = parse_operator_config(_config()).get_variant("general")
    assert not variant.matches({"M": 32, "N": 1, "K": 64})


@pytest.mark.parametrize(
    "disabled",
    ["*", "vendor/mm", "vendor/mm/general", f"{GPU_KEY}/vendor/mm/general/{DTYPE_KEY}"],
)
def test_disable_rules_accept_global_operator_and_exact_pair(monkeypatch, disabled):
    """Disable without loading a bundle at all three supported scopes."""
    monkeypatch.setenv("FLAGTUNE_DISABLE_OPS", disabled)
    proposer = predict.make_config_proposer("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)
    assert proposer(None, {}, [], {}) == []


def test_legacy_identity_alias_is_rejected():
    """Keep the operator/variant pair as the only model identity."""
    config = _config()
    legacy_key = "model" + "_" + "id"
    config["variants"]["general"][legacy_key] = "legacy/general"
    with pytest.raises(FlagTuneConfigError, match="unknown keys"):
        parse_operator_config(config)


def test_builtin_comparison_logic_and_power_operations_are_available():
    required = {
        "ident",
        "add",
        "sub",
        "mul",
        "div",
        "cdiv",
        "fdiv",
        "mod",
        "log2",
        "pow",
        "alignup",
        "aligndown",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "all",
        "any",
        "not",
    }
    assert required <= set(BUILTIN_OPS)
    assert BUILTIN_OPS["pow"](2, 5) == 32


def test_gpu_and_ordered_tensor_dtype_identity_is_canonical():
    assert make_gpu_key("NVIDIA", "NVIDIA H800 80GB HBM3", "sm90") == ("nvidia-h800-80gb-hbm3-sm90")
    assert make_gpu_key("NVIDIA", "NVIDIA H800 80GB HBM3", "sm90") == GPU_KEY
    assert normalize_dtype_name("torch.bfloat16") == "bfloat16"
    assert make_dtype_key(["bfloat16", "float16", "float32"]) == "bf16-f16-f32"
    with pytest.raises(ModelIdentityError, match="unsupported tensor dtype"):
        make_dtype_key(["float8_unknown"])


def test_yaml_loading_forwards_to_stateless_compiler(tmp_path):
    pytest.importorskip("yaml")
    config_path = tmp_path / "operator.yaml"
    config_path.write_text(
        """
op_id: vendor/add
variants:
  general:
    inputs:
      N: {}
    params:
      BLOCK: {values: [32]}
    features:
      - N
      - {name: BLOCK, op: ident, args: [BLOCK]}
""".strip(),
        encoding="utf-8",
    )
    info = load_operator_config(config_path)
    assert info.op_id == "vendor/add"
    assert artifact_key(GPU_KEY, info.op_id, "general", DTYPE_KEY) == (f"{GPU_KEY}/vendor/add/general/{DTYPE_KEY}")


def test_registration_rejects_unknown_variables_and_unsafe_identities():
    bad_feature = _config()
    bad_feature["variants"]["general"]["features"].append({"name": "bad", "op": "ident", "args": ["missing"]})
    with pytest.raises(FlagTuneConfigError, match="unknown symbol"):
        parse_operator_config(bad_feature)

    bad_op = _config()
    bad_op["op_id"] = "../outside"
    with pytest.raises(FlagTuneConfigError, match="segment"):
        parse_operator_config(bad_op)

    bad_variant = _config()
    bad_variant["variants"]["bad/name"] = bad_variant["variants"].pop("general")
    with pytest.raises(FlagTuneConfigError, match="variants key"):
        parse_operator_config(bad_variant)


def test_operator_variant_resolves_as_nested_local_path(tmp_path, monkeypatch):
    model_path = tmp_path / GPU_KEY / "vendor" / "mm" / "general" / DTYPE_KEY / MODEL_VERSION / "model.tar.gz"
    _archive(model_path)
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))

    assert FlagTuneModelManager().resolve("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY) == model_path


def test_cache_resolution_keeps_version_below_derived_pair(tmp_path, monkeypatch):
    """Resolve versioned cache bundles beneath op_id/variant."""
    model_path = tmp_path / GPU_KEY / "vendor" / "mm" / "general" / DTYPE_KEY / "2.0.0" / "model.tar.gz"
    _archive(model_path)
    monkeypatch.delenv("FLAGTUNE_MODEL_DIR", raising=False)
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))

    assert FlagTuneModelManager().resolve("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY) == model_path


def test_remote_manifest_uses_derived_pair_key(monkeypatch):
    """Address remote artifacts with op_id/variant and retain version selection."""
    from triton.flagtune.runtime.model_sources import resolve_url

    manifest = {
        "models": {
            f"{GPU_KEY}/vendor/mm/general/{DTYPE_KEY}": {
                "latest": "1.0.0",
                "versions": {
                    "1.0.0": {"url": "https://example.invalid/old.tgz", "sha256": "1" * 64},
                    "2.0.0": {"url": "https://example.invalid/model.tgz", "sha256": "2" * 64},
                },
            }
        }
    }
    monkeypatch.setenv("FLAGTUNE_MODEL_URLS", json.dumps(manifest))
    assert resolve_url("vendor/mm", "general", gpu_key=GPU_KEY,
                       dtype_key=DTYPE_KEY) == "https://example.invalid/model.tgz"


def test_remote_download_uses_pair_and_manifest_version(tmp_path, monkeypatch):
    """Download a pair bundle into its exact versioned cache directory."""
    source = _export_model_archive(tmp_path / "source", "2.0.0")
    archive_payload = source.read_bytes()
    requests = _configure_remote_download(
        monkeypatch,
        tmp_path,
        archive_payload,
        url="https://example.invalid/model.tgz",
    )

    expected = tmp_path / "cache" / GPU_KEY / "vendor" / "mm" / "general" / DTYPE_KEY / "2.0.0" / "model.tar.gz"
    assert FlagTuneModelManager().resolve("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY) == expected
    assert expected.is_file()
    assert list(expected.parent.iterdir()) == [expected]
    assert len(requests) == 1


def test_remote_download_rejects_digest_mismatch(tmp_path, monkeypatch):
    source = _archive(tmp_path / "source" / "model.tar.gz")
    payload = source.read_bytes()

    class Response:

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return payload

    monkeypatch.delenv("FLAGTUNE_MODEL_DIR", raising=False)
    monkeypatch.delenv("FLAGTUNE_DISABLE_REMOTE", raising=False)
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(
        "FLAGTUNE_MODEL_URLS",
        json.dumps({
            "models": {
                f"{GPU_KEY}/vendor/mm/general/{DTYPE_KEY}": {
                    "versions": {
                        "2.0.0": {
                            "url": "https://example.invalid/model.tar.gz",
                            "sha256": "0" * 64,
                        },
                    },
                },
            },
        }),
    )
    monkeypatch.setattr(
        "triton.flagtune.runtime.model_sources._open_https",
        lambda *_args, **_kwargs: Response(),
    )

    with pytest.raises(FileNotFoundError):
        FlagTuneModelManager().resolve(
            "vendor/mm",
            "general",
            gpu_key=GPU_KEY,
            dtype_key=DTYPE_KEY,
        )
    assert not list((tmp_path / "cache").rglob("model.tar.gz"))
    assert not list((tmp_path / "cache").rglob("*.tmp"))


def test_remote_download_verifies_digest_and_reuses_cached_bytes(tmp_path, monkeypatch):
    source = _export_model_archive(tmp_path / "source", "2.0.0")
    payload = source.read_bytes()
    requests = _configure_remote_download(monkeypatch, tmp_path, payload)
    manager = FlagTuneModelManager()

    first = manager.resolve("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)
    second = manager.resolve("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)

    assert first == second
    assert first.read_bytes() == payload
    assert len(requests) == 1


def test_remote_download_rejects_digest_valid_corrupt_archive(tmp_path, monkeypatch):
    payload = b"not a gzip tar"
    _configure_remote_download(monkeypatch, tmp_path, payload)

    with pytest.raises(FileNotFoundError):
        FlagTuneModelManager().resolve(
            "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
        )
    assert not list((tmp_path / "cache").rglob("model.tar.gz"))
    assert not list((tmp_path / "cache").rglob("*.tmp"))


def test_refresh_rejects_incompatible_remote_before_cache_commit(tmp_path, monkeypatch):
    xgboost = pytest.importorskip("xgboost")
    yaml = pytest.importorskip("yaml")
    from triton.flagtune.training.ranker import export_ranker_model

    variant = parse_operator_config(_config()).get_variant("general")
    model = xgboost.XGBRanker(n_estimators=0)
    model.fit(np.zeros((2, len(variant.feature_names))), np.zeros(2), group=[2])
    cached = export_ranker_model(
        model,
        variant,
        tmp_path / "cache",
        {},
        identity=_identity(),
        dtypes=DTYPES,
        gpu=GPU,
        model_version="1.0.0",
    ).model_path
    remote = export_ranker_model(
        model,
        variant,
        tmp_path / "remote",
        {},
        identity=_identity(),
        dtypes=DTYPES,
        gpu=GPU,
        model_version="2.0.0",
    ).model_path
    members = read_model_archive(remote)
    config = yaml.safe_load(members["flagtune_config.yaml"])
    config["op_id"] = "vendor/other"
    members["flagtune_config.yaml"] = yaml.safe_dump(config, sort_keys=False).encode()
    write_model_archive(remote, members)
    requests = _configure_remote_download(monkeypatch, tmp_path, remote.read_bytes())
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", "1")

    loaded = FlagTuneModelManager().load(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
    )

    assert loaded.model_version == "1.0.0"
    assert loaded.model_path == cached
    assert len(requests) == 1
    assert not (cached.parents[1] / "2.0.0" / "model.tar.gz").exists()


def test_download_replaces_concurrently_cached_incompatible_bundle(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")

    remote = _export_model_archive(tmp_path / "remote", "2.0.0")
    concurrent = _export_model_archive(tmp_path / "concurrent", "2.0.0")
    concurrent_members = read_model_archive(concurrent)
    concurrent_config = yaml.safe_load(concurrent_members["flagtune_config.yaml"])
    concurrent_config["op_id"] = "vendor/other"
    concurrent_members["flagtune_config.yaml"] = yaml.safe_dump(
        concurrent_config,
        sort_keys=False,
    ).encode()
    write_model_archive(concurrent, concurrent_members)

    requests = _configure_remote_download(monkeypatch, tmp_path, remote.read_bytes())
    manager = FlagTuneModelManager()
    validate_remote = manager._load_bundle_members

    def publish_incompatible_bundle(identity, version, members, model_path):
        loaded = validate_remote(identity, version, members, model_path)
        if not model_path.exists():
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(concurrent.read_bytes())
        return loaded

    monkeypatch.setattr(manager, "_load_bundle_members", publish_incompatible_bundle)

    selected = manager.resolve(
        "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
    )

    assert selected.read_bytes() == remote.read_bytes()
    assert len(requests) == 1


@pytest.mark.parametrize("url", [
    "http://example.invalid/model.tar.gz",
    "https://example.invalid/model.zip",
])
def test_remote_download_rejects_non_https_or_non_archive_url(tmp_path, monkeypatch, url):
    payload = b"not downloaded"
    requests = _configure_remote_download(monkeypatch, tmp_path, payload, url=url)

    with pytest.raises(FileNotFoundError):
        FlagTuneModelManager().resolve(
            "vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY,
        )
    assert requests == []


def test_single_model_config_round_trip_preserves_contract(tmp_path):
    """Serialize and compile one variant with ordered inputs, params, and features."""
    yaml = pytest.importorskip("yaml")
    variant = parse_operator_config(_config()).get_variant("general")
    config = variant_to_model_config(variant, _identity(), DTYPES, GPU, MODEL_VERSION)
    path = tmp_path / "flagtune_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    loaded, raw = load_model_config(path)
    assert (loaded.op_id, loaded.name) == (variant.op_id, variant.name)
    assert loaded.feature_names == variant.feature_names
    assert loaded.param_names == variant.param_names
    assert model_config_sha256(raw) == model_config_sha256(config)
    other_version = variant_to_model_config(variant, _identity(), DTYPES, GPU, "1.2.4")
    assert model_config_sha256(other_version) != model_config_sha256(config)


def test_bundle_identity_must_match_requested_pair(tmp_path, monkeypatch):
    """Reject a bundle stored under a different canonical pair."""
    yaml = pytest.importorskip("yaml")
    wrong = _config()
    wrong["op_id"] = "vendor/other"
    wrong_variant = parse_operator_config(wrong).get_variant("general")
    config = variant_to_model_config(wrong_variant, _identity("vendor/other"), DTYPES, GPU, MODEL_VERSION)
    model_path = tmp_path / GPU_KEY / "vendor" / "mm" / "general" / DTYPE_KEY / MODEL_VERSION / "model.tar.gz"
    _archive(model_path, yaml.safe_dump(config, sort_keys=False).encode())
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))

    with pytest.raises(IncompatibleModelError, match="identity mismatch"):
        FlagTuneModelManager().load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)


def test_bundle_version_must_match_version_directory(tmp_path, monkeypatch):
    """Reject a valid format-v5 config copied below a different revision."""
    yaml = pytest.importorskip("yaml")
    variant = parse_operator_config(_config()).get_variant("general")
    config = variant_to_model_config(variant, _identity(), DTYPES, GPU, "1.0.0")
    model_path = tmp_path / GPU_KEY / "vendor" / "mm" / "general" / DTYPE_KEY / "2.0.0" / "model.tar.gz"
    _archive(model_path, yaml.safe_dump(config, sort_keys=False).encode())
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))

    with pytest.raises(IncompatibleModelError, match="version mismatch"):
        FlagTuneModelManager().load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)


def test_model_config_rejects_unknown_custom_operation():
    """Keep exported bundles independent from external Python callables."""
    config = variant_to_model_config(
        parse_operator_config(_config()).get_variant("general"),
        _identity(),
        DTYPES,
        GPU,
        MODEL_VERSION,
    )
    config["when"] = {"op": "external_policy", "args": ["inputs"]}
    with pytest.raises(FlagTuneConfigError, match="unknown operation"):
        parse_model_config(config)


@pytest.mark.parametrize(
    ("declared_gpu", "declared_dtypes"),
    [
        (H20_GPU, DTYPES),
        (GPU, ["float16", "float16", "float32"]),
    ],
)
def test_bundle_identity_isolates_gpu_and_dtype(tmp_path, monkeypatch, declared_gpu, declared_dtypes):
    yaml = pytest.importorskip("yaml")
    variant = parse_operator_config(_config()).get_variant("general")
    declared = ModelIdentity(
        declared_gpu["gpu_key"],
        variant.op_id,
        variant.name,
        make_dtype_key(declared_dtypes),
    )
    config = variant_to_model_config(variant, declared, declared_dtypes, declared_gpu, MODEL_VERSION)
    requested = tmp_path / GPU_KEY / "vendor" / "mm" / "general" / DTYPE_KEY / MODEL_VERSION / "model.tar.gz"
    _archive(requested, yaml.safe_dump(config, sort_keys=False).encode())
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))

    with pytest.raises(IncompatibleModelError, match="identity mismatch"):
        FlagTuneModelManager().load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)


def test_untrained_empty_xgboost_model_runs_the_candidate_pipeline(tmp_path, monkeypatch):
    xgboost = pytest.importorskip("xgboost")
    from triton.flagtune.training.ranker import export_ranker_model

    feature_names = ["M", "N", "tile", "grid", "ratio", "aligned", "power", "log_tile"]
    empty_model = xgboost.XGBRanker(n_estimators=0)
    empty_model.fit(np.zeros((2, len(feature_names))), np.zeros(2), group=[2])
    variant = parse_operator_config(_config()).get_variant("general")
    export_ranker_model(
        empty_model,
        variant,
        tmp_path,
        {},
        identity=_identity(),
        dtypes=DTYPES,
        gpu=GPU,
        model_version=MODEL_VERSION,
    )
    assert empty_model.get_booster().get_dump() == []

    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("FLAGTUNE_TOP_K", "2")

    proposer = predict.make_config_proposer("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)
    result = proposer(None, {"M": 33, "N": 8, "K": 64}, [], {})
    assert result == [
        {"BLOCK_M": 16, "num_warps": 4},
        {"BLOCK_M": 16, "num_warps": 8},
    ]


def test_loaded_model_cache_isolated_by_explicit_version(tmp_path, monkeypatch):
    """Keep two revisions of the same four-component identity independent."""
    xgboost = pytest.importorskip("xgboost")
    from triton.flagtune.training.ranker import export_ranker_model

    variant = parse_operator_config(_config()).get_variant("general")
    model = xgboost.XGBRanker(n_estimators=0)
    model.fit(np.zeros((2, len(variant.feature_names))), np.zeros(2), group=[2])
    for version in ("1.0.0", "2.0.0"):
        export_ranker_model(
            model,
            variant,
            tmp_path,
            {},
            identity=_identity(),
            dtypes=DTYPES,
            gpu=GPU,
            model_version=version,
        )
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))
    manager = FlagTuneModelManager()

    first = manager.load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY, model_version="1.0.0")
    second = manager.load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY, model_version="2.0.0")
    latest = manager.load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)
    assert first.model_version == "1.0.0"
    assert second.model_version == latest.model_version == "2.0.0"
    assert first.model_path != second.model_path
    assert second is latest


def test_implicit_load_reuses_first_bundle_before_reresolution(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    _export_model_archive(model_root, "1.0.0")
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(model_root))
    manager = FlagTuneModelManager()
    resolve_calls = []
    real_resolve = manager.resolve

    def counted_resolve(*args, **kwargs):
        resolve_calls.append(kwargs.get("model_version"))
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(manager, "resolve", counted_resolve)
    first = manager.load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)

    _export_model_archive(model_root, "2.0.0")
    monkeypatch.setenv("FLAGTUNE_MODEL_REFRESH", "1")
    monkeypatch.setenv("FLAGTUNE_MODEL_VERSION", "2.0.0")
    second = manager.load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)

    assert second is first
    assert second.model_version == "1.0.0"
    assert resolve_calls == [None]


def test_modified_config_is_rejected_by_embedded_model_digest(tmp_path, monkeypatch):
    """Reject a valid YAML config that no longer belongs to its XGBoost file."""
    xgboost = pytest.importorskip("xgboost")
    yaml = pytest.importorskip("yaml")
    from triton.flagtune.training.ranker import export_ranker_model

    variant = parse_operator_config(_config()).get_variant("general")
    model = xgboost.XGBRanker(n_estimators=0)
    model.fit(np.zeros((2, len(variant.feature_names))), np.zeros(2), group=[2])
    exported = export_ranker_model(
        model,
        variant,
        tmp_path,
        {},
        identity=_identity(),
        dtypes=DTYPES,
        gpu=GPU,
        model_version=MODEL_VERSION,
    )
    members = read_model_archive(exported.model_path)
    config = yaml.safe_load(members["flagtune_config.yaml"])
    config["features"].pop()
    members["flagtune_config.yaml"] = yaml.safe_dump(config, sort_keys=False).encode()
    write_model_archive(exported.model_path, members)
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))

    with pytest.raises(IncompatibleModelError, match="digest mismatch"):
        FlagTuneModelManager().load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)


def test_embedded_xgboost_feature_order_must_match_config(tmp_path, monkeypatch):
    """Reject weights whose named columns were reordered after export."""
    xgboost = pytest.importorskip("xgboost")
    from triton.flagtune.training.ranker import export_ranker_model

    variant = parse_operator_config(_config()).get_variant("general")
    model = xgboost.XGBRanker(n_estimators=0)
    model.fit(np.zeros((2, len(variant.feature_names))), np.zeros(2), group=[2])
    exported = export_ranker_model(
        model,
        variant,
        tmp_path,
        {},
        identity=_identity(),
        dtypes=DTYPES,
        gpu=GPU,
        model_version=MODEL_VERSION,
    )
    members = read_model_archive(exported.model_path)
    changed = xgboost.XGBRanker()
    changed.load_model(bytearray(members["xgboost_ranker.json"]))
    changed.get_booster().feature_names = list(reversed(variant.feature_names))
    loose_path = tmp_path / "changed.json"
    changed.save_model(str(loose_path))
    members["xgboost_ranker.json"] = loose_path.read_bytes()
    write_model_archive(exported.model_path, members)
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(tmp_path))

    with pytest.raises(IncompatibleModelError, match="feature order mismatch"):
        FlagTuneModelManager().load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)
