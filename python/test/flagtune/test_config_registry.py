from __future__ import annotations

import json
import io
import math
import tarfile
import numpy as np
import pytest

from triton.flagtune import predict
from triton.flagtune.artifacts import read_model_archive, write_model_archive
from triton.flagtune.identity import (
    ModelIdentity,
    ModelIdentityError,
    artifact_key,
    gpu_metadata,
    make_dtype_key,
    make_gpu_key,
    normalize_dtype_name,
)
from triton.flagtune.model_manager import FlagTuneModelManager, IncompatibleModelError
from triton.flagtune.registry import (
    BUILTIN_OPS,
    FlagTuneConfigError,
    load_model_config,
    load_operator_config,
    model_config_sha256,
    parse_model_config,
    parse_operator_config,
    variant_to_model_config,
)

GPU = dict(gpu_metadata("nvidia", "NVIDIA H800 80GB HBM3", (9, 0)))
H20_GPU = dict(gpu_metadata("nvidia", "NVIDIA H20-3e", (9, 0)))
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


@pytest.fixture(autouse=True)
def clean_model_manager(monkeypatch):
    """Reset lazy runtime caches and environment overrides around each test."""
    predict._MODEL_MANAGER = None
    predict._TOP_K_CACHE = None
    monkeypatch.delenv("FLAGTUNE_DISABLE_OPS", raising=False)
    monkeypatch.delenv("TRITON_FLAGTUNE_TOP_K", raising=False)
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
    assert make_gpu_key("NVIDIA", "NVIDIA H800 80GB HBM3", (9, 0)) == ("nvidia-h800-80gb-hbm3-sm90")
    assert make_gpu_key("NVIDIA", "NVIDIA H800 80GB HBM3", (9, 0)) == GPU_KEY
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
    monkeypatch.setenv("TRITON_FLAGTUNE_MODEL_DIR", str(tmp_path))

    assert FlagTuneModelManager().resolve("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY) == model_path


def test_cache_resolution_keeps_version_below_derived_pair(tmp_path, monkeypatch):
    """Resolve versioned cache bundles beneath op_id/variant."""
    model_path = tmp_path / GPU_KEY / "vendor" / "mm" / "general" / DTYPE_KEY / "2.0.0" / "model.tar.gz"
    _archive(model_path)
    monkeypatch.delenv("TRITON_FLAGTUNE_MODEL_DIR", raising=False)
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))

    assert FlagTuneModelManager().resolve("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY) == model_path


def test_remote_manifest_uses_derived_pair_key(monkeypatch):
    """Address remote artifacts with op_id/variant and retain version selection."""
    from triton.flagtune.model_urls import resolve_url

    manifest = {
        "models": {
            f"{GPU_KEY}/vendor/mm/general/{DTYPE_KEY}": {
                "latest": "1.0.0",
                "versions": {
                    "1.0.0": {"url": "https://example.invalid/old.tgz"},
                    "2.0.0": {"url": "https://example.invalid/model.tgz"},
                },
            }
        }
    }
    monkeypatch.setenv("FLAGTUNE_MODEL_URLS", json.dumps(manifest))
    assert resolve_url("vendor/mm", "general", gpu_key=GPU_KEY,
                       dtype_key=DTYPE_KEY) == "https://example.invalid/model.tgz"


def test_remote_download_uses_pair_and_manifest_version(tmp_path, monkeypatch):
    """Download a pair bundle into its exact versioned cache directory."""
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as handle:
        for name, payload in (
            ("flagtune_config.yaml", b"{}"),
            ("xgboost_ranker.json", b"{}"),
            ("training_summary.json", b"{}"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))

    class Response:

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return archive.getvalue()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path))
    monkeypatch.setenv(
        "FLAGTUNE_MODEL_URLS",
        json.dumps({
            "models": {
                f"{GPU_KEY}/vendor/mm/general/{DTYPE_KEY}": {
                    "latest": "2.0.0",
                    "versions": {"2.0.0": {"url": "https://example.invalid/model.tgz"}},
                }
            }
        }),
    )

    expected = tmp_path / GPU_KEY / "vendor" / "mm" / "general" / DTYPE_KEY / "2.0.0" / "model.tar.gz"
    assert FlagTuneModelManager().resolve("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY) == expected
    assert expected.is_file()
    assert list(expected.parent.iterdir()) == [expected]


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
    monkeypatch.setenv("TRITON_FLAGTUNE_MODEL_DIR", str(tmp_path))

    with pytest.raises(IncompatibleModelError, match="identity mismatch"):
        FlagTuneModelManager().load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)


def test_bundle_version_must_match_version_directory(tmp_path, monkeypatch):
    """Reject a valid format-v4 config copied below a different revision."""
    yaml = pytest.importorskip("yaml")
    variant = parse_operator_config(_config()).get_variant("general")
    config = variant_to_model_config(variant, _identity(), DTYPES, GPU, "1.0.0")
    model_path = tmp_path / GPU_KEY / "vendor" / "mm" / "general" / DTYPE_KEY / "2.0.0" / "model.tar.gz"
    _archive(model_path, yaml.safe_dump(config, sort_keys=False).encode())
    monkeypatch.setenv("TRITON_FLAGTUNE_MODEL_DIR", str(tmp_path))

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
    monkeypatch.setenv("TRITON_FLAGTUNE_MODEL_DIR", str(tmp_path))

    with pytest.raises(IncompatibleModelError, match="identity mismatch"):
        FlagTuneModelManager().load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)


def test_untrained_empty_xgboost_model_runs_the_candidate_pipeline(tmp_path, monkeypatch):
    xgboost = pytest.importorskip("xgboost")
    from triton.flagtune.training import export_ranker_model

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

    monkeypatch.setenv("TRITON_FLAGTUNE_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_FLAGTUNE_TOP_K", "2")

    proposer = predict.make_config_proposer("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)
    result = proposer(None, {"M": 33, "N": 8, "K": 64}, [], {})
    assert result == [
        {"BLOCK_M": 16, "num_warps": 4},
        {"BLOCK_M": 16, "num_warps": 8},
    ]


def test_loaded_model_cache_isolated_by_explicit_version(tmp_path, monkeypatch):
    """Keep two revisions of the same four-component identity independent."""
    xgboost = pytest.importorskip("xgboost")
    from triton.flagtune.training import export_ranker_model

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
    monkeypatch.setenv("TRITON_FLAGTUNE_MODEL_DIR", str(tmp_path))
    manager = FlagTuneModelManager()

    first = manager.load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY, model_version="1.0.0")
    second = manager.load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY, model_version="2.0.0")
    latest = manager.load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)
    assert first.model_version == "1.0.0"
    assert second.model_version == latest.model_version == "2.0.0"
    assert first.model_path != second.model_path
    assert second is latest


def test_modified_config_is_rejected_by_embedded_model_digest(tmp_path, monkeypatch):
    """Reject a valid YAML config that no longer belongs to its XGBoost file."""
    xgboost = pytest.importorskip("xgboost")
    yaml = pytest.importorskip("yaml")
    from triton.flagtune.training import export_ranker_model

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
    monkeypatch.setenv("TRITON_FLAGTUNE_MODEL_DIR", str(tmp_path))

    with pytest.raises(IncompatibleModelError, match="digest mismatch"):
        FlagTuneModelManager().load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)


def test_embedded_xgboost_feature_order_must_match_config(tmp_path, monkeypatch):
    """Reject weights whose named columns were reordered after export."""
    xgboost = pytest.importorskip("xgboost")
    from triton.flagtune.training import export_ranker_model

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
    monkeypatch.setenv("TRITON_FLAGTUNE_MODEL_DIR", str(tmp_path))

    with pytest.raises(IncompatibleModelError, match="feature order mismatch"):
        FlagTuneModelManager().load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)


def test_migration_reexports_weights_under_derived_pair_path(tmp_path, monkeypatch):
    """Preserve predictions while refreshing config identity and digest."""
    xgboost = pytest.importorskip("xgboost")
    yaml = pytest.importorskip("yaml")
    from triton.flagtune.migration import LEGACY_MODEL_VARIANTS, migrate_ranker_model

    assert LEGACY_MODEL_VARIANTS == {
        "mm_general_tma": "general_tma",
        "gemv": "gemv",
        "mm_splitk": "splitk",
    }

    operator_config = tmp_path / "operator.yaml"
    combined_config = {
        "schema_version": 3,
        **_config(),
        "pretune": {"shape": {}, "dispatch": {}, "benchmark": {}},
    }
    operator_config.write_text(yaml.safe_dump(combined_config, sort_keys=False), encoding="utf-8")
    variant = parse_operator_config(_config()).get_variant("general")
    features = np.arange(4 * len(variant.feature_names), dtype=float).reshape(4, len(variant.feature_names))
    ranker = xgboost.XGBRanker(n_estimators=2, max_depth=2, n_jobs=1)
    ranker.fit(features, np.asarray([0.0, 1.0, 2.0, 3.0]), group=[4])
    expected = ranker.predict(features)
    source = tmp_path / "source.json"
    ranker.save_model(str(source))

    output_root = tmp_path / "models"
    target = migrate_ranker_model(
        source,
        operator_config,
        "general",
        output_root,
        gpu=GPU,
        dtypes=DTYPES,
        model_version=MODEL_VERSION,
        training_summary={"source": "unit-test"},
    )
    assert target == output_root / GPU_KEY / "vendor" / "mm" / "general" / DTYPE_KEY / MODEL_VERSION / "model.tar.gz"
    monkeypatch.setenv("TRITON_FLAGTUNE_MODEL_DIR", str(output_root))
    loaded = FlagTuneModelManager().load("vendor/mm", "general", gpu_key=GPU_KEY, dtype_key=DTYPE_KEY)
    np.testing.assert_allclose(loaded.predictor.predict(features), expected)
