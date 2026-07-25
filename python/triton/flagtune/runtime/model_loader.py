"""Resolve, validate, download, and cache FlagTune model archives.

The manager owns the runtime lifecycle of a self-contained model bundle:
selection of a SemVer revision, source precedence, archive safety checks, YAML
contract compilation, identity/version compatibility checks, XGBoost loading,
and an in-process cache keyed by complete model identity plus revision.  It is
used lazily by :mod:`predict`; callers normally use ``load_model_bundle`` or
``make_config_proposer`` rather than constructing archive paths themselves.

Environment variables:
  * ``FLAGTUNE_MODEL_DIR``: optional local model root with highest
    precedence. It must contain ``gpu_key/op_id/variant/dtype_key/version/``.
  * ``FLAGTUNE_MODEL_CACHE``: writable cache root for downloaded bundles and
    the remote URL manifest. Defaults to ``~/.flagtree/flagtune_models``.
  * ``FLAGTUNE_DISABLE_REMOTE``: any non-empty value prevents network lookup;
    only the two local roots are searched.

Remote payloads are validated before caching, but this module does not verify
checksums advertised by manifests and does not coordinate cache writes between
processes.  A corrupt, incompatible, or unavailable remote archive therefore
falls through to the usual ``FileNotFoundError`` instead of a retry protocol.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from triton.flagtune._version import __version__ as _FLAGTUNE_VERSION
from triton.flagtune.contract.archive import (
    MODEL_ARCHIVE_NAME,
    ModelArchiveError,
    parse_model_version,
    read_model_archive,
    read_model_archive_bytes,
    validate_model_version,
)
from triton.flagtune.contract.identity import ModelIdentity
from triton.flagtune.contract.operator_schema import (
    VariantInfo,
    load_model_config_bytes,
    model_config_sha256,
    model_identity_from_config,
)

logger = logging.getLogger(__name__)


def _cache_root() -> Path:
    """Return the writable cache root, without creating it during lookup."""
    env = os.environ.get("FLAGTUNE_MODEL_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".flagtree" / "flagtune_models"


def _user_model_root() -> Optional[Path]:
    """Return the optional highest-priority user model root."""
    env = os.environ.get("FLAGTUNE_MODEL_DIR", "").strip()
    return Path(env) if env else None


class IncompatibleModelError(RuntimeError):
    """Indicate that a resolved archive cannot serve the requested contract."""


@dataclass(frozen=True)
class LoadedFlagTuneModel:
    """Hold a validated identity, compiled variant, predictor, and archive path."""

    identity: ModelIdentity
    variant: VariantInfo
    predictor: Any
    model_path: Path
    model_version: str


def _model_relative_path(identity: ModelIdentity) -> Path:
    return Path(*PurePosixPath(identity.artifact_key).parts)


def _parse_compat_version(ver: str) -> tuple:
    """Compare permissive ``major.minor.patch`` compatibility fields numerically.

    Unlike artifact revisions, compatibility bounds are not strict SemVer:
    non-numeric components become zero.  This is intentionally narrow legacy
    handling and must not be used to select archive revisions.
    """
    parts = []
    for part in str(ver).split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


class FlagTuneModelManager:
    """Manage model bundles for one process.

    ``load`` searches the optional user root, then the persistent cache, then
    the URL resolver unless remote access is disabled.  If no revision was
    requested, each local source selects the highest valid SemVer it contains.
    Successfully loaded predictors are retained only in this instance, keyed
    by ``(ModelIdentity, resolved_version)``.  It does not hot-reload changed
    files or evict models, so a long-lived process needs a new manager to see
    a replacement archive.
    """

    def __init__(self) -> None:
        self._loaded: Dict[Tuple[ModelIdentity, str], LoadedFlagTuneModel] = {}

    def load(
        self,
        op_id: str,
        variant: str,
        *,
        gpu_key: str,
        dtype_key: str,
        model_version: Optional[str] = None,
    ) -> LoadedFlagTuneModel:
        """Resolve and load an exact revision, or the highest available SemVer.

        ``gpu_key``, ``op_id``, ``variant``, and ``dtype_key`` form an exact
        compatibility boundary.  The returned bundle includes the compiled
        variant contract and an XGBoost-compatible predictor.
        """
        identity = ModelIdentity(gpu_key, op_id, variant, dtype_key)
        requested = validate_model_version(model_version) if model_version is not None else None
        if requested is not None:
            cached = self._loaded.get((identity, requested))
            if cached is not None:
                return cached

        model_path = self.resolve(
            op_id,
            variant,
            gpu_key=gpu_key,
            dtype_key=dtype_key,
            model_version=requested,
        )
        resolved_version = model_path.parent.name
        cached = self._loaded.get((identity, resolved_version))
        if cached is not None:
            return cached
        loaded = self._load_bundle(identity, resolved_version, model_path)
        self._loaded[(identity, resolved_version)] = loaded
        return loaded

    def resolve(
        self,
        op_id: str,
        variant: str,
        *,
        gpu_key: str,
        dtype_key: str,
        model_version: Optional[str] = None,
    ) -> Path:
        """Return the selected ``model.tar.gz`` path without unpacking it.

        Search order is user root, persistent cache, then remote download.
        This selection verifies path layout and SemVer directory names, but it
        does not validate archive contents until :meth:`load` calls
        :meth:`_load_bundle`.
        """
        identity = ModelIdentity(gpu_key, op_id, variant, dtype_key)
        requested = validate_model_version(model_version) if model_version is not None else None
        relative = _model_relative_path(identity)

        user_root = _user_model_root()
        if user_root is not None:
            candidate = self._select_local_archive(user_root / relative, requested)
            if candidate is not None:
                logger.info("FlagTune model found at user path: %s", candidate)
                return candidate

        candidate = self._select_local_archive(_cache_root() / relative, requested)
        if candidate is not None:
            logger.info("FlagTune model found in cache: %s", candidate)
            return candidate

        if not os.environ.get("FLAGTUNE_DISABLE_REMOTE"):
            downloaded = self._download(identity, requested)
            if downloaded is not None:
                return downloaded

        suffix = f" at version {requested!r}" if requested is not None else ""
        raise FileNotFoundError(f"FlagTune model not found for {identity.artifact_key!r}{suffix}. "
                                f"Checked user dir, cache ({_cache_root()}), and remote.")

    def _select_local_archive(self, identity_root: Path, requested: Optional[str]) -> Optional[Path]:
        """Choose a non-symlink archive under one identity root.

        An explicit version must exist exactly.  Otherwise invalid directories,
        symlinks, and archives missing the expected filename are ignored and
        the highest valid SemVer directory wins.
        """
        if requested is not None:
            candidate = identity_root / requested / MODEL_ARCHIVE_NAME
            return candidate if candidate.is_file() and not candidate.is_symlink() else None
        if not identity_root.is_dir():
            return None

        candidates = []
        for entry in identity_root.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                logger.warning("Ignoring non-version FlagTune entry: %s", entry)
                continue
            try:
                version = parse_model_version(entry.name)
            except ValueError:
                logger.warning("Ignoring invalid FlagTune version directory: %s", entry)
                continue
            archive = entry / MODEL_ARCHIVE_NAME
            if not archive.is_file() or archive.is_symlink():
                logger.warning("Ignoring FlagTune version without %s: %s", MODEL_ARCHIVE_NAME, entry)
                continue
            candidates.append((version.selection_key, archive))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def _validate_flagtune_version(self, config: Dict[str, Any], model_path: Path) -> None:
        """Enforce archive minimum runtime version and warn on an old maximum."""
        min_ver = config.get("flagtune_version_min")
        if min_ver and _parse_compat_version(min_ver) > _parse_compat_version(_FLAGTUNE_VERSION):
            raise IncompatibleModelError(
                f"Model at {model_path} requires FlagTune >= {min_ver}, current version is {_FLAGTUNE_VERSION}")
        max_ver = config.get("flagtune_version_max")
        if max_ver and _parse_compat_version(max_ver) < _parse_compat_version(_FLAGTUNE_VERSION):
            logger.warning(
                "Model at %s was built for FlagTune <= %s, current version is %s",
                model_path,
                max_ver,
                _FLAGTUNE_VERSION,
            )

    def _load_bundle(self, identity: ModelIdentity, model_version: str, model_path: Path) -> LoadedFlagTuneModel:
        """Read one archive and enforce its identity, revision, schema, and model contract."""
        try:
            members = read_model_archive(model_path)
        except (OSError, ModelArchiveError) as exc:
            raise IncompatibleModelError(f"invalid FlagTune model archive at {model_path}: {exc}") from exc
        variant, config = load_model_config_bytes(members["flagtune_config.yaml"],
                                                  source=f"{model_path}:flagtune_config.yaml")
        declared = model_identity_from_config(config)
        if declared != identity:
            raise IncompatibleModelError(
                f"model identity mismatch for {model_path}: requested {identity.artifact_key!r}, "
                f"config declares {declared.artifact_key!r}")
        declared_version = config.get("model_version")
        if declared_version != model_version:
            raise IncompatibleModelError(f"model version mismatch for {model_path}: path={model_version!r}, "
                                         f"config={declared_version!r}")
        self._validate_flagtune_version(config, model_path)
        predictor = _XGBoostPredictorCompat(
            members["xgboost_ranker.json"],
            variant,
            model_config_sha256(config),
            model_path,
        )
        return LoadedFlagTuneModel(identity, variant, predictor, model_path, model_version)

    def _download(self, identity: ModelIdentity, model_version: Optional[str]) -> Optional[Path]:
        """Resolve, validate, and atomically cache one remote archive, if configured."""
        from triton.flagtune.runtime.model_sources import resolve_artifact

        artifact = resolve_artifact(
            identity.op_id,
            identity.variant,
            gpu_key=identity.gpu_key,
            dtype_key=identity.dtype_key,
            version=model_version,
        )
        if artifact is None:
            logger.info("No remote URL configured for %s", identity.artifact_key)
            return None
        version, url = artifact
        path = urlparse(url).path.lower()
        if not (path.endswith(".tar.gz") or path.endswith(".tgz")):
            logger.warning("FlagTune model URL must reference a gzip tar archive: %s", url)
            return None
        destination = (_cache_root() / _model_relative_path(identity) / version / MODEL_ARCHIVE_NAME)
        logger.info("Downloading FlagTune model for %s from %s", identity.artifact_key, url)
        try:
            from urllib.request import Request, urlopen

            request = Request(url, headers={"User-Agent": f"FlagTune/{_FLAGTUNE_VERSION}"})
            with urlopen(request, timeout=120) as response:
                payload = response.read()
            read_model_archive_bytes(payload, source=url)
            _atomic_write_bytes(destination, payload)
            logger.info("FlagTune model cached to %s", destination)
            return destination
        except Exception:
            logger.warning(
                "Failed to download model for %s from %s",
                identity.artifact_key,
                url,
                exc_info=True,
            )
            return None


class _XGBoostPredictorCompat:
    """Adapt an XGBoost ranker while enforcing its embedded schema contract.

    The booster must carry the SHA-256 digest of the bundled YAML and expose
    the same ordered feature names and count as the compiled variant.  This
    rejects a model trained against a reordered feature schema before any
    prediction.  It requires XGBoost at runtime and intentionally exposes only
    ``predict`` rather than the full estimator API.
    """

    def __init__(
        self,
        model_payload: bytes,
        variant: VariantInfo,
        config_digest: str,
        model_path: Path,
    ) -> None:
        from xgboost import XGBRanker

        self.feature_cols: List[str] = list(variant.feature_names)
        self._model = XGBRanker()
        self._model.load_model(bytearray(model_payload))
        booster = self._model.get_booster()
        stored_digest = booster.attr("flagtune_config_sha256")
        if stored_digest != config_digest:
            raise IncompatibleModelError(f"FlagTune config digest mismatch at {model_path}: "
                                         f"model={stored_digest!r}, config={config_digest!r}")
        stored_features = list(booster.feature_names or [])
        if stored_features and stored_features != self.feature_cols:
            raise IncompatibleModelError(f"FlagTune feature order mismatch at {model_path}: "
                                         f"model={stored_features}, config={self.feature_cols}")
        if int(booster.num_features()) != len(self.feature_cols):
            raise IncompatibleModelError(f"FlagTune feature count mismatch at {model_path}: "
                                         f"model={booster.num_features()}, config={len(self.feature_cols)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return ranking scores for a feature matrix already in config order."""
        return self._model.predict(X)
