# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
  * ``FLAGTUNE_MANIFEST_URL``: optional anonymous HTTPS manifest endpoint;
    defaults to ``https://models.example.com/flagtune/manifest.json``.
  * ``FLAGTUNE_MODEL_VERSION``: optional strict-SemVer exact version pin. An
    explicit ``model_version=`` API argument takes precedence.
  * ``FLAGTUNE_MODEL_REFRESH``: refreshes remote metadata only when its
    whitespace-stripped value is exactly ``1``. Other values keep cache-first
    behavior and do not perform periodic remote checks.
  * ``FLAGTUNE_DISABLE_REMOTE``: any non-empty value prevents network lookup;
    only the two local roots are searched.

Remote artifacts and redirects must use HTTPS, and artifacts must carry a
lowercase SHA-256 digest. The digest and complete bundle contract are validated
before an atomic cache write. This module does not coordinate cache writes
between processes. A corrupt, incompatible, or unavailable remote archive
therefore falls back to a valid cache candidate during refresh, or to the usual
``FileNotFoundError``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
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

if TYPE_CHECKING:
    from triton.flagtune.runtime.model_sources import RemoteArtifact

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


def _environment_model_version(explicit: Optional[str]) -> Optional[str]:
    """Resolve an explicit API version before the process environment pin."""
    if explicit is not None:
        return validate_model_version(explicit)
    configured = os.environ.get("FLAGTUNE_MODEL_VERSION", "").strip()
    return validate_model_version(configured) if configured else None


def _refresh_requested() -> bool:
    """Return whether the caller explicitly requested a remote refresh."""
    return os.environ.get("FLAGTUNE_MODEL_REFRESH", "").strip() == "1"


def _archive_is_valid(path: Path) -> bool:
    try:
        read_model_archive(path)
    except (OSError, ModelArchiveError):
        return False
    return True


def _archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FlagTuneModelManager:
    """Manage model bundles for one process.

    ``load`` searches the optional user root, then the persistent cache, then
    the URL resolver unless remote access is disabled. If no revision was
    requested, each source selects its own highest valid SemVer; source
    priority is never replaced by a cross-source version comparison. A cache
    hit performs no periodic remote check unless refresh is exactly ``1``.
    Successfully loaded predictors are retained only in this instance, keyed
    by ``(ModelIdentity, resolved_version)``. The first implicit selection is
    additionally retained by identity. The manager does not hot-reload changed
    files or evict models. Set refresh and version controls before the first
    model load, and start a new process to apply a replacement archive or a
    changed environment pin to an already loaded model.
    """

    def __init__(self) -> None:
        self._loaded: Dict[Tuple[ModelIdentity, str], LoadedFlagTuneModel] = {}
        self._implicit_loaded: Dict[ModelIdentity, LoadedFlagTuneModel] = {}

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
        implicit = model_version is None
        if implicit:
            cached = self._implicit_loaded.get(identity)
            if cached is not None:
                return cached
        requested = _environment_model_version(model_version)
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
            if implicit:
                self._implicit_loaded[identity] = cached
            return cached
        loaded = self._load_bundle(identity, resolved_version, model_path)
        self._loaded[(identity, resolved_version)] = loaded
        if implicit:
            self._implicit_loaded[identity] = loaded
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

        Search order is user root, persistent cache, then remote download. A
        version pin checks only that exact revision in each source. Normal
        cache hits do not query the manifest; explicit refresh validates the
        cache and compares its version and digest with freshly resolved remote
        metadata. Local identity and predictor compatibility remain deferred to
        :meth:`load`; downloads pass the complete bundle contract before the
        atomic cache commit.
        """
        identity = ModelIdentity(gpu_key, op_id, variant, dtype_key)
        requested = _environment_model_version(model_version)
        relative = _model_relative_path(identity)
        refresh = _refresh_requested()

        user_root = _user_model_root()
        if user_root is not None:
            candidate = self._select_local_archive(user_root / relative, requested)
            if candidate is not None:
                logger.info("FlagTune model found at user path: %s", candidate)
                return candidate

        candidate = self._select_local_archive(
            _cache_root() / relative,
            requested,
            require_valid=refresh,
        )
        if candidate is not None and not refresh:
            logger.info("FlagTune model found in cache: %s", candidate)
            return candidate

        if not refresh:
            if not os.environ.get("FLAGTUNE_DISABLE_REMOTE"):
                downloaded = self._download(identity, requested)
                if downloaded is not None:
                    return downloaded
        elif requested is not None and candidate is not None and _archive_is_valid(candidate):
            logger.info("FlagTune pinned model found in cache: %s", candidate)
            return candidate
        elif os.environ.get("FLAGTUNE_DISABLE_REMOTE"):
            logger.warning("FlagTune refresh skipped because remote access is disabled")
            if candidate is not None and _archive_is_valid(candidate):
                return candidate
        else:
            from triton.flagtune.runtime.model_sources import resolve_artifact_info

            remote = resolve_artifact_info(
                identity.op_id,
                identity.variant,
                gpu_key=identity.gpu_key,
                dtype_key=identity.dtype_key,
                version=requested,
                force_refresh=True,
            )
            valid_cache = candidate is not None and _archive_is_valid(candidate)
            if requested is None and valid_cache and remote is not None:
                cache_version = parse_model_version(candidate.parent.name)
                remote_version = parse_model_version(remote.version)
                if remote_version.selection_key < cache_version.selection_key:
                    logger.info("Keeping newer cached FlagTune model: %s", candidate)
                    return candidate
                if remote_version.selection_key == cache_version.selection_key:
                    if _archive_sha256(candidate) == remote.sha256:
                        logger.info("Reusing matching cached FlagTune model: %s", candidate)
                    else:
                        logger.warning(
                            "Ignoring changed digest for immutable FlagTune model version: %s",
                            candidate,
                        )
                    return candidate
            downloaded = self._download_artifact(identity, remote) if remote is not None else None
            if downloaded is not None:
                return downloaded
            if valid_cache:
                logger.warning("Using cached FlagTune model after refresh failure: %s", candidate)
                return candidate

        suffix = f" at version {requested!r}" if requested is not None else ""
        raise FileNotFoundError(f"FlagTune model not found for {identity.artifact_key!r}{suffix}. "
                                f"Checked user dir, cache ({_cache_root()}), and remote.")

    def _select_local_archive(
        self,
        identity_root: Path,
        requested: Optional[str],
        *,
        require_valid: bool = False,
    ) -> Optional[Path]:
        """Choose a non-symlink archive under one identity root.

        An explicit version must exist exactly.  Otherwise invalid directories,
        symlinks, and archives missing the expected filename are ignored and
        the highest valid SemVer directory wins.
        """
        if requested is not None:
            candidate = identity_root / requested / MODEL_ARCHIVE_NAME
            if not candidate.is_file() or candidate.is_symlink():
                return None
            return candidate if not require_valid or _archive_is_valid(candidate) else None
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
        if require_valid:
            for _, archive in sorted(candidates, key=lambda item: item[0], reverse=True):
                if _archive_is_valid(archive):
                    return archive
            return None
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
        return self._load_bundle_members(identity, model_version, members, model_path)

    def _load_bundle_members(
        self,
        identity: ModelIdentity,
        model_version: str,
        members: Dict[str, bytes],
        model_path: Path,
    ) -> LoadedFlagTuneModel:
        """Enforce the complete bundle contract for already-read archive members."""
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

    def _download(
        self,
        identity: ModelIdentity,
        model_version: Optional[str],
        *,
        force_refresh: bool = False,
    ) -> Optional[Path]:
        """Resolve, validate, and atomically cache one remote archive, if configured."""
        from triton.flagtune.runtime.model_sources import resolve_artifact_info

        artifact = resolve_artifact_info(
            identity.op_id,
            identity.variant,
            gpu_key=identity.gpu_key,
            dtype_key=identity.dtype_key,
            version=model_version,
            force_refresh=force_refresh,
        )
        if artifact is None:
            logger.info("No remote URL configured for %s", identity.artifact_key)
            return None
        return self._download_artifact(identity, artifact)

    def _download_artifact(self, identity: ModelIdentity, artifact: RemoteArtifact) -> Optional[Path]:
        """Download one manifest artifact after HTTPS, digest, and archive checks."""
        parsed = urlparse(artifact.url)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            logger.warning("FlagTune model URL must use HTTPS: %s", artifact.url)
            return None
        if not (parsed.path.lower().endswith(".tar.gz") or parsed.path.lower().endswith(".tgz")):
            logger.warning("FlagTune model URL must reference a gzip tar archive: %s", artifact.url)
            return None
        destination = (
            _cache_root()
            / _model_relative_path(identity)
            / artifact.version
            / MODEL_ARCHIVE_NAME
        )
        logger.info("Downloading FlagTune model for %s from %s", identity.artifact_key, artifact.url)
        try:
            from urllib.request import Request

            from triton.flagtune.runtime.model_sources import _open_https

            request = Request(
                artifact.url,
                headers={"User-Agent": f"FlagTune/{_FLAGTUNE_VERSION}"},
            )
            with _open_https(request, timeout=120) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != artifact.sha256:
                logger.warning("FlagTune model SHA-256 mismatch for %s", artifact.url)
                return None
            members = read_model_archive_bytes(payload, source=artifact.url)
            self._load_bundle_members(identity, artifact.version, members, destination)
            if destination.is_file() and not destination.is_symlink():
                try:
                    self._load_bundle(identity, artifact.version, destination)
                except Exception:
                    logger.warning(
                        "Replacing incompatible concurrently cached FlagTune model: %s",
                        destination,
                        exc_info=True,
                    )
                else:
                    logger.warning("Keeping concurrently cached immutable FlagTune model: %s", destination)
                    return destination
            _atomic_write_bytes(destination, payload)
            logger.info("FlagTune model cached to %s", destination)
            return destination
        except Exception:
            logger.warning(
                "Failed to download model for %s from %s",
                identity.artifact_key,
                artifact.url,
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
