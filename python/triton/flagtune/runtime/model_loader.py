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
  * ``FLAGTUNE_MODEL_DIR``: optional local model root with highest precedence.
    It contains flat ``<platform_key>_v<version>.tar.gz`` packages.
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
before an atomic create-if-absent cache publication. Concurrent publishers may
reuse identical bytes, but a same-version package is never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from triton.flagtune._version import __version__ as _FLAGTUNE_VERSION
from triton.flagtune.contract.archive import (
    ModelArchiveError,
    PlatformPackage,
    parse_model_version,
    read_model_archive_bytes,
    read_platform_package,
    read_platform_package_bytes,
    platform_package_name,
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
    from triton.flagtune.runtime.model_sources import RemotePackage

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
    package_path: Path
    model_member: str
    model_version: str


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


def _package_is_valid(path: Path, platform_key: str, version: str) -> bool:
    try:
        read_platform_package(path, expected_platform_key=platform_key, expected_version=version)
    except (OSError, ModelArchiveError):
        return False
    return True


def _archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_package_bytes(path: Path, payload: bytes, expected_digest: str) -> None:
    """Atomically publish immutable package bytes without replacing a winner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink():
                raise IncompatibleModelError(f"FlagTune package cache destination is a symlink: {path}")
            try:
                winner_digest = _archive_sha256(path)
            except OSError as exc:
                raise IncompatibleModelError(
                    f"cannot inspect concurrent FlagTune package winner at {path}: {exc}") from exc
            if winner_digest != expected_digest:
                raise IncompatibleModelError(
                    f"immutable FlagTune package at {path} already exists with a different SHA-256")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class FlagTuneModelManager:
    """Resolve one versioned package per platform and load indexed child models."""

    def __init__(self) -> None:
        self._loaded: Dict[Tuple[ModelIdentity, str], LoadedFlagTuneModel] = {}
        self._implicit_loaded: Dict[ModelIdentity, LoadedFlagTuneModel] = {}
        self._packages: Dict[Tuple[str, str, str], PlatformPackage] = {}

    @staticmethod
    def _package_version(path: Path, platform_key: str) -> str:
        prefix = f"{platform_key}_v"
        suffix = ".tar.gz"
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            raise IncompatibleModelError(f"noncanonical FlagTune platform package filename: {name!r}")
        version = validate_model_version(name[len(prefix):-len(suffix)])
        if name != platform_package_name(platform_key, version):
            raise IncompatibleModelError(f"noncanonical FlagTune platform package filename: {name!r}")
        return version

    @staticmethod
    def _select_user_package(root: Path, platform_key: str, requested: Optional[str]) -> Optional[Path]:
        if requested is not None:
            candidate = root / platform_package_name(platform_key, requested)
            return candidate if candidate.is_file() and not candidate.is_symlink() else None
        if not root.is_dir():
            return None
        candidates = []
        prefix = f"{platform_key}_v"
        for entry in root.iterdir():
            if entry.is_symlink() or not entry.is_file() or not entry.name.startswith(prefix):
                continue
            try:
                version = FlagTuneModelManager._package_version(entry, platform_key)
                parsed = parse_model_version(version)
            except (ValueError, IncompatibleModelError):
                logger.warning("Ignoring invalid FlagTune platform package: %s", entry)
                continue
            candidates.append((parsed.selection_key, entry))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    @staticmethod
    def _select_cache_package(root: Path, platform_key: str, requested: Optional[str]) -> Optional[Path]:
        platform_root = root / "packages" / platform_key
        if requested is not None:
            candidate = platform_root / requested / platform_package_name(platform_key, requested)
            return candidate if candidate.is_file() and not candidate.is_symlink() else None
        if not platform_root.is_dir():
            return None
        candidates = []
        for entry in platform_root.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                continue
            try:
                parsed = parse_model_version(entry.name)
            except ValueError:
                logger.warning("Ignoring invalid FlagTune package version directory: %s", entry)
                continue
            package = entry / platform_package_name(platform_key, parsed.text)
            if package.is_file() and not package.is_symlink():
                candidates.append((parsed.selection_key, package))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def load(
        self,
        op_id: str,
        variant: str,
        *,
        platform_key: str,
        dtype_key: str,
        model_version: Optional[str] = None,
    ) -> LoadedFlagTuneModel:
        """Load one child model from the selected outer platform package."""
        identity = ModelIdentity(platform_key, op_id, variant, dtype_key)
        implicit = model_version is None
        if implicit and identity in self._implicit_loaded:
            return self._implicit_loaded[identity]
        requested = _environment_model_version(model_version)
        if requested is not None and (identity, requested) in self._loaded:
            return self._loaded[(identity, requested)]

        package_path = self.resolve(
            op_id,
            variant,
            platform_key=identity.platform_key,
            dtype_key=identity.dtype_key,
            model_version=requested,
        )
        resolved_version = self._package_version(package_path, identity.platform_key)
        cached = self._loaded.get((identity, resolved_version))
        if cached is not None:
            if implicit:
                self._implicit_loaded[identity] = cached
            return cached

        digest = _archive_sha256(package_path)
        package_key = (identity.platform_key, resolved_version, digest)
        package = self._packages.get(package_key)
        if package is None:
            try:
                package = read_platform_package(
                    package_path,
                    expected_platform_key=identity.platform_key,
                    expected_version=resolved_version,
                )
            except (OSError, ModelArchiveError) as exc:
                raise IncompatibleModelError(f"invalid FlagTune platform package at {package_path}: {exc}") from exc
            self._packages[package_key] = package

        entry = package.models.get(identity.artifact_key)
        if entry is None:
            raise IncompatibleModelError(
                f"FlagTune platform package {package_path} has no model for {identity.artifact_key!r}")
        member = entry["path"]
        try:
            members = read_model_archive_bytes(
                package.archives[identity.artifact_key],
                source=f"{package_path}:{member}",
            )
        except (KeyError, ModelArchiveError) as exc:
            raise IncompatibleModelError(f"invalid FlagTune child model at {package_path}:{member}: {exc}") from exc
        loaded = self._load_bundle_members(
            identity,
            resolved_version,
            members,
            package_path,
            member,
        )
        self._loaded[(identity, resolved_version)] = loaded
        if implicit:
            self._implicit_loaded[identity] = loaded
        return loaded

    def resolve(
        self,
        op_id: str,
        variant: str,
        *,
        platform_key: str,
        dtype_key: str,
        model_version: Optional[str] = None,
    ) -> Path:
        """Return the selected outer platform package without extracting it."""
        identity = ModelIdentity(platform_key, op_id, variant, dtype_key)
        requested = _environment_model_version(model_version)

        user_root = _user_model_root()
        if user_root is not None:
            user = self._select_user_package(user_root, identity.platform_key, requested)
            if user is not None:
                logger.info("FlagTune platform package found at user path: %s", user)
                return user

        cache_root = _cache_root()
        cached = self._select_cache_package(cache_root, identity.platform_key, requested)
        refresh = _refresh_requested()
        if cached is not None and not refresh:
            logger.info("FlagTune platform package found in cache: %s", cached)
            return cached

        cached_version = self._package_version(cached, identity.platform_key) if cached is not None else None
        cached_valid = (cached is not None and cached_version is not None
                        and _package_is_valid(cached, identity.platform_key, cached_version))
        if os.environ.get("FLAGTUNE_DISABLE_REMOTE"):
            if cached_valid:
                return cached
        else:
            from triton.flagtune.runtime.model_sources import resolve_package_info

            remote = resolve_package_info(
                identity.platform_key,
                version=requested,
                force_refresh=refresh,
            )
            if remote is not None and cached is not None and cached_version is not None:
                cached_parsed = parse_model_version(cached_version)
                remote_parsed = parse_model_version(remote.version)
                if requested is None and cached_parsed.selection_key > remote_parsed.selection_key and cached_valid:
                    return cached
                if cached_parsed.selection_key == remote_parsed.selection_key:
                    cached_digest = _archive_sha256(cached)
                    if cached_digest != remote.sha256:
                        raise IncompatibleModelError(f"immutable FlagTune package {identity.platform_key!r} version "
                                                     f"{remote.version!r} changed SHA-256")
                    if cached_valid:
                        return cached
            if remote is not None:
                downloaded = self._download_package(identity.platform_key, remote)
                if downloaded is not None:
                    return downloaded
            if cached_valid:
                logger.warning("Using cached FlagTune platform package after refresh failure: %s", cached)
                return cached

        suffix = f" at version {requested!r}" if requested is not None else ""
        raise FileNotFoundError(f"FlagTune model not found for {identity.artifact_key!r}{suffix}. "
                                f"Checked flat user packages, package cache ({cache_root}), and remote.")

    def _validate_flagtune_version(self, config: Dict[str, Any], source: str) -> None:
        min_ver = config.get("flagtune_version_min")
        if min_ver and _parse_compat_version(min_ver) > _parse_compat_version(_FLAGTUNE_VERSION):
            raise IncompatibleModelError(
                f"Model at {source} requires FlagTune >= {min_ver}, current version is {_FLAGTUNE_VERSION}")
        max_ver = config.get("flagtune_version_max")
        if max_ver and _parse_compat_version(max_ver) < _parse_compat_version(_FLAGTUNE_VERSION):
            logger.warning(
                "Model at %s was built for FlagTune <= %s, current version is %s",
                source,
                max_ver,
                _FLAGTUNE_VERSION,
            )

    def _load_bundle_members(
        self,
        identity: ModelIdentity,
        model_version: str,
        members: Dict[str, bytes],
        package_path: Path,
        model_member: str,
    ) -> LoadedFlagTuneModel:
        source = f"{package_path}:{model_member}"
        variant, predictor = self._validate_bundle_members(identity, model_version, members, source)
        return LoadedFlagTuneModel(identity, variant, predictor, package_path, model_member, model_version)

    def _validate_bundle_members(
        self,
        identity: ModelIdentity,
        model_version: str,
        members: Dict[str, bytes],
        source: str,
    ) -> tuple[VariantInfo, Any]:
        """Validate config, training summary, and Booster binding for one child."""
        variant, config = load_model_config_bytes(
            members["flagtune_config.yaml"],
            source=f"{source}:flagtune_config.yaml",
        )
        declared = model_identity_from_config(config)
        if declared != identity:
            raise IncompatibleModelError(f"model identity mismatch for {source}: requested {identity.artifact_key!r}, "
                                         f"config declares {declared.artifact_key!r}")
        declared_version = config.get("model_version")
        if declared_version != model_version:
            raise IncompatibleModelError(
                f"model version mismatch for {source}: package={model_version!r}, config={declared_version!r}")
        self._validate_flagtune_version(config, source)
        config_digest = model_config_sha256(config)
        predictor = _XGBoostPredictorCompat(
            members["xgboost_ranker.json"],
            variant,
            config_digest,
            Path(source),
        )
        try:
            summary = json.loads(members["training_summary.json"].decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IncompatibleModelError(f"invalid FlagTune training summary at {source}: {exc}") from exc
        if not isinstance(summary, dict):
            raise IncompatibleModelError(f"FlagTune training summary must be an object at {source}")
        expected_features = list(variant.feature_names)
        expected_summary = {
            "feature_cols": expected_features,
            "feature_count": len(expected_features),
            "model_config_sha256": config_digest,
            "model_version": model_version,
        }
        for name, expected in expected_summary.items():
            actual = summary.get(name)
            if name == "feature_count" and type(actual) is not int:
                actual = None
            if actual != expected:
                raise IncompatibleModelError(f"FlagTune training summary {name} mismatch at {source}: "
                                             f"{actual!r} != {expected!r}")
        for name, expected in (("op_id", identity.op_id), ("variant", identity.variant)):
            if name in summary and summary[name] != expected:
                raise IncompatibleModelError(f"FlagTune training summary {name} mismatch at {source}: "
                                             f"{summary[name]!r} != {expected!r}")
        return variant, predictor

    def _validate_package_for_cache(
        self,
        package: PlatformPackage,
        source: str,
    ) -> None:
        """Validate every H20 child and the complete published identity set."""
        if package.platform_key == "nvidia-h20":
            required = {
                ModelIdentity(
                    "nvidia-h20",
                    "flaggems/mm",
                    variant,
                    "bf16-bf16-bf16",
                ).artifact_key
                for variant in ("gemv", "general_tma", "splitk")
            }
            actual = set(package.models)
            missing = sorted(required - actual)
            unexpected = sorted(actual - required)
            if missing:
                raise IncompatibleModelError(f"FlagTune package has missing required H20 models: {missing}")
            if unexpected:
                raise IncompatibleModelError(f"FlagTune package has unexpected H20 models: {unexpected}")
        for artifact in sorted(package.models):
            identity_parts = artifact.split("/")
            identity = ModelIdentity(
                identity_parts[0],
                "/".join(identity_parts[1:-2]),
                identity_parts[-2],
                identity_parts[-1],
            )
            member = package.models[artifact]["path"]
            try:
                members = read_model_archive_bytes(package.archives[artifact], source=f"{source}:{member}")
            except (KeyError, ModelArchiveError) as exc:
                raise IncompatibleModelError(f"invalid FlagTune child model at {source}:{member}: {exc}") from exc
            self._validate_bundle_members(
                identity,
                package.package_version,
                members,
                f"{source}:{member}",
            )

    def _download_package(self, platform_key: str, package: RemotePackage) -> Optional[Path]:
        parsed = urlparse(package.url)
        expected_name = platform_package_name(platform_key, package.version)
        if parsed.scheme.lower() != "https" or not parsed.netloc or Path(parsed.path).name != expected_name:
            logger.warning("FlagTune platform package URL is invalid: %s", package.url)
            return None
        destination = _cache_root() / "packages" / platform_key / package.version / expected_name
        try:
            from urllib.request import Request

            from triton.flagtune.runtime.model_sources import _open_https

            request = Request(package.url, headers={"User-Agent": f"FlagTune/{_FLAGTUNE_VERSION}"})
            with _open_https(request, timeout=120) as response:
                payload = response.read()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != package.sha256:
                logger.warning("FlagTune platform package SHA-256 mismatch for %s", package.url)
                return None
            try:
                parsed_package = read_platform_package_bytes(
                    payload,
                    expected_platform_key=platform_key,
                    expected_version=package.version,
                    source=package.url,
                )
            except ModelArchiveError as exc:
                raise IncompatibleModelError(f"invalid FlagTune platform package from {package.url}: {exc}") from exc
            self._validate_package_for_cache(parsed_package, package.url)
            if destination.is_file() and not destination.is_symlink():
                if _archive_sha256(destination) != package.sha256:
                    raise IncompatibleModelError(
                        f"immutable FlagTune package {platform_key!r} version {package.version!r} "
                        "already exists with a different SHA-256")
                return destination
            _publish_package_bytes(destination, payload, package.sha256)
            self._packages[(platform_key, package.version, digest)] = parsed_package
            logger.info("FlagTune platform package cached to %s", destination)
            return destination
        except IncompatibleModelError:
            raise
        except Exception:
            logger.warning(
                "Failed to download FlagTune platform package for %s from %s",
                platform_key,
                package.url,
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
