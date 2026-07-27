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
"""Create, validate, and order self-contained FlagTune model archives.

An exported model is one deterministic ``model.tar.gz`` rather than an
unpacked directory.  Its required root-level members are the XGBoost ranker,
the compiled YAML contract, and the training summary.  :mod:`training` writes
these archives and :mod:`model_manager` reads them before loading a predictor.

This module deliberately validates an archive in memory and never calls
``TarFile.extract*``.  That prevents path traversal, links, duplicate members,
and other archive-layout surprises, but it also means every member is held in
memory and archives may contain only root-level regular files.
"""

from __future__ import annotations

import gzip
import io
import os
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Tuple

MODEL_ARCHIVE_NAME = "model.tar.gz"
REQUIRED_MODEL_MEMBERS = (
    "xgboost_ranker.json",
    "flagtune_config.yaml",
    "training_summary.json",
)

# Strict Semantic Versioning 2.0 grammar used for artifact directory names.
# Examples accepted: ``1.2.3``, ``1.2.3-rc.1``, and ``1.2.3+build.7``.
# Examples rejected: ``v1.2.3``, ``1.2``, and ``01.2.3``.  Build metadata is
# retained as text for deterministic selection, but does not affect SemVer
# precedence; therefore ``1.2.3+cpu`` and ``1.2.3+gpu`` have equal precedence.
_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\."
                        r"(0|[1-9][0-9]*)\."
                        r"(0|[1-9][0-9]*)"
                        r"(?:-((?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
                        r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*))?"
                        r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$")


class ModelArchiveError(ValueError):
    """Report an unsafe, incomplete, or malformed model archive."""


@dataclass(frozen=True)
class SemanticVersion:
    """A strict SemVer 2.0 value with precedence-aware ordering data."""

    text: str
    major: int
    minor: int
    patch: int
    prerelease: Tuple[str, ...]

    @property
    def precedence_key(self) -> tuple:
        """Return a key implementing SemVer precedence (excluding build data)."""
        identifiers = tuple((0, int(value)) if value.isdigit() else (1, value) for value in self.prerelease)
        # A release has higher precedence than every prerelease of that release.
        return (self.major, self.minor, self.patch, not self.prerelease, identifiers)

    @property
    def selection_key(self) -> tuple:
        """Add the complete text as a deterministic tie-break for build metadata."""
        return (*self.precedence_key, self.text)


def parse_model_version(value: str) -> SemanticVersion:
    """Parse a strict SemVer 2.0 model revision without normalizing its text.

    Args:
        value: Candidate archive revision, such as ``"1.4.0-rc.2"``.

    Returns:
        Parsed numeric components and prerelease identifiers for selection.

    Raises:
        ValueError: If ``value`` is not strict SemVer 2.0.
    """
    if not isinstance(value, str) or not _SEMVER_RE.fullmatch(value):
        raise ValueError(f"model version must be strict SemVer 2.0: {value!r}")
    match = _SEMVER_RE.fullmatch(value)
    assert match is not None
    prerelease = tuple(match.group(4).split(".")) if match.group(4) else ()
    return SemanticVersion(value, int(match.group(1)), int(match.group(2)), int(match.group(3)), prerelease)


def validate_model_version(value: str) -> str:
    """Validate and return a model version suitable for one path segment."""
    return parse_model_version(value).text


def _validate_member(member: tarfile.TarInfo, seen: set[str]) -> str:
    name = member.name
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or len(relative.parts) != 1 or name != relative.name
            or relative.parts[0] in (".", "..") or "\\" in name):
        raise ModelArchiveError(f"model archive member must be a root-level file: {name!r}")
    if name in seen:
        raise ModelArchiveError(f"duplicate model archive member: {name!r}")
    if not member.isfile():
        raise ModelArchiveError(f"model archive member is not a regular file: {name!r}")
    seen.add(name)
    return name


def read_model_archive_bytes(payload: bytes, *, source: str = "model archive") -> Dict[str, bytes]:
    """Validate a gzip tar payload and return every safe root member in memory.

    Args:
        payload: Complete gzip-compressed tar payload.
        source: Diagnostic label included in validation errors.

    Returns:
        A mapping from archive member name to its exact bytes.

    Raises:
        ModelArchiveError: If decompression fails, a member is unsafe, or a
            required member is missing.
    """
    members: Dict[str, bytes] = {}
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            for member in archive.getmembers():
                name = _validate_member(member, seen)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ModelArchiveError(f"cannot read model archive member: {name!r}")
                with stream:
                    members[name] = stream.read()
    except ModelArchiveError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise ModelArchiveError(f"invalid gzip tar model archive at {source}: {exc}") from exc
    missing = [name for name in REQUIRED_MODEL_MEMBERS if name not in members]
    if missing:
        raise ModelArchiveError(f"model archive at {source} is missing required members: {missing}")
    return members


def read_model_archive(path: Path | str) -> Dict[str, bytes]:
    """Read and validate one on-disk ``model.tar.gz`` without extracting it."""
    archive_path = Path(path)
    try:
        payload = archive_path.read_bytes()
    except OSError:
        raise
    return read_model_archive_bytes(payload, source=str(archive_path))


def write_model_archive(path: Path | str, members: Mapping[str, bytes]) -> Path:
    """Atomically write a reproducible gzip tar containing root-level files.

    Required entries must be supplied as bytes.  Member order, timestamps,
    ownership, permissions, and gzip metadata are normalized so identical
    inputs produce identical bytes.  The destination is replaced atomically on
    the same filesystem; concurrent writers still race at the final replace.
    """
    target = Path(path)
    missing = [name for name in REQUIRED_MODEL_MEMBERS if name not in members]
    if missing:
        raise ModelArchiveError(f"cannot write model archive without required members: {missing}")
    names = list(REQUIRED_MODEL_MEMBERS) + sorted(set(members) - set(REQUIRED_MODEL_MEMBERS))
    for name in names:
        fake = tarfile.TarInfo(name)
        fake.type = tarfile.REGTYPE
        _validate_member(fake, set())
        if not isinstance(members[name], bytes):
            raise TypeError(f"model archive member {name!r} must be bytes")

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                    for name in names:
                        payload = members[name]
                        info = tarfile.TarInfo(name)
                        info.size = len(payload)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mode = 0o644
                        archive.addfile(info, io.BytesIO(payload))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return target
