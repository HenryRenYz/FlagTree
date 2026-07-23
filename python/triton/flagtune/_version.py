"""Define the installed FlagTune package version.

``__version__`` is sent in model/manifest download user agents and is compared
with optional ``flagtune_version_min`` and ``flagtune_version_max`` fields in
an archive's YAML contract by :class:`triton.flagtune.model_manager.FlagTuneModelManager`.
It is the runtime implementation compatibility version, not the independently
versioned model artifact revision stored in each archive path.
"""

__version__ = "0.2.0"
