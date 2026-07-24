# flagtune/core: generic auto-tuning core algorithms
"""Small backend-neutral primitives shared by FlagTune contracts and runtime.

Only data interfaces and the generic genetic-search implementation belong
here.  This package must not depend on model storage, device probing, YAML I/O,
or any FlagGems-specific operator behavior.
"""
