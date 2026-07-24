"""Stable, side-effect-free FlagTune model contracts.

This package owns identity normalization, safe YAML expressions, archive
validation, and operator-schema compilation.  It never probes devices, loads
XGBoost, downloads files, or launches benchmarks; those runtime concerns live
under :mod:`triton.flagtune.runtime`.
"""
