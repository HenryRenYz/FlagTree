"""Runtime integration for loading FlagTune models and proposing configs.

The modules here may inspect the active device, read environment variables,
load XGBoost, or access model storage.  Offline schema and archive contracts
remain in :mod:`triton.flagtune.contract`.
"""
