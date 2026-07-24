"""Offline ranking-data preparation, XGBoost fitting, and model export.

Training consumes already measured JSONL records and an explicit compiled
operator variant.  It intentionally contains no FlagGems kernel dispatch or
GPU collection logic.
"""
