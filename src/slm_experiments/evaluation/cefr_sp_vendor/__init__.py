"""Vendored CEFR-SP Lightning modules (Arase et al., EMNLP 2022).

This package ``__init__`` is import-safe (no torch). Importing
``cefr_sp_vendor.model`` / ``model_base`` pulls torch, transformers, and
pytorch-lightning. Callers must import lazily — see
``slm_experiments.evaluation.cefr_sp``.
"""
