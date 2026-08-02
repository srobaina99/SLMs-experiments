"""Shared intervention-config label helper.

Used by generation summaries (``run_store``), legacy human export, and
assessment bundling. Label semantics are stable: ``control``,
``weighting_only``, ``prompting_only``, ``both``.
"""

from __future__ import annotations


def config_label(config_weighting: bool, config_prompting: bool) -> str:
    """Map intervention flags to a config label."""
    if config_weighting and config_prompting:
        return "both"
    if config_weighting:
        return "weighting_only"
    if config_prompting:
        return "prompting_only"
    return "control"
