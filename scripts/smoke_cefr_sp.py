#!/usr/bin/env python3
"""Smoke-check CEFR-SP extras / architecture / optional checkpoint.

Usage (from repo root, with project venv):

    ./venv/bin/python scripts/smoke_cefr_sp.py
    ./venv/bin/python scripts/smoke_cefr_sp.py --score   # needs data/cefr_sp/level_estimator.ckpt

Does not download the ~1.2GB Zenodo checkpoint. Architecture check only imports
the vendored Lightning class (needs ``pip install -e '.[cefr-sp]'``) and does not
instantiate the full BERT encoder unless --score is passed.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO_ROOT / "data" / "cefr_sp" / "level_estimator.ckpt"


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def check_extras() -> int:
    missing = [m for m in ("torch", "transformers", "pytorch_lightning") if not _has(m)]
    if missing:
        print(f"FAIL: missing CEFR-SP extras: {', '.join(missing)}")
        print('  Install with: ./venv/bin/pip install -e ".[cefr-sp]"')
        return 1
    print("OK: torch / transformers / pytorch_lightning importable")
    return 0


def check_architecture_import() -> int:
    """Import vendored contrastive class without loading a checkpoint."""
    try:
        from slm_experiments.evaluation.cefr_sp_vendor.model import (  # noqa: WPS433
            LevelEstimaterContrastive,
        )
    except ImportError as exc:
        print(f"FAIL: could not import LevelEstimaterContrastive: {exc}")
        return 1
    print(f"OK: architecture import ({LevelEstimaterContrastive.__name__})")
    return 0


def check_score(ckpt: Path, device: str) -> int:
    if not ckpt.is_file():
        print(f"SKIP score: checkpoint not found at {ckpt}")
        print("  Download with: ./venv/bin/python scripts/download_cefr_sp_ckpt.py")
        return 0

    from slm_experiments.evaluation.cefr_sp import compute_cefr_sp_metrics

    metrics = compute_cefr_sp_metrics(
        "I like cats. They are soft and nice.",
        enabled=True,
        ckpt_path=str(ckpt),
        device=device,
    )
    print("OK: scored sample →", {k: metrics[k] for k in sorted(metrics)})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score",
        action="store_true",
        help="Run full scoring if level_estimator.ckpt is present",
    )
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    rc = check_extras()
    if rc != 0:
        return rc
    rc = check_architecture_import()
    if rc != 0:
        return rc
    if args.score:
        return check_score(args.ckpt, args.device)
    print("Tip: re-run with --score once the Zenodo ckpt is downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
