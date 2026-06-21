#!/usr/bin/env python3
"""Build KVL lookup JSON files from BEA 2026 shared task CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "kvl"
BEA_REPO_URL = "https://github.com/britishcouncil/bea2026st"
SPLITS = ("train", "dev", "test")
L1S = ("es", "de", "cn")


def merge_csvs(source_dir: Path, l1: str) -> dict[str, list[float]]:
    """Merge train/dev/test CSVs and return word -> GLMM scores."""
    word_scores: dict[str, list[float]] = defaultdict(list)

    for split in SPLITS:
        csv_path = source_dir / split / l1 / f"kvl_shared_task_{l1}_{split}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping", file=sys.stderr)
            continue

        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                word = row["en_target_word"].lower().strip()
                word_scores[word].append(float(row["GLMM_score"]))

    return word_scores


def build_lookup(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for l1 in L1S:
        word_scores = merge_csvs(source_dir, l1)
        lookup = {
            word: round(sum(scores) / len(scores), 4)
            for word, scores in sorted(word_scores.items())
        }

        out_path = output_dir / f"kvl_lookup_{l1}.json"
        out_path.write_text(json.dumps(lookup, indent=0), encoding="utf-8")
        print(f"Wrote {out_path} ({len(lookup)} words)")


def resolve_source_dir(source_dir: Path | None, clone_dir: Path) -> Path:
    if source_dir is not None:
        return source_dir

    data_dir = clone_dir / "data"
    if data_dir.exists():
        return data_dir

    print(f"Cloning BEA repo to {clone_dir}...")
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", BEA_REPO_URL, str(clone_dir)],
        check=True,
    )
    return clone_dir / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build KVL lookup files from BEA 2026 shared task CSVs."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Path to bea2026st/data (train/dev/test subdirs).",
    )
    parser.add_argument(
        "--clone-dir",
        type=Path,
        default=Path("/tmp/bea2026st"),
        help="Directory to clone BEA repo when --source-dir is not set.",
    )
    args = parser.parse_args()

    source_dir = resolve_source_dir(args.source_dir, args.clone_dir)
    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    build_lookup(source_dir, OUTPUT_DIR)


if __name__ == "__main__":
    main()
