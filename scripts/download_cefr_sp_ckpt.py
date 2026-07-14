#!/usr/bin/env python3
"""Download the official CEFR-SP level_estimator.ckpt from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ZENODO_DOI = "10.5281/zenodo.7234096"
ZENODO_URL = (
    "https://zenodo.org/records/7234096/files/level_estimator.ckpt?download=1"
)
EXPECTED_MD5 = "2448ff49f6e8a9c504ac8ba02116e043"
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent / "data" / "cefr_sp" / "level_estimator.ckpt"
)
DOWNLOAD_TIMEOUT_SEC = 600


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` via a sibling ``.partial`` file, then replace.

    URL is fixed by this script (not user-controlled). ``--out`` may point
    anywhere the user can write; callers should keep the default under
    ``data/cefr_sp/`` unless they intentionally override.
    """
    dest = dest.expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"Downloading {url}")
    print(f"  → {dest}")
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SEC) as resp:
            with tmp.open("wb") as out_fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out_fh.write(chunk)
        tmp.replace(dest)
    except (OSError, urllib.error.URLError, ValueError):
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download CEFR-SP level_estimator.ckpt from Zenodo "
            f"(DOI {ZENODO_DOI}) and verify MD5."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target file already exists",
    )
    args = parser.parse_args(argv)

    out: Path = args.out.expanduser().resolve()
    if out.exists() and not args.force:
        digest = md5_file(out)
        if digest == EXPECTED_MD5:
            print(f"Already present with matching MD5: {out}")
            return 0
        print(
            f"ERROR: {out} exists but MD5 is {digest}, expected {EXPECTED_MD5}. "
            "Re-run with --force to replace.",
            file=sys.stderr,
        )
        return 1

    try:
        download(ZENODO_URL, out)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        print(f"ERROR: download failed: {exc}", file=sys.stderr)
        return 1

    digest = md5_file(out)
    if digest != EXPECTED_MD5:
        print(
            f"ERROR: MD5 mismatch for {out}: got {digest}, expected {EXPECTED_MD5}",
            file=sys.stderr,
        )
        try:
            out.unlink()
        except OSError:
            pass
        return 1

    size_gb = out.stat().st_size / (1024**3)
    print(f"OK: {out} ({size_gb:.2f} GiB), md5={digest}")
    print(f"License: CC BY 4.0 — DOI {ZENODO_DOI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
