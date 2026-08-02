#!/usr/bin/env python3
"""Prefetch pinned TSAR ModernBERT revisions and/or NLTK corpora for offline use.

Used at Docker build time so ClusterUY compute nodes can run with
``TRANSFORMERS_OFFLINE=1`` and no Hub / NLTK download at run time.

SHAs are read from ``src/.../cefr_tsar.py`` (``TSAR_MODELS``) via AST — no
package import — so we do not trigger ``assessment/__init__`` → metrics →
``nltk.download`` side effects during the image build.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# CEFR-SP encoder base (Arase ckpt overlays weights). Optional — ``assess`` does
# not re-score CEFR-SP, but baking this helps offline CEFR-SP loads if desired.
CEFR_SP_BASE_REPO = "bert-base-cased"

# Corpora required for KVL v2 (English WordNet lemmatizer + POS tagger).
# ``resolve_lemmatizer_backend`` prefers NLTK whenever the package imports, even
# when WordNet data is missing — without these, every KVL v2 row is status=error.
NLTK_RESOURCES: Tuple[str, ...] = (
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "wordnet",
    "omw-1.4",
)

# (resource_name, nltk.data.find path)
NLTK_FIND_PATHS: Tuple[Tuple[str, str], ...] = (
    ("punkt", "tokenizers/punkt"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ("wordnet", "corpora/wordnet"),
)


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[2] / "src"


def load_tsar_models() -> Sequence[dict]:
    """Parse ``TSAR_MODELS`` from cefr_tsar.py without importing the package."""
    path = (
        _repo_src()
        / "slm_experiments"
        / "evaluation"
        / "assessment"
        / "cefr_tsar.py"
    )
    if not path.is_file():
        raise SystemExit(f"Cannot find TSAR_MODELS source: {path}")

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        target_id = None
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_id = node.target.id
            value = node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            t0 = node.targets[0]
            if isinstance(t0, ast.Name):
                target_id = t0.id
                value = node.value
        if target_id != "TSAR_MODELS" or value is None:
            continue
        models = ast.literal_eval(value)
        if not isinstance(models, (list, tuple)) or not models:
            raise SystemExit(f"TSAR_MODELS in {path} is empty or not a sequence")
        for spec in models:
            if "model_id" not in spec or "revision" not in spec:
                raise SystemExit(f"TSAR_MODELS entry missing model_id/revision: {spec!r}")
        return list(models)

    raise SystemExit(f"TSAR_MODELS assignment not found in {path}")


def prefetch_hf(
    targets: Sequence[Tuple[str, Optional[str]]],
    *,
    hf_home: str,
) -> None:
    os.environ["HF_HOME"] = hf_home
    os.makedirs(hf_home, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is required. Install transformers / huggingface_hub first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    for model_id, revision in targets:
        label = f"{model_id}@{revision}" if revision else model_id
        print(f"Prefetching {label}")
        kwargs: dict = {"repo_id": model_id}
        if revision:
            kwargs["revision"] = revision
        path = snapshot_download(**kwargs)
        print(f"  → {path}")


def _unzip_nltk_archives(nltk_data: Path) -> None:
    """Extract any still-zipped NLTK packages (wordnet often lands as .zip only)."""
    for kind in ("corpora", "tokenizers", "taggers"):
        kind_dir = nltk_data / kind
        if not kind_dir.is_dir():
            continue
        for zipped in sorted(kind_dir.glob("*.zip")):
            dest_name = zipped.stem
            dest = kind_dir / dest_name
            if dest.exists():
                continue
            print(f"  Unzipping {zipped.name} → {kind}/")
            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(kind_dir)


def prefetch_nltk(nltk_data: str) -> None:
    root = Path(nltk_data)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["NLTK_DATA"] = str(root)

    try:
        import nltk
    except ImportError:
        print("nltk is required to prefetch corpora.", file=sys.stderr)
        raise SystemExit(1)

    # Prefer our bake path first.
    if str(root) in nltk.data.path:
        nltk.data.path.remove(str(root))
    nltk.data.path.insert(0, str(root))

    for resource in NLTK_RESOURCES:
        print(f"Downloading NLTK resource: {resource} → {root}")
        ok = nltk.download(resource, download_dir=str(root), quiet=False)
        if not ok:
            if resource == "punkt_tab":
                print(
                    f"  warning: nltk.download({resource!r}) returned False "
                    "(ok if this nltk version does not ship it)",
                    file=sys.stderr,
                )
                continue
            raise SystemExit(f"Failed to download NLTK resource: {resource}")

    _unzip_nltk_archives(root)

    for resource, find_path in NLTK_FIND_PATHS:
        try:
            found = nltk.data.find(find_path)
            print(f"  verified {resource}: {found}")
        except LookupError as exc:
            raise SystemExit(
                f"{resource} not findable under {root} ({find_path}): {exc}"
            ) from exc

    # Functional check: WordNet lemmatizer must work offline.
    from nltk.stem import WordNetLemmatizer

    lemma = WordNetLemmatizer().lemmatize("friends", "n")
    if lemma != "friend":
        raise SystemExit(f"WordNet lemmatizer smoke failed: friends→{lemma!r}")
    print(f"NLTK_DATA verified at {root} (lemmatize friends→{lemma})")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Docker build default: TSAR pins + NLTK corpora\n"
            "  python scripts/clusteruy/prefetch_tsar_models.py \\\n"
            "    --hf-home /opt/hf_home --nltk-data /usr/share/nltk_data\n"
            "\n"
            "  # Also stage bert-base-cased (does NOT skip TSAR)\n"
            "  python scripts/clusteruy/prefetch_tsar_models.py \\\n"
            "    --hf-home /opt/hf_home --include-cefr-sp-base\n"
            "\n"
            "  # NLTK only\n"
            "  python scripts/clusteruy/prefetch_tsar_models.py \\\n"
            "    --skip-tsar --nltk-data /usr/share/nltk_data\n"
        ),
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", "/opt/hf_home"),
        help="HF cache root (default: $HF_HOME or /opt/hf_home)",
    )
    parser.add_argument(
        "--include-cefr-sp-base",
        action="store_true",
        help="Also prefetch bert-base-cased (additive; never skips TSAR)",
    )
    parser.add_argument(
        "--skip-tsar",
        action="store_true",
        help="Do not prefetch the three pinned TSAR ModernBERT revisions",
    )
    parser.add_argument(
        "--nltk-data",
        default=None,
        metavar="DIR",
        help="Download/verify NLTK corpora into DIR (sets NLTK_DATA)",
    )
    args = parser.parse_args(argv)

    if args.skip_tsar and not args.include_cefr_sp_base and not args.nltk_data:
        parser.error(
            "nothing to do: pass --nltk-data and/or --include-cefr-sp-base, "
            "or drop --skip-tsar"
        )

    did_work = False

    hf_targets: List[Tuple[str, Optional[str]]] = []
    if not args.skip_tsar:
        for spec in load_tsar_models():
            hf_targets.append((spec["model_id"], spec["revision"]))
    if args.include_cefr_sp_base:
        hf_targets.append((CEFR_SP_BASE_REPO, None))

    if hf_targets:
        prefetch_hf(hf_targets, hf_home=args.hf_home)
        did_work = True
        print(f"Done HF prefetch. HF_HOME={args.hf_home}")

    if args.nltk_data:
        prefetch_nltk(args.nltk_data)
        did_work = True

    if not did_work:
        parser.error("nothing to do")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
