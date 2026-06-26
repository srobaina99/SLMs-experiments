#!/usr/bin/env python3
"""Print KVL token index collision stats per model for manual inspection."""

from __future__ import annotations

import argparse
import sys

from slm_experiments.evaluation.kvl import SUPPORTED_L1S, KvlLookup
from slm_experiments.models.kvl_token_index import KvlTokenIndex
from slm_experiments.models.wrappers import MODEL_REGISTRY, get_model_wrapper


def parse_models(raw: str) -> list[str]:
    models = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = sorted(set(models) - set(MODEL_REGISTRY))
    if unknown:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}. Available: {available}")
    return models


def audit_model(model_name: str, *, l1: str, show_collisions: int) -> int:
    wrapper = get_model_wrapper(model_name)
    try:
        if not wrapper.model_loaded or wrapper.llm is None:
            print(f"{model_name}: model not loaded (skip)")
            return 1

        index = KvlTokenIndex.build(wrapper.llm, KvlLookup(), l1)
        stats = index.collision_stats()
        print(f"\n{model_name} (l1={l1})")
        print(f"  lemmas indexed:           {stats['lemma_count']}")
        print(f"  unique token IDs:           {stats['unique_token_ids']}")
        print(f"  collision token IDs:        {stats['collision_token_ids']}")
        print(f"  collision lemma refs:       {stats['collision_lemma_refs']}")
        print(f"  empty tokenization lemmas:  {stats['empty_tokenization_lemmas']}")
        print(f"  collision rate:             {stats['collision_rate']:.4f}")

        if show_collisions > 0:
            collisions = index.collision_token_ids()
            print(f"  sample collisions (max {show_collisions}):")
            for token_id, lemmas in list(sorted(collisions.items()))[:show_collisions]:
                print(f"    token {token_id}: {', '.join(lemmas)}")
        return 0
    finally:
        wrapper.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit KVL lemma ↔ token ID collisions per GGUF model."
    )
    parser.add_argument(
        "--models",
        default=",".join(sorted(MODEL_REGISTRY)),
        help=f"Comma-separated model names (default: all). Choices: {', '.join(sorted(MODEL_REGISTRY))}",
    )
    parser.add_argument(
        "--kvl-l1",
        default="es",
        choices=list(SUPPORTED_L1S),
        help="KVL lookup L1 (default: es)",
    )
    parser.add_argument(
        "--show-collisions",
        type=int,
        default=10,
        help="Print up to N example collision token IDs per model (default: 10, 0 to hide)",
    )
    args = parser.parse_args(argv)

    try:
        models = parse_models(args.models)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    exit_code = 0
    for model_name in models:
        exit_code = max(exit_code, audit_model(
            model_name,
            l1=args.kvl_l1,
            show_collisions=args.show_collisions,
        ))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
