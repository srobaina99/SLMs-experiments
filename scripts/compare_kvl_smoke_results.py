#!/usr/bin/env python3
"""Compare sweep JSON against reference /tmp JSON from prior subagent runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Reference files from parallel subagent runs
REFERENCES: dict[tuple[str, float | None], Path] = {
    ("baseline", None): Path("/tmp/kvl_variant_baseline.json"),
    ("always_stop", None): Path("/tmp/kvl_variant_always_stop.json"),
    ("length_penalty", 0.15): Path("/tmp/kvl_variant_length_penalty.json"),
    ("prefer_finished", None): Path("/tmp/kvl_variant_prefer_finished.json"),
    ("stop_pref", None): Path("/tmp/kvl_variant_stop_pref.json"),
    ("max80", None): Path("/tmp/kvl_variant_max80.json"),
    ("stop_length", 0.15): Path("/tmp/kvl_alpha_stop_0.15.json"),
    ("stop_length", 0.25): Path("/tmp/kvl_alpha_stop_0.25.json"),
    ("stop_length", 0.35): Path("/tmp/kvl_alpha_stop_0.35.json"),
    ("stop_length", 0.5): Path("/tmp/kvl_alpha_stop_0.5.json"),
    ("stop_length", 0.75): Path("/tmp/kvl_alpha_stop_0.75.json"),
    ("stop_length", 1.0): Path("/tmp/kvl_alpha_stop_1.0.json"),
    ("stop_length", 1.5): Path("/tmp/kvl_alpha_stop_1.5.json"),
    ("length_penalty", 0.5): Path("/tmp/kvl_alpha_len_0.5.json"),
    ("length_penalty", 1.0): Path("/tmp/kvl_alpha_len_1.0.json"),
    ("length_penalty", 1.5): Path("/tmp/kvl_alpha_len_1.5.json"),
}

COMPARE_FIELDS = (
    "steps_total",
    "finished_pool_size",
    "words_scored",
    "kvl_beam_running_mean",
    "response_chars",
    "hit_max_tokens",
)


def load_observations(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text())
    return {obs["prompt_index"]: obs for obs in data.get("observations", [])}


def normalize_alpha(alpha: float | None) -> float | None:
    if alpha is None:
        return None
    return round(alpha, 10)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: compare_kvl_smoke_results.py <sweep.json>", file=sys.stderr)
        return 2

    sweep_path = Path(sys.argv[1])
    sweep = json.loads(sweep_path.read_text())
    if isinstance(sweep, dict):
        sweep = [sweep]

    mismatches = []
    matched = 0
    missing_ref = []

    for result in sweep:
        variant = result["variant"]
        alpha = normalize_alpha(result.get("length_penalty_alpha"))
        if alpha == 0.15 and variant == "length_penalty":
            key = (variant, 0.15)
        elif variant in ("baseline", "always_stop", "prefer_finished", "stop_pref", "max80"):
            key = (variant, None)
        else:
            key = (variant, alpha)

        ref_path = REFERENCES.get(key)
        if ref_path is None or not ref_path.exists():
            missing_ref.append(f"{variant} alpha={alpha}")
            continue

        ref_obs = load_observations(ref_path)
        for obs in result.get("observations", []):
            idx = obs["prompt_index"]
            ref = ref_obs.get(idx)
            if ref is None:
                mismatches.append(f"{variant} a={alpha} P{idx}: missing prompt in reference")
                continue

            for field in COMPARE_FIELDS:
                got = obs.get(field)
                exp = ref.get(field)
                if field == "kvl_beam_running_mean":
                    if got is None and exp is None:
                        continue
                    if got is None or exp is None or abs(got - exp) > 1e-4:
                        mismatches.append(
                            f"{variant} a={alpha} P{idx} {field}: got={got} exp={exp}"
                        )
                elif got != exp:
                    mismatches.append(
                        f"{variant} a={alpha} P{idx} {field}: got={got} exp={exp}"
                    )

            preview_got = obs.get("response_preview", "")[:120]
            preview_exp = ref.get("response_preview", "")[:120]
            if preview_got != preview_exp:
                mismatches.append(
                    f"{variant} a={alpha} P{idx} preview[:120] differs"
                )

            if not any(m.startswith(f"{variant} a={alpha} P{idx}") for m in mismatches):
                matched += 1

    print(f"Compared {matched} observations against reference")
    if missing_ref:
        print(f"Missing references: {missing_ref}")
    if mismatches:
        print(f"MISMATCHES ({len(mismatches)}):")
        for m in mismatches:
            print(f"  - {m}")
        return 1

    print("ALL MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
