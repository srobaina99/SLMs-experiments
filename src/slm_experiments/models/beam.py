"""Beam search generator with A1-ratio and probability selection (Step 6 generalizes)."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BeamCandidate:
    """Single beam candidate sequence."""

    token_ids: List[int]
    cumulative_log_prob: float
    sequence_text: str


class BeamSearchGenerator:
    """
    Simplified beam search for llama.cpp models.

    Maintains multiple candidate sequences and selects by A1 ratio or log prob.
    Full multi-model integration is completed in Phase 2 (Step 6).
    """

    def __init__(
        self,
        llm,
        beam_width: int = 4,
        max_length: int = 200,
        length_penalty: float = 1.0,
    ):
        self.llm = llm
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_penalty = length_penalty

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> Dict[str, Any]:
        """Generate beam_width candidate sequences via stochastic sampling."""
        beams: List[BeamCandidate] = []

        for _ in range(self.beam_width):
            try:
                output = self.llm(
                    prompt=prompt,
                    max_tokens=self.max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    echo=False,
                )

                if output and "choices" in output and output["choices"]:
                    response_text = output["choices"][0]["text"]
                    token_ids = self.llm.tokenize(response_text.encode("utf-8"))
                    response_length = len(token_ids)
                    if response_length > 0 and temperature > 0:
                        cumulative_log_prob = -(
                            math.log(temperature) * response_length
                        )
                    else:
                        cumulative_log_prob = 0.0

                    beams.append(
                        BeamCandidate(
                            token_ids=token_ids,
                            cumulative_log_prob=cumulative_log_prob,
                            sequence_text=prompt + response_text,
                        )
                    )
            except Exception:
                continue

        if not beams:
            beams.append(
                BeamCandidate(
                    token_ids=[],
                    cumulative_log_prob=0.0,
                    sequence_text=prompt,
                )
            )

        return {"beams": beams}

    def calculate_a1_ratio(
        self,
        text: str,
        a1_vocab: List[str],
        content_words_set: set,
    ) -> Tuple[float, int, int]:
        """Weighted A1-to-content-word ratio (1.5× weight on A1 hits)."""
        words = text.lower().split()
        a1_count = 0
        content_count = 0
        a1_vocab_lower = {word.lower() for word in a1_vocab}

        for word in words:
            word_clean = "".join(c for c in word if c.isalnum())
            if word_clean in content_words_set:
                content_count += 1
                if word_clean in a1_vocab_lower:
                    a1_count += 1

        if content_count == 0:
            return 0.0, a1_count, content_count

        weighted_ratio = (a1_count * 1.5) / content_count
        return weighted_ratio, a1_count, content_count

    def select_best_beams(
        self,
        beams: List[BeamCandidate],
        a1_vocab: List[str],
        content_words_set: set,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Return best beams by A1 ratio and by cumulative log probability."""
        scored_beams: List[Dict[str, Any]] = []

        for beam in beams:
            a1_ratio, a1_count, content_count = self.calculate_a1_ratio(
                beam.sequence_text, a1_vocab, content_words_set
            )
            scored_beams.append(
                {
                    "beam": beam,
                    "a1_ratio": a1_ratio,
                    "a1_count": a1_count,
                    "content_count": content_count,
                    "cumulative_log_prob": beam.cumulative_log_prob,
                }
            )

        if not scored_beams:
            return {"best_by_a1_ratio": None, "best_by_probability": None}

        return {
            "best_by_a1_ratio": max(scored_beams, key=lambda x: x["a1_ratio"]),
            "best_by_probability": max(
                scored_beams, key=lambda x: x["cumulative_log_prob"]
            ),
        }
