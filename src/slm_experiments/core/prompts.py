"""Standard prompts and model registry for experiments."""

# Standard prompts for English learning experiments
# Grounded in CEFR 2001 §4.2 thematic areas (van Ek & Trim, Threshold Level 1990)
# and CEFR 2020 Companion Volume A1 descriptors.
# See: https://rm.coe.int/16802fc1bf (CEFR 2001)
#      https://rm.coe.int/common-european-framework-of-reference-for-languages-learning-teaching/16809ea0d4 (CEFR 2020)
STANDARD_PROMPTS = [
    # 13. Language
    "What does the word 'library' mean?",
    # 1. Personal identification (CEFR 2020 A1: "Can introduce him/herself")
    "How do I introduce myself in English?",
    # 10. Food and drink
    "Can you explain what 'breakfast' is?",
    # 13. Language
    "What is the difference between 'big' and 'large'?",
    # 3. Daily life
    "Can you describe what happens in the morning?",
    # 13. Language
    "What does 'happy' mean?",
    # 1. Personal identification (CEFR 2020 A1 descriptor)
    "How do you say goodbye in English?",
    # 14. Weather
    "What is the weather like today?",
    # 6. Relations with other people
    "What is a 'friend'?",
    # 8. Education (CEFR 2020 educational domain)
    "What do you do at school?",
    # 13. Language
    "What is the difference between 'hot' and 'cold'?",
    # 6. Relations with other people (CEFR 2020: "people he/she knows")
    "Can you describe your family?",
    # 10. Food and drink
    "What foods do you eat for lunch?",
    # 2. House and home, environment
    "What rooms are in a house?",
    # 2. House and home, environment (pets as part of domestic life)
    "What animals can be pets?",
    # 4. Free time, entertainment
    "What do you do on the weekend?",
    # 4. Free time, entertainment
    "What can people do for fun?",
    # 5. Travel
    "How do people travel to school or work?",
    # 7. Health and body care
    "What do you do when you feel sick?",
    # 9. Shopping
    "How do I ask for something at a shop?",
    # 11. Services
    "What happens when you go to the doctor?",
    # 12. Places
    "What can you see in a town?",
    # 3. Daily life
    "What does a person do every day?",
    # 1. Personal identification - occupation (CEFR 2020 occupational domain)
    "What do people do at work?",
    # 13. Language
    "What is the difference between 'this' and 'that'?",
]

CONTEXT_BLOCK = (
    "# Context\n"
    "Please respond using simple words that a young non-English speaking "
    "student can understand. Use vocabulary from basic English learning "
    "materials. Keep sentences short and clear. Avoid complex grammar "
    "structures and difficult words.\n\n"
)

SHOT_EXAMPLES = [
    (
        "Question: What is a cat?\n"
        "Answer: A cat is a small animal. It is soft and likes to play and sleep."
    ),
    (
        "Question: What does 'happy' mean?\n"
        "Answer: Happy means you feel good. You smile when you are happy. "
        "Happy is a nice feeling."
    ),
    (
        "Question: What is water?\n"
        "Answer: Water is a drink. We need water every day."
    ),
]


def build_contextual_prompt(user_prompt: str, num_shots: int = 0) -> str:
    """
    Build a contextual prompt with optional in-context examples.

    num_shots: 0 (zero-shot), 1 (one-shot), or 3 (few-shot).
    """
    if num_shots < 0:
        raise ValueError(f"num_shots must be >= 0, got {num_shots}")
    if num_shots > len(SHOT_EXAMPLES):
        raise ValueError(
            f"num_shots {num_shots} exceeds available examples ({len(SHOT_EXAMPLES)})"
        )

    if num_shots == 0:
        return CONTEXT_BLOCK + user_prompt

    parts = [CONTEXT_BLOCK]
    if num_shots == 1:
        parts.append("# Example\n")
        parts.append(SHOT_EXAMPLES[0])
        parts.append("\n\n")
    else:
        parts.append("# Examples\n")
        for example in SHOT_EXAMPLES[:num_shots]:
            parts.append(example)
            parts.append("\n\n")

    parts.append(user_prompt)
    return "".join(parts)


MODEL_CONFIGS = {
    "Phi3": {
        "model_name": "Phi3",
        "model_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
    },
    "Qwen2": {
        "model_name": "Qwen2",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    },
    "Qwen3": {
        "model_name": "Qwen3",
        "model_id": "ggml-org/Qwen3-0.6B-GGUF",
    },
    "TinyLlama": {
        "model_name": "TinyLlama",
        "model_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    },
}
