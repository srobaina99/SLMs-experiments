"""Response formatting utilities for cleaning LLM responses before evaluation."""

import re


class ResponseFormatter:
    """Clean LLM responses before readability scoring."""

    THINKING_TAG_PATTERN = re.compile(
        r"<think>.*?</think>",
        flags=re.DOTALL,
    )

    def __init__(self):
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        self.arrow_pattern = re.compile(r"[→←↑↓↔↕➜➤➡⬅⬆⬇⬌⬍⟶⟵⟷⟸⟹⟺]")
        self.bullet_pattern = re.compile(r"[•·‣⁃▪▫◦‰‱]")
        self.special_chars_pattern = re.compile(r"[★☆✓✗✘✔✖⚡⚠⭐🔥💡📝📊📈📉]")
        self.whitespace_pattern = re.compile(r"\s+")
        self.markdown_pattern = re.compile(r"[*_`#]+")
        self.symbol_parentheses_pattern = re.compile(r"\([^a-zA-Z0-9\s,.-]+\)")

    def strip_qwen3_thinking_tags(self, text: str) -> str:
        """Remove Qwen3 thinking blocks and trailing ChatML markers."""
        if not text:
            return ""

        cleaned = self.THINKING_TAG_PATTERN.sub("", text)

        if "<think>" in cleaned and "</think>" in cleaned:
            think_end = cleaned.find("</think>")
            cleaned = cleaned[think_end + len("</think>") :].strip()

        for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            cleaned = cleaned.replace(marker, "")

        return cleaned.strip()

    def remove_emojis(self, text: str) -> str:
        return self.emoji_pattern.sub("", text)

    def remove_arrows_and_bullets(self, text: str) -> str:
        text = self.arrow_pattern.sub("", text)
        return self.bullet_pattern.sub("", text)

    def remove_special_formatting(self, text: str) -> str:
        return self.special_chars_pattern.sub("", text)

    def remove_markdown_formatting(self, text: str) -> str:
        return self.markdown_pattern.sub("", text)

    def remove_symbol_parentheses(self, text: str) -> str:
        return self.symbol_parentheses_pattern.sub("", text)

    def normalize_whitespace(self, text: str) -> str:
        return self.whitespace_pattern.sub(" ", text).strip()

    def clean_response_for_evaluation(self, response: str) -> str:
        """Comprehensive cleaning of LLM response for text evaluation."""
        if not response or not response.strip():
            return ""

        cleaned_text = response
        cleaned_text = self.strip_qwen3_thinking_tags(cleaned_text)
        cleaned_text = self.remove_emojis(cleaned_text)
        cleaned_text = self.remove_arrows_and_bullets(cleaned_text)
        cleaned_text = self.remove_special_formatting(cleaned_text)
        cleaned_text = self.remove_markdown_formatting(cleaned_text)
        cleaned_text = self.remove_symbol_parentheses(cleaned_text)
        return self.normalize_whitespace(cleaned_text)

    def clean_preserving_structure(self, response: str) -> str:
        """Light cleaning that preserves paragraphs and sentences."""
        if not response or not response.strip():
            return ""

        cleaned_text = self.strip_qwen3_thinking_tags(response)
        cleaned_text = self.remove_emojis(cleaned_text)
        cleaned_text = self.remove_special_formatting(cleaned_text)
        cleaned_text = self.remove_symbol_parentheses(cleaned_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        cleaned_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned_text)
        return cleaned_text.strip()


def clean_for_evaluation(response: str) -> str:
    return ResponseFormatter().clean_response_for_evaluation(response)
