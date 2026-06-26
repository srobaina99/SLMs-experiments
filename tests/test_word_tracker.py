"""Tests for incremental word-boundary tracking."""

from slm_experiments.models.word_tracker import WordTracker


class TestWordTracker:
    def test_multi_token_word_completes_at_boundary(self):
        tracker = WordTracker()

        assert tracker.append_token_text("some") == []
        assert tracker.pending_word == "some"

        assert tracker.append_token_text("one ") == ["someone"]
        assert tracker.pending_word == ""

    def test_punctuation_boundary_emits_clean_word(self):
        tracker = WordTracker()

        assert tracker.append_token_text("friend.") == ["friend"]
        assert tracker.pending_word == ""

    def test_partial_word_does_not_emit(self):
        tracker = WordTracker()

        assert tracker.append_token_text("fund") == []
        assert tracker.pending_word == "fund"

    def test_flush_pending_emits_trailing_word(self):
        tracker = WordTracker()
        tracker.append_token_text("play")

        assert tracker.flush_pending() == ["play"]
        assert tracker.pending_word == ""

    def test_is_content_word_checks_cleaned_form(self):
        content_words = {"friend", "play"}
        assert WordTracker.is_content_word("Friend.", content_words) is True
        assert WordTracker.is_content_word("the", content_words) is False
