"""Smoke tests for repo scaffold — no GGUF required."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestScaffold:
    def test_docs_exist(self):
        for doc in [
            "AGENT.md",
            "README.md",
            "ExperimentDesign.md",
            "docs/metrics.md",
            "docs/models.md",
            "docs/interventions.md",
        ]:
            assert (REPO_ROOT / doc).exists(), f"Missing {doc}"

    def test_vocabulary_exists(self):
        vocab = REPO_ROOT / "data/vocabularies/filtered_starters_vocab.txt"
        assert vocab.exists()
        lines = [l.strip() for l in vocab.read_text().splitlines() if l.strip()]
        assert len(lines) == 493

    def test_requirements(self):
        assert (REPO_ROOT / "requirements.txt").exists()
        assert (REPO_ROOT / "requirements-dev.txt").exists()
        runtime = (REPO_ROOT / "requirements.txt").read_text()
        assert "llama-cpp-python" in runtime
        assert "torch" not in runtime
        assert "transformers" not in runtime

    def test_package_importable(self):
        import slm_experiments
        assert slm_experiments.__version__ == "0.1.0"

    def test_cli_parses_help(self):
        from slm_experiments.cli import main
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
