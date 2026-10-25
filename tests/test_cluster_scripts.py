"""Smoke tests for ClusterUY generation sbatch + image recipes (no GGUF / ckpt)."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CLUSTER = REPO_ROOT / "scripts" / "clusteruy"

GEN_SBATCH = (
    "smoke_test.sh",
    "run_phase2_weights.sh",
    "run_phase2_prompting.sh",
    "run_phase2_guided.sh",
    "run_phase2_kvl_beam.sh",
)

CKPT_BIND = '"$CKPT_HOST":/workspace/data/cefr_sp/level_estimator.ckpt:ro'
OFFLINE_EXPORTS = (
    "TRANSFORMERS_OFFLINE=1",
    "HF_HUB_OFFLINE=1",
    "HF_HOME=/opt/hf_home",
    "NLTK_DATA=/usr/share/nltk_data",
    "PYTHONPATH=/workspace/src",
)


def _read(name: str) -> str:
    path = CLUSTER / name
    assert path.is_file(), f"missing {path}"
    return path.read_text(encoding="utf-8")


class TestPhase2GuidedSbatch:
    def test_guided_script_exists_and_is_formal_sweep(self):
        text = _read("run_phase2_guided.sh")
        assert "phase2 guided" in text
        assert "--prompts all" in text
        assert "--no-plot" in text
        assert "phase2 beam" not in text
        assert "run_phase2_beam.sh" not in text

    def test_guided_binds_ckpt(self):
        text = _read("run_phase2_guided.sh")
        assert CKPT_BIND in text
        assert 'CKPT_HOST="${CEFR_SP_CKPT:-$PROJECT_DIR/data/cefr_sp/level_estimator.ckpt}"' in text


class TestGenerationSbatchCefrSp:
    @pytest.mark.parametrize("script", GEN_SBATCH)
    def test_requires_ckpt_and_binds_it(self, script: str):
        text = _read(script)
        assert 'CKPT_HOST="${CEFR_SP_CKPT:-$PROJECT_DIR/data/cefr_sp/level_estimator.ckpt}"' in text
        assert CKPT_BIND in text
        assert "if [[ ! -f" in text
        assert "exit 1" in text
        assert "--no-enable-cefr-sp" not in text

    @pytest.mark.parametrize("script", GEN_SBATCH)
    def test_offline_hf_and_nltk_env(self, script: str):
        text = _read(script)
        for flag in OFFLINE_EXPORTS:
            assert flag in text, f"{script} missing {flag}"

    def test_assess_ckpt_bind_stays_optional(self):
        text = _read("run_assess.sh")
        assert CKPT_BIND in text
        assert "if [[ -f \"$CKPT_HOST\" ]]" in text
        assert "if [[ ! -f" not in text


class TestGenerationDockerfile:
    def test_recipe_and_build_script_exist(self):
        assert (CLUSTER / "Dockerfile").is_file()
        assert (CLUSTER / "build_gen_image.sh").is_file()
        build = _read("build_gen_image.sh")
        assert "scripts/clusteruy/Dockerfile" in build
        assert "srobaina99/slm-thesis:latest" in build
        assert "--platform=linux/amd64" in build

    def test_includes_llama_cpp_p100_and_cefr_sp_runtime(self):
        text = _read("Dockerfile")
        assert "llama-cpp-python" in text
        assert "CMAKE_CUDA_ARCHITECTURES=60" in text
        assert "torch==2.4.1" in text
        assert "download.pytorch.org/whl/cu121" in text
        assert "sm_60" in text
        assert "transformers==4.55.4" in text
        assert "pytorch-lightning==2.4.0" in text
        assert "--include-cefr-sp-base" in text
        assert "--skip-tsar" in text
        assert "bert-base-cased" in text or "--include-cefr-sp-base" in text
        assert "punkt_tab" in text
        assert "wordnet" in text
        assert "cmudict" in text
        assert "HF_HOME=/opt/hf_home" in text
        assert "TRANSFORMERS_OFFLINE=1" in text
        assert "HF_HUB_OFFLINE=1" in text
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("COPY") and "ckpt" in stripped.lower():
                pytest.fail(f"generation Dockerfile must not COPY a ckpt: {stripped}")
            if stripped.startswith("RUN") and "flash-attn" in stripped:
                pytest.fail(f"generation Dockerfile must not install flash-attn: {stripped}")

    def test_eval_image_stays_llama_cpp_free(self):
        text = _read("Dockerfile.eval")
        assert "llama-cpp-python" in text  # mentioned as skipped
        assert "grep -v '^llama-cpp-python'" in text
        assert "CMAKE_CUDA_ARCHITECTURES" not in text
        assert "--include-cefr-sp-base" not in text
