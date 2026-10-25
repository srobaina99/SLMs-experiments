"""Smoke tests for ClusterUY generation sbatch + image recipes (no GGUF / ckpt)."""

import json
import os
from pathlib import Path
import subprocess
import sys

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
        assert "llama-cpp-python==0.3.16" in text
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


@pytest.fixture
def generation_stubs(tmp_path):
    """Run the launchers' inner shell with stubbed hardware/container/Python tools."""
    project = tmp_path / "project with spaces"
    project.mkdir()
    checkpoint = project / "data" / "cefr_sp" / "level_estimator.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("default checkpoint")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "invocation.json"
    python_capture = tmp_path / "python.json"
    (bin_dir / "nvidia-smi").write_text("#!/bin/bash\nexit 0\n")
    # Emulate only the workspace cd. Execute the actual inner shell unchanged
    # otherwise, including exports, so a leaked host CEFR_SP_CKPT is observable.
    (bin_dir / "singularity").write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps(args))\n"
        "assert args[-3:-1] == ['bash', '-c']\n"
        "command = args[-1].replace('cd /workspace', 'cd \"' + "
        "os.environ['SLURM_SUBMIT_DIR'] + '\"')\n"
        "sys.exit(subprocess.run(['bash', '-c', command]).returncode)\n"
    )
    (bin_dir / "python").write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['PYTHON_CAPTURE']).write_text(json.dumps({\n"
        "    'args': sys.argv[1:],\n"
        "    'checkpoint': os.environ.get('CEFR_SP_CKPT'),\n"
        "    'offline': os.environ.get('HF_HUB_OFFLINE'),\n"
        "}))\n"
    )
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    env = os.environ.copy()
    env.pop("CEFR_SP_CKPT", None)
    env.update(
        PATH=f"{bin_dir}{os.pathsep}{env['PATH']}",
        HOME=str(tmp_path),
        SLURM_SUBMIT_DIR=str(project),
        SLURM_JOB_ID="test-job",
        SLURM_NODELIST="test-node",
        CAPTURE=str(capture),
        PYTHON_CAPTURE=str(python_capture),
    )
    return project, checkpoint, capture, python_capture, env


@pytest.mark.parametrize("script", GEN_SBATCH)
@pytest.mark.parametrize("custom_checkpoint", [False, True])
def test_generation_launcher_maps_host_checkpoint_inside_container(
    script, custom_checkpoint, generation_stubs
):
    project, checkpoint, capture, python_capture, env = generation_stubs
    if custom_checkpoint:
        checkpoint = project.parent / "external checkpoint.ckpt"
        checkpoint.write_text("chosen checkpoint")
        env["CEFR_SP_CKPT"] = str(checkpoint)

    result = subprocess.run(
        ["bash", str(CLUSTER / script)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(capture.read_text())
    binds = [args[i + 1] for i, value in enumerate(args) if value == "--bind"]
    assert f"{project}:/workspace" in binds
    assert f"{checkpoint}:/workspace/data/cefr_sp/level_estimator.ckpt:ro" in binds
    inner_python = json.loads(python_capture.read_text())
    assert inner_python["checkpoint"] == "/workspace/data/cefr_sp/level_estimator.ckpt"
    assert inner_python["offline"] == "1"
    assert inner_python["args"][:3] == ["-m", "slm_experiments", "phase2"]
    assert "--no-plot" in inner_python["args"]
    if script == "smoke_test.sh":
        assert inner_python["args"][inner_python["args"].index("--prompts") + 1] == "1"
    else:
        assert inner_python["args"][inner_python["args"].index("--prompts") + 1] == "all"


@pytest.mark.parametrize("script", GEN_SBATCH)
def test_generation_launcher_rejects_missing_override_without_fallback(
    script, generation_stubs
):
    project, default_checkpoint, capture, python_capture, env = generation_stubs
    assert default_checkpoint.is_file()
    missing = project.parent / "missing override.ckpt"
    env["CEFR_SP_CKPT"] = str(missing)
    result = subprocess.run(
        ["bash", str(CLUSTER / script)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 1
    assert str(missing) in result.stderr
    assert not capture.exists()
    assert not python_capture.exists()
