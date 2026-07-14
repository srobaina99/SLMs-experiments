"""Phase 2 beam width sweep runner (deprecated — use kvl_beam or guided)."""

from __future__ import annotations

import gc
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tqdm import tqdm

from slm_experiments.core.config import ExperimentConfig, cefr_sp_config_kwargs
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.prompts import MODEL_CONFIGS
from slm_experiments.core.result import ExperimentResult
from slm_experiments.core.run_store import RunStore, make_run_id
from slm_experiments.models import get_model_wrapper
from slm_experiments.models.base import REPO_ROOT
from slm_experiments.phase1.configs import DEFAULT_SYSTEM_PROMPT
from slm_experiments.phase1.runner import parse_models, parse_prompts

DEFAULT_BEAM_WIDTH_GRID = [4, 8, 10]
_BEAM_WIDTH_PATTERN = re.compile(r"_beam_w(\d+)$")


def parse_widths(widths_arg: str) -> List[int]:
    """Parse comma-separated beam widths."""
    if not widths_arg.strip():
        raise ValueError("widths argument must not be empty")

    widths: List[int] = []
    for part in widths_arg.split(","):
        part = part.strip()
        if not part:
            continue
        width = int(part)
        if width <= 0:
            raise ValueError(f"beam width must be > 0, got {width}")
        widths.append(width)

    if not widths:
        raise ValueError("widths argument must contain at least one value")
    return widths


def beam_width_from_config(config: ExperimentConfig) -> int:
    """Extract beam width encoded in experiment_name (e.g. Qwen3_beam_w8)."""
    match = _BEAM_WIDTH_PATTERN.search(config.experiment_name)
    if not match:
        raise ValueError(
            f"Cannot parse beam width from experiment_name: {config.experiment_name}"
        )
    return int(match.group(1))


def create_beam_configs(beam_widths: List[int]) -> List[ExperimentConfig]:
    """
    Create beam sweep configs: prompting ON (zero-shot), weighting OFF.

    Returns 4 models × len(beam_widths) ExperimentConfig objects.
    """
    configs: List[ExperimentConfig] = []

    for model_name, model_info in MODEL_CONFIGS.items():
        for width in beam_widths:
            configs.append(
                ExperimentConfig(
                    model_name=model_info["model_name"],
                    model_id=model_info["model_id"],
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    config_weighting=False,
                    config_prompting=True,
                    weight_factor=1.0,
                    num_shots=0,
                    temperature=0.0,
                    top_k=50,
                    max_new_tokens=200,
                    experiment_name=f"{model_name}_beam_w{width}",
                    description=(
                        f"Beam sweep: {model_name} with contextual prompting "
                        f"(width={width}, A1-ratio selection)"
                    ),
                )
            )

    return configs


class BeamSweepRunner:
    """Run Phase 2 beam width sweep across models and prompts."""

    def __init__(
        self,
        results_root: Optional[Union[Path, str]] = None,
        pipeline: Optional[ExperimentPipeline] = None,
    ):
        root = Path(results_root) if results_root is not None else Path(REPO_ROOT) / "results"
        self.run_store = RunStore(root)
        self.pipeline = pipeline or ExperimentPipeline()

    def run(
        self,
        widths: str = "4,8,10",
        prompts: Union[str, int] = "3",
        models: str = "all",
        seed: int = 42,
        no_plot: bool = False,
        cli_args: Optional[List[str]] = None,
        enable_cefr_sp: bool = True,
        cefr_sp_ckpt_path: str = "",
        cefr_sp_device: str = "cpu",
    ) -> Tuple[str, Path]:
        """
        Execute the beam width sweep and write a run bundle.

        Returns:
            (run_id, output_directory)
        """
        width_list = parse_widths(widths)
        prompt_list = parse_prompts(prompts)
        model_list = parse_models(models)
        cefr_sp_fields = cefr_sp_config_kwargs(
            enable_cefr_sp=enable_cefr_sp,
            cefr_sp_ckpt_path=cefr_sp_ckpt_path,
            cefr_sp_device=cefr_sp_device,
        )

        all_configs = create_beam_configs(width_list)
        configs_by_model = {
            model_name: [c for c in all_configs if c.model_name == model_name]
            for model_name in model_list
        }

        for model_name in model_list:
            if not configs_by_model[model_name]:
                raise ValueError(f"No beam configs found for model '{model_name}'")

        total_observations = len(prompt_list) * sum(
            len(configs_by_model[m]) for m in model_list
        )
        started_at = datetime.now(timezone.utc)
        run_id = make_run_id(2, "beam", started_at=started_at.replace(tzinfo=None))

        results: List[ExperimentResult] = []

        with tqdm(total=total_observations, desc="Phase 2 beam", unit="obs") as pbar:
            for model_name in model_list:
                wrapper = get_model_wrapper(model_name, seed=seed)
                model_configs = configs_by_model[model_name]

                for prompt_idx, prompt in enumerate(prompt_list):
                    prompt_id = f"P{prompt_idx + 1}"
                    pbar.set_description(f"{model_name} {prompt_id}")

                    for base_config in model_configs:
                        config = replace(base_config, prompt_id=prompt_id, **cefr_sp_fields)
                        beam_width = beam_width_from_config(config)
                        pbar.set_postfix(width=beam_width)

                        result = self.pipeline.run_beam(
                            prompt=prompt,
                            config=config,
                            model=wrapper,
                            beam_width=beam_width,
                            experiment_name=config.experiment_name,
                        )
                        results.append(result)
                        pbar.update(1)

                if hasattr(wrapper, "cleanup"):
                    wrapper.cleanup()
                del wrapper
                gc.collect()

        completed_at = datetime.now(timezone.utc)
        out_dir = self.run_store.write_bundle(
            run_id,
            results,
            phase=2,
            experiment="beam",
            cli_args=cli_args or [],
            models=model_list,
            prompt_count=len(prompt_list),
            started_at=started_at,
            completed_at=completed_at,
        )

        if no_plot:
            print("Plot skipped (--no-plot)")
        else:
            from slm_experiments.plot import plot_run

            plots_dir = plot_run(run_id, results_root=self.run_store.results_root)
            print(f"Plots: {plots_dir.resolve()}")

        return run_id, out_dir


__all__ = [
    "DEFAULT_BEAM_WIDTH_GRID",
    "BeamSweepRunner",
    "beam_width_from_config",
    "create_beam_configs",
    "parse_widths",
]
