"""Phase 2 prompting shot sweep runner."""

from __future__ import annotations

import gc
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tqdm import tqdm

from slm_experiments.core.config import ExperimentConfig
from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.prompts import MODEL_CONFIGS
from slm_experiments.core.result import ExperimentResult
from slm_experiments.core.run_store import RunStore, make_run_id
from slm_experiments.models import get_model_wrapper
from slm_experiments.models.base import REPO_ROOT
from slm_experiments.phase1.configs import DEFAULT_SYSTEM_PROMPT
from slm_experiments.phase1.runner import parse_models, parse_prompts

DEFAULT_SHOT_GRID = [0, 1, 3]


def parse_shots(shots_arg: str) -> List[int]:
    """Parse comma-separated shot counts."""
    if not shots_arg.strip():
        raise ValueError("shots argument must not be empty")

    shots: List[int] = []
    for part in shots_arg.split(","):
        part = part.strip()
        if not part:
            continue
        num_shots = int(part)
        if num_shots < 0:
            raise ValueError(f"shot count must be >= 0, got {num_shots}")
        shots.append(num_shots)

    if not shots:
        raise ValueError("shots argument must contain at least one value")
    return shots


def create_prompting_configs(shot_counts: List[int]) -> List[ExperimentConfig]:
    """
    Create prompting shot sweep configs: prompting ON, weighting OFF.

    Returns 4 models × len(shot_counts) ExperimentConfig objects.
    """
    configs: List[ExperimentConfig] = []

    for model_name, model_info in MODEL_CONFIGS.items():
        for num_shots in shot_counts:
            configs.append(
                ExperimentConfig(
                    model_name=model_info["model_name"],
                    model_id=model_info["model_id"],
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    config_weighting=False,
                    config_prompting=True,
                    weight_factor=1.0,
                    num_shots=num_shots,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    max_new_tokens=200,
                    experiment_name=f"{model_name}_prompting_{num_shots}shot",
                    description=(
                        f"Prompting sweep: {model_name} with {num_shots} shot(s)"
                    ),
                )
            )

    return configs


class PromptingSweepRunner:
    """Run Phase 2 prompting shot sweep across models and prompts."""

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
        shots: str = "0,1,3",
        prompts: Union[str, int] = "3",
        models: str = "all",
        seed: int = 42,
        no_plot: bool = False,
        cli_args: Optional[List[str]] = None,
    ) -> Tuple[str, Path]:
        """
        Execute the prompting shot sweep and write a run bundle.

        Returns:
            (run_id, output_directory)
        """
        shot_list = parse_shots(shots)
        prompt_list = parse_prompts(prompts)
        model_list = parse_models(models)

        all_configs = create_prompting_configs(shot_list)
        configs_by_model = {
            model_name: [c for c in all_configs if c.model_name == model_name]
            for model_name in model_list
        }

        for model_name in model_list:
            if not configs_by_model[model_name]:
                raise ValueError(
                    f"No prompting configs found for model '{model_name}'"
                )

        total_observations = len(prompt_list) * sum(
            len(configs_by_model[m]) for m in model_list
        )
        started_at = datetime.now(timezone.utc)
        run_id = make_run_id(2, "prompting", started_at=started_at.replace(tzinfo=None))

        results: List[ExperimentResult] = []

        with tqdm(total=total_observations, desc="Phase 2 prompting", unit="obs") as pbar:
            for model_name in model_list:
                wrapper = get_model_wrapper(model_name, seed=seed)
                model_configs = configs_by_model[model_name]

                for prompt_idx, prompt in enumerate(prompt_list):
                    prompt_id = f"P{prompt_idx + 1}"
                    pbar.set_description(f"{model_name} {prompt_id}")

                    for base_config in model_configs:
                        config = replace(base_config, prompt_id=prompt_id)
                        pbar.set_postfix(shots=config.num_shots)

                        result = self.pipeline.run(
                            prompt=prompt,
                            config=config,
                            model=wrapper,
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
            experiment="prompting",
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
    "DEFAULT_SHOT_GRID",
    "PromptingSweepRunner",
    "create_prompting_configs",
    "parse_shots",
]
