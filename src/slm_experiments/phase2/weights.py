"""Phase 2 weight factor sweep runner."""

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

DEFAULT_WEIGHT_GRID = [1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0]


def parse_weights(weights_arg: str) -> List[float]:
    """Parse comma-separated weight factors."""
    if not weights_arg.strip():
        raise ValueError("weights argument must not be empty")

    weights: List[float] = []
    for part in weights_arg.split(","):
        part = part.strip()
        if not part:
            continue
        weight = float(part)
        if weight <= 0:
            raise ValueError(f"weight factor must be > 0, got {weight}")
        weights.append(weight)

    if not weights:
        raise ValueError("weights argument must contain at least one value")
    return weights


def create_weight_configs(weight_factors: List[float]) -> List[ExperimentConfig]:
    """
    Create weight sweep configs: weighting ON + prompting ON for each factor.

    Returns 4 models × len(weight_factors) ExperimentConfig objects.
    """
    configs: List[ExperimentConfig] = []

    for model_name, model_info in MODEL_CONFIGS.items():
        for weight in weight_factors:
            weight_str = str(weight).replace(".", "_")
            configs.append(
                ExperimentConfig(
                    model_name=model_info["model_name"],
                    model_id=model_info["model_id"],
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    config_weighting=True,
                    config_prompting=True,
                    weight_factor=weight,
                    num_shots=0,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    max_new_tokens=200,
                    experiment_name=f"{model_name}_weighted_prompted_{weight_str}",
                    description=(
                        f"Weight sweep: {model_name} with weighting + prompting "
                        f"(factor={weight})"
                    ),
                )
            )

    return configs


class WeightSweepRunner:
    """Run Phase 2 weight factor sweep across models and prompts."""

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
        weights: str = "1.0,1.3,1.5,2.0,2.5,3.0,4.0",
        prompts: Union[str, int] = "3",
        models: str = "all",
        seed: int = 42,
        no_plot: bool = False,
        cli_args: Optional[List[str]] = None,
    ) -> Tuple[str, Path]:
        """
        Execute the weight sweep and write a run bundle.

        Returns:
            (run_id, output_directory)
        """
        weight_list = parse_weights(weights)
        prompt_list = parse_prompts(prompts)
        model_list = parse_models(models)

        all_configs = create_weight_configs(weight_list)
        configs_by_model = {
            model_name: [c for c in all_configs if c.model_name == model_name]
            for model_name in model_list
        }

        for model_name in model_list:
            if not configs_by_model[model_name]:
                raise ValueError(f"No weight configs found for model '{model_name}'")

        total_observations = len(prompt_list) * sum(
            len(configs_by_model[m]) for m in model_list
        )
        started_at = datetime.now(timezone.utc)
        run_id = make_run_id(2, "weights", started_at=started_at.replace(tzinfo=None))

        results: List[ExperimentResult] = []

        with tqdm(total=total_observations, desc="Phase 2 weights", unit="obs") as pbar:
            for model_name in model_list:
                wrapper = get_model_wrapper(model_name, seed=seed)
                model_configs = configs_by_model[model_name]

                for prompt_idx, prompt in enumerate(prompt_list):
                    prompt_id = f"P{prompt_idx + 1}"
                    pbar.set_description(f"{model_name} {prompt_id}")

                    for base_config in model_configs:
                        config = replace(base_config, prompt_id=prompt_id)
                        pbar.set_postfix(
                            weight=config.weight_factor,
                            prompting=config.config_prompting,
                        )

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
            experiment="weights",
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
