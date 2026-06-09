"""Phase 1 factorial experiment runner."""

from __future__ import annotations

import gc
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tqdm import tqdm

from slm_experiments.core.pipeline import ExperimentPipeline
from slm_experiments.core.prompts import MODEL_CONFIGS, STANDARD_PROMPTS
from slm_experiments.core.result import ExperimentResult
from slm_experiments.core.run_store import RunStore, make_run_id
from slm_experiments.models import get_model_wrapper
from slm_experiments.models.base import REPO_ROOT
from slm_experiments.phase1.configs import create_factorial_configs


def parse_prompts(prompts_arg: Union[str, int]) -> List[str]:
    """Select prompts: integer count or 'all'."""
    if isinstance(prompts_arg, str) and prompts_arg.lower() == "all":
        return list(STANDARD_PROMPTS)

    count = int(prompts_arg)
    if count < 1:
        raise ValueError(f"prompt count must be >= 1, got {count}")
    if count > len(STANDARD_PROMPTS):
        raise ValueError(
            f"prompt count {count} exceeds available prompts ({len(STANDARD_PROMPTS)})"
        )
    return STANDARD_PROMPTS[:count]


def parse_models(models_arg: str) -> List[str]:
    """Select models: 'all' or comma-separated subset."""
    if models_arg.lower() == "all":
        return list(MODEL_CONFIGS.keys())

    selected = [name.strip() for name in models_arg.split(",") if name.strip()]
    unknown = [name for name in selected if name not in MODEL_CONFIGS]
    if unknown:
        available = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown model(s): {unknown}. Available: {available}")
    return selected


class FactorialRunner:
    """Run the Phase 1 factorial design: models × interventions × prompts."""

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
        prompts: Union[str, int] = "3",
        models: str = "all",
        seed: int = 42,
        no_plot: bool = False,
        cli_args: Optional[List[str]] = None,
    ) -> Tuple[str, Path]:
        """
        Execute the factorial experiment and write a run bundle.

        Returns:
            (run_id, output_directory)
        """
        prompt_list = parse_prompts(prompts)
        model_list = parse_models(models)

        all_configs = create_factorial_configs()
        configs_by_model = {
            model_name: [c for c in all_configs if c.model_name == model_name]
            for model_name in model_list
        }

        for model_name in model_list:
            if not configs_by_model[model_name]:
                raise ValueError(f"No factorial configs found for model '{model_name}'")

        total_observations = len(prompt_list) * sum(
            len(configs_by_model[m]) for m in model_list
        )
        started_at = datetime.now(timezone.utc)
        run_id = make_run_id(1, "factorial", started_at=started_at.replace(tzinfo=None))

        results: List[ExperimentResult] = []

        with tqdm(total=total_observations, desc="Phase 1 factorial", unit="obs") as pbar:
            for model_name in model_list:
                wrapper = get_model_wrapper(model_name, seed=seed)
                model_configs = configs_by_model[model_name]

                for prompt_idx, prompt in enumerate(prompt_list):
                    prompt_id = f"P{prompt_idx + 1}"
                    pbar.set_description(f"{model_name} {prompt_id}")

                    for base_config in model_configs:
                        config = replace(base_config, prompt_id=prompt_id)
                        pbar.set_postfix(
                            weighting=config.config_weighting,
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
            phase=1,
            experiment="factorial",
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
