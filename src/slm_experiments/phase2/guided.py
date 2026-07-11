"""Phase 2 guided decoding top-K pool sweep runner.

Carrier: prompting ON (zero-shot), weighting OFF. Pool ``0`` is an in-run
unconstrained baseline (``config_guided=False`` → plain greedy).
"""

from __future__ import annotations

import gc
import re
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

DEFAULT_TOP_K_POOL_GRID = [0, 5, 10, 20]
_GUIDED_TOP_K_PATTERN = re.compile(r"_guided_k(\d+)$")


def parse_top_k_pools(top_k_pools_arg: str) -> List[int]:
    """Parse comma-separated guided top-K pool sizes (0 = unconstrained baseline)."""
    if not top_k_pools_arg.strip():
        raise ValueError("top-k-pools argument must not be empty")

    pools: List[int] = []
    for part in top_k_pools_arg.split(","):
        part = part.strip()
        if not part:
            continue
        pool_size = int(part)
        if pool_size < 0:
            raise ValueError(f"guided top-k pool size must be >= 0, got {pool_size}")
        pools.append(pool_size)

    if not pools:
        raise ValueError("top-k-pools argument must contain at least one value")
    return pools


def guided_top_k_from_config(config: ExperimentConfig) -> int:
    """Extract guided top-K pool size encoded in experiment_name (e.g. Qwen3_guided_k10)."""
    match = _GUIDED_TOP_K_PATTERN.search(config.experiment_name)
    if not match:
        raise ValueError(
            f"Cannot parse guided top-k from experiment_name: {config.experiment_name}"
        )
    return int(match.group(1))


def create_guided_configs(
    top_k_pools: List[int],
    guided_mode: str = "flat",
) -> List[ExperimentConfig]:
    """
    Create guided decoding sweep configs: prompting ON (zero-shot), weighting OFF.

    Pool size 0 is an in-run unconstrained baseline (``config_guided=False``).
    Returns 4 models × len(top_k_pools) ExperimentConfig objects.
    """
    configs: List[ExperimentConfig] = []

    for model_name, model_info in MODEL_CONFIGS.items():
        for pool_size in top_k_pools:
            is_baseline = pool_size == 0
            if is_baseline:
                description = (
                    f"Guided decode sweep baseline: {model_name} with contextual "
                    f"prompting, unconstrained greedy (guided OFF)"
                )
            else:
                description = (
                    f"Guided decode sweep: {model_name} with contextual prompting "
                    f"(top_k_pool={pool_size}, mode={guided_mode})"
                )
            configs.append(
                ExperimentConfig(
                    model_name=model_info["model_name"],
                    model_id=model_info["model_id"],
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    config_weighting=False,
                    config_prompting=True,
                    config_guided=not is_baseline,
                    weight_factor=1.0,
                    num_shots=0,
                    guided_top_k=pool_size,
                    guided_mode=guided_mode,
                    temperature=0.0,
                    top_k=50,
                    max_new_tokens=200,
                    experiment_name=f"{model_name}_guided_k{pool_size}",
                    description=description,
                )
            )

    return configs


def _stamp_guided_baseline_metadata(
    result: ExperimentResult,
    guided_mode: str,
) -> ExperimentResult:
    """Ensure baseline rows bucket under by_guided_top_k=0."""
    result.guided_top_k = 0
    result.guided_mode = guided_mode
    result.guided_steps_a1_chosen = 0
    result.guided_steps_total = 0
    result.guided_intervention_rate = 0.0
    return result


class GuidedSweepRunner:
    """Run Phase 2 guided decoding top-K pool sweep across models and prompts."""

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
        top_k_pools: str = "0,5,10,20",
        prompts: Union[str, int] = "3",
        models: str = "all",
        seed: int = 42,
        no_plot: bool = False,
        cli_args: Optional[List[str]] = None,
        mode: str = "flat",
    ) -> Tuple[str, Path]:
        """
        Execute the guided top-K pool sweep and write a run bundle.

        Returns:
            (run_id, output_directory)
        """
        pool_list = parse_top_k_pools(top_k_pools)
        prompt_list = parse_prompts(prompts)
        model_list = parse_models(models)

        all_configs = create_guided_configs(pool_list, guided_mode=mode)
        configs_by_model = {
            model_name: [c for c in all_configs if c.model_name == model_name]
            for model_name in model_list
        }

        for model_name in model_list:
            if not configs_by_model[model_name]:
                raise ValueError(f"No guided configs found for model '{model_name}'")

        total_observations = len(prompt_list) * sum(
            len(configs_by_model[m]) for m in model_list
        )
        started_at = datetime.now(timezone.utc)
        run_id = make_run_id(2, "guided", started_at=started_at.replace(tzinfo=None))

        results: List[ExperimentResult] = []

        with tqdm(total=total_observations, desc="Phase 2 guided", unit="obs") as pbar:
            for model_name in model_list:
                wrapper = get_model_wrapper(model_name, seed=seed)
                model_configs = configs_by_model[model_name]

                for prompt_idx, prompt in enumerate(prompt_list):
                    prompt_id = f"P{prompt_idx + 1}"
                    pbar.set_description(f"{model_name} {prompt_id}")

                    for base_config in model_configs:
                        config = replace(base_config, prompt_id=prompt_id)
                        guided_k = guided_top_k_from_config(config)
                        pbar.set_postfix(top_k=guided_k, mode=mode)

                        if not config.config_guided:
                            result = self.pipeline.run(
                                prompt=prompt,
                                config=config,
                                model=wrapper,
                                experiment_name=config.experiment_name,
                            )
                            result = _stamp_guided_baseline_metadata(result, mode)
                        else:
                            result = self.pipeline.run_guided(
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
            experiment="guided",
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
    "DEFAULT_TOP_K_POOL_GRID",
    "GuidedSweepRunner",
    "create_guided_configs",
    "guided_top_k_from_config",
    "parse_top_k_pools",
]
