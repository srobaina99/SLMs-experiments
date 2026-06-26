"""Phase 2 KVL beam width sweep runner."""

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
from slm_experiments.phase2.beam import parse_widths

DEFAULT_KVL_BEAM_WIDTH_GRID = [4, 8]
_KVL_BEAM_WIDTH_PATTERN = re.compile(r"_kvl_beam_w(\d+)$")


def kvl_beam_width_from_config(config: ExperimentConfig) -> int:
    """Extract KVL beam width encoded in experiment_name (e.g. Qwen3_kvl_beam_w8)."""
    match = _KVL_BEAM_WIDTH_PATTERN.search(config.experiment_name)
    if not match:
        raise ValueError(
            f"Cannot parse KVL beam width from experiment_name: {config.experiment_name}"
        )
    return int(match.group(1))


def create_kvl_beam_configs(
    beam_widths: List[int],
    branch_factor: int = 10,
    kvl_l1: str = "es",
) -> List[ExperimentConfig]:
    """
    Create KVL beam sweep configs: prompting ON (zero-shot), weighting OFF.

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
                    config_kvl_beam=True,
                    kvl_beam_width=width,
                    kvl_branch_factor=branch_factor,
                    kvl_l1=kvl_l1,
                    weight_factor=1.0,
                    num_shots=0,
                    temperature=0.0,
                    top_k=50,
                    top_p=0.95,
                    max_new_tokens=200,
                    experiment_name=f"{model_name}_kvl_beam_w{width}",
                    description=(
                        f"KVL beam sweep: {model_name} with contextual prompting "
                        f"(width={width}, branch_factor={branch_factor}, l1={kvl_l1})"
                    ),
                )
            )

    return configs


class KvlBeamSweepRunner:
    """Run Phase 2 KVL beam width sweep across models and prompts."""

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
        widths: str = "4,8",
        branch_factor: int = 10,
        kvl_l1: str = "es",
        prompts: Union[str, int] = "3",
        models: str = "all",
        seed: int = 42,
        no_plot: bool = False,
        cli_args: Optional[List[str]] = None,
    ) -> Tuple[str, Path]:
        """
        Execute the KVL beam width sweep and write a run bundle.

        Returns:
            (run_id, output_directory)
        """
        width_list = parse_widths(widths)
        prompt_list = parse_prompts(prompts)
        model_list = parse_models(models)

        all_configs = create_kvl_beam_configs(
            width_list, branch_factor=branch_factor, kvl_l1=kvl_l1
        )
        configs_by_model = {
            model_name: [c for c in all_configs if c.model_name == model_name]
            for model_name in model_list
        }

        for model_name in model_list:
            if not configs_by_model[model_name]:
                raise ValueError(f"No KVL beam configs found for model '{model_name}'")

        total_observations = len(prompt_list) * sum(
            len(configs_by_model[m]) for m in model_list
        )
        started_at = datetime.now(timezone.utc)
        run_id = make_run_id(2, "kvl_beam", started_at=started_at.replace(tzinfo=None))

        results: List[ExperimentResult] = []

        with tqdm(total=total_observations, desc="Phase 2 kvl_beam", unit="obs") as pbar:
            for model_name in model_list:
                wrapper = get_model_wrapper(model_name, seed=seed, timeout_seconds=7200)
                model_configs = configs_by_model[model_name]

                for prompt_idx, prompt in enumerate(prompt_list):
                    prompt_id = f"P{prompt_idx + 1}"
                    pbar.set_description(f"{model_name} {prompt_id}")

                    for base_config in model_configs:
                        config = replace(base_config, prompt_id=prompt_id)
                        beam_width = kvl_beam_width_from_config(config)
                        pbar.set_postfix(width=beam_width, l1=kvl_l1)

                        result = self.pipeline.run_kvl_beam(
                            prompt=prompt,
                            config=config,
                            model=wrapper,
                            beam_width=beam_width,
                            branch_factor=config.kvl_branch_factor,
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
            experiment="kvl_beam",
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
    "DEFAULT_KVL_BEAM_WIDTH_GRID",
    "KvlBeamSweepRunner",
    "create_kvl_beam_configs",
    "kvl_beam_width_from_config",
    "parse_widths",
]
