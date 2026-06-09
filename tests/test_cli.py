"""Tests for CLI dispatch."""

from pathlib import Path
from unittest.mock import patch

import pytest

from slm_experiments.cli import main


class TestCliPhase1Run:
    @patch("slm_experiments.phase1.runner.FactorialRunner")
    def test_phase1_run_dispatches(self, mock_runner_cls, capsys):
        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = ("20260606_120000_phase1_factorial", Path("/tmp/run"))

        main(["phase1", "--prompts", "3", "--seed", "7"])

        mock_runner.run.assert_called_once_with(
            prompts="3",
            models="all",
            seed=7,
            no_plot=False,
            cli_args=["phase1", "--prompts", "3", "--seed", "7"],
        )

        captured = capsys.readouterr()
        assert "20260606_120000_phase1_factorial" in captured.out
        assert "/tmp/run" in captured.out

    @patch("slm_experiments.phase1.runner.FactorialRunner")
    def test_phase1_run_no_plot_flag(self, mock_runner_cls):
        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = ("run_id", Path("/tmp/run"))

        main(["phase1", "--no-plot"])

        mock_runner.run.assert_called_once()
        assert mock_runner.run.call_args.kwargs["no_plot"] is True

    @patch("slm_experiments.phase1.runner.FactorialRunner")
    def test_phase1_run_legacy_command(self, mock_runner_cls):
        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = ("run_id", Path("/tmp/run"))

        main(["phase1", "run", "--prompts", "1"])

        mock_runner.run.assert_called_once_with(
            prompts="1",
            models="all",
            seed=42,
            no_plot=False,
            cli_args=["phase1", "--prompts", "1"],
        )

    @patch("slm_experiments.plot.plot_run")
    def test_plot_command_dispatches(self, mock_plot_run, capsys):
        mock_plot_run.return_value = Path("/tmp/run/plots")

        main(["plot", "--run-id", "20260606_120000_phase1_factorial"])

        mock_plot_run.assert_called_once_with("20260606_120000_phase1_factorial")
        captured = capsys.readouterr()
        assert "Plots written to:" in captured.out
        assert "/tmp/run/plots" in captured.out


class TestCliRuns:
    def test_runs_list_empty(self, tmp_path: Path, capsys, monkeypatch):
        monkeypatch.setattr(
            "slm_experiments.models.base.REPO_ROOT",
            str(tmp_path),
        )
        main(["runs", "list"])
        captured = capsys.readouterr()
        assert "No runs found." in captured.out

    def test_runs_list_and_show(self, tmp_path: Path, capsys, monkeypatch):
        monkeypatch.setattr(
            "slm_experiments.models.base.REPO_ROOT",
            str(tmp_path),
        )

        from slm_experiments.core.pipeline import ExperimentPipeline
        from slm_experiments.core.config import ExperimentConfig
        from slm_experiments.core.run_store import RunStore, make_run_id

        pipeline = ExperimentPipeline()
        config = ExperimentConfig(model_name="Qwen3", prompt_id="P1")
        result = pipeline.run("What is a dog?", config, MockSuccessModel())
        store = RunStore(tmp_path / "results")
        run_id = make_run_id(1, "factorial")
        store.write_bundle(run_id, [result], phase=1, experiment="factorial")

        main(["runs", "list"])
        list_out = capsys.readouterr().out
        assert run_id in list_out
        assert "factorial" in list_out

        main(["runs", "show", run_id])
        show_out = capsys.readouterr().out
        assert f"Run: {run_id}" in show_out
        assert "Phase: 1" in show_out
        assert "flesch_kincaid_grade" in show_out

    def test_runs_show_missing_exits(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "slm_experiments.models.base.REPO_ROOT",
            str(tmp_path),
        )
        with pytest.raises(SystemExit) as exc:
            main(["runs", "show", "missing_run"])
        assert exc.value.code == 1


class MockSuccessModel:
    def generate(self, prompt, config):
        return {
            "response": "A dog is an animal. It is small.",
            "response_time_seconds": 1.0,
            "generation_successful": True,
        }


class TestCliPhase2Run:
    @patch("slm_experiments.phase2.weights.WeightSweepRunner")
    def test_phase2_run_weights_dispatches(self, mock_runner_cls, capsys):
        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = ("20260606_120000_phase2_weights", Path("/tmp/run"))

        main(["phase2", "weights", "--prompts", "3", "--seed", "7"])

        mock_runner.run.assert_called_once_with(
            weights="1.0,1.3,1.5,2.0,2.5,3.0,4.0",
            prompts="3",
            models="all",
            seed=7,
            no_plot=False,
            cli_args=["phase2", "weights", "--prompts", "3", "--seed", "7"],
        )

        captured = capsys.readouterr()
        assert "20260606_120000_phase2_weights" in captured.out

    @patch("slm_experiments.phase2.weights.WeightSweepRunner")
    def test_phase2_run_weights_custom_grid(self, mock_runner_cls):
        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = ("run_id", Path("/tmp/run"))

        main(["phase2", "weights", "--weights", "1.5,2.0"])

        assert mock_runner.run.call_args.kwargs["weights"] == "1.5,2.0"

    @patch("slm_experiments.phase2.prompting.PromptingSweepRunner")
    def test_phase2_run_prompting_dispatches(self, mock_runner_cls, capsys):
        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = (
            "20260606_120000_phase2_prompting",
            Path("/tmp/run"),
        )

        main(["phase2", "prompting", "--shots", "0,1,3"])

        mock_runner.run.assert_called_once_with(
            shots="0,1,3",
            prompts="3",
            models="all",
            seed=42,
            no_plot=False,
            cli_args=["phase2", "prompting", "--shots", "0,1,3"],
        )

        captured = capsys.readouterr()
        assert "20260606_120000_phase2_prompting" in captured.out

    @patch("slm_experiments.phase2.prompting.PromptingSweepRunner")
    def test_phase2_run_prompting_no_plot(self, mock_runner_cls):
        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = ("run_id", Path("/tmp/run"))

        main(["phase2", "prompting", "--no-plot"])

        assert mock_runner.run.call_args.kwargs["no_plot"] is True

    @patch("slm_experiments.phase2.beam.BeamSweepRunner")
    def test_phase2_run_beam_dispatches(self, mock_runner_cls, capsys):
        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = (
            "20260606_120000_phase2_beam",
            Path("/tmp/run"),
        )

        main(["phase2", "beam", "--widths", "4,8,10", "--seed", "7"])

        mock_runner.run.assert_called_once_with(
            widths="4,8,10",
            prompts="3",
            models="all",
            seed=7,
            no_plot=False,
            cli_args=["phase2", "beam", "--widths", "4,8,10", "--seed", "7"],
        )

        captured = capsys.readouterr()
        assert "20260606_120000_phase2_beam" in captured.out

    @patch("slm_experiments.phase2.beam.BeamSweepRunner")
    def test_phase2_run_beam_no_plot(self, mock_runner_cls):
        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = ("run_id", Path("/tmp/run"))

        main(["phase2", "beam", "--no-plot"])

        assert mock_runner.run.call_args.kwargs["no_plot"] is True


class TestCliHuman:
    @patch("slm_experiments.human.export.HumanExporter")
    def test_human_export_dispatches(self, mock_exporter_cls, capsys):
        mock_exporter = mock_exporter_cls.return_value
        mock_exporter.export.return_value = (Path("/tmp/run/human_review.csv"), 12)

        main(["human", "export", "--run-id", "20260606_120000_phase1_factorial", "--sample", "12"])

        mock_exporter.export.assert_called_once_with(
            run_id="20260606_120000_phase1_factorial",
            sample=12,
        )

        captured = capsys.readouterr()
        assert "Exported 12 rows" in captured.out
        assert "human_review.csv" in captured.out

    @patch("importlib.import_module")
    def test_human_import_dispatches(self, mock_import_module, capsys):
        mock_module = mock_import_module.return_value
        mock_importer = mock_module.HumanImporter.return_value
        mock_importer.import_tags.return_value = 5

        main(
            [
                "human",
                "import",
                "--run-id",
                "20260606_120000_phase1_factorial",
                "--tags",
                "/tmp/human_review.csv",
            ]
        )

        mock_import_module.assert_called_once_with("slm_experiments.human.import")
        mock_module.HumanImporter.assert_called_once()
        mock_importer.import_tags.assert_called_once_with(
            run_id="20260606_120000_phase1_factorial",
            tags_path="/tmp/human_review.csv",
        )

        captured = capsys.readouterr()
        assert "Updated 5 rows" in captured.out
