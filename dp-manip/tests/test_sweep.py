from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

from dp_manip.config import default_run_name, load_experiment


ROOT = Path(__file__).resolve().parents[1]


def load_sweep_module():
    script = ROOT / "scripts" / "sweep.py"
    module_spec = importlib.util.spec_from_file_location("sweep", script)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


class SweepTest(unittest.TestCase):
    def test_sweep_cells_come_from_experiment_spec(self) -> None:
        module = load_sweep_module()
        experiment = ROOT / "configs" / "experiments" / "data_size.toml"
        declared = load_experiment(experiment)
        task_runs = [run for run in module.runs(experiment) if run.task == "pickcube"]
        actual = [(run.value, run.seed) for run in task_runs]
        expected = [
            (value, seed)
            for value in declared.values
            for seed in declared.seeds_for(value)
        ]
        self.assertEqual(actual, expected)
        self.assertTrue(all(run.config.parent.name == "tasks" for run in task_runs))

    def test_run_names_match_the_trainer_default(self) -> None:
        module = load_sweep_module()
        experiment = ROOT / "configs" / "experiments" / "data_size.toml"
        for run in module.runs(experiment):
            with self.subTest(run=run):
                resolved = run.resolve()
                self.assertEqual(run.name, default_run_name(resolved))
                self.assertEqual(
                    run.name,
                    f"{run.task}_rgb_{resolved.policy.backbone}_n{run.value}_s{run.seed}",
                )

    def test_backbone_grid_runs_through_the_sweep(self) -> None:
        module = load_sweep_module()
        experiment = ROOT / "configs" / "experiments" / "backbone.toml"
        declared = load_experiment(experiment)
        grid = module.runs(experiment)
        self.assertEqual(
            len(grid),
            len(module.TASKS) * sum(len(declared.seeds_for(value)) for value in declared.values),
        )
        names = [run.name for run in grid]
        self.assertEqual(len(set(names)), len(names), "two backbone cells share a run directory")

        run = next(run for run in grid if run.value == "transformer")
        self.assertEqual(run.resolve().policy.backbone, "transformer")
        args = argparse.Namespace(action="train", output_root=ROOT / "runs", data_root=None)
        command = module.build_command(args, run)
        self.assertEqual(command[command.index("--experiment-value") + 1], "transformer")
        self.assertIn("_rgb_transformer_", run.name)

        # The unet arm is the data-size N=100 cell: same config, same directory,
        # so whichever array runs second finds final.pt and skips it.
        data_size = {run.name for run in module.runs(ROOT / "configs" / "experiments" / "data_size.toml")}
        unet_names = {run.name for run in grid if run.value == "unet"}
        self.assertTrue(unet_names <= data_size)

    def test_train_split_eval_uses_experiment_diagnostic_budget(self) -> None:
        module = load_sweep_module()
        experiment = ROOT / "configs" / "experiments" / "data_size.toml"
        declared = load_experiment(experiment)
        run = module.runs(experiment)[-1]

        def eval_command(split: str, episodes: int | None) -> list[str]:
            args = argparse.Namespace(
                action="eval",
                experiment=experiment,
                output_root=ROOT / "runs",
                checkpoint="final.pt",
                split=split,
                episodes=episodes,
                num_envs=None,
                render_backend=None,
            )
            return module.build_command(args, run)

        def episodes_flag(command: list[str]) -> str | None:
            return command[command.index("--episodes") + 1] if "--episodes" in command else None

        self.assertGreater(run.value, declared.train_eval_episodes)
        self.assertEqual(episodes_flag(eval_command("train", None)), str(declared.train_eval_episodes))
        self.assertEqual(episodes_flag(eval_command("train", 3)), "3")
        self.assertIsNone(episodes_flag(eval_command("test", None)))


if __name__ == "__main__":
    unittest.main()
