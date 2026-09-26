from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path

from dp_manip.config import default_run_name, from_dict, load, load_experiment


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs" / "baseline.toml"
TASKS = ROOT / "configs" / "tasks"


class LayeredConfigTest(unittest.TestCase):
    def test_merge_precedence(self) -> None:
        task_text = """
[task]
name = "precedence"
env_id = "PickCube-v1"
control_mode = "pd_ee_delta_pos"
max_episode_steps = 100

[data]
train_path = "train.h5"
val_path = "val.h5"
"""
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.toml"
            task.write_text(task_text, encoding="utf-8")
            experiment = Path(directory) / "experiment.toml"
            experiment.write_text(
                "[task]\nmax_episode_steps = 150\n\n[data]\nnum_demos = 300\n",
                encoding="utf-8",
            )
            with BASELINE.open("rb") as stream:
                baseline_demos = tomllib.load(stream)["data"]["num_demos"]
            resolved = load(task, baseline=BASELINE)
            self.assertEqual(resolved.data.num_demos, baseline_demos)
            self.assertEqual(resolved.task.max_episode_steps, 100)
            resolved = load(task, baseline=BASELINE, experiment=experiment)
            self.assertEqual(resolved.data.num_demos, 300)
            self.assertEqual(resolved.task.max_episode_steps, 150)
            resolved = load(
                task,
                ["data.num_demos=400"],
                baseline=BASELINE,
                experiment=experiment,
            )
            self.assertEqual(resolved.data.num_demos, 400)

    def test_task_layer_cannot_shadow_baseline(self) -> None:
        header = """
[task]
name = "drift"
env_id = "PickCube-v1"
control_mode = "pd_ee_delta_pos"
max_episode_steps = 100

[data]
train_path = "train.h5"
val_path = "val.h5"
"""
        cases = {
            "baseline section": header + "\n[train]\nbatch_size = 128\n",
            "baseline key in [data]": header + "num_demos = 200\n",
            "baseline key in [task]": header.replace(
                "max_episode_steps = 100", 'max_episode_steps = 100\nsim_backend = "physx_cuda"'
            ),
        }
        for label, text in cases.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                task = Path(directory) / "task.toml"
                task.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "baseline.toml"):
                    load(task, baseline=BASELINE)

    def test_task_configs_only_contain_task_specific_values(self) -> None:
        expected_task_keys = {"name", "env_id", "control_mode", "max_episode_steps"}
        expected_data_keys = {"train_path", "val_path"}
        for task_path in TASKS.glob("*.toml"):
            with self.subTest(task=task_path.name):
                with task_path.open("rb") as stream:
                    raw = tomllib.load(stream)
                self.assertEqual(set(raw), {"task", "data"})
                self.assertEqual(set(raw["task"]), expected_task_keys)
                self.assertEqual(set(raw["data"]), expected_data_keys)

        pick = load(TASKS / "pickcube.toml")
        stack = load(TASKS / "stackcube.toml")
        for section in ("vision", "policy", "train", "ema", "diffusion", "eval"):
            self.assertEqual(getattr(pick, section), getattr(stack, section))

    def test_legacy_entry_redirects_to_task_config(self) -> None:
        path = ROOT / "configs" / "pickcube_rgb.toml"
        with path.open("rb") as stream:
            self.assertEqual(set(tomllib.load(stream)), {"legacy"})
        self.assertEqual(load(path), load(TASKS / "pickcube.toml"))

    def test_num_demos_accepts_any_positive_integer(self) -> None:
        task = TASKS / "pickcube.toml"
        self.assertEqual(load(task, ["data.num_demos=37"]).data.num_demos, 37)
        for invalid in (0, -1, 2.5, True):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "must be positive"):
                value = str(invalid).lower() if isinstance(invalid, bool) else str(invalid)
                load(task, [f"data.num_demos={value}"])

    def test_data_size_experiment_definition_and_resolution(self) -> None:
        path = ROOT / "configs" / "experiments" / "data_size.toml"
        spec = load_experiment(path)
        self.assertEqual(spec.variable, "data.num_demos")
        self.assertEqual(spec.values, (25, 50, 100, 200))
        # The train-seed diagnostic must use seeds shared by every nested subset.
        self.assertEqual(spec.train_eval_episodes, min(spec.values))
        optional = load_experiment(ROOT / "configs" / "experiments" / "data_size_optional400.toml")
        self.assertEqual(optional.train_eval_episodes, spec.train_eval_episodes)
        resolved = load(TASKS / "pickcube.toml", experiment=path, experiment_value=50)
        self.assertEqual(resolved.data.num_demos, 50)

    def test_backbone_experiment_definition_and_resolution(self) -> None:
        path = ROOT / "configs" / "experiments" / "backbone.toml"
        spec = load_experiment(path)
        self.assertEqual(spec.name, "backbone")
        self.assertEqual(spec.variable, "policy.backbone")
        self.assertEqual(spec.values, ("unet", "transformer", "mlp"))
        for value in spec.values:
            # Track B trains every arm with seeds 1-5 (docs/final-plan.md §6).
            self.assertEqual(spec.seeds_for(value), (1, 2, 3, 4, 5))
        # The overfitting diagnostic reuses the first 25 seeds shared by the
        # data-size subsets and by both N_B choices.
        self.assertEqual(spec.train_eval_episodes, 25)

        with BASELINE.open("rb") as stream:
            baseline_demos = tomllib.load(stream)["data"]["num_demos"]
        normalized_arms = []
        for value in spec.values:
            resolved = load(TASKS / "pickcube.toml", experiment=path, experiment_value=value)
            self.assertEqual(resolved.policy.backbone, value)
            # N_B defaults to the canonical baseline size; hard tasks escalate
            # to 200 through the pre-registered rule, not through this file.
            self.assertEqual(resolved.data.num_demos, baseline_demos)
            # Gate B: normalizing the declared variable must make every arm
            # resolve to exactly the same config.
            arm = resolved.to_dict()
            arm["policy"]["backbone"] = spec.values[0]
            normalized_arms.append(arm)
        for arm in normalized_arms[1:]:
            self.assertEqual(arm, normalized_arms[0])

        with self.assertRaisesRegex(ValueError, "is not in experiment"):
            load(TASKS / "pickcube.toml", experiment=path, experiment_value="banana")

    def test_invalid_diagnostics_are_rejected(self) -> None:
        spec = """
[experiment]
name = "bad"
variable = "data.num_demos"
values = [10]

[replicates]
"10" = [1]

[diagnostics]
"""
        for body in ("train_eval_episodes = 0", "train_eval_episodes = true", "other = 1"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "experiment.toml"
                path.write_text(spec + body + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "diagnostics"):
                    load_experiment(path)

    def test_run_name_separates_backbone_arms(self) -> None:
        task = TASKS / "pickcube.toml"
        unet = load(task, ["data.num_demos=50", "train.seed=2"])
        # Existing UNet run directories keep their historical names.
        self.assertEqual(default_run_name(unet), "pickcube_rgb_unet_n50_s2")
        other = load(task, ["data.num_demos=50", "train.seed=2", 'policy.backbone="transformer"'])
        self.assertEqual(default_run_name(other), "pickcube_rgb_transformer_n50_s2")

    def test_version_one_checkpoint_config_is_adapted(self) -> None:
        expected = load(TASKS / "pickcube.toml")
        legacy = expected.to_dict()
        legacy["train"]["ema_decay"] = legacy.pop("ema")["decay"]
        legacy["policy"].update(legacy.pop("diffusion"))
        self.assertEqual(from_dict(legacy), expected)


if __name__ == "__main__":
    unittest.main()
