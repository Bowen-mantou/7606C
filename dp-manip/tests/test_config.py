from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path

from dp_manip.config import from_dict, load, load_experiment


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
num_demos = 200
"""
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.toml"
            task.write_text(task_text, encoding="utf-8")
            experiment = Path(directory) / "experiment.toml"
            experiment.write_text("[data]\nnum_demos = 300\n", encoding="utf-8")
            self.assertEqual(load(task, baseline=BASELINE).data.num_demos, 200)
            self.assertEqual(
                load(task, baseline=BASELINE, experiment=experiment).data.num_demos,
                300,
            )
            resolved = load(
                task,
                ["data.num_demos=400"],
                baseline=BASELINE,
                experiment=experiment,
            )
            self.assertEqual(resolved.data.num_demos, 400)

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
        resolved = load(TASKS / "pickcube.toml", experiment=path, experiment_value=50)
        self.assertEqual(resolved.data.num_demos, 50)

    def test_version_one_checkpoint_config_is_adapted(self) -> None:
        expected = load(TASKS / "pickcube.toml")
        legacy = expected.to_dict()
        legacy["train"]["ema_decay"] = legacy.pop("ema")["decay"]
        legacy["policy"].update(legacy.pop("diffusion"))
        self.assertEqual(from_dict(legacy), expected)


if __name__ == "__main__":
    unittest.main()
