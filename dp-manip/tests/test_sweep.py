from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from dp_manip.config import load_experiment


ROOT = Path(__file__).resolve().parents[1]


class SweepTest(unittest.TestCase):
    def test_sweep_cells_come_from_experiment_spec(self) -> None:
        script = ROOT / "scripts" / "sweep.py"
        module_spec = importlib.util.spec_from_file_location("sweep", script)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)

        experiment = ROOT / "configs" / "experiments" / "data_size.toml"
        declared = load_experiment(experiment)
        task_runs = [run for run in module.runs(experiment) if run.task == "pickcube"]
        actual = [(run.num_demos, run.seed) for run in task_runs]
        expected = [
            (value, seed)
            for value in declared.values
            for seed in declared.seeds_for(value)
        ]
        self.assertEqual(actual, expected)
        self.assertTrue(all(run.config.parent.name == "tasks" for run in task_runs))


if __name__ == "__main__":
    unittest.main()
