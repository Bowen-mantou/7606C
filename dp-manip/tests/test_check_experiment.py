"""Phase 13 regression tests: Gate B is checked, not just documented.

The checker compares every resolved cell of an experiment matrix against one
reference and reports any difference outside the declared variable, replicate
seed, runtime paths and architecture definitions. The declared data-size and
backbone matrices must pass their own check.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from dp_manip.config import default_run_name, load, load_experiment

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "configs" / "tasks"
EXPERIMENTS = ROOT / "configs" / "experiments"


def load_checker():
    path = ROOT / "scripts" / "check_experiment.py"
    spec = importlib.util.spec_from_file_location("check_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ConfigDifferenceTest(unittest.TestCase):
    def test_nested_differences_are_reported_by_dotted_path(self) -> None:
        checker = load_checker()
        left = {"train": {"seed": 1, "batch_size": 64}, "policy": {"backbone": "unet"}}
        right = {"train": {"seed": 2, "batch_size": 128}, "policy": {"backbone": "mlp"}}
        self.assertEqual(
            checker.config_differences(left, right),
            {
                "policy.backbone": ("unet", "mlp"),
                "train.batch_size": (64, 128),
                "train.seed": (1, 2),
            },
        )

    def test_without_keys_prunes_only_the_allowed_paths(self) -> None:
        checker = load_checker()
        raw = {"train": {"seed": 1, "batch_size": 64}, "data": {"root": "/a", "num_demos": 100}}
        self.assertEqual(
            checker.without_keys(raw, {"train.seed", "data.root"}),
            {"train": {"batch_size": 64}, "data": {"num_demos": 100}},
        )


class MatrixCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.checker = load_checker()

    def cell(self, label: str, *overrides: str):
        return self.checker.Cell(label, load(TASKS / "pickcube.toml", list(overrides)))

    def test_replicate_seed_is_not_drift(self) -> None:
        cells = [self.cell("s1", "train.seed=1"), self.cell("s2", "train.seed=2")]
        self.assertEqual(self.checker.check_cells(cells, {"train.seed"}), [])

    def test_unexpected_key_is_reported_with_both_values(self) -> None:
        cells = [
            self.cell("unet s1", "train.seed=1"),
            self.cell("unet s2", "train.seed=2", "train.batch_size=128"),
        ]
        drifts = self.checker.check_cells(cells, {"train.seed"})
        self.assertEqual([drift.key for drift in drifts], ["train.batch_size"])
        self.assertEqual(drifts[0].reference_label, "unet s1")
        self.assertEqual(drifts[0].reference_value, 64)
        self.assertEqual(drifts[0].cell_label, "unet s2")
        self.assertEqual(drifts[0].cell_value, 128)

    def test_control_hash_ignores_allowed_keys_only(self) -> None:
        allowed = {"train.seed", "policy.backbone"}
        left = self.cell("unet", "train.seed=1")
        arm = self.cell("transformer", "train.seed=2", "policy.backbone=transformer")
        self.assertEqual(
            self.checker.control_hash(left.config, allowed),
            self.checker.control_hash(arm.config, allowed),
        )
        drifted = self.cell("batch", "train.seed=2", "train.batch_size=128")
        self.assertNotEqual(
            self.checker.control_hash(left.config, allowed),
            self.checker.control_hash(drifted.config, allowed),
        )

    def test_declared_matrices_have_no_drift(self) -> None:
        for name in ("data_size", "data_size_optional400", "backbone"):
            experiment = EXPERIMENTS / f"{name}.toml"
            spec = load_experiment(experiment)
            allowed = self.checker.allowed_keys(spec)
            for task_path in sorted(TASKS.glob("*.toml")):
                with self.subTest(experiment=name, task=task_path.stem):
                    cells = self.checker.matrix_cells(task_path, experiment, spec)
                    self.assertGreaterEqual(len(cells), 2)
                    self.assertEqual(self.checker.check_cells(cells, allowed), [])
                    hashes = {self.checker.control_hash(cell.config, allowed) for cell in cells}
                    self.assertEqual(len(hashes), 1)

    def test_drift_report_matches_the_plan_format(self) -> None:
        cells = [
            self.cell("unet s1", "train.seed=1"),
            self.cell("transformer s1", "train.seed=1", "train.batch_size=128"),
        ]
        report = self.checker.format_drifts("pickcube", self.checker.check_cells(cells, set()))
        self.assertIn("ERROR: Unexpected experiment config drift in pickcube", report)
        self.assertIn("train.batch_size:", report)
        self.assertIn("reference unet s1 = 64", report)
        self.assertIn("transformer s1 = 128", report)

    def test_cli_checks_a_single_seed_and_rejects_empty_selections(self) -> None:
        checker = load_checker()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = checker.main(["--experiment", "backbone", "--task", "pickcube", "--seed", "1"])
        self.assertEqual(status, 0)
        self.assertIn("Gate B ok", output.getvalue())

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = checker.main(["--experiment", "data_size", "--task", "pickcube", "--seed", "5"])
        self.assertEqual(status, 0)
        self.assertIn("2 cells ok", output.getvalue())

        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            status = checker.main(["--experiment", "data_size", "--task", "pickcube", "--seed", "9"])
        self.assertEqual(status, 1)

    def test_structural_keys_are_only_allowed_in_the_backbone_experiment(self) -> None:
        structural = {"policy.unet_dims", "policy.mlp_hidden_dim", "policy.transformer_layers"}
        backbone = self.checker.allowed_keys(load_experiment(EXPERIMENTS / "backbone.toml"))
        data_size = self.checker.allowed_keys(load_experiment(EXPERIMENTS / "data_size.toml"))
        self.assertTrue(structural <= backbone)
        self.assertFalse(structural & data_size)

        cells = [
            self.cell("100 s1", "train.seed=1"),
            self.cell("200 s1", "train.seed=1", "data.num_demos=200", "policy.unet_dims=[128, 256]"),
        ]
        drifts = self.checker.check_cells(cells, data_size)
        self.assertEqual([drift.key for drift in drifts], ["policy.unet_dims"])

    def test_run_root_checks_the_recorded_configs(self) -> None:
        checker = load_checker()
        spec = load_experiment(EXPERIMENTS / "backbone.toml")
        declared = checker.matrix_cells(TASKS / "pickcube.toml", EXPERIMENTS / "backbone.toml", spec, seed=1)

        def write_runs(root: Path, edits: dict[str, dict[str, dict]] = {}, skip: tuple[str, ...] = ()):
            for cell in declared:
                if cell.label in skip:
                    continue
                raw = cell.config.to_dict()
                raw["data"]["root"] = "/scratch/somewhere"  # runtime path, never drift
                for section, values in edits.get(cell.label, {}).items():
                    raw[section].update(values)
                run_dir = root / default_run_name(cell.config)
                run_dir.mkdir(parents=True)
                (run_dir / "run.json").write_text(json.dumps({"config": raw}), encoding="utf-8")

        def run(root: Path) -> tuple[int, str]:
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                status = checker.main(
                    ["--experiment", "backbone", "--task", "pickcube", "--seed", "1", "--run-root", str(root)]
                )
            return status, output.getvalue()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_runs(root)
            status, output = run(root)
            self.assertEqual(status, 0, output)
            self.assertIn("3 cells ok", output)

        # An OOM workaround on one arm is exactly the drift the declaration
        # check cannot see: it only exists in that run's run.json.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_runs(root, {"transformer s1": {"train": {"batch_size": 32}}})
            status, output = run(root)
            self.assertEqual(status, 1)
            self.assertIn("run.json differs from the declared config", output)
            self.assertIn("Unexpected experiment config drift", output)
            self.assertIn("transformer s1 run.json = 32", output)

        # A baseline changed after every run finished leaves the arms mutually
        # consistent but stale against the current declaration.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = {"train": {"lr": 3e-4}}
            write_runs(root, {label: stale for label in ("unet s1", "transformer s1", "mlp s1")})
            status, output = run(root)
            self.assertEqual(status, 1)
            self.assertIn("train.lr:", output)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_runs(root, skip=("mlp s1",))
            status, output = run(root)
            self.assertEqual(status, 0, output)
            self.assertIn("2/3 cells have run.json; not run: mlp s1", output)

        with tempfile.TemporaryDirectory() as directory:
            status, _ = run(Path(directory))
            self.assertEqual(status, 1)

    def test_cli_reports_unknown_task_and_experiment(self) -> None:
        checker = load_checker()
        with self.assertRaisesRegex(FileNotFoundError, "unknown task"):
            checker.main(["--experiment", "backbone", "--task", "banana"])
        with self.assertRaisesRegex(FileNotFoundError, "unknown experiment"):
            checker.main(["--experiment", "banana", "--task", "pickcube"])


if __name__ == "__main__":
    unittest.main()
