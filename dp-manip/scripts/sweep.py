#!/usr/bin/env python3
"""Map stable Slurm array indices to a declared experiment grid."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip.config import ExperimentSpec, load_experiment  # noqa: E402


TASKS = (
    "pickcube",
    "stackcube",
    "pushcube",
    "pullcube",
    "peginsertionside",
    "plugcharger",
)
DEFAULT_EXPERIMENT = ROOT / "configs" / "experiments" / "data_size.toml"


@dataclass(frozen=True)
class Run:
    task: str
    num_demos: int
    seed: int
    experiment: Path

    @property
    def config(self) -> Path:
        return ROOT / "configs" / "tasks" / f"{self.task}.toml"

    @property
    def name(self) -> str:
        return f"{self.task}_rgb_unet_n{self.num_demos}_s{self.seed}"


def runs(experiment_path: Path = DEFAULT_EXPERIMENT) -> list[Run]:
    spec: ExperimentSpec = load_experiment(experiment_path)
    cells = tuple(
        (int(value), seed)
        for value in spec.values
        for seed in spec.seeds_for(value)
    )
    return [Run(task, num_demos, seed, experiment_path) for task in TASKS for num_demos, seed in cells]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("show", "train", "eval"))
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--index", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--render-backend")
    return parser.parse_args()


def selected_run(args: argparse.Namespace) -> Run:
    grid = runs(args.experiment)
    if args.index is None or not 0 <= args.index < len(grid):
        raise ValueError(f"--index must be in [0, {len(grid) - 1}]")
    return grid[args.index]


def main() -> None:
    args = parse_args()
    grid = runs(args.experiment)
    if args.action == "show":
        for index, run in enumerate(grid):
            print(f"{index:03d} {run.name} {run.config.relative_to(ROOT)}")
        print(f"{len(grid)} runs")
        return

    run = selected_run(args)
    if args.action == "train":
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_dp.py"),
            "--config",
            str(run.config),
            "--experiment",
            str(run.experiment),
            "--experiment-value",
            str(run.num_demos),
            "--seed",
            str(run.seed),
            "--exp",
            run.name,
            "--output-root",
            str(args.output_root),
            "--resume",
            "auto",
        ]
        if args.data_root is not None:
            command.extend(("--data-root", str(args.data_root)))
    else:
        checkpoint = args.output_root / run.name / "checkpoints" / args.checkpoint
        command = [
            sys.executable,
            str(ROOT / "scripts" / "eval_dp.py"),
            str(checkpoint),
            "--split",
            args.split,
        ]
        if args.episodes is not None:
            command.extend(("--episodes", str(args.episodes)))
        if args.num_envs is not None:
            command.extend(("--num-envs", str(args.num_envs)))
        if args.render_backend is not None:
            command.extend(("--render-backend", args.render_backend))
    print(f"[{args.index}] {run.name}", flush=True)
    process = subprocess.Popen(command, cwd=ROOT)

    def forward_signal(signum, _frame) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, forward_signal)
    raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
