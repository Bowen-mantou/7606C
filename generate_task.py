#!/usr/bin/env python3
"""Generate the demos of ONE task: expert -> rgb/state conversion -> frame-0 fix -> export.

    uv run python generate_task.py pickcube                    # everything, plan defaults
    uv run python generate_task.py plugcharger --stages expert # one stage
    uv run python generate_task.py peginsertionside --control-mode pd_joint_pos

Stages (in order; --stages picks a subset):
  expert  run_cpu.py: motion planning, pd_joint_pos, obs none, successes only. Pool demos
          from seed 0, validation demos from seed 4000 (tasks.py has the counts).
  rgb     replay_trajectory --use-first-env-state -c <mode> -o rgb --shader minimal
          (the camera shader of a plain gym.make env, i.e. of the teammates' evaluation)
  state   the same conversion recording state. CPU physics is deterministic, so both agree
          step by step; export_demos.py checks it. (--use-env-states is not usable: in
          mani-skill 3.0.1 it records one-step predictions, not the states.)
  first   first_frame_obs.py: correct frame 0 of both conversions (3.0.1 records stale
          contacts, e.g. PickCube is_grasped, and a shifted env_states[0]).
  export  export_demos.py: first 400 usable pool demos and first 50 validation demos in the
          official ManiSkill layout, plus sample.png. Demos whose conversion failed or that
          contain an env reset (3.0.1 can glue failed attempts into one episode) are skipped.
  stats   replay_stats.py: expert/conversion success, dropped seeds, lengths, cameras.

Layout under --out (default ./data); different tasks use different <Env> directories, so
all tasks can run as parallel jobs on one --out:

  work/<Env>/motionplanning/{train,val}.h5 + .json                     raw expert demos
  work/<Env>/motionplanning/{split}.{rgb,state}.<mode>.physx_cpu.h5     conversions (+ .first_obs.h5)
  dataset/{train,val}/<Env>/motionplanning/trajectory.state.<mode>.physx_cpu.h5 (+ .json, .export_info.json)
  dataset/train/<Env>/motionplanning/sample.png
  logs/<task>.log                                                     step summary of every run

Every step writes <output>.done when it succeeds and is skipped while that marker exists,
so a job can be resubmitted after a timeout. Output without a marker belongs to an
interrupted or concurrent run: the step fails instead of overwriting it. Inspect and
remove it by hand.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from tasks import EXPORT_TRAIN, EXPORT_VAL, TASKS, TRAIN_START, VAL_START

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
STAGES = ("expert", "rgb", "state", "first", "export", "stats")
NVIDIA_ICD = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
LAVAPIPE_ICD = Path("/usr/share/vulkan/icd.d/lvp_icd.json")


class Run:
    def __init__(self, log: Path, timeout: int):
        self.log, self.timeout, self.results = log, timeout, []
        log.parent.mkdir(parents=True, exist_ok=True)

    def note(self, status: str, label: str, detail: str = "") -> None:
        line = f"{status:5s} {label}" + (f": {detail}" if detail else "")
        self.results.append(line)
        print(line, flush=True)

    def call(self, label: str, command: list) -> bool:
        print(f"=== [{datetime.now():%H:%M:%S}] {label}: {' '.join(map(str, command))}", flush=True)
        start = time.time()
        try:
            code = subprocess.run([str(part) for part in command], cwd=ROOT, timeout=self.timeout).returncode
        except subprocess.TimeoutExpired:
            self.note("FAIL", label, f"timed out after {self.timeout}s")
            return False
        seconds = time.time() - start
        self.note("OK" if code == 0 else "FAIL", label, f"{seconds:.0f}s" if code == 0 else f"exit {code}, {seconds:.0f}s")
        return code == 0

    def once(self, label: str, output: Path, command: list, complete_episodes: int | None = None) -> bool:
        """Run command unless output is marked done; never touch unmarked output."""
        marker = output.with_name(output.name + ".done")
        if marker.exists():
            self.note("SKIP", label, "done")
            return True
        meta = output.with_suffix(".json")
        if output.exists() or meta.exists():
            found = episode_count(meta)
            if complete_episodes is not None and found == complete_episodes:
                marker.write_text(f"{datetime.now().isoformat()} complete: {found} episodes (marked afterwards)\n")
                self.note("SKIP", label, f"all {found} episodes present, marked done")
                return True
            self.note("FAIL", label, f"{output} exists but is not marked done ({found} episodes); "
                                     "an interrupted or concurrent run? inspect and remove it by hand")
            return False
        if not self.call(label, command):
            return False
        marker.write_text(f"{datetime.now().isoformat()} {' '.join(map(str, command))}\n")
        return True

    def finish(self, header: str) -> int:
        failed = any(line.startswith("FAIL") for line in self.results)
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(f"{header}\n" + "\n".join(self.results) + f"\nresult: {'FAIL' if failed else 'OK'}\n\n")
        print("=== summary\n" + "\n".join(self.results), flush=True)
        return 1 if failed else 0


def episode_count(meta: Path) -> int | None:
    try:
        return len(json.loads(meta.read_text(encoding="utf-8"))["episodes"])
    except (OSError, ValueError, KeyError):
        return None


def choose_vulkan_icd() -> str:
    """Vulkan driver for SAPIEN: an explicit VK_ICD_FILENAMES, else NVIDIA, else lavapipe (CPU)."""
    if os.environ.get("VK_ICD_FILENAMES"):
        return f"VK_ICD_FILENAMES={os.environ['VK_ICD_FILENAMES']} (given)"
    for icd, name in ((NVIDIA_ICD, "NVIDIA"), (LAVAPIPE_ICD, "lavapipe, software rendering on the CPU")):
        if icd.exists():
            os.environ["VK_ICD_FILENAMES"] = str(icd)
            return f"VK_ICD_FILENAMES={icd} ({name})"
    return "no Vulkan ICD found under /usr/share/vulkan/icd.d; rendering will fail unless SAPIEN finds a driver itself"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {','.join(STAGES)}")
    parser.add_argument("--control-mode", help="override the plan's control mode (tasks.py)")
    parser.add_argument("--n-train", type=int, help="expert pool demos to generate (default: tasks.py)")
    parser.add_argument("--n-val", type=int, help="expert validation demos to generate (default: tasks.py)")
    parser.add_argument("--export-train", type=int, default=EXPORT_TRAIN, help="usable pool demos to export")
    parser.add_argument("--export-val", type=int, default=EXPORT_VAL, help="usable validation demos to export")
    parser.add_argument("--train-start", type=int, default=TRAIN_START)
    parser.add_argument("--val-start", type=int, default=VAL_START)
    parser.add_argument("--timeout", type=int, default=8 * 3600, help="seconds per step")
    args = parser.parse_args()

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        parser.error(f"unknown stages {unknown}; choose from {STAGES}")
    task = TASKS[args.task]
    mode = args.control_mode or task.control_mode
    counts = {"train": args.n_train or task.raw_train, "val": args.n_val or task.raw_val}
    exports = {"train": args.export_train, "val": args.export_val}
    starts = {"train": args.train_start, "val": args.val_start}
    out = args.out.resolve()
    work = out / "work" / task.env_id / "motionplanning"
    python = sys.executable

    run = Run(out / "logs" / f"{args.task}.log", args.timeout)
    header = (f"=== generate_task {args.task} {datetime.now().isoformat(timespec='seconds')} env {task.env_id} "
              f"mode {mode} raw {counts} export {exports} stages {stages} out {out}")
    print(header, flush=True)
    print(choose_vulkan_icd(), flush=True)

    for split in ("train", "val"):
        raw = work / f"{split}.h5"
        rgb = work / f"{split}.rgb.{mode}.physx_cpu.h5"
        state = work / f"{split}.state.{mode}.physx_cpu.h5"
        demos = out / "dataset" / split / task.env_id / "motionplanning" / f"trajectory.state.{mode}.physx_cpu.h5"
        if "expert" in stages and not run.once(
                f"{split}: expert ({counts[split]} from seed {starts[split]})", raw,
                [python, SCRIPTS / "run_cpu.py", "-e", task.env_id, "-b", "physx_cpu", "--only-count-success",
                 "-n", counts[split], "--start-seed", starts[split], "--traj-name", split,
                 "--record-dir", out / "work"], complete_episodes=counts[split]):
            continue
        if set(stages) & {"rgb", "state", "first", "export"}:
            marker = raw.with_name(raw.name + ".done")
            if not marker.exists() and episode_count(raw.with_suffix(".json")) != counts[split]:
                run.note("FAIL", f"{split}: conversions", f"{raw} is missing or has not {counts[split]} episodes yet")
                continue
            if not marker.exists():
                marker.write_text(f"{datetime.now().isoformat()} complete: {counts[split]} episodes (marked afterwards)\n")
        replay = [python, "-m", "mani_skill.trajectory.replay_trajectory", "--traj-path", raw, "-b", "physx_cpu",
                  "--use-first-env-state", "-c", mode, "--save-traj", "--num-envs", "1"]
        if "rgb" in stages and not run.once(f"{split}: rgb conversion", rgb, [*replay, "-o", "rgb", "--shader", "minimal"]):
            continue
        if "state" in stages and not run.once(f"{split}: state conversion", state, [*replay, "-o", "state"]):
            continue
        if "first" in stages:
            sidecars = [path.with_name(path.stem + ".first_obs.h5") for path in (rgb, state)]
            if all(path.exists() for path in sidecars):
                run.note("SKIP", f"{split}: frame 0", "sidecars exist")
            elif not run.call(f"{split}: frame 0", [python, SCRIPTS / "first_frame_obs.py", rgb, state]):
                continue
        if "export" in stages:
            if not run.once(f"{split}: export first {exports[split]}", demos,
                            [python, SCRIPTS / "export_demos.py", "--rgb", rgb, "--state", state, "-o", demos,
                             "--split", split, "--num-demos", exports[split]]):
                continue
            if split == "train":
                run.call("train: sample.png", [python, SCRIPTS / "preview_rgb.py", demos, demos.parent / "sample.png"])
    if "stats" in stages:
        run.call("stats", [python, SCRIPTS / "replay_stats.py", out / "work", "--env", task.env_id,
                           "--name", "train", "val"])
    return run.finish(header)


if __name__ == "__main__":
    sys.exit(main())
