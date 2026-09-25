"""The six tasks of dp-manip docs/final-plan.md §1 and their generation defaults.

control_mode is the plan's; the two pd_ee_delta_pose tasks keep only 60-80% of their
demos through the control-mode conversion (dp-manip docs/0925-smoke.md §五), so they
generate a surplus. ``--control-mode`` of generate_task.py overrides the plan's choice
(e.g. pd_joint_pos, which needs no conversion).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    env_id: str
    control_mode: str
    raw_train: int  # expert demos generated from seed 0; the export keeps the first usable ones
    raw_val: int    # expert demos generated from seed 4000


TASKS = {
    "pickcube": Task("PickCube-v1", "pd_ee_delta_pos", 440, 55),
    "stackcube": Task("StackCube-v1", "pd_ee_delta_pos", 440, 55),
    "pushcube": Task("PushCube-v1", "pd_ee_delta_pos", 440, 55),
    "pullcube": Task("PullCube-v1", "pd_ee_delta_pos", 440, 55),
    "peginsertionside": Task("PegInsertionSide-v1", "pd_ee_delta_pose", 800, 100),
    "plugcharger": Task("PlugCharger-v1", "pd_ee_delta_pose", 800, 100),
    "liftpegupright": Task("LiftPegUpright-v1", "pd_ee_delta_pose", 800, 100),  # fallback for PlugCharger
}

# docs/final-plan.md §2.2: training pool (seeds 0-3999, first 400 usable) and validation
# demos (seeds 4000-4999, first 50 usable). Test and validation rollout seeds never become demos.
TRAIN_START, VAL_START = 0, 4000
EXPORT_TRAIN, EXPORT_VAL = 400, 50
