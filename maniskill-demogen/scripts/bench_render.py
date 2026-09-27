#!/usr/bin/env python3
"""Per-step cost of rgb observations on this machine: the same env with obs_mode state and
rgb, 200 random steps each, for a 1-camera (PickCube) and a 2-camera (PlugCharger) task.
The difference is what rendering adds to every step of an rgb replay. Reference (9.25):
lavapipe 14 / 63 ms per step, NVIDIA GTX 1080 Ti 2 / 4 ms per step.
"""

import os
import time

import gymnasium as gym
import mani_skill.envs  # noqa: F401  registers the envs
import numpy as np

STEPS = 200


def main() -> None:
    print(f"VK_ICD_FILENAMES={os.environ.get('VK_ICD_FILENAMES', '(not set)')}, {os.cpu_count()} CPUs", flush=True)
    for env_id, mode in (("PickCube-v1", "pd_ee_delta_pos"), ("PlugCharger-v1", "pd_joint_pos")):
        per_step = {}
        for obs_mode in ("state", "rgb"):
            env = gym.make(env_id, obs_mode=obs_mode, control_mode=mode, sim_backend="physx_cpu",
                           render_backend="cpu", sensor_configs=dict(shader_pack="minimal"))
            env.reset(seed=0)
            rng = np.random.default_rng(0)
            low, high = env.action_space.low, env.action_space.high
            for _ in range(10):  # warm-up
                env.step(rng.uniform(low, high).astype(np.float32))
            start = time.perf_counter()
            for _ in range(STEPS):
                env.step(rng.uniform(low, high).astype(np.float32))
            per_step[obs_mode] = (time.perf_counter() - start) / STEPS * 1000
            env.close()
        print(f"{env_id:16s} state {per_step['state']:6.1f} ms/step, rgb {per_step['rgb']:6.1f} ms/step, "
              f"rendering adds {per_step['rgb'] - per_step['state']:6.1f} ms/step", flush=True)


if __name__ == "__main__":
    main()
