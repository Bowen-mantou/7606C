#!/usr/bin/env bash
# Check a new machine end to end before submitting real jobs (a few minutes, CPU + Vulkan):
#   1. versions and the mplib patch (setup.sh must have run);
#   2. which Vulkan driver rendering will use (nvidia-smi and /usr/share/vulkan/icd.d);
#   3. a tiny run of generate_task.py for PickCube (1 camera, 4-dim actions) and PlugCharger
#      (2 cameras, 7-dim actions, conversion failures) into ./data-check, all stages;
#   4. the exported files: demo counts, dims, and images that are not blank;
#   5. rendering speed (scripts/bench_render.py), to size the jobs.
# Run it inside a job/allocation on the kind of node the real jobs will use.
set -uo pipefail
cd "$(dirname "$0")"
unset UV_PROJECT_ENVIRONMENT
OUT=${OUT:-data-check}
PY=.venv/bin/python
[ -x "$PY" ] || { echo "no .venv: run ./setup.sh first" >&2; exit 1; }

echo "=== host: $(hostname), $(nproc) CPUs, $(free -g 2>/dev/null | awk '/Mem:/{print $2" GB RAM"}')"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || echo "nvidia-smi: not available"
echo "Vulkan ICDs: $(ls /usr/share/vulkan/icd.d/ 2>/dev/null | tr '\n' ' ')"
[ -n "${VK_ICD_FILENAMES:-}" ] && echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES (given)"

echo "=== python packages and patch"
"$PY" - <<'PY' || exit 1
import inspect
import mani_skill, mplib, sapien, torch
from mani_skill.examples.motionplanning.base_motionplanner import motionplanner
assert "mplib.pymp.Pose(p=self.base_pose.p" in inspect.getsource(motionplanner), "mplib patch missing: run ./setup.sh"
print(f"mani_skill {mani_skill.__version__}, sapien {sapien.__version__}, mplib {mplib.__version__}, torch {torch.__version__}")
PY

echo "=== tiny generation into $OUT"
status=0
"$PY" generate_task.py pickcube --out "$OUT" --n-train 3 --n-val 2 --export-train 2 --export-val 1 || status=1
"$PY" generate_task.py plugcharger --out "$OUT" --n-train 4 --n-val 2 --export-train 1 --export-val 1 || status=1

echo "=== exported files"
"$PY" - "$OUT" <<'PY' || status=1
import glob, json, sys
import h5py, numpy as np
files = sorted(glob.glob(f"{sys.argv[1]}/dataset/*/*/motionplanning/*.h5"))
assert len(files) == 4, f"expected 4 exported files, found {files}"
for path in files:
    info = json.load(open(path[:-3] + ".export_info.json"))
    with h5py.File(path) as f:
        images = f["traj_0/obs_rgb/rgb"][()]
        assert images.std() > 1, f"{path}: images look blank"
    print(f"{path.split('/dataset/')[1]}: {info['num_demos']} demos, obs {info['obs_dim']}, action {info['action_dim']}, "
          f"images {info['obs_rgb_image_shape']} from {info['cameras']}, pixel std {images.std():.1f}, "
          f"frame-0 fixes {info['first_frame_fix']['episodes_changed']}, rejected {len(info['rejected'])}")
PY
echo "=== rendering speed"
"$PY" - <<'PY'
import os, sys
sys.path.insert(0, ".")
from generate_task import choose_vulkan_icd
print(choose_vulkan_icd())
os.execv(sys.executable, [sys.executable, "scripts/bench_render.py"])
PY
if [ "$status" = 0 ]; then
  echo "=== check passed: this machine can generate. Look at $OUT/dataset/train/*/motionplanning/sample.png"
else
  echo "=== check FAILED (see above)"
fi
exit "$status"
