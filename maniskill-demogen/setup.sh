#!/usr/bin/env bash
# Build the generation environment in ./.venv (no sudo, no conda).
#   1. uv sync --frozen: exactly the versions in uv.lock (Python 3.11 is fetched by uv).
#   2. Apply patches/mani_skill_mplib_0_2_1.patch to the installed mani_skill: its motion
#      planner calls the mplib 0.1.1 API; 0.2.1 takes Pose objects and has no
#      use_point_cloud argument. Skipped when already applied, so the script can be rerun.
#   3. Print the versions that matter.
# Then run ./check_env.sh.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it into your home directory:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
# A UV_PROJECT_ENVIRONMENT set elsewhere would install into (and patch) another venv.
unset UV_PROJECT_ENVIRONMENT

uv sync --frozen

site=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
patch_file=patches/mani_skill_mplib_0_2_1.patch
if command -v patch >/dev/null 2>&1; then
  if patch -p1 -R --dry-run -s -f -d "$site" < "$patch_file" >/dev/null 2>&1; then
    echo "mplib patch: already applied"
  else
    patch -p1 -N -d "$site" < "$patch_file"
    echo "mplib patch: applied"
  fi
else
  # No patch(1): git can apply outside a repository.
  if git apply -p1 -R --check --unsafe-paths --directory="$site" "$patch_file" 2>/dev/null; then
    echo "mplib patch: already applied"
  else
    git apply -p1 --unsafe-paths --directory="$site" "$patch_file"
    echo "mplib patch: applied (git apply)"
  fi
fi

.venv/bin/python - <<'PY'
import inspect

import mani_skill
import mplib
import numpy
import sapien
import torch
from mani_skill.examples.motionplanning.base_motionplanner import motionplanner

source = inspect.getsource(motionplanner)
assert "mplib.pymp.Pose(p=self.base_pose.p" in source, "mplib patch missing from mani_skill"
print(f"python ok: mani_skill {mani_skill.__version__}, sapien {sapien.__version__}, mplib {mplib.__version__}, "
      f"numpy {numpy.__version__}, torch {torch.__version__}")
PY
echo "Environment ready. Next: ./check_env.sh"
