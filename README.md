# maniskill-demogen

在任意 Linux x86_64 机器（集群节点）上为 DASC7606C Track 3 生成 ManiSkill 3 示范数据：运动规划专家 → 转换到计划的控制模式（rgb 与 state 各一次）→ 修正第 0 帧 → 导出成与官方示范相同的格式，**一个任务一个作业**。

代码与流程来自 dp-manip 仓库，9.25 在 ubuntu 上验证过（见 dp-manip `docs/0925-smoke.md`）；文中的 final-plan 指 dp-manip `docs/final-plan.md`。

## 需要什么

| | 说明 |
| --- | --- |
| 系统 | Linux x86_64（mplib 只有这个平台的 wheel），无需 sudo、conda、CUDA |
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh`；安装时要能访问 PyPI 和 download.pytorch.org |
| CPU / 内存 | 每个任务作业 4 核、16 GB 足够（仿真和运动规划都在 CPU 上） |
| **渲染（只有 rgb 需要）** | 一个 Vulkan 设备，二选一：<br>• NVIDIA GPU + 带 Vulkan 的驱动（`/usr/share/vulkan/icd.d/nvidia_icd.json`）：快<br>• 纯 CPU 节点 + Mesa lavapipe（`/usr/share/vulkan/icd.d/lvp_icd.json`）：慢 7–17 倍，但 4 核就够 |

不需要 GPU 算力：torch 是 CPU 版，GPU 只用来渲染。

## 用法

```bash
./setup.sh              # uv sync --frozen（Python 3.11 由 uv 下载）+ mplib 补丁
./check_env.sh          # 在作业节点上跑：生成 PickCube 和 PlugCharger 各几条并检查，几分钟

# 一个任务（全部阶段；中断后重跑会从断点继续）
.venv/bin/python generate_task.py pickcube
# SLURM：一个任务一个作业，按集群改分区、GPU 行
for t in pickcube stackcube pushcube pullcube peginsertionside plugcharger; do
  sbatch --job-name=gen-$t --export=TASK=$t slurm/generate_task.sbatch
done
```

常用参数（`generate_task.py --help` 有完整说明）：

| 参数 | 作用 |
| --- | --- |
| `--stages expert,rgb,state,first,export,stats` | 只跑其中几个阶段 |
| `--control-mode pd_joint_pos` | 覆盖计划的控制模式（7 维任务的转换成功率只有 60–80%） |
| `--n-train / --n-val` | 生成的原始专家条数（默认见 `tasks.py`，7 维任务多生成） |
| `--export-train 400 --export-val 50` | 导出前 N 条可用示范（final-plan §2.2） |
| `--out` | 输出根目录，默认 `./data`；六个任务可以共用同一个 |

## 输出

```
data/work/<Env>/motionplanning/{train,val}.h5                        原始专家（pd_joint_pos，无观测）
data/work/<Env>/motionplanning/{split}.{rgb,state}.<mode>.physx_cpu.h5 两次转换 + .first_obs.h5 旁路文件
data/dataset/{train,val}/<Env>/motionplanning/trajectory.state.<mode>.physx_cpu.h5   ← 交给训练的数据
data/dataset/{train,val}/<Env>/motionplanning/trajectory.state.<mode>.physx_cpu.json      官方字段
data/dataset/{train,val}/<Env>/motionplanning/trajectory.state.<mode>.physx_cpu.export_info.json  维度、种子、丢弃与修正记录、哈希
data/dataset/train/<Env>/motionplanning/sample.png                   前 5 条的首帧与末帧
data/logs/<task>.log                                                 每次运行的步骤汇总
```

HDF5 与队友验证过的官方示范同结构（`traj_N/obs, actions, success, terminated, truncated, env_states`），另加：

- `traj_N/obs_rgb/rgb`：(T+1, 128, 128, 3×相机数) uint8；StackCube、PegInsertionSide、PlugCharger 有腕部相机，为 6 通道；
- `traj_N/obs_rgb/state`：obs_mode=rgb 的 agent + extra（非特权），与评估时 `FlattenRGBDObservationWrapper(env, rgb=True, depth=False)` 的 `obs["state"]` 逐项对应。

**RGB 策略用 `obs_rgb/rgb` + `obs_rgb/state`，state 策略用 `obs`**（`obs` 含物体位姿）。`traj_N` 按种子排序，前 N 条即嵌套子集。

## 每个阶段做什么

| 阶段 | 脚本 | 说明 |
| --- | --- | --- |
| expert | `scripts/run_cpu.py` | ManiSkill 官方运动规划，只保留成功的；训练池从种子 0、验证示范从 4000 开始 |
| rgb | `mani_skill.trajectory.replay_trajectory -c <mode> -o rgb --shader minimal` | 相机 shader 与普通 `gym.make` 的评估环境一致 |
| state | 同上，`-o state` | CPU 物理是确定性的，两次转换逐步一致，导出时校验 |
| first | `scripts/first_frame_obs.py` | 修正第 0 帧 |
| export | `scripts/export_demos.py`、`scripts/preview_rgb.py` | 丢弃转换失败和中途重置的示范，取前 N 条 |
| stats | `scripts/replay_stats.py` | 专家成功率、转换成功率、丢弃的种子、长度、相机（final-plan §2.3） |

每个步骤成功后写 `<输出>.done`，有标记就跳过；有输出却没有标记（被打断或另一个作业在写）时报错，不覆盖，需要人工检查后删除。

## ManiSkill 3.0.1 的已知问题（本仓库已处理）

1. `replay_trajectory --use-env-states` 录下的是「从设定状态走一步」的预测，不是状态本身 → state 与 rgb 各自独立转换。
2. 控制模式转换可能把失败的尝试和最后成功的一次拼成一条示范（PlugCharger 上见到），重置那一步的动作是随机的 → 单步关节跳变超过 0.2 rad 的示范在导出时丢弃。
3. 第 0 帧的接触类观测是上一条示范的残留（PickCube `is_grasped`）→ `first_frame_obs.py` 重算。
4. 转换文件的 `env_states[0]` 错了一位（记的是 t=1）→ 同上一起替换。

官方下载的示范（队友用过的）同样带有 3、4 两个问题。

## 环境

`pyproject.toml` / `uv.lock` 复刻 dp-manip ubuntu 专家环境的版本：Python 3.11、mani-skill 3.0.1、sapien 3.0.3、mplib 0.2.1（覆盖 ManiSkill 要求的 0.1.1）、numpy 1.26.4、gymnasium 1.3.0、torch 2.14.0 CPU 版。`setup.sh` 再对 `.venv` 里的 mani_skill 打 `patches/mani_skill_mplib_0_2_1.patch`（mplib 0.2.1 的 API 变化）；uv 用复制模式安装，补丁不会改到 uv 缓存。

## 已验证

- ubuntu（GTX 1080 Ti，NVIDIA Vulkan）：dp-manip 的同一套脚本生成 6 个任务，VariDP 原版代码训练和评估跑通。
- wsl（无 NVIDIA Vulkan，lavapipe）：本仓库从零 `setup.sh` + `check_env.sh` 通过。与 ubuntu 生成的数据相比，同一种子的示范长度、转换成败相同，数值只有浮点级差异（PickCube ≤ 1e-4，PlugCharger ≤ 0.03），图像平均差 0.3 个灰度级。**最终数据应全部在同一种机器上生成。**
- 渲染耗时（每步）：lavapipe 1 路相机 14 ms、2 路 63 ms；NVIDIA 1080 Ti 2–4 ms。按 lavapipe 估算，4 维单相机任务每个不到 1 小时，PegInsertionSide、PlugCharger 各约 3–4 小时；有 NVIDIA Vulkan 时快得多。

`scripts/check_rgb_obs.py`（可选）在训练机上比较导出文件的首帧和评估环境 reset 出来的观测。
