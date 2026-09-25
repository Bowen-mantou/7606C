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

## 把仓库放到服务器上

仓库只有代码（不含 `.venv` 和数据，约 100 KB），环境由 `setup.sh` 在服务器上联网安装，数据在服务器上生成，不需要上传大文件：

```bash
# 本机
git bundle create maniskill-demogen.bundle --all
scp maniskill-demogen.bundle <server>:
# 服务器
git clone maniskill-demogen.bundle maniskill-demogen && cd maniskill-demogen
```

服务器不能联网时，`setup.sh` 装不了依赖，需要另想办法（例如在能联网的同类机器上装好后整体拷过去）。

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

**作业数量受限时**（HKU 集群就是这样），用一个作业跑全部任务，同时跑的个数等于申请的 CPU 数：

```bash
sbatch slurm/run_all.sbatch                                  # 一个作业，4 CPU + 1 GPU
srun --jobid=<已有分配的 id> --overlap ./run_all.sh           # 或在已有的交互式分配里（建议放在 tmux 里）
```

每个任务的完整输出在 `data/logs/<task>.out`，步骤汇总在 `data/logs/<task>.log`。被中断后重新执行同一条命令，会从断点继续。

**集群实测（9.25，gpu-4080-402，RTX 4080 SUPER，4 CPU）**：`check_env.sh` 通过，渲染每步只多 1.0 ms（单相机）/ 1.7 ms（双相机）。按 ubuntu 满负载时的速度估算，4 个任务并行，全部 6 个任务约 1.5–2.5 小时；磁盘约 15 GB（导出约 6.5 GB，`work/` 里的中间文件和它差不多大），放不下家目录时用 `OUT=` 或 `--out` 指到 scratch。

常用参数（`generate_task.py --help` 有完整说明）：

| 参数 | 作用 |
| --- | --- |
| `--stages expert,rgb,state,first,export,stats` | 只跑其中几个阶段 |
| `--control-mode MODE` | 覆盖 `tasks.py` 里的控制模式 |
| `--n-train / --n-val` | 生成的原始专家条数（默认见 `tasks.py`，PlugCharger 多生成） |
| `--export-train 400 --export-val 50` | 导出前 N 条可用示范（final-plan §2.2） |
| `--out` | 输出根目录，默认 `./data`；六个任务可以共用同一个 |

## 控制模式与条数（`tasks.py`）

| 任务 | 控制模式 | 动作维 | 原始条数（训练 + 验证） | 重放成功率（wsl 实测） |
| --- | --- | --- | --- | --- |
| PickCube、StackCube、PushCube、PullCube | `pd_ee_delta_pos` | 4 | 440 + 55 | ≈100% |
| PegInsertionSide | `pd_joint_pos` | 8 | 440 + 55 | 97% |
| PlugCharger | `pd_joint_pos` | 8 | 600 + 80 | 80% |

7 维任务 9.25 定为 `pd_joint_pos`（转成 `pd_ee_delta_pose` 只剩 60–73%）。**它的动作是关节绝对目标角（弧度），约 30% 的数值在 [-1, 1] 之外**：训练代码必须对动作做归一化，执行前也不能把动作裁剪到 [-1, 1]（VariDP `train/eval.py` 第 156 行正是这样裁剪的；7606-train-template 不做动作归一化，两者都要改）。

导出取每个 split 的前 400 / 50 条可用示范。万一不够，导出会报「only N usable demos」：删掉该任务 `work/` 下对应 split 的文件（和 `.done`），用更大的 `--n-train` / `--n-val` 重跑；生成是确定性的，前面的示范会原样再生成。

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
| rgb | `mani_skill.trajectory.replay_trajectory -c <mode> --allow-failure -o rgb --shader minimal` | 相机 shader 与普通 `gym.make` 的评估环境一致；每条示范单独保存，失败的在导出时丢弃 |
| state | 同上，`-o state` | CPU 物理是确定性的，两次转换逐步一致，导出时校验 |
| first | `scripts/first_frame_obs.py` | 修正第 0 帧 |
| export | `scripts/export_demos.py`、`scripts/preview_rgb.py` | 丢弃重放未成功的示范，取前 N 条 |
| stats | `scripts/replay_stats.py` | 专家成功率、转换成功率、丢弃的种子、长度、相机（final-plan §2.3） |

每个步骤成功后写 `<输出>.done`，有标记就跳过；有输出却没有标记（被打断或另一个作业在写）时报错，不覆盖，需要人工检查后删除。

## ManiSkill 3.0.1 的已知问题（本仓库已处理）

1. `replay_trajectory --use-env-states` 录下的是「从设定状态走一步」的预测，不是状态本身 → state 与 rgb 各自独立转换。
2. 某条示范重放失败时，录制缓冲区不清空，它的步骤会被拼到下一条保存的示范前面（重置处的动作是随机的）→ 重放一律加 `--allow-failure`，每条单独保存，导出时丢弃失败的；另外单步关节跳变超过 0.2 rad 的示范也丢弃（第二道保险）。
3. 第 0 帧的接触类观测是上一条示范的残留（PickCube `is_grasped`）→ `first_frame_obs.py` 重算。
4. 转换文件的 `env_states[0]` 错了一位（记的是 t=1）→ 同上一起替换。

官方下载的示范（队友用过的）同样带有 3、4 两个问题。

## 环境

`pyproject.toml` / `uv.lock` 复刻 dp-manip ubuntu 专家环境的版本：Python 3.11、mani-skill 3.0.1、sapien 3.0.3、mplib 0.2.1（覆盖 ManiSkill 要求的 0.1.1）、numpy 1.26.4、gymnasium 1.3.0、torch 2.14.0 CPU 版；OpenCV 用 `opencv-python-headless`（mani-skill 默认依赖的 `opencv-python` 需要系统的 `libGL.so.1`，HKU 集群登录节点没有）。`setup.sh` 再对 `.venv` 里的 mani_skill 打 `patches/mani_skill_mplib_0_2_1.patch`（mplib 0.2.1 的 API 变化）；uv 用复制模式安装，补丁不会改到 uv 缓存。

## 已验证

- ubuntu（GTX 1080 Ti，NVIDIA Vulkan）：dp-manip 的同一套脚本生成 6 个任务，VariDP 原版代码训练和评估跑通。
- wsl（无 NVIDIA Vulkan，lavapipe）：本仓库从零 `setup.sh` + `check_env.sh` 通过。与 ubuntu 生成的数据相比，同一种子的示范长度、转换成败相同，数值只有浮点级差异（PickCube ≤ 1e-4，PlugCharger ≤ 0.03），图像平均差 0.3 个灰度级。**最终数据应全部在同一种机器上生成。**
- 渲染耗时（每步）：lavapipe 1 路相机 14 ms、2 路 63 ms；NVIDIA 1080 Ti 2–4 ms。wsl 上 lavapipe 的 rgb 重放：PegInsertionSide 约 10 s/条、PlugCharger 约 12 s/条。估算单任务作业：4 维单相机任务不到 1 小时，StackCube 约 1–2 小时，PegInsertionSide 约 1.5 小时，PlugCharger 约 2.5 小时；有 NVIDIA Vulkan 时快得多。

`scripts/check_rgb_obs.py`（可选）在训练机上比较导出文件的首帧和评估环境 reset 出来的观测。
