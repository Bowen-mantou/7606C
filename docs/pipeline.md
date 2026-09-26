# 数据集生成流水线

本文说明 `generate_task.py` 生成的每一步：**有几个阶段、各自怎么计数、进入下一阶段的是什么数据、什么会导致失败**，以及按当前设定每一步大概产生多少数据。命令和参数见 [README](../README.md)，代码位置以 `file:line` 标注。

## 总览

一个任务生成两份数据（两个 split）：`train` 与 `val`。每个 split 依次跑 6 个阶段（`generate_task.py:53`）：

```
expert ──▶ rgb ──▶ state ──▶ first ──▶ export ──▶ stats
  │         │        │         │         │          │
raw.h5   rgb.h5   state.h5  .first_obs  dataset/   打印报告
                             .h5         *.h5
```

计数遵循三段式，这是理解全部数量的关键：

1. **前段（expert）成功才计数** —— 失败不写盘，一直做到攒够 N 个成功；
2. **中段（rgb / state）全量落盘、事后判成败** —— 每个 episode 都保存，成败写在 JSON 里；
3. **末段（export）取前 N 条可用，不够就报错** —— 不做自动补生成。

`--stages` 可以只跑其中几个阶段（`generate_task.py:142-145`）。每个阶段用 `Run.once()` 包裹：成功后写 `<输出>.done`，有标记就跳过；**有输出却没标记**（被中断或并发）时报错、不覆盖，需人工删除（`generate_task.py:80-99`）。因此中断后重跑同一条命令会从断点继续。单步默认超时 8 小时（`generate_task.py:139`）。

## 计数流转表

| 阶段 | 输入 | 计数方式 | 输出 | 数量（train / val） |
| --- | --- | --- | --- | --- |
| expert | 无（自己运动规划） | **成功才 +1**，攒够 N 才停 | `work/<Env>/motionplanning/<split>.h5`（+`.json`） | **恰好 440 / 55 条成功** |
| rgb | raw `.h5` | 全量重放、全存（含失败） | `<split>.rgb.<mode>.physx_cpu.h5`（+`.json`） | 440 / 55 个 episode（含失败） |
| state | 同一 raw `.h5` | 同上，只跑物理、无渲染 | `<split>.state.<mode>.physx_cpu.h5`（+`.json`） | 440 / 55 个 episode |
| first | rgb + state + raw | 逐 episode 重算第 0 帧，各存 1 帧 | `<转换文件名>.first_obs.h5`（sidecar） | 440 / 55 组 |
| export | rgb + state + sidecar | 丢弃失败与中途 reset，**取前 N 条可用**；不足 N 断言失败 | `dataset/<split>/<Env>/motionplanning/trajectory.state.<mode>.physx_cpu.h5` | **恰好 400 / 50** |
| stats | raw + 转换产物 | 纯统计、只打印 | 无 | 报告 |

数量常量在 `tasks.py`：`raw_train / raw_val`（`tasks.py:29-35`）与 `EXPORT_TRAIN / EXPORT_VAL`（`tasks.py:40-41`），`generate_task.py:148-149` 读取。

## 逐阶段详解

### 1. expert —— 运动规划生成专家轨迹

- **命令**：`run_cpu.py -e <Env> -b physx_cpu --only-count-success -n <N> --start-seed <s> --traj-name <split> --record-dir <out>/work`（`generate_task.py:166-170`）。
- **计数**：`run_cpu.py:87-125` 的循环里，`solve(seed)` 成功才 `flush_trajectory()` 保存并 `passed += 1`；失败则 `flush_trajectory(save=False)`、`seed += 1`、`continue`（`run_cpu.py:102-107`）。`passed == num_traj` 才 break（`:124-125`）。所以 **`-n 440` 指 440 个成功，而不是 440 次尝试**。
- **seed**：train 从 0、val 从 4000 连续试，直到攒够成功数。失败也消耗 seed。
- **输出**：`work/<Env>/motionplanning/<split>.h5`，恰好 N 条成功示范，按 seed 递增。
- **失败与损耗**：
  - 规划失败（抛异常或 `success=False`）是**预期损耗**，只消耗 seed、不写盘；本次实测 PegInsertionSide 成功率约 0.70，即要试约 `440 / 0.70 ≈ 630` 个 seed。
  - 真正导致阶段失败：始终凑不满 N（死循环）、单步超时（默认 8h）、半截文件无 `.done`。
  - seed 越界（跑出 train 0–3999 / val 4000–4999）不会在这里报错，但会在 export 断言失败（见下）。

### 2. rgb —— 重放并录制 RGB 观测

- **命令**：`replay_trajectory --traj-path <raw> -b physx_cpu --use-first-env-state -c <mode> --allow-failure --save-traj --num-envs 1 -o rgb --shader minimal`（`generate_task.py:179-181`）。
- **计数**：raw 的 N 条**全部重放、全部落盘**，每条是否成功记录在 JSON 的 `success` 字段；失败 episode 保留，留给 export 丢弃。这是 `--allow-failure` 的作用，也规避 ManiSkill 3.0.1 “失败 episode 的步被粘到下一条前面”的 bug。
- **输出**：`work/<Env>/motionplanning/<split>.rgb.<mode>.physx_cpu.h5`。
- **失败与损耗**：
  - 重放未成功（PegInsertionSide 约 3%，`tasks.py:12`；PlugCharger 约 20%）——**预期损耗**，export 段丢弃；
  - 真正导致阶段失败：**没有可用的 Vulkan 渲染设备**（登录节点 `gpu2gate1` 必然失败，必须在 GPU 节点跑；`generate_task.py:116-124` 按 `VK_ICD_FILENAMES` → NVIDIA → lavapipe 选择）、OOM、磁盘满。

### 3. state —— 同一条重放，只录 state

- **命令**：与 rgb 相同，仅 `-o state`（`generate_task.py:183`），纯物理、无渲染。
- **计数**：同样全量落盘。
- **输出**：`work/<Env>/motionplanning/<split>.state.<mode>.physx_cpu.h5`。
- **失败与损耗**：物理异常；此外它必须与 rgb **逐步一致**——CPU 物理确定性使同一 seed 的 `actions` 完全相等、`env_states` 最大差 ≤1e-5、agent 观测一致，export 会逐项校验（`export_demos.py:204-263`）。

### 4. first —— 修正第 0 帧

- **命令**：`first_frame_obs.py <rgb.h5> <state.h5>`（`generate_task.py:185-190`）。
- **作用**：修 ManiSkill 3.0.1 的两个第 0 帧缺陷——第 0 帧的接触类观测是上一条 episode 的残留（如 PickCube `is_grasped`）、`env_states[0]` 错记成源数据 t=1 的状态。
- **计数**：对重放 JSON 里的**每个 episode** 用一个从未 step 过的环境 `reset(seed)`，校验其状态与 raw 的 `env_states[0]` 相差 ≤1e-6（`first_frame_obs.py:92`），写出 1 帧观测与状态。
- **输出**：sidecar `<转换文件名>.first_obs.h5`（`first_frame_obs.py:74-75`），每个 episode 一组。
- **失败与损耗**：`reset(seed)` 复现不出 raw 初始状态（跨机器/随机性，>1e-6）；找不到对应 raw（命名规则或 `--raw`）；半截 sidecar 已存在且需覆盖 → 断言 `exists; pass --overwrite`（`generate_task.py:185-190` 只在两个 sidecar 都在时才跳过，否则直接调用，容易踩）。

### 5. export —— 合并、筛选、定稿

- **命令**：`export_demos.py --rgb <...> --state <...> -o <dataset.h5> --split <split> --num-demos <N>`（`generate_task.py:191-197`）。
- **处理**（`export_demos.py`）：
  1. 按 seed 配对 rgb 与 state，要求两边 seed 集合完全一致（`:337-338`）；
  2. 断言所有 seed 落在 split 区间内（train 0–3999 / val 4000–4999，`:67`、`:341-343`）；
  3. **丢弃 rgb 或 state 重放未成功的 seed**（`:364-370`）；
  4. **丢弃 episode 中途 env reset** 的 seed（关节跳变 >0.2 rad，`:75`、`:161-170`、`:372-377`）；
  5. 按 seed **取前 N 条可用**，并断言可用数 ≥ N（`:378-380`）；
  6. 组装 `traj_i/obs`（state）、`traj_i/obs_rgb/{rgb,state}`、`actions`、`success/terminated/truncated`、`env_states`（第 0 帧用 sidecar 替换），再自校验一遍（`verify()`，`:293-311`）。
- **输出**：`dataset/<split>/<Env>/motionplanning/trajectory.state.<mode>.physx_cpu.h5`（+官方 `.json`、`.export_info.json`）；train 另出 `sample.png`（`preview_rgb.py`）。
- **失败（hard fail，无法自动恢复）**：
  - 可用数 < 400/50 → `only N usable demos, need 400`；
  - seed 越界；rgb/state 的 seed 集不一致；
  - `actions`/成败标志不相等、`env_states` 差 >1e-5、agent 观测不一致；
  - sidecar 缺失或布局不符、第 0 帧对不上；
  - 各条 demo 维度不一致、出现非有限值；
  - 输出已存在且未加 `--overwrite`。
  - `sample.png` 生成失败只记 FAIL，不会删除已写好的数据。

### 6. stats —— 只读体检

- **命令**：`replay_stats.py <out>/work --env <Env> --name train val`（`generate_task.py:198-200`）。
- **输出**：无文件，打印专家成功率、重放成功率、丢弃的 seed、中途 reset、可用率 `usable/440`、长度、相机数（`replay_stats.py:55-73`）。
- **失败**：只影响报告，读损坏文件才报错，不影响数据正确性。

## 按当前设定，每一步可能有多少数据

以本次生成的 PegInsertionSide（`pd_joint_pos`，原始 440 + 55）为例。成功率取自 `tasks.py:12` 的实测与本次运行（`success_rate ≈ 0.70`）。

| 步骤 | 设定 | 预计 / 实测（train） | 预计（val） | 损耗来源 |
| --- | --- | --- | --- | --- |
| expert raw | 成功数写死 | **恰好 440**（实测 OK，698s；试约 630 个 seed） | **恰好 55**（试约 80 个 seed） | 规划失败约 30% 不写盘 |
| rgb 重放 | 全量 | episode **440**，成功 **约 427** | **55**，成功约 53 | 重放失败约 3%（保留但标记失败） |
| state 重放 | 全量 | episode **440**，成功 **约 427** | **55**，约 53 | 同上 |
| first 第 0 帧 | 逐 episode | sidecar **440 组** | **55 组** | 重置不匹配（应接近 0） |
| export 可用 | 取前 N | **取 400**（可用约 420–427） | **取 50**（可用约 53） | 失败重放 + 中途 reset 的交集 |
| 最终数据 | 写死 | **恰好 400 条** | **恰好 50 条** | — |

**余量与门槛**：train 需要 ≥400/440，即两段重放合计成功率不得低于约 **91%**；val 需要 ≥50/55，同样约 **91%**。PegInsertionSide 实测重放 97%，余量约 20–27 条。磁盘每个任务约 **2–4 GB**（128×128×6 通道图像占约 98%，`README` 已注明）。

## 跨阶段机制与常见故障

- **`.done` 续跑**：每个 `run.once()` 步骤成功后写标记，重跑跳过；有输出无标记则报错，需确认无并发后人工删除（`generate_task.py:80-99`）。
- **超时**：单步超过 `--timeout`（默认 8h）标记 FAIL。
- **渲染依赖**：只有 rgb 阶段需要 Vulkan；无 NVIDIA 驱动时退到 lavapipe（CPU，慢 7–17 倍），登录节点两者都没有，必须 `srun` 到 GPU 节点。
- **seed 越界**：极低成功率会耗尽 train 0–3999 / val 4000–4999，导出阶段直接失败。

## 失败阈值速查

| 现象 | 阶段 | 处理 |
| --- | --- | --- |
| `only N usable demos, need 400` | export | 删掉该 split 的 raw/转换文件与 `.done`，用更大的 `--n-train/--n-val` 重跑 |
| `... exists but is not marked done` | 任意 | 确认无并发进程后，删掉报错文件再重跑 |
| 渲染报错 / 找不到 Vulkan | rgb | 确认在 GPU 节点、`/usr/share/vulkan/icd.d/nvidia_icd.json` 存在，必要时 `export VK_ICD_FILENAMES=<路径>` |
| rgb 与 state 不一致 | export | 两次转换不可重复，删除该 split 的转换与 sidecar 重跑 |

## 输出布局

```
data/work/<Env>/motionplanning/<split>.h5                              原始专家（无观测）
data/work/<Env>/motionplanning/<split>.{rgb,state}.<mode>.physx_cpu.h5 两次转换 + .first_obs.h5 旁路
data/dataset/{train,val}/<Env>/motionplanning/trajectory.state.<mode>.physx_cpu.h5   ← 交给训练的数据
data/dataset/train/<Env>/motionplanning/sample.png                     首帧/末帧预览
data/logs/<task>.log                                                   每步汇总
```

（完整字段说明见 [README](../README.md) 的“输出”一节。）
