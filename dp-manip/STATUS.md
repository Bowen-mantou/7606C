# 当前状态

- 已切换为 `maniskill-demogen` RGB schema：`obs_rgb/rgb + obs_rgb/state`。
- 六任务配置已按最终任务表建立；PegInsertionSide/PlugCharger 使用 `pd_joint_pos`。
- 训练已改为集群优先：HDF5 懒加载、worker DataLoader、AMP、EMA、可恢复 checkpoint。
- Slurm 核心 96 组与条件 N=400 数组入口已建立。
- 闭环评估已改为 RGB 环境，并固定使用与数据一致的 `physx_cpu`。
- Phase 0 已冻结 commit `834be80` 的 PickCube RGB baseline，并实测完成最小
  `train → save → load → evaluate` 链路；manifest 见
  `baselines/phase0/pickcube_rgb.json`。
- Phase 4 已把 data-size 子集固定为按 `episode_seed` 升序取前 N 条，并拒绝重复的
  `episode_id`/`episode_seed`：`run.json` 记录 `data_selection.demo_seeds`，
  `tests/test_data_nesting.py` 验证 `25 ⊂ 50 ⊂ 100 ⊂ 200`，
  `inspect_dataset.py` 在提交前做同样预检。
- Phase 5 已让 `resume.pt`（format 3）保存/恢复 Python、NumPy、torch CPU/CUDA RNG
  state，并把训练 batch 改为按 `(seed, step)` 确定性采样；`run.json` 记录
  `sampler.scheme=step_seeded_with_replacement`，`tests/test_rng_resume.py` 验证
  连续训练与被抢占续训的随机轨迹一致。已在本机 CPU 上用合成数据实测
  `SIGUSR1 → resume.pt → 续训`：final EMA 权重与 raw model/optimizer state 均与
  连续训练 bitwise 相同。旧 `resume.pt` 缺少 `rng` 字段时仍可续训并打印 warning。
- Phase 6 已把 observation encoding 抽成独立 `ObservationEncoder`
  （`dp_manip/observation_encoder.py`）：统一接口 `observation_encoder(rgb, proprio)`
  固定返回 `(B, To, Dobs)`，`Dobs = num_cameras * feature_dim + proprio_dim`；policy 只在
  交给 Conditional UNet 前 flatten，encoder 不感知 UNet/Transformer/MLP。
  `tests/test_observation_encoder.py` 验证输出 shape、RGB+proprio 融合、
  share/per-camera encoder、policy 边界 flatten，以及旧 checkpoint
  （`image_encoders.*`、`state_*`/`proprio_*` → `observation_encoder.*`）仍可加载。
  本机 CPU 用合成数据实测 `train → save → load → get_action` 和 legacy-key resume 全部通过。
- Phase 7 已引入统一 `NoisePredictor` 接口（`dp_manip/backbones/`）：`policy.backbone`
  （baseline 默认 `"unet"`）经 registry 构建 backbone，policy 把 `(B, To, Dobs)`
  observation 序列原样传入，flatten 只发生在 `UNetBackbone` 内部，encoder / trainer /
  scheduler / EMA / evaluator 未变。`adapt_legacy_state_dict` 新增
  `noise_pred_net.* → noise_predictor.unet.*` 映射，旧 checkpoint 仍可加载。
  Phase 0 manifest 升到 schema version 2，仅新增结构选择器记录，不改任何超参。
  本机 CPU 合成数据实测 train→save→load→get_action 与 legacy resume 通过，
  loss 与 Phase 6 逐位一致。
- Phase 9 已从 VariDP 迁移 Transformer backbone（`dp_manip/backbones/transformer.py`）：
  忠实移植官方 `TransformerForDiffusion` 与 VariDP 的 DP-T lowdim 配置（8 层 / 4 头 /
  256 维 / causal attention / attn dropout 0.3），`TransformerBackbone` 把共享
  observation encoder 产出的 `(B, To, Dobs)` 原样作为条件 token 送入 decoder，不重新
  编码 RGB/proprio，也不引入 VariDP 的 trainer/dataset/scheduler。结构参数
  `policy.transformer_*` 来自 `baseline.toml`，与 UNet arm 的 resolved config 只在
  `policy.backbone` 上不同。Phase 0 manifest 升到 schema version 3，仅新增 Transformer
  结构字段，不改任何科学超参。本机 CPU 合成数据实测 transformer arm 的
  train→save→resume→load→get_action 通过，`tests/test_backbone_smoke.py` 覆盖
  forward/backward/optimizer step/sampling。
- Phase 10 已从 VariDP 迁移 MLP backbone（`dp_manip/backbones/mlp.py`）：忠实移植
  VariDP 本地 baseline 的 `MLPNoisePred`（展平动作 ‖ 时间嵌入 ‖ 展平观测 → 3 层
  Mish/LayerNorm MLP），时间嵌入移入 backbone 以保持统一的原始 timestep contract，
  观测条件为共享 encoder 的 `(B, To, Dobs)` flatten 后再经 VariDP 的 observation MLP
  （`To*Dobs → 256 → 256`，Phase 11 验收时按 `docs/final-plan.md` §6 B2 补回，宽度由
  `policy.mlp_obs_feat_dim` 控制）。结构参数 `policy.mlp_*` 来自 `baseline.toml`。Phase 0 manifest 升到
  schema version 4，仅新增 MLP 结构字段，不改任何科学超参。本机 CPU 合成数据实测
  unet/transformer/mlp 三个 arm 的 train→save→resume→load→get_action 全部通过；
  三个 resolved config 只在 `policy.backbone` 上不同（Gate B），
  `tests/test_backbone_smoke.py` 对三个 backbone 逐一执行 forward/backward/
  optimizer step/sampling。
- Phase 11 已建立 canonical backbone experiment 定义
  `configs/experiments/backbone.toml`（轨道 B）：`variable = "policy.backbone"`、
  `values = ["unet", "transformer", "mlp"]`，三个 arm 都使用种子 1–5；N_B 默认沿用
  baseline 的 100 条，难任务按 `docs/final-plan.md` §6 的预注册规则在运行时改用 200 条；
  `[diagnostics] train_eval_episodes = 25` 与 data-size grid 共用前 25 个训练 seed 的
  过拟合诊断预算。`tests/test_config.py` 验证 spec 解析、未知 value 被拒绝，以及把
  `policy.backbone` 归一后三个 arm 的 resolved config 完全相同（Gate B）；
  `configs/README.md` 记录了解析方式和统一入口的调用方式。
- Phase 12 已建立统一实验入口 `scripts/run_experiment.py`（`--task` / `--experiment` /
  `--value` / `--seed`）：它只按 canonical 分层解析 config，然后调用唯一的训练 pipeline
  `dp_manip/trainer.py::run_training`。`config.load` 现在会把 CLI 字符串值匹配到 experiment
  spec 声明的类型，所以整数 data-size 与字符串 backbone 共用同一个入口；入口内没有任何按
  experiment 名称的分支，新实验只需新增 config（测试用临时 spec 验证）。`scripts/train_dp.py`
  缩成同一 trainer 的薄 CLI，继续供 `sweep.py`/Slurm 使用，训练循环逐行未变。
  `tests/test_run_experiment.py` 验证 value 匹配、两类实验的解析、与 sweep cell 和旧 CLI 的
  resolved config 完全一致，以及未知 task/experiment/value 被拒绝。本机 CPU 合成数据实测
  data_size N=25 与 backbone unet/transformer/mlp 三 arm 均经统一入口完成
  train→save→resume→load→get_action，三个 backbone run config 只在 `policy.backbone`
  上不同（Gate B）。
- Phase 13 已实现 Gate B 自动检查器 `scripts/check_experiment.py`：对每个任务解析 experiment
  spec 声明的全部 `(value, seed)` cell 并逐一对比 resolved config，只允许声明的实验变量、
  replicate seed（`train.seed`）、运行时 `data.root` 和 backbone 结构键
  （`policy.unet_*` / `policy.transformer_*` / `policy.mlp_*`）不同，其余差异按 per-key
  矩阵格式报错并返回非零状态；同时按 §19 输出每个任务的 `control_hash`（同一矩阵所有 cell
  必须相同）。`dp_manip.config.resolve_config_path` 抽出了入口共用的 name-or-path 解析。
  `tests/test_check_experiment.py` 覆盖 diff/prune/control_hash、真实三套矩阵
  （data_size / data_size_optional400 / backbone，六任务共 216 cells）、drift 报告格式与
  CLI 退出码；本机实测三套矩阵全部 `Gate B ok`。验收时补充：结构键豁免只在
  `policy.backbone` 实验生效；`--run-root` 读取各 cell 的 `run.json`，与其他 cell 和当前
  声明对比，并列出尚未运行的 cell（本机用带 `--set` 缩小预算的合成 run 实测能检出）。
- 本机没有项目的 ManiSkill/GPU 环境；完整数据检查与正式 GPU smoke 仍需在集群完成。
  本机临时 venv（torch/diffusers/h5py）仅用于 CPU 单元测试与合成数据 smoke，不是项目环境。
