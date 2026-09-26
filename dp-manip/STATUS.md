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
- 本机没有项目的 ManiSkill/GPU 环境；完整数据检查与正式 GPU smoke 仍需在集群完成。
  本机临时 venv（torch/diffusers/h5py）仅用于 CPU 单元测试与合成数据 smoke，不是项目环境。
