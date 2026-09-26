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
- 本机没有项目的 torch/ManiSkill 环境；完整数据检查与正式 GPU smoke 仍需在集群完成。
