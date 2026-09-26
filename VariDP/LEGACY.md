# VariDP：已冻结的 donor 实现

本目录是 **backbone 对照的 donor**，不是第二条实验 pipeline。三种噪声预测主干
（官方 `ConditionalUnet1D`、官方 DP-T `TransformerForDiffusion`、本地 `MLPNoisePred`
及 MLP arm 的 observation MLP）已按 `dp-manip/REFACTOR_PLAN.md` Phase 9–10 迁移到
`dp-manip/dp_manip/backbones/`，成为 canonical 实现。

VariDP 自己的 trainer / dataset / scheduler / EMA / evaluator **没有被迁移**，也不应再跑：

- 正式实验只从 `dp-manip` 发起（`scripts/run_experiment.py`、`scripts/sweep.py`、`slurm/`）。
- 不要从 `dp-manip` import 本目录的任何代码；`dp-manip/tests/test_legacy_boundary.py`
  会静态检查这个边界。
- 这里只作 debug / 实现对照参考。需要改结构或训练口径时，改 `dp-manip/` 的 canonical
  实现和 config，不要在这里另起一套。
- 保留到集群上 RGB UNet / Transformer / MLP 三条 arm 全部通过 Gate A 为止；删除前确认
  git tag `pre-unified-pipeline`（指向重构前 Phase 0 冻结的 RGB baseline）仍然存在。
