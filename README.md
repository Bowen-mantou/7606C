# 7606 proj of Group11

DASC7606C 小组项目。**数据量**与**模型结构**两条研究轨道现在都由 `dp-manip` 的同一套 RGB
pipeline 运行；`VariDP` 保留为 backbone 实现的 donor，已冻结，不参与正式实验。

| 目录 | 做什么 |
| --- | --- |
| [maniskill-demogen](./maniskill-demogen) | **Demo Gen**：按任务生成 ManiSkill RGB + state 示范数据，支持断点续跑与 Slurm 批处理 |
| [dp-manip](./dp-manip) | **统一 pipeline**：data-efficiency（示范量 25 / 50 / 100 / 200，按条件加 400）与 backbone（UNet / Transformer / MLP） |
| [VariDP](./VariDP) | 历史 donor：UNet / DP-T / MLP 的实现来源，三者已迁移进 `dp-manip/dp_manip/backbones/`，目录本身冻结（见 [VariDP/LEGACY.md](./VariDP/LEGACY.md)） |

- 实验参数、种子口径与分工：[dp-manip/docs/final-plan.md](./dp-manip/docs/final-plan.md)
- 旧 state-based 工作流归档：[dp-manip/legacy/](./dp-manip/legacy/)

详细参见两个目录下的readme.
