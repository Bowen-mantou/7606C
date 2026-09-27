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

## dp-manip upstream / subtree workflow

`dp-manip/` 通过 **Git subtree**（`--squash`）vendored 进本仓库。它的 canonical upstream 是
[`hollinsStuart/dp-manip`](https://github.com/hollinsStuart/dp-manip)；本仓库 `hryang1130/7606C`
是**课程级集成仓库**——`dp-manip/` 的代码与实验改动先在 upstream 完成，再同步回这里，
不要直接修改 vendored 目录（除了本仓库自己的集成与文档细节）。

注意：Git remote 是**本地配置**，新 clone 不会自动继承，需要在本地加一次：

```bash
git remote add dp-manip git@github.com:hollinsStuart/dp-manip.git
git fetch dp-manip
git switch -c integration/update-dp-manip
git subtree pull --prefix=dp-manip dp-manip main --squash
```

同步之后必须：在 `dp-manip/` 下装环境并跑完整测试、检查 `git diff`（`dp-manip/` 之外不得有变化），
然后创建 PR；PR 使用**普通 merge commit**，不要 squash 或 rebase。
贡献者与署名依据见 [dp-manip/CONTRIBUTORS.md](./dp-manip/CONTRIBUTORS.md)。
