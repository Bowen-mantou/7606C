# Layered RGB experiment configs

正式入口按以下顺序解析，并将完整结果保存进 checkpoint：

```text
baseline.toml
  + tasks/<task>.toml
  + experiment override
  + CLI runtime override
```

`baseline.toml` 是仿真/渲染、vision、policy、train、EMA、diffusion、evaluation
和通用 data 默认值的唯一权威来源。`tasks/*.toml` 只保存环境、控制模式、回合长度
和数据路径。根目录的 `*_rgb.toml` 仅为旧命令提供跳转，不含第二份 baseline 参数。

正式 data-size grid 定义在 `experiments/data_size.toml`，其中声明
`variable = "data.num_demos"`、实验 values 和各 value 的 replicate seeds。core 只要求
`data.num_demos` 为正整数。条件式 N=400 follow-up 单独放在
`experiments/data_size_optional400.toml`，不属于正式 grid。

`data.num_demos=N` 固定选择按 `episode_seed` 升序排序后的前 N 条示范，与 HDF5 导出
顺序和 `episode_id` 无关，因此 `25 ⊂ 50 ⊂ 100 ⊂ 200` 对任何导出结果都成立；
`episode_id` 与 `episode_seed` 在同一 split 内必须唯一，重复会直接报错。
`run.json` 的 `data_selection` 记录实际选中的 `demo_seeds`，
`tests/test_data_nesting.py` 与 `scripts/inspect_dataset.py` 负责校验该不变量。

例如，解析 PickCube 的 N=50 数据量实验：

```bash
python scripts/train_dp.py \
  --config configs/tasks/pickcube.toml \
  --experiment configs/experiments/data_size.toml \
  --experiment-value 50
```

`--set SECTION.KEY=VALUE`、`--seed`、`--num-demos` 是最后应用的运行时覆盖。
集群上可用 `--data-root` 覆盖数据根目录。临时 smoke 可用
`--set train.total_iters=...`；正式实验仍使用 baseline 的固定训练预算。
