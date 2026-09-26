# Layered RGB configs

正式入口按以下顺序解析，并将完整结果保存进 checkpoint：

```text
baseline.toml
  + tasks/<task>.toml
  + CLI runtime override
```

`baseline.toml` 是仿真/渲染、vision、policy、train、EMA、diffusion、evaluation
和通用 data 默认值的唯一权威来源。`tasks/*.toml` 只保存环境、控制模式、回合长度
和数据路径。根目录的 `*_rgb.toml` 仅为旧命令提供跳转，不含第二份 baseline 参数。

例如：

```bash
python scripts/train_dp.py --config configs/tasks/pickcube.toml
```

`--set SECTION.KEY=VALUE`、`--seed`、`--num-demos` 是运行时覆盖；集群上可用
`--data-root` 覆盖数据根目录。临时 smoke 可用 `--set train.total_iters=...`。
