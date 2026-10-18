# scripts/ — 分组实验启动脚本

本目录包含 MMLSv2 论文协议（`configs/mmlsv2_paper_protocol.json`，`pmdfnet_mmlsv2_remote_sensing_v4_ce_only_extended_baselines`）的分组启动入口。
所有脚本都必须在仓库根目录之外的任意位置以 `bash scripts/<脚本>.sh` 方式调用；它们会自行定位仓库根目录。

## 分组

| 脚本 | 分组 | 实验 | runs |
|---|---|---|---|
| `run_core.sh` | core | M1–M4 代表性基线 | 12 |
| `run_ablations.sh` | ablations | A1、A2、A5、B2、B3、C2、L1、L2、L3 | 27 |
| `run_extended.sh` | extended | M6 UPerNet、M7 FPN、M8 MANet、M9 U-Net++、M10 OCRNet+HRNet18、M11 M3LSNet | 18 |
| `run_all.sh` | all | 上述三组顺序执行 | 57 |

三个分组**互斥且并集完整**（`core` = M1–M4、`extended` = M6–M11、`ablations` = 其余），
分组是协议里每个实验显式存储的 `group` 字段。因此 `run_all.sh` 会让每个实验**恰好训练一次**，共 19 × 3 = 57 个 run。

分组选择逻辑只有一处实现（`train_ablation.experiment_in_group`），训练器、显存探针和启动脚本共用，三者不会对"某个组包含哪些实验"产生分歧。

`run_all.sh` 按组顺序执行；若想用单进程跑完所有组，用 `GROUP=all bash scripts/train_paper_experiments.sh`。

## 校验与评估

| 脚本 | 作用 |
|---|---|
| `run_data_smoke.sh` | 只读数据冒烟：CLI 解析、data-root 解析、dataset 导入、读取一个样本（不建模型、不训练） |
| `run_vram_smoke.sh` | 有界合成显存探针：逐个协议条目建模型并跑一个训练步，报告峰值显存（不读数据、不训练、不存 checkpoint） |
| `run_test_eval.sh` | 冻结后的批量 Test 评价，写入各 run 的 `test_metrics.json` |
| `run_complexity_eval.sh` | 对冻结的主基线统一测量 Params、GFLOPs、前向延迟/FPS 和峰值推理显存 |
| `run_robustness_eval.sh` | 对 M4 与 L1/M5 执行 clean + 5 个固定 Test 扰动/模态缺失条件，并保存逐图像 TP/FP/FN/TN |

## 复杂度与效率测量

`run_complexity_eval.sh` 默认测量 M1–M4、M6–M11 和 L1（L1 是论文表中的 M5），使用各正式 run 的 `best.pth`，不重新训练。结果写入：

```text
outputs_experiments/complexity/complexity_results.json
outputs_experiments/complexity/complexity_results.csv
```

指标定义固定为：Params 是模型全部参数量；GFLOPs 是 batch=1 的前向 FLOPs/1e9，由 `torch.profiler` 对支持 FLOP 公式的算子统计；FPS 是同步前向吞吐，延迟是 warmup 后的 forward-only 平均时间；显存是推理阶段峰值 allocated/reserved CUDA memory。自定义算子可能不被 profiler 统计，因此 GFLOPs 字段同时记录 `flops_backend`，不能把它误读成对所有算子的精确硬件指令数。

示例：

```bash
DEVICE=cuda:0 COMPLEXITY_SEEDS=41 bash scripts/run_complexity_eval.sh
```

## 受控扰动与模态缺失

固定协议在 `configs/mmlsv2_robustness_protocol.json`。它包含五个条件：DEM 偏置、DEM 高斯噪声、光学通道高斯噪声、物理模态缺失（Thermal/Slope/DEM）和光学模态缺失（Grayscale/R/G/B）；每个条件都带有预先声明的等级。所有扰动都在固定 MARS Test normalization 之后施加，模态缺失是归一化空间置零，即用该归一化空间的均值替代，不是修改原始 TIFF。每个模型/seed 还会评价 clean 作为配对参照。

输出为：

```text
outputs_experiments/robustness/robustness_results.json
outputs_experiments/robustness/robustness_results.csv
outputs_experiments/robustness/robustness_per_image.jsonl
```

JSONL 保存每张 Test 图像的 TP、FP、FN、TN、mIoU、F1、预测前景面积和目标前景面积，因此可以直接做同一图像上的模型配对比较、bootstrap 和 FP/FN 失败案例筛选。

```bash
DATA_ROOT=/data/.../mmlsv2_mapped_mars_ls bash scripts/run_robustness_eval.sh
```

## M11 与可复现性记录

M11 的 `mamba_ssm` selective-scan 和内置 `gated_conv_simulator` 是两个不同的后端，不能只记录“模型名为 M11”。复杂度和 robustness 输出会写入 `m3lsnet_backend` 与 `mamba_ssm_available`；正式训练/论文冻结时也应将这个字段连同环境信息归档。

当前历史 57 个 run 的 `git_dirty=true` 表示运行开始时工作树有未提交改动：summary 中的 commit 只能恢复当时的 HEAD，不能恢复这些未提交差异。最终冻结建议先提交全部代码/协议并确认 `git status --short` 为空，再重新跑至少论文主表和关键消融；如果暂时不重跑，则必须归档 `git diff HEAD`、暂存区差异、未跟踪文件、协议 JSON、环境/依赖清单以及对应输出目录，并把结果标记为 dirty-snapshot。该过程不会自动删除或重置现有结果。

## 环境变量

`scripts/_env.sh` 被各启动脚本 source，并自动加载仓库根目录的 `.env`。`.env`
中的变量作为默认值，启动命令中已经 export 的变量优先级更高。
训练、评估、数据 smoke test 和显存 smoke test 都使用同一套解析逻辑。

统一解析以下变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATA_ROOT` | `$MARS_DATA_ROOT` / `$MAPPED_DATA_ROOT` / `$MMLSV2_DATA_ROOT` / `.env` | mapped MMLSv2 数据根目录 |
| `SEEDS` | `41 42 43` | 随机种子 |
| `DEVICE` | `cuda:0` | torch 设备 |
| `BATCH_SIZE` | 协议默认（32） | 训练 batch |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `EXPERIMENTS` | 空（整组） | 组内子集，逗号分隔，如 `M4` 或 `L1,L2,L3` |
| `CONFIG` | `configs/mmlsv2_paper_protocol.json` | 协议文件 |
| `PYTHON_BIN` | `python` | Python 解释器 |
| `RUN_ROOT` | `outputs_experiments/pmdfnet_mmlsv2` | 仅评价脚本使用 |
| `EXPECTED_RUNS` | `57` | 仅评价脚本使用；部分矩阵评估时下调 |
| `DRY_RUN` / `SKIP_MODEL_INIT` | `0` | 数据/构建检查开关 |

`GROUP` 可覆盖单组脚本的分组（例如用 `run_core.sh` 这个入口去跑 `GROUP=ablations`），但可读性差，建议直接用对应脚本。

## 常用示例

```bash
# 补齐 E1 主表：先跑最关键的对照 M4，再跑新增基线
DATA_ROOT=/data/.../mmlsv2_mapped_mars_ls EXPERIMENTS=M4 bash scripts/run_core.sh
DATA_ROOT=/data/.../mmlsv2_mapped_mars_ls bash scripts/run_extended.sh

# 显存不够时先探针，再决定 batch
GROUP=extended BATCH_SIZE=32 bash scripts/run_vram_smoke.sh
GROUP=extended BATCH_SIZE=16 bash scripts/run_extended.sh

# 只跑损失消融
EXPERIMENTS=L1,L2,L3 bash scripts/run_ablations.sh

# 单 seed 快速核查
SEEDS=41 DRY_RUN=1 bash scripts/run_core.sh

# 全矩阵（57 runs）
DATA_ROOT=/data/.../mmlsv2_mapped_mars_ls bash scripts/run_all.sh

# 评价
DATA_ROOT=/data/.../mmlsv2_mapped_mars_ls bash scripts/run_test_eval.sh
```

## 注意

- `L1` 同时是论文主模型 M5，以及 A4/B1/C1 的来源，**只训练一次**，在 `run_ablations.sh` 里。
- `EXPERIMENTS` 只在所选组内过滤；给某个组传别组的 id 会被训练器拒绝并报错。
- 正式运行集为 **57** 个 run；`run_test_eval.sh` 会校验该数量（可用 `EXPECTED_RUNS` 下调）。
  该脚本按 `RUN_ROOT` 锚定匹配，`_backup_*`、`_archive*`、`_old*`、隐藏目录以及形如 `<run>_s41_..._DEBUG` 的手工副本都会被排除，不会混入统计。
- CE-only 重跑的实验名带 `_ce` 后缀，与旧 combined-loss 的历史 run 目录区分，两代不得混用。
- `DATA_ROOT` 可由 `DATA_ROOT` / `MARS_DATA_ROOT` / `MAPPED_DATA_ROOT` / `MMLSV2_DATA_ROOT` 或 `.env` 提供。训练与评估脚本不再内置远程主机路径。
