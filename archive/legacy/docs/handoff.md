# PMDFNet：Physically Motivated Dual-Stream Fusion for Multimodal Landslide Segmentation

项目论文名称和项目目录已统一为 PMDFNet。历史代码键 `dual_segformer_*`、已有结果目录仍保留，以保证旧实验可追溯、可复现。

更新时间：2026-10-14
当前阶段：已完成一次 Small 单模型基线链路验证和公开 test 评估；当前正在迁移到 A100，论文级固定协议和多 seed 正式实验尚未启动
核心顺序：**先复现比赛结果，再冻结基线，最后进行新模型尝试**

## 1. 工作目标

本阶段不直接追求新的网络结构，而是先回答两个问题：

1. 当前代码、数据预处理和训练配置能否稳定复现旧版比赛方案；
2. 在旧版结果可复核之后，新模型是否带来可重复、可解释的提升。

实验结论必须区分以下三类结果：

~~~text
历史比赛结果：官方/线上榜单结果，作为历史参考，不重新包装成本地验证结果
复现结果：使用固定代码、固定数据和固定协议重新训练得到的结果
新模型结果：在复现基线冻结后，使用相同协议进行的新增实验
~~~

## 2. 当前环境与路径

### 2.1 本地仓库

~~~text
仓库：E:\repository\PMDFNet
本地比赛原始数据：Z:\huilin\bdd\cp_data\mars_seg
本地 MMLSv2：Z:\huilin\bdd\cp_data\mmlsv2
本地映射数据：Z:\huilin\bdd\cp_data\mmlsv2_mapped_mars_ls
~~~

映射脚本：

~~~text
tools/data/map_mmlsv2_to_mars.py
~~~

映射报告：

~~~text
Z:\huilin\bdd\cp_data\mmlsv2_mapped_mars_ls\mapping_manifest.json
~~~

### 2.2 远端训练环境

~~~text
SSH Host：A100-1
GPU：NVIDIA A100 80GB PCIe，使用 GPU 0
代码：/data/huilin/projects/PMDFNet
远端原始数据：/data/huilin/scrinvme/huilin/bdd/cp_data/mmlsv2
远端映射数据：/data/huilin/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls
训练入口：/data/huilin/projects/PMDFNet/train.py
当前远端仓库：ablation，commit fc21b43；工作区干净
本地论文实验分支：ablation，commit fc21b43；已同步到 A100
~~~

远端数据规模：

~~~text
train：465 images / 465 masks
val：   66 images /  66 masks
test： 133 images / 133 masks
~~~

远端 train.py 已修改为读取环境变量：

~~~bash
export MARS_DATA_ROOT=/data/huilin/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls
~~~

优先级为：

~~~text
MARS_DATA_ROOT > DATA_ROOT > 原始比赛数据默认路径
~~~

训练启动时会在日志中打印最终使用的数据根目录。当前 A100 的 Python 环境尚待确认，不能继续沿用旧服务器的环境路径。

旧服务器的正式训练环境记录为：

~~~text
/home/23039356r/.conda/envs/M3LSNet/bin/python
Python 3.10.19 / PyTorch 2.5.1+cu121
CUDA_VISIBLE_DEVICES=0（物理 GPU 0）
~~~

该环境只作为旧服务器历史记录。A100 正式运行前需要重新记录 Python、PyTorch、CUDA、Git commit、GPU、数据 manifest 和 seed。

### 2.3 Shell 实验入口

顶层启动脚本为 `exp_train.sh`，默认只启动比赛复现组：

~~~bash
bash exp_train.sh reproduce
bash exp_train.sh fusion
bash exp_train.sh capacity
bash exp_train.sh new_model
bash exp_train.sh all
~~~

实验组的默认顺序是：比赛复现、旧模型融合消融、模型容量对比，最后才是 SKG 新模型。路径、seed、设备和 epoch 通过环境变量覆盖，具体配置见各组脚本。

### 2.4 训练输出、日志和 checkpoint 规则

训练期间，单次运行先写入代码项目下的暂存目录，例如：

~~~text
outputs_experiments/reproduction/<model>/seed42/<run_id>/
~~~

训练成功结束后，整个运行目录会复制到数据集根目录的：

~~~text
<MARS_DATA_ROOT>/outputs/<run_id>/
~~~

暂存目录默认保留，便于失败排查和恢复。最终目录包含：

- `checkpoints/best.pth`：验证集 mIoU 最高的模型；
- `checkpoints/last.pth`：最后一个 epoch 的模型；
- `logs/train_log_*.txt`：配置、每个 epoch 的 train/val 指标、best 更新和完成状态；
- `tensorboard/`：batch loss、epoch loss 和验证指标；
- `run_config.json`：数据路径、seed、超参数、输出路径和 checkpoint 规则；
- `results.json`：best/last epoch、指标和最终结果路径。

模型权重仍只保留 `best.pth` 和 `last.pth` 两类。`best.pth` 的选择依据是验证集 `mIoU`，不是 test 指标；若关闭验证或验证集为空，则不会生成 `best.pth`。文本日志记录 epoch 级训练和验证信息，batch 级 loss 记录在 TensorBoard 中。

### 2.5 已完成结果清点（2026-10-14）

旧服务器 `rtx6000-2` 已回收，但最终运行目录保存在共享数据盘，A100 上仍可访问：

~~~text
/data/huilin/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls/outputs/
~~~

已找到两个 Small seed=42 运行目录。最新目录为：

~~~text
dual_segformer_convnextsmall_chv1_add_seed42_20261004_005845/
~~~

该目录完整保留：

~~~text
checkpoints/best.pth
checkpoints/last.pth
logs/train_log_20261004_005845.txt
run_config.json
results.json
test_metrics.json
tensorboard/
~~~

最新运行结果（global pixel aggregation）：

~~~text
best epoch：70
val mIoU：0.84491
public converted test mIoU：0.84443
public converted test batch-average mIoU（辅助记录）：0.84937
~~~

该结果是公开转换后的 MMLSv2 test 评估，不是官方隐藏 leaderboard 结果。旧目录
`dual_segformer_convnextsmall_chv1_add_seed42_20261003_232622/` 也保留，但正式汇总优先使用最新目录。

### 2.6 论文级可复现性设置

本轮已将各训练入口的 seed 逻辑统一到 `reproducibility.py`：

- Python、NumPy、PyTorch、CUDA 和 DataLoader worker 使用同一实验 seed；
- 固定 `CUBLAS_WORKSPACE_CONFIG`，关闭 cuDNN benchmark 和 TF32；
- `STRICT_DETERMINISM=1` 时，遇到不支持确定性的 CUDA 算子直接报错，而不是静默继续；
- 启动脚本在 Python 进程启动前设置 `PYTHONHASHSEED`；
- legacy、SKG 和 ablation run 记录 `reproducibility.json`、`runtime.json`，legacy 另外记录 `mapping_manifest` SHA-256 和 Git 状态。

严格可复现的边界是“同一代码、同一数据文件、同一依赖环境、同一 GPU 和同一运行配置”。跨 GPU 或跨 PyTorch/CUDA 版本不承诺 bit-wise 完全一致。

## 3. 数据映射结论

映射后的内部通道顺序为旧比赛顺序：

~~~text
[Thermal, Slope, DEM, Grayscale, Red, Green, Blue]
~~~

MMLSv2 原始顺序为：

~~~text
[Red, Green, Blue, DEM, Slope, Thermal, Grayscale]
~~~

映射同时完成通道重排和数值反归一化，使数据可以被旧版 MarsSegDataset 按比赛数据方式读取。

### 3.1 适用范围

映射数据可以用于：

- 复用旧版比赛代码；
- 训练和验证旧版 PMDFNet；
- 在公开 test mask 上进行冻结后的本地评估；
- 对比旧模型和新模型。

但论文中必须称为：

~~~text
MMLSv2 converted to the original Mars-LS channel ordering and radiometric representation
~~~

不能把它描述为新的独立数据集，也不能把公开 MMLSv2 test 结果直接称为官方隐藏测试集成绩。

### 3.2 已知差异

- train mask 有一个像素与本地比赛版本不同；
- Slope 通道存在少量边界零值差异；
- MMLSv2 的 test mask 是公开标签，比赛原始 test 没有本地公开 mask；
- 映射数据用于复现输入格式，不代表可以重新获得官方线上 leaderboard 评价。

## 4. 阶段 A：复现比赛结果

这一阶段完成前，不开始新模型结构搜索。

### A0. 环境和数据冒烟测试

目标：确认训练环境、数据读取、归一化和 GPU 都正确。

检查项：

- train.py 能通过语法检查；
- MARS_DATA_ROOT 生效，日志打印映射数据路径；
- train/val 数据数量分别为 465/66；
- 图像读取后形状为 7 x 128 x 128；
- mask 只包含 0/1；
- 经过比赛 mean/std 后不存在 NaN 或 Inf；
- cuda:0 能正常初始化；
- 只运行一个 batch，确认 loss 可以反向传播。

建议命令模板：

~~~bash
cd /data/huilin/projects/PMDFNet
export MARS_DATA_ROOT=/data/huilin/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls
python3 -m py_compile train.py scripts/train_competition_reproduction.py
~~~

正式运行前需要先激活正确的 Python 环境。train.py 当前主程序会顺序运行多个模型；正式复现应使用本地单模型入口 scripts/train_competition_reproduction.py，避免一次提交完整模型列表。

### A1. 单模型复现

首先只复现历史核心模型：

~~~text
dual_segformer_convnexttiny_chv1_add
dual_segformer_convnextsmall_chv1_add
~~~

固定配置：

~~~text
数据：mmlsv2_mapped_mars_ls
训练 split：train
验证 split：val
输入尺寸：128 x 128
Epoch：100
Dual 模型 batch size：16
优化器：AdamW
学习率：1e-4
weight decay：5e-4
Scheduler：CosineAnnealingLR
随机种子：42
设备：cuda:0
通道划分：chv1，即 0,1,2,3 / 4,5,6
增强：沿用旧版代码，包括 MosaicCastDataset
~~~

注意：比赛 test 不参与 checkpoint 选择，验证必须使用 val。

单模型启动命令：

~~~bash
python scripts/train_competition_reproduction.py \
  --model-name dual_segformer_convnexttiny_chv1_add \
  --seed 42 \
  --device cuda:0
~~~

### A2. 历史模型矩阵复现

单模型链路正常后，再依次运行：

~~~text
dual_segformer_convnexttiny_chv1_add
dual_segformer_convnextsmall_chv1_add
dual_segformer_convnextbase_chv1_add
dual_segformer_convnextlarge_chv1_add
~~~

每个模型必须单独保存：

~~~text
训练配置
Git commit
随机种子
数据路径和 mapping_manifest
训练日志
best checkpoint
last checkpoint
验证指标
~~~

### A3. 复现判定标准

历史线上结果只作为参考，不要求本地 val 数值完全相等。复现通过应满足：

- 数据读取和归一化链路一致；
- 模型结构、通道划分、增强、loss 和优化器一致；
- 同一 seed 重复运行结果接近；
- 训练曲线和前景预测没有明显坍缩；
- best checkpoint 和验证指标可以从日志中追溯；
- 至少 Tiny 和 Small 两个核心模型成功完成。

历史结果参考：

~~~text
比赛 dataA 的历史 chv1_add online score：0.8665
旧记录中的 MMLSv2 train -> test chv1_cat：mIoU 0.8619，IoU_fg 0.8210，F1 0.9017
~~~

以上数值均为历史记录，不能替代本轮按 train -> val 协议产生的复现结果。

### A4. 复现阶段报告

复现阶段至少生成一张表：

| Model | Channels | Fusion | Seed | Best mIoU | Best IoU_fg | Best F1 | Best epoch | Checkpoint |
|---|---|---|---:|---:|---:|---:|---:|---|
| Tiny | chv1 | add | 42 |  |  |  |  |  |
| Small | chv1 | add | 42 |  |  |  |  |  |
| Base | chv1 | add | 42 |  |  |  |  |  |
| Large | chv1 | add | 42 |  |  |  |  |  |

只有这张表和对应 checkpoint 都齐全后，才进入阶段 B。

## 5. 阶段 B：冻结基线后的新模型尝试

### B0. 冻结事项

阶段 A 完成后冻结以下内容：

- 映射数据版本和 mapping_manifest.json；
- train/val/test split；
- 输入尺寸和基础增强；
- 评价指标；
- baseline checkpoint 选择规则；
- 训练 seed 集合；
- test 只评估一次的流程。

任何新模型都必须与冻结后的 baseline 使用同一数据和评价协议。

### B1. 新模型主线

当前仓库的新模型主线为 SKG-PMDFNet，包括：

~~~text
single-stream / dual-stream baseline
spectral descriptor
train-only prototype bank
knowledge-guided fusion
knowledge consistency loss
~~~

建议顺序：

1. 先运行与旧模型同规模的 single-stream 和 dual-stream baseline；
2. 加入 spectral descriptor；
3. 加入 prototype prior；
4. 加入 knowledge-guided fusion；
5. 最后加入 consistency loss；
6. 每次只改变一个主要因素。

### B2. 新模型实验矩阵

最小矩阵：

| 编号 | 模型 | 目的 |
|---|---|---|
| B-01 | Single SegFormer | 单流基线 |
| B-02 | PMDFNet | 双流基线 |
| B-03 | Dual + descriptor | 检查显式光谱描述子 |
| B-04 | Dual + prototype | 检查类别原型先验 |
| B-05 | Dual + knowledge fusion | 检查知识引导融合 |
| B-06 | Full SKG-PMDFNet | 完整方法 |
| B-07 | Full - descriptor | 消融 |
| B-08 | Full - prototype | 消融 |
| B-09 | Full - consistency | 消融 |

### B3. 新模型训练协议

正式结果至少使用三个 seed：

~~~text
42, 123, 7
~~~

报告：

~~~text
mean ± std
mIoU
IoU_fg
F1
precision
recall
参数量
FLOPs 或推理时间
显存占用
失败案例
~~~

归一化、光谱统计量、prototype 和聚类中心只能由 train split 计算。val 用于 checkpoint 选择和模型决策，test 必须在所有设计冻结后进行一次最终评估。

如果新模型使用映射后的 raw-like 数据，不能使用 SKG 代码默认的 normalization=auto/none 直接输入，建议使用 train-only z-score，并在实验记录中明确说明。

## 6. 数据与评价边界

必须遵守：

- MMLSv2 是公开的比赛数据发布版本，不是独立外部数据集；
- 转换后的数据可作为比赛格式复现数据，但不能制造新的 leaderboard 结果；
- 不使用 test mask 选择模型、调参或确定消融项；
- 不将历史 top-2 线上成绩和本地 val/test 指标混在同一张表中；
- 论文中单独标注“历史比赛结果”“本地复现结果”“新模型结果”；
- 所有结果必须记录 seed、代码 commit、配置、数据版本和 checkpoint 路径。

## 7. 当前待办清单

### 阶段 A：复现比赛结果

- [x] 确认 A100-1、代码路径和共享数据路径
- [x] 核对旧服务器结果已保存在共享数据盘
- [x] 将本地 `ablation` 分支（commit `fc21b43`）同步到 A100 代码目录
- [x] 完成 MMLSv2 到比赛格式的映射
- [x] 导出映射数据并完成数量检查
- [x] 修改远端 train.py 使用环境变量
- [x] 远端 train.py 语法检查通过
- [x] 编写按实验目的分组的 exp_train.sh 启动入口
- [ ] 激活远端正确 Conda/虚拟环境
- [ ] 完成单 batch 数据和 GPU smoke test
- [ ] 复现 Tiny chv1_add
- [x] 复现 Small chv1_add（旧服务器 seed=42；A100 clean repeat 待完成）
- [ ] 复现 Base/Large chv1_add
- [ ] 固定复现 baseline checkpoint
- [ ] 整理 train -> val 结果表

### 阶段 B：新模型

- [ ] 固定 baseline 配置和 checkpoint
- [ ] 运行 single/dual baseline
- [ ] 运行 descriptor ablation
- [ ] 运行 prototype ablation
- [ ] 运行 knowledge fusion ablation
- [ ] 运行 consistency loss ablation
- [ ] 使用 42/123/7 三个 seed
- [ ] 对冻结模型进行一次 test 评估
- [ ] 统计 mean ± std、参数量、速度和显存
- [ ] 保存可视化和失败案例
- [ ] 更新论文实验表和方法限制

## 8. 结果记录模板

每个 run 至少记录：

~~~text
run_id:
date:
remote_host:
git_commit:
data_root:
mapping_manifest:
train_split:
val_split:
test_split:
model:
encoder:
channels1:
channels2:
fusion:
loss:
augmentation:
normalization:
seed:
epochs:
batch_size:
learning_rate:
weight_decay:
best_checkpoint:
best_miou:
best_iou_fg:
best_f1:
test_evaluated: yes/no
notes:
~~~

## 9. 下一步

下一步先做“确定性审计”，不启动完整实验矩阵：

1. 在同一 commit、同一 GPU、同一数据 manifest、同一环境下，用同一 seed 重复两个短 run；
2. 比较 `reproducibility.json`、训练曲线、验证指标和 checkpoint hash；
3. 若 `STRICT_DETERMINISM=1` 报出不支持的算子，先定位并处理，再开始正式复现；
4. 审计通过后，冻结论文协议，再运行阶段 A 的正式多 seed 实验。

## 10. 论文级实验重做方案（2026-10-04）

### 10.1 Material Passport

~~~text
Origin Skill: academic-research-suite / experiment-agent
Origin Mode: experiment planning
Origin Date: 2026-10-04
Verification Status: code-level reproducibility hardening completed; protocol runs pending
Version Label: pmdfnet_paper_protocol_v1
~~~

工作假设：`SKG-PMDFNet` 是待投稿的主方法，旧版 `PMDFNet` 是历史/公平基线。如果最终论文主方法仍是旧版模型，只需交换主方法和 baseline 的叙述，不改变数据、seed 和评价规则。

### 10.2 两条实验轨道

不能用一套结果同时回答“是否复现比赛”和“新方法是否公平提升”。因此分成：

| 轨道 | 目的 | 预处理 | 结果用途 |
|---|---|---|---|
| H：历史兼容轨道 | 尽量复现旧版训练链路和历史结果 | 保留 legacy 原有 split-specific normalization、MosaicCast 和旧配置 | 复现/历史对照，不作为最严格的主表公平比较 |
| P：论文公平轨道 | 比较 baseline、SKG 和消融 | 所有方法统一使用 train-only channel statistics；train/val/test 使用同一个 normalizer | 论文主表、消融表和最终 test |

P 轨道的 train-only normalizer、descriptor 和 prototype 只能由 465 张 train 图像计算；val 只用于 checkpoint/模型决策；test 在协议冻结后才加载。当前训练代码已经统一使用 global pixel-level mIoU 选择 best checkpoint；若使用映射后的 raw-like 数据，启动 SKG 时仍必须显式指定 `--normalization train_zscore`，不能沿用 `auto/none`。

### 10.3 固定评价协议

- 主指标：全像素汇总 confusion matrix 计算的 global mIoU；
- 次指标：IoU_fg、IoU_bg、F1、precision、recall；
- 所有方法均使用 validation global mIoU 选择 `best` checkpoint；`last` 只表示最后一个 epoch；
- test 不参与调参、消融选择、early stopping 或 checkpoint 选择；
- 论文主表报告 3 个 seed 的 `mean ± std`，不要只报告最优 seed；
- 历史旧 run 的逐 batch 平均 mIoU 和历史线上 top-2 分数保留为兼容性记录，但不能与 P 轨道 global mIoU 混成同一列。

### 10.4 Seed 和确定性策略

| 用途 | Seed | 说明 |
|---|---|---|
| 调试/冒烟 | 42 | 只验证代码，不进入论文统计 |
| 正式结果 | 42、123、7 | baseline、主方法和关键消融使用相同 seed 集合 |
| 确定性审计 | 固定 42 | 同配置重复两次，要求指标/曲线一致或差异可解释 |

正式运行统一固定：`CUDA_VISIBLE_DEVICES=0`、`num_workers=4`、Python 环境、Git commit、数据 manifest hash、normalizer 文件和训练超参数。严格模式默认 `STRICT_DETERMINISM=1`；若某个模型确实包含无法确定性的算子，必须在日志中记录，不能把“近似可复现”写成完全可复现。

### 10.5 分阶段实验矩阵

#### Stage 0：确定性审计

用 P 轨道配置、同一 seed 42 运行两个 2–5 epoch 的短实验。检查：

- 两次的输入文件顺序、DataLoader generator 和 worker seed 一致；
- loss/val 指标逐 epoch 一致；
- `run_config`、normalizer、Git commit、manifest hash 一致；
- 若保存权重，比较 checkpoint SHA-256。

#### Stage A：历史基线和公平基线

1. H 轨道：legacy Tiny/Small `chv1_add`，seed 42，作为历史兼容复现；
2. P 轨道：Single-stream SegFormer；
3. P 轨道：Dual-stream `add`；
4. P 轨道：Dual-stream `cat`；
5. P 轨道：选定的 legacy Small `chv1_add`，作为参数规模/历史模型对照。

先用 seed 42 做配置检查；确认协议无误后，最终进入主表的 baseline 使用 42/123/7。Tiny/Small/Base/Large 容量扫描可先用单 seed，只有论文要比较的代表模型才做三 seed。

#### Stage B：SKG 主方法和组件消融

每次只增加一个主要组件，推荐顺序：

| 编号 | 配置 | 要验证的问题 |
|---|---|---|
| B0 | Dual baseline | 新模型相对于双流基础结构的增益 |
| B1 | + spectral descriptor | 显式光谱描述是否有效 |
| B2 | + train-only prototype prior | 类别原型先验是否有效 |
| B3 | + knowledge-guided fusion | 知识引导融合是否有效 |
| B4 | + consistency loss | 一致性约束是否带来独立增益 |
| B5 | Full SKG-PMDFNet | 完整方法最终效果 |
| B6 | Full - descriptor | 组件必要性 |
| B7 | Full - prototype | 组件必要性 |
| B8 | Full - consistency | 组件必要性 |

探索性超参数（prototype K、temperature、loss weight、early stopping）只用 train/val 和固定 exploratory seed，不能反复查看 test 后再选择。最终配置冻结后，用 42/123/7 重跑主方法和关键 baseline。

#### Stage C：效率、鲁棒性和可解释性

对最终保留模型统一测量参数量、FLOPs 或单图推理时间、峰值显存和吞吐；保存代表性 TP/FP/FN/TN 图、不同模态缺失/噪声情况下的失败案例，以及 descriptor/prototype/gate 可视化。效率测试不得改变训练协议。

#### Stage D：最终 test

只有当数据、normalizer、模型结构、超参数、seed 集合和 checkpoint 选择全部冻结后，才对每个最终 run 的 `best` checkpoint 进行一次 public test 评估。映射后的 MMLSv2 test 有公开 mask，可以作为本地补充实验；论文中必须写成 converted/public test evaluation，不能称为官方隐藏 leaderboard 结果。

### 10.6 论文结果表和质量门槛

预期表格：

1. 数据与训练协议表：split、通道顺序、normalization、输入尺寸、增强、optimizer、epoch、seed；
2. 主结果表：Single/Dual/legacy/SKG，global mIoU 和次指标的 mean ± std；
3. 组件消融表：B0–B8，保持同一 encoder、通道划分和训练预算；
4. 效率表：参数、FLOPs/延迟、显存、精度；
5. 可视化图：预测、边界错误、知识门控/原型响应和失败案例。

进入下一阶段的门槛：

- Stage 0 同 seed 重复通过；
- 每个 run 能追溯到 commit、manifest、normalizer、seed 和 checkpoint；
- 统一 validation global mIoU 选择规则；
- 主表模型至少 3 个 seed；
- test 未参与任何设计决策；
- 结果同时报告均值、标准差和样本数量，不以单次最好结果替代统计结果。
