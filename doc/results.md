# PMDFNet：MMLSv2 实验结果记录

> 对应实验计划：[`PMDFNet_MMLSv2_Experiment_Plan_v2.md`](PMDFNet_MMLSv2_Experiment_Plan_v2.md)
> 协议：`pmdfnet_mmlsv2_remote_sensing_v5_ce_only_groups_split`
> 更新时间：2026-10-18
> 结果来源：A100-1（`cuda:0`），远程目录 `/data/huilin/projects/PMDFNet/outputs_experiments/pmdfnet_mmlsv2/`

本文档记录当前 protocol v5 的正式实验结果。表中 Test 指 converted MMLSv2 的带公开标签 `test` split，不是原比赛的隐藏 Test2 榜单。

## 1. 当前结论

- 正式矩阵共 19 个协议运行项、3 个随机种子（41/42/43），共 57 个 run；训练和 Test 评估均已完成。
- 主模型是 `L1`，论文展示 ID 为 `M5`：PMDFNet-Small、`3+4` 双流、Add Fusion、CE-only。
- L1/M5 的 Test mIoU 为 **0.8594 ± 0.0034**，是当前 19 个正式运行项中的最高均值。
- 相同输入的单流直接对照 M4 为 `0.8470 ± 0.0100`；L1 相对 M4 的描述性提升为 `+0.0124` mIoU。
- 原始通道分组 L1 为 `0.8594`，B2 为 `0.8520`，B3 为 `0.8460`；当前结果支持“本轮原始分组表现最好”的描述，但尚未完成显著性检验。
- Add Fusion 的 L1 为 `0.8594`，Concat Fusion 的 C2 为 `0.8462`；在当前训练协议下 Add 表现更好。
- CE-only（L1）高于 CE + Dice（L3，`0.8534`）和 Dice-only（L2，`0.8323`）。这只是同一架构、同一数据划分下的描述性结果。
- E5 受控扰动、图像级 bootstrap/配对检验以及固定规则的 Test 定性/FP-FN 案例分析已在 A100 上完成；这些结果支持当前 M4 vs. L1/M5 的配对比较，但不替代跨 seed 的训练随机性汇总，也未对全部消融组合做成对检验。
- CAM 与 GT/Pred 原型 t-SNE 解释性分析已在 A100 上完成；这是定性机制分析，不改变主表指标，也不作为显著性或因果证据。

## 2. 数据、映射与评价边界

### 2.1 数据版本

| 项目 | 当前记录 |
|---|---|
| 训练数据根目录 | `/data/huilin/projects/PMDFNet/data/mmlsv2_mapped` |
| 通道顺序 | Thermal、Slope、DEM、Grayscale、R、G、B |
| Train / Val / Test | 465 / 66 / 133 |
| 图像尺寸 | `128 × 128` |
| 当前 Test | converted MMLSv2 的带 mask 的公开 `test` split |
| 官方隐藏 Test2 | 未用于本轮数值评价 |

主模型的两条输入流为：

- Stream 1：`0,1,2`，Thermal + Slope + DEM；
- Stream 2：`3,4,5,6`，Grayscale + R + G + B。

### 2.2 映射核验

映射报告使用像素绝对误差阈值 `atol=1e-3`。以下百分比是所有比较值中落在阈值内的比例，不等价于逐文件完全相等：

| Split | 图像像素阈值内比例 | Mask |
|---|---:|---|
| train | 99.9986% | 100.0000% |
| val | 99.9999% | 100.0000% |
| test | 99.9966% | 参考 mask 不可用 |

按波段统计如下：

| 波段 | train | val | test |
|---|---:|---:|---:|
| Thermal | 100.0000% | 100.0000% | 100.0000% |
| Slope | 99.9899% | 99.9995% | 99.9765% |
| DEM | 100.0000% | 100.0000% | 100.0000% |
| Grayscale | 100.0000% | 100.0000% | 100.0000% |
| R | 100.0000% | 100.0000% | 100.0000% |
| G | 100.0000% | 100.0000% | 100.0000% |
| B | 100.0000% | 100.0000% | 100.0000% |

由于 Slope 仍有少量超出阈值的值，且 Test 没有对应参考 mask，本次映射报告的 `complete_match` 仍为 `false`。因此论文中不能写成“所有像素逐值完全一致”；映射细节以远程目录中的 `mapping_manifest.json` 和 `mapping_match_report.json` 为准。

## 3. 统一训练与评价协议

| 项目 | 设置 |
|---|---|
| 输入 | 7 通道，`128×128` |
| 训练预算 | 100 epochs |
| Batch size | 32 |
| 优化器 | AdamW |
| 学习率 | `1e-4` |
| Weight decay | `5e-4` |
| Scheduler | cosine |
| Augmentation | `mars`，`aug_prob=0.5`，`mosaic_prob=0.5` |
| Workers | 4 |
| Mixed precision | false |
| Seeds | 41、42、43 |
| Checkpoint | 每个 epoch 验证，按 Val mIoU 保存 `best.pth` |
| Test 预测 | `argmax`，无 TTA、无 Ensemble、无 Test 阈值搜索 |
| Test 主指标 | 所有 Test 像素汇总混淆矩阵后的 global-pixel metrics |
| Test 样本数 | 133，每个 run 5 个 batch |

主协议使用 CE-only；仅 L2 覆盖为 Dice-only，L3 覆盖为代码实现的 CE + Dice。公开 Test 评估使用 `tools/evaluation/evaluate_test_metrics.py`，同时保存 global-pixel 指标和 batch-average 指标，本文表格统一使用前者。

M6/UPerNet 因当前 CUDA `adaptive_avg_pool2d` 反向传播没有确定性实现，协议显式使用 `deterministic=false`；其余 54 个 run 使用 `deterministic=true`。这不是评估失败，但 M6 的逐位可复现性弱于其他配置。

## 4. 正式矩阵完成情况

| 分组 | 正式运行项 | run 数量 | 状态 |
|---|---|---:|---|
| core | M1、M2、M3、M4 | 12 | 已完成 |
| ablations | A1、A2、A5、B2、B3、C2、L1、L2、L3 | 27 | 已完成 |
| extended | M6、M7、M8、M9、M10、M11 | 18 | 已完成 |
| 合计 | 19 个运行项 | **57** | **57/57 完成** |

训练结束后每个 run 均保存 `best.pth`、`last.pth`、`config.json`、`metrics.csv`、日志和运行记录；Test 阶段已为 57 个 run 生成 `test_metrics.json` 和 `results.json`。

为避免重复训练，论文显示 ID 与协议运行 ID 的关系为：

| 论文显示 ID | 实际运行 ID |
|---|---|
| M5、A4、B1、C1 | L1 |
| A3 | M4 |

这些是同一配置的显示别名，不代表额外训练。

## 5. 主基线结果

下表为 3 个 seed 的均值 ± 样本标准差。Val 列是每个 seed 的最佳验证 mIoU 再跨 seed 汇总；Test 指标是全 Test 像素汇总后的 global-pixel 指标。

| ID | 方法 | Val best mIoU | Test mIoU | Test IoU-FG | Test IoU-BG | Test F1 |
|---|---|---:|---:|---:|---:|---:|
| M1 | U-Net + ResNet50 | 0.8135 ± 0.0073 | 0.8295 ± 0.0071 | 0.7790 ± 0.0081 | 0.8799 ± 0.0062 | 0.8757 ± 0.0051 |
| M2 | DeepLabV3+ + ResNet50 | 0.8062 ± 0.0017 | 0.8176 ± 0.0108 | 0.7629 ± 0.0150 | 0.8723 ± 0.0068 | 0.8654 ± 0.0097 |
| M3 | SegFormer MiT-B2 | 0.8260 ± 0.0017 | 0.8276 ± 0.0071 | 0.7757 ± 0.0101 | 0.8796 ± 0.0043 | 0.8737 ± 0.0064 |
| M4 | 单流 ConvNeXt-Small + MLP | 0.8316 ± 0.0081 | 0.8470 ± 0.0100 | 0.7998 ± 0.0132 | 0.8941 ± 0.0068 | 0.8887 ± 0.0081 |
| **M5 = L1** | **PMDFNet-Small，双流 Add，CE-only** | **0.8209 ± 0.0096** | **0.8594 ± 0.0034** | **0.8166 ± 0.0041** | **0.9022 ± 0.0028** | **0.8990 ± 0.0025** |
| M6 | UPerNet + ResNet50 | 0.7997 ± 0.0057 | 0.8100 ± 0.0099 | 0.7524 ± 0.0131 | 0.8677 ± 0.0068 | 0.8587 ± 0.0085 |
| M7 | FPN + ResNet50 | 0.8013 ± 0.0109 | 0.8099 ± 0.0106 | 0.7513 ± 0.0151 | 0.8686 ± 0.0061 | 0.8579 ± 0.0099 |
| M8 | MANet + ResNet50 | 0.8108 ± 0.0065 | 0.8245 ± 0.0050 | 0.7720 ± 0.0072 | 0.8771 ± 0.0028 | 0.8713 ± 0.0046 |
| M9 | U-Net++ + ResNet50 | 0.8149 ± 0.0049 | 0.8121 ± 0.0094 | 0.7532 ± 0.0125 | 0.8710 ± 0.0065 | 0.8592 ± 0.0081 |
| M10 | OCRNet + HRNet-W18 | 0.8153 ± 0.0068 | 0.8214 ± 0.0167 | 0.7675 ± 0.0235 | 0.8753 ± 0.0099 | 0.8683 ± 0.0151 |
| M11 | M3LSNet | 0.8134 ± 0.0022 | 0.8176 ± 0.0027 | 0.7630 ± 0.0042 | 0.8722 ± 0.0024 | 0.8656 ± 0.0027 |

## 6. 消融结果

### 6.1 模态、容量与通道分组

| ID | 配置 | Test mIoU | Test IoU-FG | Test IoU-BG | Test F1 |
|---|---|---:|---:|---:|---:|
| A1 | 光学单流：Grayscale + RGB | 0.8331 ± 0.0054 | 0.7821 ± 0.0076 | 0.8842 ± 0.0037 | 0.8777 ± 0.0048 |
| A2 | 物理单流：Thermal + Slope + DEM | 0.8425 ± 0.0060 | 0.7964 ± 0.0086 | 0.8885 ± 0.0044 | 0.8867 ± 0.0053 |
| A3 | 7 通道单流 | **复用 M4** | **复用 M4** | **复用 M4** | **复用 M4** |
| A4 | 原始双流 | **复用 L1/M5** | **复用 L1/M5** | **复用 L1/M5** | **复用 L1/M5** |
| A5 | 单流 ConvNeXt-Base，近似容量对照 | 0.8307 ± 0.0064 | 0.7796 ± 0.0088 | 0.8818 ± 0.0048 | 0.8761 ± 0.0055 |
| B1 | 原始 3+4 物理分组 | **复用 L1/M5** | **复用 L1/M5** | **复用 L1/M5** | **复用 L1/M5** |
| B2 | RGB vs. 物理 + Grayscale | 0.8520 ± 0.0033 | 0.8066 ± 0.0049 | 0.8974 ± 0.0017 | 0.8929 ± 0.0030 |
| B3 | Thermal + DEM + Grayscale vs. Slope + RGB | 0.8460 ± 0.0199 | 0.7973 ± 0.0286 | 0.8948 ± 0.0112 | 0.8870 ± 0.0178 |

A5 是近似容量对照，不是严格参数量匹配；Params/GFLOPs/延迟/FPS/峰值推理显存已统一测量，见第 8 节，但仍不能将 A5 差异完全归因于模态解耦。

### 6.2 Fusion 与损失

| ID | 变量 | Test mIoU | Test IoU-FG | Test IoU-BG | Test F1 |
|---|---|---:|---:|---:|---:|
| C1 | Add Fusion | **复用 L1/M5** | **复用 L1/M5** | **复用 L1/M5** | **复用 L1/M5** |
| C2 | Concat + 1×1 Conv | 0.8462 ± 0.0077 | 0.7977 ± 0.0104 | 0.8946 ± 0.0052 | 0.8875 ± 0.0065 |
| L1 / M5 | CE-only | **0.8594 ± 0.0034** | **0.8166 ± 0.0041** | **0.9022 ± 0.0028** | **0.8990 ± 0.0025** |
| L2 | Dice-only | 0.8323 ± 0.0054 | 0.7812 ± 0.0099 | 0.8834 ± 0.0008 | 0.8771 ± 0.0062 |
| L3 | CE + Dice | 0.8534 ± 0.0143 | 0.8089 ± 0.0188 | 0.8979 ± 0.0097 | 0.8943 ± 0.0116 |

## 7. Precision 与 Recall 补充结果

| ID | Precision | Recall |
|---|---:|---:|
| M1 | 0.8729 ± 0.0135 | 0.8788 ± 0.0071 |
| M2 | 0.8702 ± 0.0095 | 0.8610 ± 0.0215 |
| M3 | 0.8782 ± 0.0068 | 0.8694 ± 0.0160 |
| M4 | 0.8987 ± 0.0053 | 0.8791 ± 0.0116 |
| M5 = L1 | **0.9014 ± 0.0075** | **0.8967 ± 0.0054** |
| M6 | 0.8697 ± 0.0073 | 0.8480 ± 0.0127 |
| M7 | 0.8763 ± 0.0043 | 0.8406 ± 0.0208 |
| M8 | 0.8740 ± 0.0039 | 0.8688 ± 0.0118 |
| M9 | 0.8849 ± 0.0119 | 0.8351 ± 0.0155 |
| M10 | 0.8743 ± 0.0022 | 0.8628 ± 0.0314 |
| M11 | 0.8696 ± 0.0129 | 0.8619 ± 0.0159 |
| A1 | 0.8875 ± 0.0112 | 0.8683 ± 0.0153 |
| A2 | 0.8760 ± 0.0182 | 0.8983 ± 0.0243 |
| A5 | 0.8808 ± 0.0154 | 0.8719 ± 0.0189 |
| B2 | 0.9004 ± 0.0033 | 0.8856 ± 0.0090 |
| B3 | 0.9089 ± 0.0125 | 0.8672 ± 0.0430 |
| C2 | 0.9080 ± 0.0092 | 0.8680 ± 0.0135 |
| L2 | 0.8857 ± 0.0197 | 0.8697 ± 0.0317 |
| L3 | 0.8964 ± 0.0069 | 0.8921 ± 0.0166 |

## 8. 统一复杂度与效率测量

测量使用各正式 run 的 `best.pth`，输入为 `1×7×128×128`，设备为 NVIDIA A100 80GB PCIe，`eval + inference_mode`，warm-up 20 次，计时 100 次，前向同步计时。GFLOPs 使用 `torch.profiler` 的支持算子统计；峰值显存为推理阶段 CUDA allocated/reserved，不是训练反向传播峰值。

| ID | Params (M) | GFLOPs | 延迟 (ms/image) | FPS | 峰值推理显存 allocated (GiB) | 峰值推理显存 reserved (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 32.5338 | 5.4238 | 6.6597 | 150.16 | 0.1540 | 0.1797 |
| M2 | 26.6904 | 4.6744 | 8.6000 | 116.28 | 0.1160 | 0.1230 |
| M3 | 24.7352 | 2.7030 | 25.5113 | 39.20 | 0.1359 | 0.1465 |
| M4 | 50.0921 | 6.3177 | 14.3013 | 69.92 | 0.2306 | 0.2656 |
| **M5 = L1** | **99.5407** | **11.9812** | **28.6428** | **34.91** | **0.4175** | **0.4688** |
| M6 | 37.2975 | 9.7044 | 7.0759 | 141.33 | 0.2351 | 0.2637 |
| M7 | 26.1287 | 3.9965 | 6.4304 | 155.51 | 0.1431 | 0.1582 |
| M8 | 147.4527 | 9.3941 | 8.7927 | 113.73 | 0.7281 | 0.7598 |
| M9 | 48.9984 | 28.8280 | 8.0349 | 124.46 | 0.2523 | 0.3145 |
| M10 | 13.0181 | 51.5493 | 35.6207 | 28.07 | 0.3500 | 0.3906 |
| M11 | 9.1204 | 16.6363 | 11.6522 | 85.82 | 0.1202 | 0.1582 |

M11 在当前 A100 环境实际使用 `gated_conv_simulator`；测量脚本中的自动元数据导入提示失败，但已通过独立模块探针确认，Params/GFLOPs/速度/显存数值不受影响。原始机器可复核文件为 `outputs_experiments/complexity/complexity_results.json` 和 `.csv`。

## 9. E5 受控扰动、配对统计与定性分析

这些补充实验不直接并入主结果表；正文只保留实验范围、完成状态和结果索引，详细数值与图像保存在以下文件中。

| 补充实验 | 状态 | 简介与结果路径 |
|---|---|---|
| E5 受控扰动与模态缺失 | 已完成 | M4 vs L1/M5，3 个 seed，clean + 五类条件，共 72 条记录；原始汇总见 [`robustness_results.json`](../outputs_experiments/robustness/robustness_results.json) / [`robustness_results.csv`](../outputs_experiments/robustness/robustness_results.csv)，逐图像 TP/FP/FN/TN 见 [`robustness_per_image.jsonl`](../outputs_experiments/robustness/robustness_per_image.jsonl)。 |
| 图像级 bootstrap 与配对检验 | 已完成 | 按相同 seed、条件和 Test 图像配对，执行 10,000 次 bootstrap 与 10,000 次 sign-flip 检验；详细结果见 [`robustness_paired_stats.json`](../outputs_experiments/robustness/analysis/robustness_paired_stats.json) / [`robustness_paired_stats.csv`](../outputs_experiments/robustness/analysis/robustness_paired_stats.csv)。 |
| FP/FN 与失败案例分析 | 已完成 | 固定 seed 41 的 clean Test，按双方优劣、双方困难和最大差异规则选择 14 张图；选择清单见 [`failure_cases.json`](../outputs_experiments/robustness/analysis/failure_cases.json)。 |
| Test 定性可视化 | 已完成 | 每张图包含 RGB、GT、M4、L1/M5 预测及两者 FP/FN 图；PNG 和渲染清单见 [`qualitative_cases.json`](../outputs_experiments/robustness/qualitative/qualitative_cases.json) 及同目录文件。 |
| CAM 分析 | 已完成 | A100 上使用 M4 与 L1/M5、seed 41、公开 Test 固定抽取的 12 张图，前景类 `class_id=1`、`gt_mask` 目标和自动 decoder feature layer；逐模型 CAM、RGB/GT/Pred/FP-FN 与 metadata 见 [`M4_s41`](../outputs_experiments/interpretability/cam/M4_s41/) / [`L1_s41`](../outputs_experiments/interpretability/cam/L1_s41/)。 |
| CAM 对比图 | 已完成 | 将两模型同一批样本排版为 `RGB | GT | Pred | CAM | FP/FN` 对比图；见 [`cam_comparison_s41.png`](../outputs_experiments/interpretability/cam/cam_comparison_s41.png) 和 [`cam_comparison_s41_manifest.json`](../outputs_experiments/interpretability/cam/cam_comparison_s41_manifest.json)。 |
| GT/Pred 特征原型 t-SNE | 已完成 | 共 448 个区域原型；M4/L1 decoder 特征均为 256 维，经各模型 PCA 对齐到 32 维后联合 t-SNE；图和坐标见 [`tsne/M4,L1_s41`](../outputs_experiments/interpretability/tsne/M4,L1_s41/)，包括 `tsne_all.png`、`tsne_gt.png`、`tsne_pred.png`、`prototypes.csv` 和 `tsne_coordinates.csv`。 |

执行入口：[`run_robustness_eval.sh`](../scripts/run_robustness_eval.sh)、[`run_robustness_analysis.sh`](../scripts/run_robustness_analysis.sh) 和 [`run_interpretability.sh`](../scripts/run_interpretability.sh)。固定扰动协议：[`mmlsv2_robustness_protocol.json`](../configs/mmlsv2_robustness_protocol.json)；解释性分析规则见实验计划 E7.1。

## 10. 每个 seed 的 Test mIoU

| ID | seed 41 | seed 42 | seed 43 |
|---|---:|---:|---:|
| M1 | 0.8296 | 0.8365 | 0.8223 |
| M2 | 0.8210 | 0.8263 | 0.8055 |
| M3 | 0.8286 | 0.8342 | 0.8201 |
| M4 | 0.8582 | 0.8438 | 0.8389 |
| M5 = L1 | 0.8634 | 0.8576 | 0.8573 |
| M6 | 0.8215 | 0.8038 | 0.8049 |
| M7 | 0.8181 | 0.7980 | 0.8137 |
| M8 | 0.8267 | 0.8188 | 0.8281 |
| M9 | 0.8035 | 0.8221 | 0.8108 |
| M10 | 0.8029 | 0.8259 | 0.8353 |
| M11 | 0.8159 | 0.8207 | 0.8163 |
| A1 | 0.8378 | 0.8344 | 0.8272 |
| A2 | 0.8493 | 0.8381 | 0.8399 |
| A5 | 0.8306 | 0.8371 | 0.8243 |
| B2 | 0.8543 | 0.8482 | 0.8536 |
| B3 | 0.8249 | 0.8486 | 0.8645 |
| C2 | 0.8545 | 0.8391 | 0.8450 |
| L2 | 0.8279 | 0.8306 | 0.8383 |
| L3 | 0.8635 | 0.8596 | 0.8371 |

没有删除或挑选 seed；所有 41/42/43 均纳入均值和标准差计算。

## 11. 结果解读边界

### 11.1 双流与模态贡献

L1/M5 比直接单流对照 M4 高 `0.0124` mIoU，A1 和 A2 均低于 L1，说明在当前协议下两类输入共同使用具有描述性收益。A5 的结果也低于 L1，但 A5 只是近似容量对照；结合复杂度表可见 L1 参数量和计算量明显更高，因此不能把差异单独解释为双流结构本身的因果收益。

### 11.2 通道分组

L1/B1 的 `0.8594` 高于 B2 的 `0.8520` 和 B3 的 `0.8460`。当前结果与原始物理分组的设计动机一致，但 B3 跨 seed 波动较大，且尚未完成配对统计检验，因此结论应写成“本轮三 seed 实验中原始分组表现最好”，不写成普遍最优。

### 11.3 Fusion 与损失

L1 的 Add Fusion 高于 C2 的 Concat Fusion；CE-only 高于 L3 和 L2。由于这些结果来自单一固定数据划分，结论限于当前实现和训练预算。

### 11.4 基线比较

M4 是当前最强的单流直接对照，M1/M3/M8 等其他基线低于 M4；M6、M7、M9、M10、M11 的均值也低于 L1。所有比较使用同一 Test split、相同 3 个 seed 和统一的 global-pixel 指标，但 M6 的确定性设置例外；复杂度表是 seed 41 的单次推理测量，不代表三 seed 的速度均值。

## 12. 结果文件与复现记录

每个正式 run 的目录结构包含：

```text
outputs_experiments/pmdfnet_mmlsv2/
  {experiment}_s{41,42,43}_{timestamp}/
    config.json
    command.txt
    metrics.csv
    logs/train.log
    checkpoints/best.pth
    checkpoints/last.pth
    results.json
    test_metrics.json
```

当前核对结果：

- `summary.csv`：57 条 completed 记录；
- `test_metrics.json`：57 份；
- `results.json`：57 份；
- Test 使用各 run 的 `best.pth`，选择依据是 Val mIoU；
- 评估脚本：[`scripts/run_test_eval.sh`](../scripts/run_test_eval.sh)；
- 测试评估实现：[`tools/evaluation/evaluate_test_metrics.py`](../tools/evaluation/evaluate_test_metrics.py)。

补充评价入口与已归档输出：

- 复杂度：[`scripts/run_complexity_eval.sh`](../scripts/run_complexity_eval.sh)，实现见 [`tools/evaluation/measure_model_complexity.py`](../tools/evaluation/measure_model_complexity.py)；
- E5 鲁棒性：[`scripts/run_robustness_eval.sh`](../scripts/run_robustness_eval.sh)，协议见 [`configs/mmlsv2_robustness_protocol.json`](../configs/mmlsv2_robustness_protocol.json)；
- 配对统计和失败案例：[`scripts/run_robustness_analysis.sh`](../scripts/run_robustness_analysis.sh)，实现见 [`tools/evaluation/analyze_robustness.py`](../tools/evaluation/analyze_robustness.py)；
- 定性图：[`tools/evaluation/render_qualitative_cases.py`](../tools/evaluation/render_qualitative_cases.py)。
- CAM 与原型 t-SNE：[`scripts/run_interpretability.sh`](../scripts/run_interpretability.sh)，实现见 [`tools/evaluation/cam_analysis.py`](../tools/evaluation/cam_analysis.py)、[`tools/evaluation/tsne_prototype_analysis.py`](../tools/evaluation/tsne_prototype_analysis.py) 和 [`tools/evaluation/make_cam_comparison_figure.py`](../tools/evaluation/make_cam_comparison_figure.py)；结果见 `../outputs_experiments/interpretability/`。

历史 57 个训练/Test run 的记录使用 Git commit `653b81dc76db4d0774d2126cad98641ca32390a9`，但 `summary.csv` 中 57 个 run 的 `git_dirty` 均为 `true`，因此这些结果仍应视为 dirty 工作树快照。复杂度、E5 鲁棒性、配对统计和定性渲染是在 A100 的同步 commit `f5c3374a00985749d751e17bef4120a2cc43578b` 上执行；本次 CAM/t-SNE 分析是在 clean commit `162d9a23d4031308057ed4654a298154ec08966e` 上执行。投稿或最终归档前仍应在 clean commit 下重新冻结协议并记录完整 hash。

M11 的配置明确为 `pretrain=false`，但历史本轮结果文件没有单独记录 `mamba_ssm` selective-scan 与 fallback 实现路径。新训练的 `runtime.json`、复杂度结果和鲁棒性结果会记录该字段；历史 57 个 run 仍不能事后假定其后端。

## 13. 尚未完成的工作

- 公开数据集的来源、引用、通道和映射规则需继续随论文材料归档；本轮不把数据泄漏、场景独立性和完整数据审计作为阻塞项；
- 当前只对 M4 vs L1/M5 执行了 E5 的图像级配对统计；若论文需要对 B2/B3/C2/L2/L3 等所有消融做成对检验，需要另外指定比较集合并运行同一分析脚本；
- 受控扰动的曲线图仍需在论文排版阶段从 `robustness_results.csv` 绘制，原始指标和定性 PNG 已归档；
- 历史 57 个 run 的 M6 非确定性影响的补充记录；M11 当前 A100 环境已确认使用 fallback simulator，但历史 run 尚未在 `runtime.json` 中显式记录；
- 在 clean commit 下重新冻结，或归档历史工作树的完整代码差异、未跟踪文件、环境信息和原始输出；
- 官方隐藏 Test2 的提交或榜单结果。当前结果不能替代官方隐藏测试评价。

在上述工作完成前，本文档支持的是“当前 protocol v5 下的 57-run 标准 Test 结果记录”，不是最终论文的完整 Results/Discussion 定稿。
