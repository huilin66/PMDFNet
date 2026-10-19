# PMDFNet：Physically Motivated Dual-Stream Fusion for Multimodal Landslide Segmentation（MMLSv2 期刊实验计划 v2，按 protocol v5 对齐）

本文件中的 PMDFNet 是项目论文统一名称；旧版代码中的 `dual_segformer_*` 仅保留为历史兼容键。

> **目标期刊**：MDPI *Remote Sensing*  
> **研究定位**：保留既有 PMDFNet，不增加新的主网络模块；通过公平基线、模态分组消融、统计重复和可复现的数据转换协议，检验双流设计的实际价值。  
> **主数据集**：`mmlsv2_mapped_mars_ls`（由 MMLSv2 转换/映射获得；论文中称为 *converted MMLSv2*，**不是新的独立数据集**）。  
> **主模型**：`PMDFNet-Small`，两条独立 ConvNeXt-Small 编码器 + 四尺度 Add Fusion + SegFormer All-MLP Decoder；当前 protocol v5 的主损失为 **CE-only**。**在开始正式测试集比较前冻结此选择**。
> **对照模型**：单流 ConvNeXt-Small + 同类 MLP Decoder；其他经典分割基线。  
> **明确排除**：TRB-Net（缺少可重复执行的代码/权重/协议，本计划不将其列为必做对比）。  
> **实验范围**：仅包含 MMLSv2；Landslide4Sense 另立计划。  
> **状态**：以 `configs/mmlsv2_paper_protocol.json`（`pmdfnet_mmlsv2_remote_sensing_v5_ce_only_groups_split`）和当前 `scripts/`、`tools/evaluation/` 实现为执行蓝本；参数、样本映射与预训练来源须经 E0/实现核验后最终冻结。

> **实现对齐说明（2026-10-19）**：本文档中旧版计划与当前代码不一致的地方，以 protocol v5、`configs/mmlsv2_robustness_protocol.json` 和实际运行脚本为准。当前正式矩阵使用 batch size `32`，主模型 M5 由协议运行项 `L1` 实现并使用 CE-only；`L2`、`L3` 仅作为损失消融覆盖项。当前公开 Test 评估脚本使用 MARS Test 统计量。主基线复杂度由 `scripts/run_complexity_eval.sh` 统一测量；E5 已由 `scripts/run_robustness_eval.sh` 在 A100 上执行，图像级统计和定性渲染由 `scripts/run_robustness_analysis.sh` 完成。E7 已在 A100 上完成四套冻结 Test 分析：历史固定样本的 M4 与 L1/M5 分析、按 L1/M5 相对 M1/M2/M3/M10/M11 逐图像 mIoU 优势筛选并冻结样本清单的 M1/M3/M10/M11/M5 性能条件化 CAM、同一批优势样本的 RGB/Slope/模型预测/GT 实例对比，以及覆盖全部 133 张公开 Test 图像的既有 M1–M4/L1 GT 前景/背景 t-SNE；当前两类对比图均采用 `RGB | Slope | M1 | M3 | M10 | M11 | M5 | GT`，结果归档于 `outputs_experiments/interpretability_selected_m1_m3_m10_m11_m5/`。新增的单通道训练协议已在 A100-1 完成 S1–S4 × `41/42/43` 共 12 个 run；Test 诊断已完成 M4/L1 共 6 个冻结 run 和 S1–S4 共 12 个冻结 run，结果分别归档于 `outputs_experiments/test_diagnostics/` 与 `outputs_experiments/test_diagnostics_single_channel/`。

## 0. 研究问题、边界与待验证主张

文章拟验证的中心假设（**不是已被实验结果证实的结论**）：

> A physically motivated dual-stream decomposition can improve segmentation from heterogeneous optical and terrain-related Martian remote-sensing inputs without requiring complex cross-modal interaction modules.

对应研究问题：

- **RQ1：效果**——在统一 MMLSv2 协议下，PMDFNet-Small 相较代表性分割方法能否获得稳定的性能改善？
- **RQ2：容量控制**——双流优点是否超出“两个编码器导致参数量增加”的影响？
- **RQ3：输入分组**——原始物理动机的 `3+4` 通道分组是否优于相同通道宽度的其他分组？
- **RQ4：设计合理性**——简单 Add Fusion 与 CE-only 主训练协议是否能提供有竞争力的准确率—复杂度权衡；CE + Dice 作为 L3 对照时是否带来额外收益？
- **RQ5：可靠性**——性能差异是否跨随机种子稳定，在预设的受控扰动和模态缺失下表现如何？

**结论边界**：即便取得性能优势，也只可声称“物理含义驱动的通道分组（physically motivated modality grouping）”，不应将它描述为通过物理方程约束训练的严格 *physics-informed learning*。如果容量匹配对照或替代分组更好，须如实收缩方法主张。

### 0.1 主模型为何选择 Small

本轮以**当前已验证的 `PMDFNet-Small`**为主模型，配套使用 ConvNeXt-Small 单流直接对照；当前 protocol v5 不包含 PMDFNet-Tiny 正式运行项，不允许依据 Test 表现事后切换论文主模型。

应在实验记录中分别交代：

1. **原 Workshop 稿**：Table 2 同时包含 Tiny 和 Small，Tiny 具有完整 Test1/Test2 报告，而 Small 在该表仅报告 Test2；不能笼统宣称“原稿以 Small 为唯一主模型”。
2. **本轮期刊研究**：研究开始前选择 Small 作为主配置的原因、已验证的训练配置及代码版本；保存主模型冻结记录。
3. **Tiny 扩展**：如未来加入，必须作为新的版本化辅助实验预先规定，不作为后验挑选的主结果。

### 0.2 固定网络定义

原始 7 通道（以 **E0 最终核实的索引映射**为准）：

| 分支 | 物理含义 | 通道数 |
|---|---|---:|
| Stream 1 | Thermal Inertia + Slope + DEM | 3 |
| Stream 2 | Grayscale + R + G + B | 4 |

两条权重独立的 ConvNeXt-Small 在相应尺度上产生特征，逐尺度相加后交给 SegFormer All-MLP Decoder；当前正式 protocol 的默认损失为 **CE-only**。`L2` 使用 Dice-only，`L3` 使用实现中的 CE + Dice；其余基线和消融均使用 CE-only。主体网络不修改。对照实验允许更改**基线**骨干规模、首层输入通道数、融合消融算子或损失项，但须声明这些是实验对照而非最终方法升级。

---

## E0. 数据版本与转换映射记录【保留必要的来源与映射记录】

**目的**：建立 `mmlsv2_mapped_mars_ls` 可追溯、可复现的来源链，明确公开数据集来源、输入顺序和映射规则。由于本项目使用公开发布数据集，数据泄漏、场景独立性和完整数据审计不作为本轮实验的阻塞项；不在本计划中新增独立审计工程。

### E0.1 转换与通道审计

- [ ] 明确原始 MMLSv2 下载来源、发布日期/commit/数据发布版本；记录本地转换脚本的 Git commit。
- [ ] 列出转换前后的 **7 个通道顺序、名称、物理单位、数据类型、NoData、取值范围**。
- [ ] 记录几何对齐、重采样、裁剪、插值、异常值处理、归一化及映射规则。
- [ ] 计算 Train / Val / Test 各通道的 min/max、均值、标准差、分位数和直方图。
- [ ] 当前 protocol v5 的训练/验证配置使用 `normalization=train`，即 MARS Train 统计量用于 Train/Val；公开 Test 评估脚本 `tools/evaluation/evaluate_test_metrics.py` 使用 `MARS_MEAN_TEST/MARS_STD_TEST`。Test 统计量不得参与训练参数拟合、checkpoint 选择或超参数选择，并须在论文中明确这一实现口径。
- [ ] 所有归一化参数、裁剪阈值或校正系数必须明确来源；当前实现中的 Train/Test 统计量版本和生成方式必须随协议存档。
- [ ] 校验每张图像与 GT 的配对、mask 值域（前景/背景/ignore）、空间尺寸、波段次序。

### E0.2 数据泄漏与场景独立性

本轮不实施单独的数据泄漏、scene-level 独立性和完整数据审计；公开数据集的官方来源、划分名称和许可/引用信息在论文中照常注明。不得把本轮结果外推为已证明跨场景泛化，也不把“公开”误写成“天然无泄漏”；但这些内容不阻塞当前训练、Test、复杂度或鲁棒性实验。

### E0.3 最低交付物

- `channel_mapping.yaml`（如需论文补充材料）：各通道来源索引、单位与映射方法；
- 固定 `train/val/test` 文件列表和完整转换脚本；当前映射程序生成的 `mapping_manifest.json` 与 `mapping_match_report.json` 作为映射核验附件保存。
- 训练使用 `.env` 解析出的 `MMLSV2_DATA_ROOT`/`DATA_ROOT` 映射数据根目录；`MARS_LSC_ROOT` 仅作为映射比较参考，不能作为训练输入。参考数据版本必须与映射时使用的版本一致并记录。

**论文写法**：*Converted MMLSv2*（转换后的 MMLSv2）。只有证据支持时才能说明“与竞赛数据基本一致”；不得据此称其为新数据集。

---

## 1. 全实验通用协议【必须先冻结】

### 1.1 数据与模型选择

1. **Train**：仅用于优化模型参数、拟合归一化统计和训练增强。
2. **Validation**：用于 checkpoint 和预定义超参数选择；当前训练器每个 epoch 验证一次，并按全局 mIoU 保存 `best.pth`。当前测试评估固定使用 `argmax`，没有额外的 Val 阈值搜索；受控扰动的**类型/强度/生成规则也先在这里确定并冻结**。
3. **Test**：当前使用 converted MMLSv2 中带公开 mask 的 `test` split，不等同于官方隐藏竞赛 Test2；仅在全部协议和模型配置冻结后执行最终评价。不得依据 Test 分数重新选择主模型、调整分组、阈值、扰动强度或模型配置。
4. 数据或映射后续发生变化，须生成新的版本号并在实验日志中说明；此前 Test 评估不能当作未使用。

### 1.2 基础训练设置

当前正式执行以 protocol v5 和现有代码为准：`128×128`、`100 epochs`、`batch size=32`、`AdamW`、初始学习率 `1e-4`、weight decay `5e-4`、cosine schedule。训练使用 `MarsAugmentor`，`aug_prob=0.5`、`mosaic_prob=0.5`，包括水平/垂直翻转、90° 倍数旋转、裁剪和 Mosaic 等操作。Train/Val 使用 `normalization=train`；公开 Test 评估按当前脚本使用 MARS Test 统计量。

> **实现注意**：JSON 配置文件的说明文字仍保留“Train 统计量应用到 validation/test”的历史表述，但当前 `tools/evaluation/evaluate_test_metrics.py` 实际读取 `MARS_MEAN_TEST/MARS_STD_TEST`。本计划记录实际评估代码行为；正式论文提交前必须将该口径作为明确的 protocol 决策，并同步配置说明或代码。

- 训练 DataLoader 当前使用 `drop_last=True`；batch size 为 32 时每个 epoch 仅使用完整 batch，需在运行记录中保留该设置。
- 除 M6 外，正式运行项使用 `deterministic=true`；M6 的 UPerNet 因 CUDA `adaptive_avg_pool2d` 反向传播缺少确定性实现，协议显式设置 `deterministic=false`。所有正式 seeds 仍固定为 `41, 42, 43`，默认 `num_workers=4`，默认设备为 `cuda:0`，不使用 mixed precision；M6 的非确定性例外必须记录并披露。
- 当前 protocol 的通用损失为 **CE-only**；仅 `L2` 覆盖为 Dice-only，`L3` 覆盖为 CE + Dice（代码实现为 `0.5*CE + 0.5*Dice`）。
- 默认 `pretrain=true`；`M11` 的协议项显式使用 `pretrain=false`，并记录其实际 M3LSNet selective-scan 或 fallback 实现路径。

- 主方法 **Small 与单流 Small** 的优化配置和训练预算尽可能一致。
- 单独记录每个模型的初始化（ImageNet/其他预训练）、首层通道适配方法和可训练参数。
- 主表仅用**单模型、无 TTA、无 Ensemble、无额外 Test 分布校正**的结果；TTA 单独补充。
- 当前二分类预测统一使用 `torch.argmax`；本轮不执行额外概率阈值搜索。若未来引入概率阈值，必须另行建立版本化协议，并不得覆盖本轮既有 Test 结果。
- 正式实验随机种子固定为 `41, 42, 43`，在正式运行后不再依据结果更改。

### 1.3 统一评价口径

| 指标 | 定义/用途 |
|---|---|
| mIoU ↑ | 由全测试集累计混淆矩阵分别计算前景/背景 IoU 后取均值；**主要指标** |
| Foreground IoU ↑ | 滑坡类别 IoU；**主要指标** |
| Precision / Recall / F1 ↑ | 均以 Landslide 为正类 |
| Params (M) ↓ | 推理模型总参数数 |
| GFLOPs ↓ | 同一 `128×128` 输入、相同计量工具 |
| FPS / latency ↓ | 同一 GPU、batch、precision、warm-up、计时策略（FPS ↑，latency ↓） |

**不要混淆**：全数据集混淆矩阵汇总的 mIoU 与逐图像 mIoU 的均值不同，论文和统计检验必须标明所用定义。当前评估至少保存 TP/FP/FN/TN、`argmax` 预测规则和全局/批次平均两套指标，便于重算；如需保存预测概率，须在运行协议中另行声明。

---

## E1. 主实验：公平基线对比【必须】

| ID | 方法 | Backbone / 架构 | 输入 | 阶段 |
|---|---|---|---|---|
| M1 | U-Net | 经典 U-Net + ResNet50 | 全部 7 通道 | core，CE-only |
| M2 | DeepLabV3+ | DeepLabV3+ + ResNet50 | 全部 7 通道 | core，CE-only |
| M3 | SegFormer | MiT-B2 | 全部 7 通道 | core，CE-only |
| M4 | **Single-stream ConvNeXt-Small + MLP Decoder** | ConvNeXt-Small | 全部 7 通道 | **core，直接对照** |
| M5 | **PMDFNet-Small（协议运行项 L1）** | ConvNeXt-Small ×2 + Add Fusion | 原始 3+4 通道 | **ablations，主方法，CE-only** |
| M6 | UPerNet | ResNet50 | 全部 7 通道 | extended，CE-only，非确定性例外 |
| M7 | FPN | ResNet50 | 全部 7 通道 | extended，CE-only |
| M8 | MANet | ResNet50 | 全部 7 通道 | extended，CE-only |
| M9 | U-Net++ | ResNet50 | 全部 7 通道 | extended，CE-only |
| M10 | OCRNet | HRNet-W18，含 auxiliary head | 全部 7 通道 | extended，CE 主/辅助损失 |
| M11 | M3LSNet | M3LSNet，自定义 Mamba/VSSD 风格编码器 | 全部 7 通道 | extended，`pretrain=false` |

**命名提醒**：M4 是 ConvNeXt 编码器配 All-MLP Decoder 的单流对照，不能把它写成原始 MiT 架构的官方 SegFormer。当前配置没有 PMDFNet-Tiny；M8 是 MANet + ResNet50，不是 Tiny 版本。

主结果至少报告 mIoU、Foreground IoU、F1、Params、GFLOPs；当前正式矩阵的 19 个协议运行项全部使用 `41/42/43` 三个 seed，共 57 个 run，并报告 mean±std。论文展示的 M5 结果来自 L1，不新增单独的 M5 训练；测试集仅在冻结后评估。论文正文不把原 Workshop 的带 TTA 比赛分数与本地无 TTA 结果混到同一公平比较列。

**交付**：`Table 2. Comparison with segmentation baselines on converted MMLSv2`。

---

## E2. 单流/双流与模态贡献消融【必须】

| ID | 网络 | 输入 | 对照目的 | 复用 |
|---|---|---|---|---|
| A1 | 单流 ConvNeXt-Small + MLP Decoder | RGB + Grayscale（4） | 仅光学信息 | 新训练 |
| A2 | 单流 ConvNeXt-Small + MLP Decoder | Thermal + Slope + DEM（3） | 仅地形/热物理信息 | 新训练 |
| A3 | 单流 ConvNeXt-Small + MLP Decoder | 7 通道 | Early Fusion | **M4（显示别名）** |
| A4 | PMDFNet-Small | 原始 3+4 | 原始双流 | **L1/M5（显示别名）** |
| A5 | **近似容量对照：单流 ConvNeXt + MLP Decoder** | 7 通道 | 控制容量/参数量混杂 | 新训练 |

A5 当前固定为单流 ConvNeXt-Base + MLP Decoder。必须实测总参数量和 GFLOPs；如无法与 M5 严格匹配，应报告百分比偏差，并将比较称为“近似容量对照”，不写成“参数完全相同”。

**主要比较**：`A4 vs A3`（双流增益）、`A4 vs A5`（排除纯容量解释）、`A1/A2 vs A3/A4`（模态互补性）。A3、A4 只用于论文展示命名，不产生新的训练目录。

**交付**：`Table 3. Effect of dual-stream encoding and input modalities`。

---

## E3. 通道分组消融【必须】

**严格控制**：所有方案都使用同一 `PMDFNet-Small`、相同融合和 Decoder，**Stream 1 固定 3 通道，Stream 2 固定 4 通道**。除首层输入映射外，不调整网络配置。

| ID | Stream 1：3 通道 | Stream 2：4 通道 | 理由 | 阶段 |
|---|---|---|---|---|
| B1 | Thermal + Slope + DEM | Grayscale + RGB | 原始物理分组；**复用 L1/M5** | 必须 |
| B2 | RGB | Thermal + Slope + DEM + Grayscale | 将 RGB 单独成组；检验灰度归属 | 必须 |
| B3 | Thermal + DEM + Grayscale | Slope + RGB | 非原始混合分组 | 必须 |
| B4 | Slope + DEM + Grayscale | Thermal + RGB | 另一种严格 3+4 混合分组 | 当前 protocol 未纳入 |

说明：旧版 B4（`Slope + Grayscale + RGB` vs `Thermal + DEM`）实际是 **5+2**，不能用于严格 3+4 对比。上表给出的新 B4（`Thermal + RGB` 与 `Slope + DEM + Grayscale`）仅作为未来可选的严格 3+4 方案，不属于当前 protocol v5，当前不训练。

- 当前 protocol v5 正式训练 B2/B3，B1 直接复用 L1/M5；B4 不在 19 个正式运行项中，如需加入必须建立新的协议版本，不允许依据 Test 结果临时增添有利分组。
- 比较 B1/B2/B3，观察 B1 是否在多个 seed 上稳定占优。
- 若 B1 不占优，论文主张应从“物理分组最优”收缩为“双流分组对输入配置敏感，原始分组具有实用性”。

**交付**：`Table 4. Ablation of channel grouping strategies`。

---

### E3.1 单通道移除补充消融【补充协议，不并入 57 个正式 run】

为补充回答“每个输入波段是否真正有贡献”，新增独立协议
`configs/mmlsv2_single_channel_protocol.json`。它沿用 PMDFNet-Small、Add Fusion、CE-only、
batch size `32`、`41/42/43` seeds 和正式训练流程，只在对应语义分支中移除一个源波段：

| ID | 移除波段 | Stream 1 | Stream 2 |
|---|---|---|---|
| S1 | Thermal | Slope + DEM | Grayscale + RGB |
| S2 | Slope | Thermal + DEM | Grayscale + RGB |
| S3 | DEM | Thermal + Slope | Grayscale + RGB |
| S4 | Grayscale | Thermal + Slope + DEM | RGB |

训练入口为 `scripts/run_single_channel_ablation.sh`，共 `4 × 3 = 12` 个补充 run。
这 12 个 run 不改变 protocol v5 的 57-run 主矩阵，不得在主表中隐式替换 B1/L1；应作为
Supplementary Table 单独报告，并同时给出 Params/GFLOPs，避免把去掉波段后的容量变化误解为
纯输入贡献。输入 TIFF 仍按固定 7 通道读取，具体波段是否进入模型由 `channels1/channels2`
显式控制。

执行状态：已在 A100-1 完成 S1–S4 × `41/42/43` 的 12 个训练 run；结果根目录为
`/data/huilin/projects/PMDFNet/outputs_experiments/pmdfnet_mmlsv2_single_channel/`，其中
`summary.csv` 记录 12 条 completed 记录，`launcher.log` 记录完整执行过程。

**交付**：Supplementary Table S2，单通道移除的 Test mIoU、Foreground IoU、F1、Boundary F1
和复杂度；必要时附逐 seed 结果。

---

## E4. 融合与损失消融【当前 protocol v5 已纳入正式矩阵】

| ID | 只改变的变量 | 配置 | 复用 |
|---|---|---|---|
| C1 | Fusion | 四尺度逐元素 Add | **L1/M5** |
| C2 | Fusion | Concat + 1×1 Conv，统一回原通道数 | 新训练 |
| L1 | Loss | CE only | **主模型 M5，训练一次** |
| L2 | Loss | Dice only | 新训练 |
| L3 | Loss | CE + Dice | 新训练 |

- C1/C2 的参数量可能不同，必须报告额外 Params 和 GFLOPs。
- Loss 对照固定 Backbone、输入、融合与训练协议。L1 是主模型 M5，同时作为 CE-only loss row；L3 是独立的 CE + Dice loss row，不能再将 L3 作为 M5 的复用项。
- 这些实验验证**原设计选择**，不意味着将 Concat 写成论文方法升级。

**交付**：`Table 5. Fusion and loss ablation`。

---

## E5. 受控扰动及模态缺失【第二轮，已执行】

当前 `scripts/run_core.sh`、`run_ablations.sh` 和 `run_extended.sh` 的 57-run 正式矩阵不包含 E5；E5 是冻结 checkpoint 后的独立 Test 评价，不重新训练。代码入口为 `scripts/run_robustness_eval.sh`，固定协议为 `configs/mmlsv2_robustness_protocol.json`，使用 JSON 而不是旧计划中尚未落地的 YAML。已在 A100 上完成 M4 vs L1/M5、3 个 seed、clean + 五类条件，共 72 条条件记录。

**先 Val 确定协议，再冻结，最后 Test 一次性评价。** 不从 Test 结果决定扰动类别、等级、随机种子、选择哪张图或采用哪个模型。

| 扰动 | 对象 | 验证目标 |
|---|---|---|
| DEM offset | DEM | 地形测量系统偏差敏感性 |
| DEM Gaussian noise | DEM | 地形测量随机噪声敏感性 |
| Optical Gaussian noise | Grayscale + RGB | 光学传感器噪声敏感性 |
| Optical modality missing | Grayscale + RGB | 光学模态依赖程度 |
| Physical modality missing | Thermal + Slope + DEM | 地形与热物理模态依赖程度 |

当前实现统一比较 **M4（单流 Small） vs M5（双流 Small，存储项 L1）**，保持 clean/perturbed 模型权重不变。协议预先固定三个 DEM/光学噪声等级（轻/中/重），并将偏置和噪声定义在 MARS Test normalization 之后；模态缺失将指定通道置为归一化空间的 0。记录协议中的 RNG seed，并保存 `robustness_protocol.json` 的 SHA-256。

报告 clean 指标、各扰动等级指标、相对/绝对性能下降；同时输出每张 Test 图像的 TP/FP/FN/TN、mIoU 和 F1。已使用 `tools/evaluation/analyze_robustness.py` 对每个 seed 的 133 张 Test 图像执行配对 bootstrap 和 sign-flip 检验，并使用 `tools/evaluation/render_qualitative_cases.py` 生成固定规则的 RGB/GT/预测/FP-FN 案例图。此实验只能支持**受控扰动鲁棒性**，不能替代真实的跨区域测试。

**交付**：`Table 7. Robustness under controlled perturbations`、`Figure 3. Robustness curves`。

---

## E6. 重复实验与复杂度【必须】

### E6.1 统计可靠性

- 当前正式矩阵的 19 个协议运行项（M1–M4、M6–M11、A1、A2、A5、B2、B3、C2、L1、L2、L3）全部使用 `41/42/43` 三个随机种子，共 57 个 run；M5、A4、B1、C1 是对 L1 的论文显示别名，不重复训练。不能把单次最优值冒充均值。
- 每次运行固定并记录训练种子、数据读取顺序、预训练权重、checkpoint 选择策略和确定性开关；M6 明确标记为 `deterministic=false`。
- 主要结果报告 mean±std；差值的区间可用按测试**图像**配对重采样的 bootstrap。应区分两类统计对象：跨 seed 的训练随机性、测试图像采样的置信区间，二者不可混为一谈。
- 即便某个 seed 表现不佳，也不得从报告中删除；不以 Test 选择“最优 seed”。

### E6.2 推理效率

| 指标 | 统一设置 |
|---|---|
| Params | 完整推理网络（编码器 + Fusion + Decoder） |
| GFLOPs | 同一 128×128 输入，双流分别按真实通道计入 |
| FPS / latency | 同一 GPU、batch、精度、warm-up 和测速次数 |
| Peak VRAM（可选） | 同一 batch size，报告测量范围 |

当前仓库的 `run_vram_smoke.sh` 仍仅用于有界显存/建模探针；正式统一测量已补充到 `tools/evaluation/measure_model_complexity.py`，入口为 `scripts/run_complexity_eval.sh`。它读取冻结 run 的 `best.pth`，记录硬件、batch、精度、warm-up、计时配置及 M11 后端；结果写入 `outputs_experiments/complexity/complexity_results.{json,csv}`。主表默认不含 TTA；如另作 TTA，单独报告预测质量和额外耗时。不同分辨率/硬件的速度不能直接比较。

**交付**：`Table 6. Model complexity and inference speed`、`Supplementary Table S1. Multi-seed results`。

---

## E7. 定性与误差分析【必须】

- **Validation 例图**：可以用于开发阶段分析和论文的解释性展示，但选择标准必须透明。
- **Test 例图**：在查看模型预测前，按预先确定的随机种子/样本 ID 哈希选择固定测试图像；不得事后只挑出 M5 优势最大的案例。
- 推荐对每张例图展示：`RGB | GT | Single-stream Small | PMDFNet-Small | FP/FN error map`。
- 分析类型包括：明显滑坡主体、易误判背景、边界复杂、漏检、误检、两种模型都失败的情形。
- 建议使用测试集的总体 FP/FN 统计或按图像分布分位数支撑定性讨论，不应从几个案例推导总体性能。

这里的“图像级 bootstrap 或配对统计检验”不是只挑几张图展示，而是对同一 Test 图像在不同模型下的逐图像指标/误差做成配对样本：bootstrap 时对图像 ID 成对有放回重采样，配对检验时比较每张图的 mIoU、F1、FP/FN 或错误面积差。当前脚本按每个 seed 的 133 张图像执行 10,000 次 bootstrap 和 10,000 次双侧 sign-flip 检验，并保存 CI、p 值和逐 seed 记录。定性可视化和失败案例分析则按预先固定的四类规则（候选模型更好、参考模型更好、双方都困难、差异最大）生成，不只展示事后最有利的图片。

**交付**：`Figure 2. Qualitative comparison`、`Figure 4. Failure analysis`。

### E7.1 CAM 与原型 t-SNE 解释性分析【建议，冻结后执行】

为复现 MRSN 中“激活区域 + 特征原型”这一类机制分析，本项目采用与 PMDFNet 结构一致的 PyTorch 实现，不直接移植 Paddle 模型代码：

- `tools/evaluation/cam_analysis.py`：从正式 run 的 `config.json` 和 `best.pth` 重建模型，在指定 decoder feature layer 上计算 Grad-CAM；默认使用 Test 的 `gt_mask`，按固定 `selection_seed` 为同一批样本生成**前景为主、背景为辅**的 per-sample 类别计划（默认 75% 前景、25% 背景），同时保存 `RGB`、原始 `Slope` 波段预览、GT、Pred、CAM overlay、FP/FN、CAM target mask、灰度 CAM 和逐图像 metadata。也支持 MRSN 风格的 `--target-plan`，用 `image.tif: 1` 或 `0: image_a.tif,image_b.tif` 显式冻结样本/类别。
- `tools/evaluation/tsne_prototype_analysis.py`：默认对全部公开 Test 图像提取每个模型的 GT/Pred 区域特征原型，每张图像、每个类别、每个来源各保留一个区域均值向量；先对所选模型联合执行维度对齐、标准化和 t-SNE，再输出联合概览图和 `by_model/<model_id>.png`。后者每个模型一张图，默认使用 GT-defined regions，并以颜色区分 Background/Foreground；传入 `--ids` 可限定图像，传入 `--sources gt,pred` 时再以 marker 区分 GT/Pred。
- `tools/evaluation/select_interpretability_samples.py`：在冻结 checkpoint 上计算逐 Test 图像 mIoU，仅为性能条件化 CAM 提供 `selected_ids.txt`；默认 t-SNE 不使用该筛选清单，而是覆盖全部 Test 图像。
- `tools/evaluation/make_cam_comparison_figure.py`：只负责读取已经生成的 CAM 目录并排版，不重新推理；保证图像选择和模型输出不会因排版过程改变。当前对比列固定为 `RGB`、`Slope`、M1/M3/M10/M11/M5 的 CAM 和末列 `GT`。
- `tools/evaluation/make_prediction_comparison_figure.py`：只读取同一批 CAM 输出目录中的 `RGB`、`Slope`、`GT` 和 `Pred`，排版为实例预测对比图，不重新推理或重新选择样本。
- `scripts/run_interpretability.sh`：默认对 seed 41 的 M1/M3/M10/M11/M5 执行性能条件化 CAM 和同批优势样本预测对比；M5 自动解析为存储运行项 L1。完整 CAM/预测图使用优势样本清单中的 12 张，论文版默认由 `INTERP_PAPER_TOP_K=6` 输出最高优势的 6 张，t-SNE 保持 M1–M4/L1/M5 的全 Test 分析。现有清单仍以 M1/M2/M3/M10/M11 作为样本选择参考，M2 不参与新的显示列。设置 `INTERP_TSNE_ALL_TEST=false` 可恢复为 CAM 同批样本；输出目录由 `INTERP_OUTPUT_ROOT` 指定。

固定规则如下：

1. CAM 默认使用**性能条件化样本**：`select_interpretability_samples.py` 在同一 Test 图像上比较目标 PMDFNet 与指定参考模型的逐图像 mIoU，只保留目标模型优势为正的样本，再按优势固定排序取前 `K` 张。这类图用于展示“PMDFNet 优势案例”，不是无偏的 Test 代表性估计。t-SNE 默认使用全部公开 Test 图像；只有设置 `INTERP_TSNE_ALL_TEST=false` 时才与 CAM 共用 `selected_ids.txt`。
2. CAM 的 `gt_mask` 仅用于定位解释区域，不改变预测和评价指标；空目标区域自动回退到整幅图，并在 metadata 中记录。默认 CAM 类别计划由固定 seed 生成，前景样本占多数、背景样本占少数；同一计划在不同模型之间共享，不按单个模型预测或 Test 分数选择。
3. t-SNE 的 GT/Pred 原型来自同一图像和同一类别的空间特征均值。默认覆盖全部公开 Test 图像；空类别区域在对应特征图分辨率下没有有效像素时跳过。所有选定模型先使用同一个联合 embedding，再生成每模型单独图，因此 `by_model/*.png` 的坐标仍可横向比较。t-SNE 是描述性可视化，不作为显著性检验；类别、来源 marker、特征层、PCA 对齐和随机种子必须随 manifest 一起保存。
4. 正文优先展示两张互补的 M1/M3/M10/M11/M5 对比图：一张为 `RGB | Slope | M1 | M3 | M10 | M11 | M5 | GT` 的实例预测汇总，另一张为相同列顺序的 CAM 激活汇总；两者使用同一批性能条件化优势样本。现有优势样本仍由 L1/M5 与 M1、M2、M3、M10、M11 的逐图像 mIoU 比较确定，M2 仅保留在冻结选择规则中，不列入新的五模型显示集合。FP/FN、CAM target 和灰度 CAM 作为逐模型原始审计材料保留，t-SNE 继续使用既有全 Test 模型集合。预测图只支持具体案例层面的可视比较，不能替代全 Test 统计；也不能仅凭 CAM 热区声称模型“学习了物理因果关系”，只能描述其关注区域与错误模式的关联。

**执行入口示例**：

```bash
DATA_ROOT=/data/.../mmlsv2_mapped_mars_ls \
  INTERP_CAM_IDS=M1,M2,M3,M4,L1 INTERP_TSNE_IDS=M1,M2,M3,M4,L1 \
  INTERP_SELECTION_TARGET_IDS=L1 INTERP_SELECTION_REFERENCE_IDS=M1,M2,M3,M4 \
  INTERP_BACKGROUND_RATIO=0.25 INTERP_TSNE_SOURCES=gt INTERP_TSNE_ALL_TEST=true \
  INTERP_SEED=41 INTERP_SPLIT=test \
  bash scripts/run_interpretability.sh
```

如需将多个 PMDFNet 变体与代表性基线进行 CAM 对比，直接将
`INTERP_CAM_IDS` 设置为已冻结的正式 run 列表，例如
`INTERP_CAM_IDS=L1,A5,C2,M4,M3`。这里的“最优”必须在分析前依据已冻结的
Val/主表规则确定，不能根据 Test 例图临时挑选；CAM 脚本本身不会依据 Test
分数自动选择模型。

**交付**：正文可使用的预测实例对比图、CAM 对比图和补充材料 t-SNE 图；原始 `cam_manifest.json`、`cam_meta.csv`、预测对比图 manifest、`tsne_manifest.json`、`prototypes.csv` 与 `tsne_coordinates.csv` 必须归档。

---

### E7.2 Boundary、逐 tile 分布、前景面积分层与不确定性【已完成，冻结后执行】

这些分析统一由 `tools/evaluation/evaluate_test_diagnostics.py` 完成，入口为
`scripts/run_test_diagnostics.sh`。它只读取冻结 run 的 `config.json` 与 `best.pth`，重新在带公开
mask 的 Test split 推理，不改写各 run 的主 `test_metrics.json`，也不重新选择 checkpoint。

- **Boundary 指标**：将二值 mask 定义为 `Dilate3x3(mask) - Erode3x3(mask)` 的一像素轮廓；默认
  允许 2 像素匹配容差，分别计算 Boundary Precision、Recall、F1 和 tolerance-dilated contour
  band IoU。输出全 Test pooled 结果和逐 tile 结果，并明确它们不等同于区域 IoU。
- **每 tile 分布**：保存每张 Test tile 的 TP/FP/FN/TN、mIoU、Foreground IoU、F1、错误率、
  Boundary F1/IoU 和分布统计（均值、标准差、中位数、四分位数）。这用于展示性能离散程度，
  不能用少数 tile 替代全测试集指标。
- **前景面积分层**：按 GT 前景面积比例排序，文件名作为平局 tie-break，再用固定的等数量切分
  形成 small/medium/large 三组。133 张 Test tile 将得到 45/44/44 张的确定性分组；每组同时
  报告区域指标、Boundary 指标和样本数，不能按模型优劣事后调整分界。
- **不确定性**：二分类 softmax 前景概率计算 binary predictive entropy（自然对数，单位 nats），
  保存每 tile 的平均熵、错误像素上的平均熵及 95 分位数，并计算熵与 tile 错误率的 Pearson
  相关。无错误像素的 tile 不参与“错误像素平均熵”相关分析；相关性是描述性证据，不是校准
  或因果证明。

默认分析 M4 与 L1/M5 的 `41/42/43` 三个 seed；单通道 S1–S4 完成训练后，可用同一脚本将
`DIAGNOSTIC_CONFIG`、`DIAGNOSTIC_RUN_ROOT` 和 `DIAGNOSTIC_IDS` 切换到补充协议。输出目录为
`outputs_experiments/test_diagnostics/`，包括 `diagnostics_manifest.json`、
`diagnostics_summary.{json,csv}`、`per_tile.csv`、`foreground_area_stratified.csv`、
`per_tile_distributions.csv`、`across_seed_summary.csv` 和三张诊断图。

执行状态：已在 A100-1 使用冻结的 M4/L1 `best.pth` 完成 6 个 Test 诊断 run（3 个 seed/模型），
并使用同一协议对 S1–S4 完成 12 个 Test 诊断 run。远程结果目录分别为
`/data/huilin/projects/PMDFNet/outputs_experiments/test_diagnostics/` 和
`/data/huilin/projects/PMDFNet/outputs_experiments/test_diagnostics_single_channel/`；两者均含
JSON/CSV 汇总、逐 tile 结果、前景面积分层结果、跨 seed 汇总和三张 PNG 诊断图。

**执行入口示例**：

```bash
DATA_ROOT=/data/.../mmlsv2_mapped_mars_ls \
  DIAGNOSTIC_IDS=M4,L1 DIAGNOSTIC_SEEDS="41 42 43" \
  bash scripts/run_test_diagnostics.sh
```

**交付**：Supplementary Table S3（Boundary 与面积分层）、Figure 2（逐 tile 分布）、
Figure 3（前景面积分层）和 Figure 4（熵—错误率关系）；正文只在统计口径和样本量清楚时引用这些补充结果。

---

## 2. 当前正式实验矩阵（protocol v5）

当前配置将正式训练矩阵拆成互斥的三个 group，共 19 个协议运行项、3 个 seed、57 个 run。`M5` 是论文显示 ID，实际协议运行项为 `L1`；`A3/A4/B1/C1` 是对 `M4/L1` 的显示别名，不重复训练。

| 分组 | 正式训练项 | run 数量 | 启动入口 |
|---|---|---:|---|
| core | M1、M2、M3、M4 | 4 × 3 = 12 | `scripts/run_core.sh` |
| ablations | A1、A2、A5、B2、B3、C2、L1、L2、L3 | 9 × 3 = 27 | `scripts/run_ablations.sh` |
| extended | M6、M7、M8、M9、M10、M11 | 6 × 3 = 18 | `scripts/run_extended.sh` |
| 合计 | 19 个协议运行项 | **57** | 三组全部执行 |

### 2.1 补充协议（不计入正式 57-run 主矩阵）

| 分组 | 协议文件 | 训练项 | run 数量 | 启动入口 |
|---|---|---|---:|---|
| single_channel | `configs/mmlsv2_single_channel_protocol.json` | S1、S2、S3、S4 | **4 × 3 = 12** | `scripts/run_single_channel_ablation.sh` |
| Test diagnostics | 无新增训练 | M4/L1（可扩展至 S1–S4） | 冻结 checkpoint 推理 | `scripts/run_test_diagnostics.sh` |

补充协议有自己的 output root、summary.csv 和 protocol version。它们用于回答更细的波段贡献、
边界质量、性能分布、前景面积敏感性和预测不确定性问题，不能与正式 57 个 run 混合计数，也
不能在看到 Test 结果后临时增加或删除某个 S1–S4 条目。

其中：

- 主方法及直接对照：M4（单流 Small）与 L1（论文 M5，双流 Small，CE-only）；
- 代表性基线：M1、M2、M3；
- 模态贡献：A1、A2；
- 容量对照：A5（单流 ConvNeXt-Base）；
- 通道分组：L1/M5（B1 别名）、B2、B3；
- 融合与损失：L1/M5（C1 别名）、C2、L2、L3；
- 扩展基线：M6–M11，均属于当前正式矩阵，不再写成未执行的“可选”结果。

**训练去重**：`M4=A3`；`L1=M5=A4=B1=C1`。同配置、同 seed、同数据版本直接共享模型与预测。A1/A2、A5、B2/B3、C2、L2、L3 需要独立训练。

第一轮结束设立**分析关口**：

1. 若 L1/M5 在 3 seeds 上对 M4 的优势不稳定，先排查数据、统计口径和训练差异，不要根据 Test 回头调模型。
2. 若 L1/M5 不优于 A5，不能把性能提升单方面归因为模态解耦；如实讨论参数与容量影响。
3. 若 B1 不优于 B2/B3，修改**论文解释与结论范围**，而不是通过 Test 导向的分组搜索寻找支持原主张的结果。
4. 若核心证据不足，可以决定暂缓期刊投稿或另启**新的实验版本**；不得把同一 Test 的多轮反馈伪装成独立最终测试。

---

## 3. 推荐执行顺序

### Phase 0：E0 数据审计与协议冻结

1. 完成映射索引、单位、直方图、配对、场景独立性/近重复检查。
2. 保存 `mapping_manifest.json`、`mapping_match_report.json`，并补齐 manifest、hash、转换脚本和数据审计记录。
3. 以 `configs/mmlsv2_paper_protocol.json` 为训练协议源文件，冻结主模型 Small、batch size 32、CE-only 主损失、随机种子、训练/验证流程、指标和 Test 的使用规则；如另建 `experiment_protocol.yaml`，必须与 JSON 一致。

### Phase 1：基础比较和 3-seed 稳定性

4. 先运行 `scripts/run_data_smoke.sh`；确认 `.env` 解析出的 `DATA_ROOT` 指向 mapped MMLSv2，而不是原始 MMLSv2。
5. 运行 `scripts/run_core.sh`、`scripts/run_ablations.sh` 和 `scripts/run_extended.sh`，完成 19 个协议运行项 × `41/42/43`，共 57 个 run。
6. 资源有限时可按上述三组分阶段执行，但不得改变协议、seed、数据版本或输出命名；先在 Validation 确认实现正确。

### Phase 2：验证核心机制

7. 核实 B2/B3、C2、L2/L3 与 L1/M5 的单变量比较；B4 不属于当前 protocol v5。
8. 运行 `scripts/run_complexity_eval.sh` 完成 Params、GFLOPs、FPS/latency 和峰值推理显存测量；`run_vram_smoke.sh` 仍只作显存/建模探针。
9. 【已完成】按独立协议运行 `scripts/run_single_channel_ablation.sh`，完成 S1–S4 × `41/42/43`；
   这组训练在 Test 诊断前已冻结，未由 Test 结果反向选择波段。
10. 核实双流、模态划分、损失、单通道贡献和参数效率方面的结论是否成立。

### Phase 3：冻结后最终测试与写作

11. 确认 57 个 formal run 均存在 `best.pth`、`config.json` 和完成日志后，运行 `scripts/run_test_eval.sh`；该脚本要求默认找到 57 个正式 run。
12. 输出标准化预测、全像素混淆矩阵和指标；记录当前 Test 使用 MARS Test 统计量的实现口径。
13. 执行 `scripts/run_robustness_eval.sh` 的已冻结扰动协议，并使用 `scripts/run_robustness_analysis.sh` 完成图像级配对统计、FP/FN 失败案例和 Test 可视化；完成效率测试和多 seed 汇总。
14. 【已完成】运行 `scripts/run_test_diagnostics.sh`，生成 M4/L1 以及 S1–S4 的 Boundary、逐 tile、前景面积分层和熵不确定性结果。
15. 在 Test 评价完成且模型/样本选择冻结后，运行 `scripts/run_interpretability.sh`，生成 M1/M3/M10/M11/M5 与 GT 的性能条件化预测实例对比、CAM 对比和既有全 Test GT 前景/背景原型 t-SNE；检查 manifest 中的 checkpoint、feature layer、selection seed、样本选择规则和 data split。两类对比图的固定列顺序为 `RGB | Slope | M1 | M3 | M10 | M11 | M5 | GT`。
16. 根据实际证据撰写 Discussion；不得把 L3 的 CE + Dice 结果写成主模型 M5/L1 的 CE-only 结果，也不得把 CAM/t-SNE、Boundary 或熵相关性的描述性结果写成因果或显著性证据。

> **注意**：理想流程是先在 Validation 完成各配置、checkpoint、阈值和实验矩阵的冻结，然后一次性评价 Test。不能完成一次 Test 后再决定是否增加某个更可能获胜的分组或改主模型。

---

## 4. 期刊图表清单

| 编号 | 图表 | 优先级 |
|---|---|---|
| Table 1 | Converted MMLSv2：数据划分、各通道统计和映射协议摘要 | 必须 |
| Table 2 | M1–M5 主基线（M5 由 L1 提供，含均值±标准差） | 必须 |
| Table 3 | A1–A5 单流/双流、模态贡献及容量控制 | 必须 |
| Table 4 | B1–B3：严格 3+4 分组比较（B1 复用 L1） | 必须 |
| Table 5 | Add vs Concat；CE/Dice Loss（L1/L2/L3） | 必须 |
| Table 6 | Params / GFLOPs / FPS | 必须 |
| Table 7 | 预设受控扰动及模态缺失 | 建议 |
| Figure 1 | 原始 PMDFNet-Small 结构图（更新输入和维度标注） | 必须 |
| Figure 2 | 逐 tile 指标分布 | 建议 |
| Figure 3 | small/medium/large 前景面积分层 | 建议 |
| Figure 4 | 预测熵与 tile 错误率关系 | 建议 |
| Figure 5 | 同一批 PMDFNet 优势样本的 RGB、Slope、M1/M3/M10/M11/M5 预测和 GT 同图比较 | 建议 |
| Figure 6 | M1/M3/M10/M11/M5 与 GT 的 RGB、Slope 和 CAM 同图比较 | 建议 |
| Figure 7 | 全 Test GT 前景/背景原型 t-SNE | 建议 |
| Supplementary Table S2 | S1–S4 单通道移除消融（逐 seed 与复杂度） | 建议 |
| Supplementary Table S3 | Boundary 指标、逐 tile 分布、前景面积分层与不确定性摘要 | 建议 |
| Supplementary | 映射 manifest、详细实现、单 seed 结果 | 建议 |
| Supplementary | GT/Pred 特征原型 t-SNE、CAM 全部样本与 metadata | 建议 |

---

## 5. 复现目录与记录建议

```text
outputs_experiments/pmdfnet_mmlsv2/
  {experiment_name}_s{41,42,43}_{timestamp}/
    config.json
    command.txt
    git.json
    reproducibility.json
    runtime.json
    metrics.csv
    logs/train.log
    checkpoints/best.pth
    checkpoints/last.pth
    results.json
    test_metrics.json

protocol_artifacts/
  experiment_protocol.yaml       # 若另行生成，必须与 JSON 一致
  robustness_protocol.json       # E5 的固定扰动协议
  channel_mapping.yaml
  mapping_manifest.json
  mapping_match_report.json

outputs_experiments/complexity/
  complexity_results.json
  complexity_results.csv

outputs_experiments/pmdfnet_mmlsv2_single_channel/
  {S1,S2,S3,S4}_..._s{41,42,43}_{timestamp}/
    config.json
    checkpoints/best.pth
    test_metrics.json
  summary.csv

outputs_experiments/robustness/
  robustness_results.json
  robustness_results.csv
  robustness_per_image.jsonl
  analysis/
    robustness_paired_stats.json
    robustness_paired_stats.csv
    failure_cases.json
    failure_cases.csv
  qualitative/
    qualitative_cases.json
    *.png

outputs_experiments/interpretability_selected_m1_m3_m10_m11_m5/
  selection/pmdfnet_advantage_M1_M2_M3_M10_M11_s41/
    selection_manifest.json
    selection_scores.csv
    selected_samples.csv
    selected_ids.txt
  cam/{M1,M3,M10,M11,M5}_s41/
    cam_manifest.json
    cam_meta.csv
    samples/rank_*/{rgb,slope,gt,pred,cam_overlay,cam_gray,fp_fn}.png
  cam/cam_comparison_rgb_slope_gt_s41.png
  cam/prediction_comparison_rgb_slope_gt_s41.png
  cam/cam_comparison_rgb_slope_gt_s41_n6.png
  cam/prediction_comparison_rgb_slope_gt_s41_n6.png
  tsne/M4,L1_s41/
    tsne_manifest.json
    prototypes.csv
    tsne_coordinates.csv
    tsne_all.png
    tsne_gt.png                 # default; tsne_pred.png is added when requested
    by_model/{M1,M2,M3,M4,L1}.png

outputs_experiments/test_diagnostics/
  diagnostics_manifest.json
  diagnostics_summary.json
  diagnostics_summary.csv
  per_tile.csv
  per_tile_distributions.csv
  foreground_area_stratified.csv
  across_seed_summary.csv
  plots/
    per_tile_distributions.png
    foreground_area_stratification.png
    uncertainty_error_scatter.png

outputs_experiments/test_diagnostics_single_channel/
  diagnostics_manifest.json
  diagnostics_summary.{json,csv}
  per_tile.csv
  per_tile_distributions.csv
  foreground_area_stratified.csv
  across_seed_summary.csv
  plots/
    per_tile_distributions.png
    foreground_area_stratification.png
    uncertainty_error_scatter.png
```

当前训练协议源文件是 `configs/mmlsv2_paper_protocol.json`；运行目录由训练器自动记录配置、命令、Git、可复现性、运行时、checkpoint 和指标。正式 Test 评估写入各 run 目录的 `test_metrics.json` 和 `results.json`。如用于公开复现，至少提供数据处理脚本、训练/评估脚本、主模型配置和运行命令。对于不能公开再分发的数据，提供合法下载途径和确定性转换步骤。

## 6. 投稿前检查清单

- [ ] 已明确主模型 Small 的版本及其与原 Workshop Tiny/Small 结果的关系；当前 protocol v5 不包含 PMDFNet-Tiny 正式运行项。
- [ ] 已记录公开数据集来源、引用、通道、单位、归一化和映射规则；本轮不以数据泄漏、场景独立性或完整数据审计作为阻塞项。
- [ ] 标注来源、数据划分、Test 使用边界已完整披露。
- [ ] 所有主基线在相同评估协议下比较；参数近似匹配对照已报告差异。
- [ ] B1–B3 都满足固定 3+4 的分支输入宽度；B4 不属于当前 protocol v5。
- [ ] 19 个正式协议运行项均使用 `41/42/43` 三个 seeds，报告 mean±std；M5/A4/B1/C1 由 L1 复用，不依据 Test 选 seed。
- [ ] 已记录 batch size `32`、`drop_last=True`、CE-only 主损失，以及 L2/L3 的损失覆盖。
- [ ] 已记录 M6 因 UPerNet CUDA 反向算子限制而使用 `deterministic=false`，不得将 M6 描述为逐位可复现。
- [ ] 已记录 Train/Val 使用 Train 统计量、公开 Test 评估使用 MARS Test 统计量的当前实现口径。
- [x] 已按独立 `single_channel` 协议完成 S1–S4 × `41/42/43`，并将其与 57-run 主矩阵分开归档。
- [x] 扰动类型和强度在 Validation 确定，最终 Test 不参与修改方案。
- [x] Test 定性样本按固定 seed、固定规则和双向案例类别选择；失败案例不只展示单方面有利样本。
- [x] 已运行并归档 `run_complexity_eval.sh` 的 Params/GFLOPs/FPS/latency/显存结果。
- [x] 已运行并归档 E5 的 clean/扰动指标、图像级 bootstrap/配对检验以及 FP/FN 定性 PNG。
- [x] 已在冻结 Test 样本规则下运行并归档历史 M4/L1 seed 41 的 CAM/GT-Pred 原型 t-SNE，以及按 L1/M5 相对 M1/M2/M3/M10/M11 逐图像 mIoU 优势筛选的 M1/M3/M10/M11/M5 seed 41 CAM 和预测实例对比；完整归档保留 12 张优势样本，论文版 CAM 和预测图按优势排序展示前 6 张，并增加 M5 与 GT 列；当前两类对比图均采用 `RGB | Slope | M1 | M3 | M10 | M11 | M5 | GT` 排版，未把解释性图误写成显著性或因果证据。
- [x] 已运行并归档 Boundary 指标、逐 tile 分布、前景面积分层和熵不确定性分析；已固定 2 像素 Boundary 容差和 3 组面积分层规则。
- [ ] 已记录 M11 实际使用的 `mamba_ssm_selective_scan` 或 `gated_conv_simulator` 后端。
- [ ] 正式冻结前工作树为 clean commit；若沿用历史 dirty run，已归档 `git diff HEAD`、未跟踪文件、环境信息和原始输出，并明确标注 dirty-snapshot。
- [ ] 不将 TTA、Ensemble、测试集校正和模型主体效果混为一谈。
- [ ] 物理分组、效率和鲁棒性方面的论文表述有对应的实证支持。
- [ ] 不把 converted MMLSv2 说成独立新数据集，不在缺乏证据时声称真实跨区域泛化。

**最后原则**：上述计划提升的是论文方法论质量与可重复性，并不保证 *Remote Sensing* 接收。是否能够围绕不升级的 PMDFNet 形成足够的期刊贡献，取决于上述对照实验实际揭示了什么。
