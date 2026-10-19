# PMDFNet：Physically Motivated Dual-Stream Fusion for Multimodal Landslide Segmentation

本文件中的 PMDFNet 是项目论文统一名称；旧版代码中的 `dual_segformer_*` 仅保留为历史兼容键。

> **目标期刊**：*Remote Sensing*  
> **核心约束**：不升级 PMDFNet 主网络；保留双 ConvNeXt Encoder、四尺度逐元素相加（Add Fusion）和 SegFormer All-MLP Decoder。  
> **主数据集**：MMLSv2（以统一的 Train / Validation / Test 划分完成训练、模型选择与最终评估）。  
> **工作目标**：在复现原始网络的基础上，通过完整基线比较、模态消融、分组消融、鲁棒性与效率评估，形成可复现、可解释的实验证据。  
> **范围**：本文件只规划 **MMLSv2** 实验；其他数据集的扩展实验另行规划。

## 1. 需要回答的研究问题

- **RQ1——性能**：PMDFNet 是否优于具有代表性的单流语义分割模型？
- **RQ2——双流有效性**：性能提升是否来源于模态解耦，而不仅是模型参数量增加？
- **RQ3——模态分组**：按物理含义划分输入通道，是否优于其他合理划分或非物理划分？
- **RQ4——设计选择**：简单的 Add Fusion 与 CE + Dice 是否具备合理性？
- **RQ5——可靠性**：模型在受控输入扰动下的表现是否稳定？性能提升是否具有重复性，计算开销如何？

### 1.1 固定的原始网络

输入 7 通道，按原稿设计划为：

| 分支 | 输入通道 | 通道数 |
|---|---|---:|
| Geometric-Physical Stream | Thermal Inertia、Slope、DEM | 3 |
| Optical-Textural Stream | Grayscale、RGB | 4 |

两个**权重独立**的 ConvNeXt 在四个尺度产生特征，并做逐尺度相加：

\[
F_i^{\mathrm{fused}} = F_i^{G} + F_i^{O}, \quad i=1,2,3,4.
\]

随后使用 SegFormer All-MLP Decoder 输出二分类掩码；默认损失为：

\[
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{CE}} + \mathcal{L}_{\mathrm{Dice}}.
\]

**约束解释**：新增对照只用于验证原始设计。测试 Concat 或改变输入分组不代表要将其写入最终网络。允许针对基线修改首层输入通道数，以及针对任务调整输出类别数。

## 2. 统一实验协议（所有实验共用）

### 2.1 训练和测试原则

1. 使用固定的 MMLSv2 Train / Validation / Test 划分；任何模型不得用 Test 选择 checkpoint、超参数或阈值。
2. 首轮复现尽量沿用原稿设置：**128 × 128** 输入、**100 epochs**、**batch size 16**、**AdamW**（初始学习率 **1×10⁻⁴**、weight decay **5×10⁻⁴**）、cosine learning-rate schedule；如因模型差异调整，单独记录。
3. 固定常规数据增强：水平/垂直翻转、90° 倍数旋转、随机裁剪、Mosaic；对特定输入通道施加的噪声增强须保持与模型对照一致，并记录具体设置。
4. 所有比较默认采用**无 TTA、无 Ensemble、无额外测试集分布校正**的单模型预测，避免混淆网络与推理增强的收益。
5. 使用 Validation 进行 checkpoint 与预测阈值选择；对比时明确二分类阈值是否统一或各模型独立选择。
6. 所有模型采用**相同评价脚本**；固定随机种子、数据增强配置及软件版本。
7. 预训练权重、输入首层适配方法（如 3→7 通道）、优化器、学习率策略等必须完整报告。统一训练预算不等于不同网络一定使用完全相同的最佳超参数，任何例外均需披露。

### 2.2 主要评价指标

| 指标 | 含义 | 建议呈现 |
|---|---|---|
| mIoU ↑ | 背景和滑坡两类 IoU 的宏平均 | **主要指标** |
| Foreground IoU ↑ | 滑坡类别 IoU | **主要指标** |
| Precision ↑ | 滑坡类别精确率 | 补充 |
| Recall ↑ | 滑坡类别召回率 | 补充 |
| F1 ↑ | 滑坡类别 F1-score | 补充 |
| Params ↓ | 可训练参数量 | 效率对比 |
| GFLOPs ↓ | 固定输入尺寸下的理论计算量 | 效率对比 |
| FPS ↑ / latency ↓ | 实际推理速度或单图时延 | 效率对比 |

建议使用**整个 Test 集累计混淆矩阵**计算总体 Precision / Recall / F1 / IoU；若另外报告逐图像平均结果，必须使用不同的指标名称并说明汇总方法。测试阈值和 GT 的 ignore-label 规则需全程一致。

## 3. E1——主实验：与代表性模型比较【必须】

### 3.1 模型配置

| ID | 方法 | Backbone | 输入 | 用途 | 优先级 |
|---|---|---|---|---|---|
| M1 | U-Net | 标准 U-Net Encoder | 全部 7 通道 | 经典 CNN 分割基线 | 必须 |
| M2 | DeepLabV3+ | ResNet50 | 全部 7 通道 | 经典强基线 | 必须 |
| M3 | SegFormer | MiT-B2 | 全部 7 通道 | Transformer 分割基线 | 必须 |
| M4 | ConvNeXt-Tiny + SegFormer MLP Decoder | ConvNeXt-Tiny（单流） | 全部 7 通道 | **直接单流对照** | 必须 |
| M5 | **PMDFNet-Tiny** | ConvNeXt-Tiny ×2（双流） | 原始 3+4 分组 | **主要方法** | 必须 |
| M6 | OCRNet | HRNet | 全部 7 通道 | 扩展基线 | 建议 |
| M7 | UPerNet | ConvNeXt-Tiny | 全部 7 通道 | 扩展基线 | 建议 |

**重要说明**：M4 是“ConvNeXt 编码器 + SegFormer MLP Decoder”的单流实现，并非官方 MiT 编码器版本的 SegFormer；在文章中应单独准确命名。

### 3.2 主要分析

- 将 M5 与 M4 直接比较，观察单流到双流的性能变化。
- 比较 M5 与 M1–M3，评估相对于经典 CNN / Transformer 模型的性能。
- 联合 Params、GFLOPs、FPS 说明精度—效率权衡，避免只强调最高 mIoU。
- 主实验中不混用带 TTA 的比赛成绩与无 TTA 的本地复现成绩。

**论文输出**：`Table 2. Comparison with representative segmentation methods on MMLSv2`。

## 4. E2——核心消融：双流结构与模态贡献【必须】

### 4.1 单流、双流和模态组合

| ID | 网络结构 | 输入 | 目的 | 优先级 |
|---|---|---|---|---|
| A1 | 单流 ConvNeXt-Tiny + MLP Decoder | RGB + Grayscale（4 通道） | 光学模态贡献 | 必须 |
| A2 | 单流 ConvNeXt-Tiny + MLP Decoder | Thermal + Slope + DEM（3 通道） | 地形/物理模态贡献 | 必须 |
| A3 | 单流 ConvNeXt-Tiny + MLP Decoder | 全部 7 通道 | Early Fusion 对照，**复用 M4** | 必须 |
| A4 | 原始 PMDFNet-Tiny | 原始 3+4 分组 | 双流对照，**复用 M5** | 必须 |
| A5 | 参数量近似匹配的单流 Encoder + 同类 Decoder | 全部 7 通道 | 控制模型容量因素 | **强烈建议** |

**A5 的原则**：选择接近 A4 参数规模的单流模型作为容量匹配对照，报告实际参数差异；不应仅通过增加训练轮数代替参数量控制。选择何种 Backbone 规模可在代码复现阶段据真实参数量确定。

### 4.2 关注的结论

- A4 vs A3：双流解耦是否优于直接输入全部 7 通道？
- A4 vs A5：双流优势是否能够超越“更多参数”的解释？
- A1/A2 vs A3/A4：两组模态是否具有互补性？

**论文输出**：`Table 3. Ablation of modality contributions and dual-stream encoding`。

## 5. E3——核心消融：物理模态分组【必须】

**原则**：固定双流 ConvNeXt-Tiny、融合位置、Add Fusion、Decoder、训练策略，仅改变 7 个输入通道的分配。由于两分支的首层输入通道数可能变化，需记录并控制由此产生的少量参数差异。

| ID | Stream 1 | Stream 2 | 目的 |
|---|---|---|---|
| B1 | Thermal + Slope + DEM | Grayscale + RGB | **原始物理分组，复用 M5/A4** |
| B2 | RGB | Thermal + Slope + DEM + Grayscale | 检验 Grayscale 的分支归属 |
| B3 | Thermal + DEM + Grayscale | Slope + RGB | 非原始分组对照 1 |
| B4 | Slope + Grayscale + RGB | Thermal + DEM | 非原始分组对照 2 |

若希望更严格地控制分支输入宽度，优先保留 **3+4** 的比较组合，避免通道数差异成为混杂因素。B2 仍可保留，但须在结果分析中说明其分支宽度相反。

**核心问题**：原始物理分组是否稳定优于替代划分？如果并不优于其他分组，应如实弱化“物理分组最优”的论断，不应为得到预期结论反复挑选分组。

**论文输出**：`Table 4. Effect of modality grouping strategies`。

## 6. E4——设计消融：Fusion 与 Loss【建议】

### 6.1 融合策略

| ID | Fusion | 其他配置 |
|---|---|---|
| C1 | **Add**（逐尺度相加） | 原始 PMDFNet，**复用 M5** |
| C2 | Concat + 1×1 Conv | 其他网络与训练设置保持一致 |

注意：C2 会增加少量融合参数。应报告新增参数与 FLOPs，不能将 C1/C2 视为严格相同计算预算的比较。无需进一步增加复杂 Cross-Attention 或新的可学习模块。

### 6.2 损失函数

| ID | Loss | 目的 |
|---|---|---|
| L1 | Cross-Entropy | 像素分类损失基线 |
| L2 | Dice | 区域重叠损失基线 |
| L3 | **Cross-Entropy + Dice** | 原始设定，**复用 M5** |

这两组实验可以合并为一张表，但应保持变量单独变化，避免同时改变 Fusion 与 Loss。

**论文输出**：`Table 5. Ablation of feature fusion and loss functions`。

## 7. E5——鲁棒性与敏感性分析【建议】

在**不参与训练和参数选择的 Test 图像**上设置固定、预先定义的扰动强度，比较 **M4（单流）** 与 **M5（双流）**。鲁棒性结果应同时报告干净测试集分数、扰动后分数及下降幅度。

| 类别 | 扰动建议 | 主要目的 |
|---|---|---|
| 地形扰动 | DEM 常数偏移、Gaussian noise | 衡量物理输入误差敏感性 |
| 光学扰动 | RGB/Grayscale 亮度与对比度变化 | 衡量光学变化敏感性 |
| 传感器噪声 | 对指定通道施加多强度 Gaussian noise | 衡量通道噪声影响 |
| 模态缺失 | 将光学或物理分支输入替换为对应训练集均值 | 衡量对某一模态的依赖程度 |

实施要求：

1. 在原始物理量空间或归一化空间中明确扰动定义，确保强度具有可解释性；不可混用不同通道的单位。
2. 缺失模态应使用一致的替换规则。若采用训练集标准化，缺失通道替换为归一化后的 0 是一种清晰的实现。
3. 记录每一级扰动强度，不在 Test 上选择“最好”的强度。
4. 此实验仅支持**受控扰动下的鲁棒性**论断，不代表真实跨区域泛化结论。

**论文输出**：`Table 7. Robustness under controlled input perturbations` 和 `Figure 3. Performance degradation across perturbation levels`。

## 8. E6——统计可靠性与计算效率【必须/建议】

### 8.1 重复训练

- **必须优先重复**：M4（单流基线）与 M5（双流方法），建议 3 个不同随机种子。
- **推荐重复**：主表其他核心模型（M1–M3）以及 B1/B2 等关键分组；资源允许时对主要结论统一报告 mean ± std。
- 对关键模型差异，可按 Test 图像配对重采样，计算 IoU 差值的 Bootstrap 95% CI，并明确“按图像平均的指标”与“全数据集累计指标”的区别；也可以采用适配累计混淆矩阵指标的图像级重采样方法。
- 在正文注明随机种子、重复次数、checkpoint 选择方式；避免只使用单次最优结果。

### 8.2 模型效率

| 指标 | 测试要求 |
|---|---|
| Parameters (M) | 统计整个推理网络 |
| FLOPs / GFLOPs | 同一输入尺寸，例如 1×7×128×128（双流分别按实际输入计） |
| FPS / latency | 同一 GPU、batch size、精度、warm-up 与测速次数 |
| Peak GPU memory（可选） | 同一 batch size 下统计 |

建议分别记录 **without TTA** 和 **with TTA** 的耗时；主表默认以 without TTA 为准。不能把多模型 Ensemble 的吞吐与单模型混在同一列中比较。

**论文输出**：`Table 6. Model complexity and inference efficiency`；必要时增加 `Table S1. Multi-seed performance`。

## 9. E7——定性比较与错误分析【必须】

选取约 **6–8 个 Test 样本**，同时覆盖典型成功与失败案例：

- 滑坡主体区域分割正确的案例；
- 单流漏检、双流正确的案例；
- 单流误检、双流改善的案例；
- 两种模型均失败的案例；
- 细长结构、边界不清和背景纹理相近的困难案例。

推荐每个案例显示：

`Optical RGB / Ground Truth / M4 Single-stream / M5 PMDFNet / FP-FN error maps`

可以给出每个案例的 Foreground IoU，并将错误描述与可见区域、误检/漏检位置直接对应；不应仅凭少量例图推断整体鲁棒性。

**论文输出**：`Figure 2. Qualitative comparison` 与 `Figure 4. Representative failure cases`。

## 10. 可选补充：TTA 与 Ensemble（不是主要贡献）

原始比赛稿包含多尺度 TTA、Tiny+Large Ensemble 和额外的测试数据分布校正策略。修订时建议：

- 主结果只使用**单模型、无 TTA**的统一评估协议；
- 额外报告 M5 **without / with TTA**，单独衡量推理增强的作用；
- 若原先 Large 与 Ensemble 已可复现，可放在附录；否则无需投入大量时间重训；
- 数据分布校正策略如果没有独立验证，不必纳入这篇论文的主要贡献。

## 11. 最终建议图表清单

| 序号 | 图表 | 重要性 |
|---|---|---|
| Table 1 | MMLSv2 数据划分、类别及通道统计 | 必须 |
| Table 2 | 主模型对比（M1–M7） | 必须 |
| Table 3 | 单流/双流与模态贡献（A1–A5） | 必须 |
| Table 4 | 模态分组消融（B1–B4） | 必须 |
| Table 5 | Fusion + Loss 消融（C1/C2、L1–L3） | 建议 |
| Table 6 | 参数量、FLOPs、FPS | 必须 |
| Table 7 | 扰动鲁棒性 | 建议 |
| Figure 1 | PMDFNet 网络架构 | 必须（保留并完善原图） |
| Figure 2 | 分割定性对比 | 必须 |
| Figure 3 | 扰动强度—性能变化曲线 | 建议 |
| Figure 4 | 典型失败案例与错误图 | 建议 |

若篇幅受限，可将部分 Loss 消融、多随机种子明细及鲁棒性数据移至 Supplementary Material，但正文需保留支持核心结论的证据。

## 12. 推荐执行顺序与训练结果复用

### Phase 1：建立主对照（最高优先级）

1. 训练 **M4/A3**：单流 ConvNeXt-Tiny + MLP Decoder，7 通道。
2. 训练 **M5/A4/B1/C1/L3**：原始 PMDFNet-Tiny，3+4 通道。
3. 用统一脚本得到 mIoU、Foreground IoU、Precision、Recall、F1、Params、GFLOPs、FPS。
4. 首先判断原始双流相较单流是否稳定有效。

### Phase 2：完成最关键的机制证据

5. 训练 **A1/A2**：光学单流和物理单流。
6. 训练 **A5**：参数量近似匹配的 7 通道单流模型。
7. 训练 **B2/B3/B4**：输入模态划分对照。
8. 汇总 Table 3–4，形成主要实证结论。

### Phase 3：补齐主实验与常规消融

9. 训练 **M1/M2/M3**；时间允许增加 **M6/M7**。
10. 训练 **C2** 以及 **L1/L2**（C1 和 L3 直接复用 M5）。
11. 完成主表和 Loss/Fusion 消融表。

### Phase 4：提升论文完整性

12. 对主要模型进行 **3-seed 重复训练**并汇总统计。
13. 完成固定扰动测试、参数和速度测量。
14. 绘制定性对比、失败案例、鲁棒性曲线，撰写 Discussion。

### 关键复用关系

| 已训练配置 | 可复用到 |
|---|---|
| M4 | A3、鲁棒性单流对照、速度测试 |
| M5 | A4、B1、C1、L3、鲁棒性双流对照、速度测试 |
| A1 / A2 | 单模态贡献分析 |
| 其他 M / B / C / L | 各自对应主表和消融表 |

## 13. 投稿前检查

- [ ] 主表所有模型均在同一数据划分与评价脚本下运行。
- [ ] Test 未被用于模型选择、阈值调整或实验方案筛选。
- [ ] 单流 7 通道与双流 3+4 通道比较完整。
- [ ] 参数量匹配对照已完成，或明确说明这一局限。
- [ ] 原始物理分组与替代分组比较完整。
- [ ] CE、Dice 以及原始组合的结果可解释。
- [ ] 主要性能结果包括重复实验统计。
- [ ] 模型效率、TTA 使用与预训练方式明确。
- [ ] 定性分析既包含成功案例，也包含失败案例。
- [ ] 结论不夸大“physics-informed”或真实跨域泛化程度。

---

**最低可投稿实验组合建议**：E1 主对比 + E2 单流/双流与模态贡献 + E3 模态分组 + E6 统计/效率 + E7 可视化。E4 Fusion/Loss 和 E5 扰动鲁棒性建议在上述关键证据成立后补齐。

**定位提醒**：这份计划可以显著提高实验完整性，但不能保证方法原创性达到特定期刊的录用要求。最终创新论证应基于实际结果，而不是预设双流或物理分组一定优越。
