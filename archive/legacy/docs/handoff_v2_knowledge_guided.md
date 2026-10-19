# SKG-PMDFNet：PMDFNet 的知识引导扩展实验交接文档

基础双流模型的论文统一名称为 PMDFNet；`dual_segformer_*` 等旧实现键仅作为兼容标识保留。

更新时间：2026-07-24  
仓库：`/localnvme/project/PMDFNet`
数据集：仅使用 `MMLSv2（dataB）`  
文档状态：新版实验主线，替代旧版以 `dataA/dataB + 融合搜索` 为核心的 handoff

---

## 1. 项目目标

本仓库用于完成 MMLSv2 火星滑坡多光谱语义分割的实验复现、方法升级、消融分析和期刊投稿准备。

本阶段不再继续围绕以下内容进行大规模搜索：

- dataA 与 dataB 的联合验证；
- add / concat / attention / MoE 的无约束融合枚举；
- Tiny / Small / Base / Large 的大规模 backbone 搜索；
- 将双分支、ConvNeXt 或 CE + Dice 本身作为主要创新。

新的研究主线为：

> **利用显式类别光谱知识指导 VIS/NIR 与 SWIR 双流特征融合，并提升火星滑坡分割在有限样本、缺失波段和光谱退化条件下的准确性、稳健性与可解释性。**

暂定方法名称：

```text
SKG-PMDFNet
Spectral Knowledge-Guided Dual-Stream Network
for Martian Landslide Segmentation
```

目标期刊优先级：

```text
EAAI > ESWA > KBS
```

其中：

- EAAI：强调实际工程问题、有限样本、传感器退化和稳健分割；
- ESWA：要求形成较完整的智能融合与可靠性分析体系；
- KBS：只有显式知识表示、知识注入、知识一致性和可解释性均充分时再考虑。

---

## 2. 数据集与固定实验协议

### 2.1 数据集

仅使用 MMLSv2：

```text
根目录：/scrinvme/huilin/bdd/cp_data/mmlsv2
像素值域：[0, 1]
任务：二分类语义分割
train：465
val：66
test：133，包含 mask
```

`.env`：

```env
MMLSV2_DATA_ROOT=/scrinvme/huilin/bdd/cp_data/mmlsv2
```

### 2.2 通道划分

当前暂定：

```text
channels1 = 0,1,2,3   # VIS/NIR 组
channels2 = 4,5,6     # SWIR 组
```

正式论文开始前必须完成以下核验：

1. 确认 7 个通道的准确名称；
2. 确认每个通道的中心波长或波段范围；
3. 确认各通道的物理含义和预处理方法；
4. 核对当前代码中的通道顺序是否与官方数据说明一致；
5. 将核验结果写入 `docs/mmlsv2_band_definition.md`。

在波段定义未核实前：

- 可以使用 `VIS/NIR group` 和 `SWIR group` 作为工作名称；
- 不要声称具体矿物、含水量或地貌物理机制；
- 不要沿用旧论文中的 `Thermal Inertia / Slope / DEM / RGB` 描述。

### 2.3 严格的数据使用规则

固定协议：

```text
train：模型训练、统计量计算、光谱知识库构建
val：checkpoint 选择、early stopping、超参数选择、消融决策
test：全部设计冻结后，仅用于最终评估
```

禁止：

- 使用 test 指标保存 best checkpoint；
- 使用 test 调整 loss 权重、prototype 数量、融合结构或阈值；
- 根据 test 结果决定是否保留某个模块；
- 使用 val 或 test 像素构建类别光谱原型；
- 将旧的 `train → test` 结果作为正式论文结果。

所有归一化统计量、类别原型和聚类中心必须仅由 train split 计算。

### 2.4 数据归一化

MMLSv2 已归一化至 `[0,1]`。当前 `MMLSV2Dataset` 跳过 raw-DN mean/std 标准化，此逻辑继续保留。

仍需增加可选的 train-only 标准化模式，用于对照：

```text
none            # 直接使用 [0,1]
train_zscore    # 每通道使用 train mean/std
robust          # 每通道使用 train median/IQR 或 percentile
```

标准化模式只能在 val 上选择，选定后固定用于 test。

---

## 3. 当前代码与历史结果的定位

主训练入口：

```text
train_ablation.py
```

当前已具备：

- 单流和双流模型；
- ConvNeXt 编码器；
- SegFormer/All-MLP 解码器；
- add、concat、attention、MoE、MoEv2 融合；
- 多种 loss；
- Mosaic 和常规增强；
- best checkpoint 与 CSV 汇总；
- smoke test。

当前 dataB 历史最佳参考配置：

```text
PMDFNet-Tiny
channels1 = 0,1,2,3
channels2 = 4,5,6
fusion = concat
best_mIoU   = 0.8619
best_IoU_fg = 0.8210
best_F1     = 0.9017
seed        = 42
```

注意：该结果来自旧的 `train → test` 模型选择流程，只能作为代码调试和相对趋势参考，不能进入正式结果表。

旧结果目录：

```text
outputs_jstar/                  # 归一化 bug 时产生，不可信
outputs_experiments/            # 部分为 train→test，仅作历史参考
```

新版正式结果统一存放：

```text
outputs_skg/
```

---

## 4. 新方法总体结构

完整方法包含四个核心部分：

```text
输入光谱关系构建
    ↓
类别光谱原型知识库
    ↓
知识引导的双流多尺度融合
    ↓
置信度控制的知识一致性学习
```

基础编码器固定使用 ConvNeXt-Tiny，除参数匹配实验外不再大规模搜索 backbone。

### 4.1 双流编码

输入：

```text
X_vn   = channels 0–3
X_swir = channels 4–6
```

双流特征：

```text
F_vn[i]   = Encoder_vn[i](X_vn)
F_swir[i] = Encoder_swir[i](X_swir)
```

两个编码器结构相同、参数独立。

基础解码器继续使用 SegFormer All-MLP decoder，避免同时修改编码、融合和解码三个部分。

### 4.2 显式光谱描述

从 7 通道输入构建知识描述 `Z`。候选组成：

```text
1. 原始 7 通道
2. 相邻通道差值
3. 相邻通道光谱斜率
4. VIS/NIR 与 SWIR 的归一化差异
5. 跨组波段比值或 log-ratio
6. 像素级光谱均值、方差和幅度
7. 局部窗口光谱统计
```

归一化差异的一般形式：

```text
ND(i,j) = (B_i - B_j) / (B_i + B_j + eps)
```

要求：

- 不无依据穷举全部组合；
- 优先依据真实波长和波段含义选择关系；
- 若物理含义无法完全确认，统一称为 `cross-band spectral relations`；
- 每一种描述都需经过独立 val 消融。

建议实现文件：

```text
knowledge/spectral_descriptor.py
```

建议输出：

```text
Z: [B, D, H, W]
```

### 4.3 类别光谱原型知识库

仅使用 train split 和 train mask，为背景与滑坡分别构建多个原型：

```text
P_bg = {p_bg_1, ..., p_bg_K}
P_ls = {p_ls_1, ..., p_ls_K}
```

推荐初始方案：

```text
特征空间：标准化后的 spectral descriptor Z
采样：每张图按类别均匀采样，防止大图或背景主导
聚类：MiniBatchKMeans
原型数 K：1 / 2 / 4 / 8，在 val 上选择
距离：cosine distance 为主，Euclidean 为对照
```

每个像素计算对各类别原型的最大或 soft aggregation 相似度，生成知识先验图：

```text
Q = [Q_bg, Q_ls]
```

原型管理方式分为三种消融：

```text
fixed       # 训练前计算，训练中固定
learnable   # 完全可学习
residual    # P = P0 + deltaP，推荐主方案
```

主方案采用：

```text
P = P0 + deltaP
L_proto_reg = ||deltaP||_2
```

要求：

- `P0` 只能由 train 计算；
- 每次 split 或 seed 必须独立构建；
- 保存 prototype 文件、配置、采样统计和哈希；
- 禁止将 validation/test 信息混入原型。

建议实现：

```text
knowledge/build_prototypes.py
knowledge/prototype_bank.py
knowledge/prototype_prior.py
```

原型缓存建议：

```text
artifacts/prototypes/<descriptor>/<K>/<seed>/prototypes.npz
```

### 4.4 知识引导融合模块

当前固定 concat 作为基础。新模块根据以下信息动态生成融合权重：

```text
F_vn[i]
F_swir[i]
K_i       # 多尺度知识特征
Q_i       # 多尺度类别先验图
```

推荐输出三个 gate：

```text
g_vn[i]    # VIS/NIR 分支贡献
g_swir[i]  # SWIR 分支贡献
g_inter[i] # 跨流交互特征贡献
```

融合形式：

```text
F_inter[i] = Conv([F_vn[i], F_swir[i]])

F_fuse[i] =
    g_vn[i]   * Proj_vn(F_vn[i])
  + g_swir[i] * Proj_swir(F_swir[i])
  + g_inter[i]* F_inter[i]
```

三个 gate 使用 softmax 归一化，使其具有明确的相对贡献解释。

需要保留以下实现对照：

```text
concat
add
ordinary_attention
knowledge_guided_gate
```

建议实现：

```text
models/fusion/knowledge_guided_fusion.py
```

建议先仅在 stage 3 和 stage 4 引入知识融合，之后再比较：

```text
late：stage 4
mid-late：stage 3 + 4
all：stage 1 + 2 + 3 + 4
```

避免第一版直接在全部尺度堆叠复杂模块。

### 4.5 知识一致性损失

基础分割损失固定为一个稳定配置，不将其作为创新点：

```text
L_seg = L_CE + L_Dice
```

知识一致性仅在知识先验高置信区域生效：

```text
L_kc = mean_x [w(x) * KL(Q(x) || P_pred(x))]
```

其中：

```text
w(x) = confidence(Q(x))
```

候选置信度定义：

```text
max probability
margin between top-1 and top-2 similarity
prototype distance threshold
entropy-based confidence
```

总损失：

```text
L_total = L_seg
        + lambda_kc * L_kc
        + lambda_proto * L_proto_reg
```

候选权重：

```text
lambda_kc    = 0 / 0.05 / 0.1 / 0.2 / 0.5
lambda_proto = 0 / 1e-4 / 1e-3 / 1e-2
```

只允许在 val 上选择。

建议实现：

```text
losses/knowledge_consistency.py
```

---

## 5. 正式实验阶段

所有实验先使用单 seed 完成功能和趋势筛选，最终关键实验使用 3–5 seeds。

### 阶段 0：数据与协议修复

目标：确保正式实验不存在 test 泄漏。

任务：

```text
D0.1 固定 train / val / test
D0.2 best checkpoint 仅根据 val 指标保存
D0.3 test evaluation 独立脚本，只加载冻结 checkpoint
D0.4 train-only 统计量与 prototype 缓存
D0.5 保存 split、seed、代码版本和配置
D0.6 核验 7 个通道定义
```

验收条件：

- 训练日志中不出现 test 指标；
- 测试脚本不执行训练和 early stopping；
- prototype metadata 明确标记来源为 train；
- 同一 checkpoint 重复测试结果一致。

建议新增：

```text
scripts/train_official_split.sh
scripts/eval_test_once.sh
scripts/verify_data_protocol.sh
```

### 阶段 1：锁定公平基础模型

| ID | 模型 | 目的 |
|---|---|---|
| B1 | 单流 7 通道 SegFormer | 早期融合基础线 |
| B2 | 单流 7 通道 ConvNeXt + SegFormer decoder | 与双流编码器体系一致 |
| B3 | PMDFNet-add | 旧论文结构 |
| B4 | PMDFNet-concat | 当前 dataB 参考最强基础线 |
| B5 | 参数匹配单流模型 | 排除双流参数翻倍因素 |
| B6 | 普通 attention fusion | 与知识融合进行公平比较 |

统一要求：

```text
encoder       = ConvNeXt-Tiny
input size    = 固定
loss          = 固定
augmentation  = 固定
optimizer     = 固定
epochs        = 固定上限
early stop    = 仅基于 val
TTA           = 关闭
ensemble      = 关闭
seed          = 相同
```

B5 应尽量匹配 B4 的：

```text
Params
FLOPs
训练 epoch
预训练状态
```

阶段 1 输出：

```text
outputs_skg/baselines/baseline_summary.csv
```

### 阶段 2：验证光谱知识是否独立有效

在不接入深度网络前，先评估知识本身。

任务：

```text
K1 统计每个通道的类别分布
K2 绘制滑坡/背景的均值、标准差、分位数
K3 计算波段间相关性和类别可分性
K4 PCA / UMAP 可视化
K5 构建 prototype bank
K6 直接用 prototype prior 对 val 像素分类
K7 可视化 Q_bg / Q_ls / uncertainty
K8 比较 K = 1 / 2 / 4 / 8
K9 比较 cosine / Euclidean
K10 比较不同 spectral descriptor
```

评价指标：

```text
Pixel AUC
AP
IoU_fg
F1
ECE 或 Brier score
类间距离
类内方差
```

阶段 2 的决策规则：

- 若 prototype prior 在 val 上明显优于随机或多数类，进入知识融合；
- 若 prior 只在少部分场景有效，使用置信度 mask，而非全局强制约束；
- 若 prior 几乎无区分能力，重新设计 descriptor，不能仅靠增加网络复杂度掩盖问题。

输出：

```text
outputs_skg/knowledge_analysis/
artifacts/prototypes/
figures/spectral_analysis/
```

### 阶段 3：核心方法消融

主消融矩阵：

| ID | 配置 | 作用 |
|---|---|---|
| A0 | B4：Dual-concat | 基础模型 |
| A1 | A0 + spectral descriptor 直接输入 decoder | 判断显式关系作为额外输入是否有效 |
| A2 | A0 + prototype prior 直接输入 decoder | 判断类别先验是否有效 |
| A3 | A0 + knowledge-guided fusion | 核心融合模块 |
| A4 | A3 + knowledge consistency loss | 验证一致性约束 |
| A5 | A4 + residual prototype learning | 完整模型 |
| A6 | A5 + prototype regularization | 完整推荐配置 |

知识真实性对照：

| ID | 配置 | 目的 |
|---|---|---|
| C1 | 随机 prototype | 排除额外参数收益 |
| C2 | 交换背景/滑坡 prototype | 验证语义正确性 |
| C3 | 打乱 prototype 与图像的对应关系 | 验证样本知识关联 |
| C4 | 相同参数量但不输入 Q | 参数公平对照 |
| C5 | 使用全局均值 prototype，不聚类 | 验证多原型必要性 |
| C6 | 完全可学习 prototype，无 P0 | 验证显式先验必要性 |

模块位置消融：

```text
stage4 only
stage3 + stage4
stage1–4
```

推荐优先级：

```text
先做 stage3 + stage4；只有效果稳定后再做 all-stage。
```

输出：

```text
outputs_skg/ablation/ablation_summary.csv
```

### 阶段 4：波段缺失与光谱退化实验

由于仅使用一个数据集，必须通过受控 corruption 证明稳健性。

测试对象：

```text
B4 Dual-concat
B6 ordinary attention
A6 full SKG-PMDFNet
```

测试过程中不重新训练，直接在冻结 test checkpoint 上施加退化。

#### 4.1 缺失模态

```text
单波段置零：band 0–6
VIS/NIR 整组置零：0–3
SWIR 整组置零：4–6
随机丢弃：1 / 2 / 3 bands
```

#### 4.2 噪声与模糊

```text
Gaussian noise sigma = 0.01 / 0.03 / 0.05
Gaussian blur kernel = 3 / 5 / 7
```

#### 4.3 辐射偏移

```text
gain = 0.8 / 0.9 / 1.1 / 1.2
bias = -0.05 / -0.02 / +0.02 / +0.05
gamma = 0.8 / 1.2
```

#### 4.4 局部异常

```text
随机矩形区域的 channel corruption
随机 stripe / dead-band 模拟
局部饱和或截断
```

评价：

```text
clean IoU_fg
corrupted IoU_fg
absolute drop
relative drop
mean corruption error
robustness AUC
```

可解释性要求：

- SWIR 受损时，检查 `g_swir` 是否下降；
- VIS/NIR 受损时，检查 `g_vn` 是否下降；
- 知识先验低置信区域是否由交互分支承担；
- 绘制 corruption severity–performance 曲线；
- 绘制 gate 变化曲线和空间热图。

输出：

```text
outputs_skg/robustness/
figures/robustness/
figures/gates/
```

### 阶段 5：最终多种子、统计检验与效率

推荐 seeds：

```text
42 123 7 3407 2026
```

最低要求：3 seeds；投稿最终版本建议 5 seeds。

最终比较模型：

```text
B1 / B2 / B4 / B5 / B6
A3 / A4 / A6
外部公开 baseline
```

最终指标：

```text
mIoU
IoU_fg
F1 / Dice
Precision
Recall
Boundary IoU 或 Boundary F1
Params
FLOPs
Peak GPU memory
FPS / latency
训练时间
```

统计：

```text
mean ± std across seeds
95% bootstrap CI
per-image paired comparison
Wilcoxon signed-rank test 或 paired bootstrap
```

最终 test 规则：

1. 每个 seed 根据 val 选择唯一 checkpoint；
2. 模型、超参数和后处理全部冻结；
3. 执行一次正式 test；
4. 不根据 test 结果回改设计；
5. 如需修改方法，必须重新定义实验版本并重新完成全部关键实验。

---

## 6. 推荐训练配置

第一轮统一配置建议：

```text
encoder          = convnext_tiny
channels1        = 0,1,2,3
channels2        = 4,5,6
input size       = 保持当前已验证设置
optimizer        = AdamW
lr               = 1e-4
weight decay     = 5e-4
scheduler        = cosine
max epochs       = 150 或 200
early patience   = 25
primary metric   = val_iou_fg
batch size       = 根据显存固定
mixed precision  = on
TTA              = off
ensemble         = off
```

增强建议分成两类：

允许用于所有模型的基础增强：

```text
horizontal / vertical flip
90-degree rotation
random crop / resize
```

需要谨慎评估的光谱增强：

```text
channel-selective noise
gain / bias jitter
channel dropout
```

光谱增强不能只用于完整模型，所有公平比较模型必须使用同一训练增强；鲁棒性训练作为单独扩展实验。

---

## 7. Checkpoint 与日志规范

每个 run 目录：

```text
outputs_skg/<stage>/<experiment>/<seed>/
├── config.yaml
├── environment.txt
├── git_commit.txt
├── split_manifest.json
├── prototype_metadata.json
├── training_metrics.csv
├── val_metrics.json
├── checkpoints/
│   ├── best_iou_fg.pth
│   ├── best_miou.pth
│   └── last.pth
├── predictions_val/
├── visualizations/
└── run.log
```

禁止以 test 指标命名或保存 checkpoint。

`summary.csv` 至少包含：

```text
experiment_id
seed
model
encoder
fusion
knowledge_descriptor
prototype_type
prototype_k
lambda_kc
params
flops
best_val_epoch
best_val_iou_fg
best_val_miou
final_val_iou_fg
test_iou_fg
test_miou
test_f1
status
git_commit
```

在正式 test 前，`test_*` 字段保持为空。

---

## 8. 推荐新增命令行参数

在 `train_ablation.py` 或新版 `train_skg.py` 中增加：

```text
--train-split train
--val-split val
--test-split test
--eval-test false

--knowledge-mode {none,descriptor,prototype,full}
--descriptor-config configs/descriptors/default.yaml
--prototype-path ...
--prototype-k 4
--prototype-distance {cosine,euclidean}
--prototype-update {fixed,learnable,residual}
--prototype-reg-weight 1e-3

--fusion {add,cat,att,knowledge_gate}
--knowledge-stages 3,4
--knowledge-consistency-weight 0.1
--knowledge-confidence {maxprob,margin,entropy,distance}
--knowledge-confidence-threshold 0.7

--save-gates
--save-prior-map
--save-feature-statistics
```

单独测试脚本：

```text
tools/evaluation/eval_skg.py
```

建议参数：

```text
--checkpoint
--split test
--corruption none
--severity 0
--save-predictions
--save-gates
--output-json
```

---

## 9. 建议脚本

```text
scripts/
├── 00_verify_dataset.sh
├── 01_train_baselines.sh
├── 02_build_prototypes.sh
├── 03_eval_prototype_prior.sh
├── 04_train_knowledge_ablation.sh
├── 05_train_control_ablation.sh
├── 06_eval_corruptions.sh
├── 07_train_multiseed.sh
├── 08_eval_final_test.sh
└── 09_summarize_paper_results.sh
```

推荐运行顺序：

```sh
sh scripts/00_verify_dataset.sh
sh scripts/01_train_baselines.sh
sh tools/knowledge/build_prototypes.sh
sh scripts/03_eval_prototype_prior.sh
sh scripts/04_train_knowledge_ablation.sh
sh scripts/05_train_control_ablation.sh
sh scripts/06_eval_corruptions.sh
sh scripts/07_train_multiseed.sh
sh tools/evaluation/08_eval_final_test.sh
sh scripts/09_summarize_paper_results.sh
```

---

## 10. 结果表规划

### Table 1：数据和波段定义

```text
split / samples / spatial size / channels / wavelength / preprocessing
```

### Table 2：主模型比较

```text
Method | Params | FLOPs | mIoU | IoU_fg | F1 | Boundary F1
```

### Table 3：核心消融

```text
Descriptor | Prototype | KG Fusion | KC Loss | IoU_fg | mIoU
```

### Table 4：知识真实性对照

```text
Correct / Random / Swapped / Shuffled / Learnable-only prototype
```

### Table 5：原型设计消融

```text
K | distance | update mode | prior AUC | final IoU_fg
```

### Table 6：退化稳健性

```text
Method | Clean | Missing VN | Missing SWIR | Noise | Gain Shift | Avg Drop
```

### Table 7：多种子统计与显著性

```text
Method | mean ± std | 95% CI | p-value
```

---

## 11. 图像与可解释性规划

必须生成：

```text
Fig. 1  SKG-PMDFNet 总体架构
Fig. 2  7 个波段定义和双流划分
Fig. 3  滑坡/背景光谱统计和 prototype 分布
Fig. 4  prototype prior map 与 uncertainty map
Fig. 5  knowledge-guided gate 可视化
Fig. 6  baseline 与完整模型定性对比
Fig. 7  缺失波段/噪声下的 gate 变化
Fig. 8  corruption severity–IoU 曲线
```

定性样本必须包含：

```text
大面积滑坡
小目标滑坡
边界模糊场景
高误检场景
高漏检场景
SWIR 受损场景
VIS/NIR 受损场景
知识先验正确与错误案例
```

不得只选择成功案例；至少展示一组 failure cases。

---

## 12. 论文贡献的固定表述方向

最终贡献不再写：

```text
双流网络
ConvNeXt backbone
SegFormer decoder
CE + Dice loss
```

推荐三项贡献：

### Contribution 1：显式类别光谱知识表示

从训练集提取多原型类别光谱知识，将滑坡和背景的跨波段统计规律组织成可解释、可复用的知识表示。

### Contribution 2：知识引导的双流融合

利用类别光谱先验动态调节 VIS/NIR、SWIR 与跨流交互特征的贡献，替代固定 add 或 concat 融合。

### Contribution 3：面向波段退化的知识一致性学习

通过置信度控制的知识一致性约束和退化评估，提高模型在缺失波段、噪声和辐射偏移下的稳健性，并提供可解释的模态贡献分析。

---

## 13. 成功判定标准

只有满足以下条件，才认为方法达到期刊投稿准备阶段。

### 最低条件

```text
1. 严格 train/val/test，无 test 泄漏
2. 完整模型在 3 个以上 seed 上稳定优于 Dual-concat
3. 参数匹配单流对照仍明显低于完整模型
4. 正确 prototype 优于随机、交换和打乱 prototype
5. 知识先验本身在 val 上具有可测量的区分能力
6. 缺失或退化波段下，完整模型性能下降更小
7. gate 对退化模态作出合理响应
8. 报告计算成本和统计显著性
```

### 建议目标

```text
Clean IoU_fg：相对 B4 提升 ≥ 1.0 个百分点
Robustness：平均性能下降相对 B4 减少 ≥ 15%
多 seed：提升方向一致
统计检验：p < 0.05 或 bootstrap CI 不跨 0
```

上述数值是实验目标，不是预设结论。若 clean 精度提升较小，但退化稳健性和可解释性显著增强，也可形成 EAAI/ESWA 的完整故事。

### 停止或转向条件

出现以下情况时暂停继续堆模块：

```text
prototype prior 接近随机
知识真实性对照无明显差异
完整模型收益仅来自参数增加
多 seed 提升不稳定
退化场景下 gate 不响应输入质量
知识一致性导致错误先验被强化
```

此时优先重新检查波段定义、descriptor 和 prototype 采样，不继续增加 attention/MoE 模块。

---

## 14. 当前最高优先级任务

按以下顺序执行，不并行扩展过多支线。

### P0：必须先完成

```text
[ ] 核实 MMLSv2 七通道定义
[ ] 修复为 train→val 选模型，test 独立评估
[ ] 新建 outputs_skg 目录体系
[ ] 重跑 B1–B6，建立公平 baseline
[ ] 实现参数匹配单流模型
```

### P1：知识可行性验证

```text
[ ] 实现 spectral descriptor
[ ] 构建 train-only prototype bank
[ ] 评估 prototype prior 的 val AUC / IoU / F1
[ ] 完成 K、距离和 descriptor 对比
[ ] 输出光谱统计和 prior map
```

### P2：核心模型

```text
[ ] 实现 knowledge-guided fusion
[ ] 实现 confidence-aware knowledge consistency loss
[ ] 完成 A0–A6
[ ] 完成 C1–C6
```

### P3：期刊级验证

```text
[ ] 受控退化测试
[ ] 3–5 seeds
[ ] 统计检验
[ ] FLOPs / Params / FPS / 显存
[ ] gate、prototype 和 failure case 可视化
```

---

## 15. 推荐目录结构

```text
PMDFNet/
├── train_skg.py
├── tools/evaluation/eval_skg.py
├── dataset.py
├── env_utils.py
├── tools/analysis/summarize_results.py
├── configs/
│   ├── baselines/
│   ├── skg/
│   ├── descriptors/
│   └── corruptions/
├── knowledge/
│   ├── spectral_descriptor.py
│   ├── build_prototypes.py
│   ├── prototype_bank.py
│   └── prototype_prior.py
├── models/
│   ├── dual_segformer.py
│   └── fusion/
│       └── knowledge_guided_fusion.py
├── losses/
│   └── knowledge_consistency.py
├── analysis/
│   ├── analyze_bands.py
│   ├── analyze_prototypes.py
│   ├── statistical_test.py
│   └── plot_robustness.py
├── scripts/
│   ├── 00_verify_dataset.sh
│   ├── 01_train_baselines.sh
│   ├── 02_build_prototypes.sh
│   ├── 03_eval_prototype_prior.sh
│   ├── 04_train_knowledge_ablation.sh
│   ├── 05_train_control_ablation.sh
│   ├── 06_eval_corruptions.sh
│   ├── 07_train_multiseed.sh
│   ├── 08_eval_final_test.sh
│   └── 09_summarize_paper_results.sh
├── docs/
│   ├── mmlsv2_band_definition.md
│   └── experiment_protocol.md
├── artifacts/
│   └── prototypes/
└── outputs_skg/
    ├── baselines/
    ├── knowledge_analysis/
    ├── ablation/
    ├── controls/
    ├── robustness/
    └── final/
```

---

## 16. 环境与依赖

当前环境：

```text
conda env：M3LSNet
python：/home/23039356r/.conda/envs/M3LSNet/bin/python
```

基础依赖：

```text
torch==2.5.1+cu121
tifffile
segmentation_models_pytorch
timm
python-dotenv
tqdm
numpy
scipy
scikit-learn
pandas
matplotlib
```

可选分析依赖：

```text
umap-learn
statsmodels
```

每个正式 run 保存：

```sh
python -V
pip freeze
nvidia-smi
git rev-parse HEAD
git diff --stat
```

---

## 17. 历史问题与继承约束

| 问题 | 原因 | 新版处理 |
|---|---|---|
| MMLSv2 精度曾约 0.55 | 对 `[0,1]` 数据错误应用 raw-DN mean/std | 保留 `MMLSV2Dataset` 跳过旧标准化逻辑 |
| 旧实验使用 train→test | test 被用于 best checkpoint 选择 | 改为 train→val，test 独立评估 |
| 多个 run 后期前景坍缩 | 训练不稳定或 checkpoint 使用错误 | 保存 best-val 与 last，并分析 collapse |
| 双流模型参数近乎翻倍 | 与单流比较不公平 | 增加参数匹配单流 baseline |
| CE + Dice 被描述为创新 | 属于常规损失，且历史结果不稳定 | 固定为训练配置，不列为贡献 |
| 通道物理含义与旧论文冲突 | 旧文使用 DEM/Slope/RGB 叙述 | 以 MMLSv2 官方波段定义为唯一依据 |
| 固定 add/cat 缺少机制解释 | 融合结果依赖数据处理域 | 改为显式知识引导动态融合 |

---

## 18. 交接总结

新版项目不再回答：

> 哪一种普通融合方式在一次实验中分数最高？

而是回答：

> **训练集中的类别光谱知识能否被显式表示，并用于动态指导 VIS/NIR 与 SWIR 特征融合，从而提高火星滑坡分割的准确性、退化稳健性和可解释性？**

实验工作的核心证据链必须完整：

```text
光谱知识存在
→ 原型先验可独立区分类别
→ 正确知识优于随机/错误知识
→ 知识引导融合优于固定融合
→ 在缺失和退化波段下更稳健
→ gate 与 prototype 提供合理解释
→ 多种子与统计检验确认结果可靠
```

在该证据链完成前，不进行大规模 backbone 扩展、TTA、ensemble 或面向排行榜的后处理。
