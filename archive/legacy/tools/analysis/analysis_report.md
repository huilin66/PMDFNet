# PMDFNet 火星滑坡分割复盘分析报告

生成日期：2026-07-06  
分析仓库：`E:\repository\PMDFNet`，对照仓库：`E:\repository\M3LSNet`

## 1. 本地文件与任务入口

用户提到的 `train.csv` / `val.csv` 在 `PMDFNet` 根目录未发现；当前实际存在的是：

- `training_metrics.csv`：240 行，包含线上提交分数、实验备注、部分本地 best mIoU。
- `val_metrics.csv`：10 行，包含后续本地验证实验，无线上分数。

后续分析将 `training_metrics.csv` 视作“线上提交记录”，将 `val_metrics.csv` 视作“后续本地验证记录”。这是一个需要人工确认的文件名差异。

## 2. PMDFNet 代码结构与实际调用链

### 2.1 训练入口

`train.py` 是当前整理版主训练入口：

- `train_pipeline(model_name, conduct_val=False)` 硬编码大部分超参：`LR=1e-4`、`WEIGHT_DECAY=5e-4`、`EPOCHS=100`、`VAL_INTERVAL=1`、`IN_CHANNELS=7`。
- 数据路径硬编码为 `/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase_updateB2`。
- `model = get_model(model_name, in_channels=7, num_classes=2)`。
- 训练集：`MarsSegDataset(split="train")` 后再包一层 `MosaicCastDataset`。
- 验证集：`MarsSegDataset(split="val")`。
- 损失函数固定为 `UnetFormerLoss()`。
- 每个 epoch 覆盖保存 `last.pth`；只有 `conduct_val=True` 且本轮 mIoU 更高时保存 `best.pth`。
- `__main__` 中会顺序训练多个模型，当前包含 `m3lsnet`、`ocrnet_hrnet_w48`、`upernet_convnexttiny`、`segformer_mitb2`、`segformer_convnexttiny` 和多个 `dual_segformer_*`。

关键问题：

- 训练入口没有 CLI 参数，不便于消融复现。
- checkpoint 只保存纯 `state_dict`，不保存 config / optimizer / scheduler / epoch / metric。
- 本地 metric 是“逐 batch 计算再平均”，不是严格的全验证集全局混淆矩阵。
- `device` 固定优先 `cuda:1`，多机迁移容易踩坑。

### 2.2 Dataset 与归一化

整理版 `dataset.py` 保留了：

- `MarsSegDataset`：训练/验证用，按 split 使用 `MARS_MEAN_TRAIN`、`MARS_MEAN_VAL`、`MARS_MEAN_TEST` 做 7 通道全局标准化。
- `MarsSegDatasetInferV0`：实际固定读取 `val/images`，更像验证分析用。
- `MarsSegDatasetInferV1`：测试用，split=test 时使用 `MARS_MEAN_TEST` / `MARS_STD_TEST`。
- `MarsSegDatasetInferV2`：测试时对 channel 2 使用 `MARS_MEAN_TESTB` / `MARS_STD_TESTB` 做分布对齐，并与常规标准化软融合，权重 `a=0.33`。
- `MosaicCastDataset`：50% 概率将 4 张图拼成 256x256 后随机裁回 128x128。

重要风险：

- 本地代码注释/统计暗示 channel 2 是 DEM；但当前公开 MMLSv2 GitHub 文档写的是 band order：B1 Red、B2 Green、B3 Blue、B4 DEM、B5 Slope、B6 Thermal inertia、B7 Grayscale。也就是说，公开文档与本地代码的通道语义存在明显不一致。需要确认比赛下载包的真实 band order，不能只看 README。
- 当前公开 MMLSv2 文档还写 image value range 为 0.0 到 1.0；本地均值/std 明显是原始尺度或后处理尺度。这说明本地 `*_updateB2` 与公开仓库当前说明可能不是完全同一预处理形态。

### 2.3 Augmentation

`MarsAugmentor` 实际启用：

- 随机垂直/水平 flip。
- 随机 90/180/270 度旋转。
- 20% 概率对部分通道加高斯噪声。
- `MMLSv2RandomResizedCropv2`，默认 `scale=(0.2, 2.0)`、`p=0.3`。

`MMLSv2ColorJitter` 已定义但在 `MarsAugmentor` 中注释掉。`training_metrics.csv` 中有 `add colorjit`，说明比赛过程有试过，但整理版主训练链路未启用。

### 2.4 Model

`networks/__init__.py::get_model()` 是模型工厂。整理版保留了大量 SMP 模型、HF OneFormer/Mask2Former/OCRNet、M3LSNet、UnetFormer，以及大量 dual wrapper 名称。

PMDFNet 的核心是：

- 两个同构 SMP Segformer 分支。
- `chv1`：`[0,1,2,3]` 与 `[4,5,6]`。
- `chv2`：`[0,1,2]` 与 `[4,5,6]`。
- `chv3`：`[0,1,2]` 与 `[3,4,5,6]`。
- 融合：`add` / `cat` / `att` / `moe` / `moev2`。
- 最终只使用主分支 decoder 和 segmentation head，辅助分支只贡献 encoder features。

从线上记录看，关键保留下来的最终方向不是 M3LSNet，而是 ConvNeXt + SegFormer/UPerNet + dual-branch fusion + TTA/ensemble。

### 2.5 Loss 与 Metric

当前训练固定用 `UnetFormerLoss`：label smoothing CE + multiclass Dice，`ignore_index=255`。但是 mask 是二值 `[0,1]`，代码没有显式处理 ignore label，除非数据里真的有 255 才生效。

本地 `calculate_metrics()`：

- `argmax(logits)` 得到 0/1。
- 计算 FG precision/recall/F1、IoU_FG、IoU_BG、mIoU。
- 在验证循环中对每个 batch 的指标取平均。

如果官方 mIoU 是全测试集像素级汇总后再算 `(IoU_BG + IoU_FG)/2`，本地 batch-average 会有偏差，尤其 batch size 不同、空前景/小目标比例不同的时候。

### 2.6 推理、TTA、后处理

整理版有三个推理脚本：

- `tools/submission/generate_submission.py`：单模型、无 TTA，`argmax` 保存 tif。
- `tools/submission/generate_submission_aug.py`：单模型 TTA，多尺度/flip/rotation，默认仍偏硬编码。
- `tools/submission/generate_submission_voting.py`：多模型 TTA 后保存概率与 mask，再 hard vote / soft vote。soft voting 阈值固定 `>0.5`。

`tools/submission/generate_submission_voting.py` 中，若 `ana=True` 默认数据根为 phase 1 dataset，并用 `MarsSegDatasetInferV0` 读取 val；若用于真实 test 需要把 `ana=False` 并切换到 `Mars_LSc_2025_test_data_2nd_phase_updateB2`。这是线上/本地混用时容易出错的位置。

## 3. PMDFNet 与 M3LSNet 差异

### 3.1 原始仓库保留了大量试验痕迹

`M3LSNet` 中存在但整理版删除/收敛的文件：

- `data_merge.py`：把 val 复制进 train，同时保留 val/test 结构。
- `data_repair.py`、`data_sta.py`、`data_sta_vis.py`：数据修复与统计。
- `generate_val_results.py`、`result_process.py`、`padding_tools/*`：验证可视化、后处理和 padding 实验。
- `train_aux.py`：额外批量训练入口。
- 一篇 `M3LSNet` PDF。

整理版把 loss 移到 `losses/` 包，并新增 `requirements.txt`、`training_metrics.csv`、`val_metrics.csv`。

### 3.2 关键逻辑差异

- 原始 `train.py` 的验证代码被整段注释掉；整理版通过 `conduct_val=True` 恢复验证和 `best.pth` 保存。
- 原始训练数据使用 `MartianLandslideDatasetV2`；整理版对应改名为 `MarsSegDataset`，逻辑基本等价。
- 原始 `dataset.py` 有大量 `MartianLandslideDatasetV21` 到 `V29`、`V3/V4/3Band`，主要试验 test/isolated test 的统计对齐与通道策略；整理版只保留 `InferV0/V1/V2` 三个推理变体。
- 原始 `tools/submission/generate_submission_aug.py` 中写死真实 phase 2 test 根目录；整理版 `tools/submission/generate_submission_voting.py` 默认回到了 phase 1 根目录，真实提交路径被注释。这是整理版与比赛过程的关键不一致。
- 原始仓库中 `data_merge.py` 明确写了“复制 val 到 new_train”，说明比赛过程可能存在 train+val 合并训练策略；整理版主训练入口没有保留这个开关。

### 3.3 可能是最终提交关键改动

结合 `training_metrics.csv`，更可能贡献线上成绩的因素：

- ConvNeXt backbone 的 SegFormer / UPerNet。
- PMDFNet branch（legacy key `dual_segformer_*`），尤其 `dual_segformer_convnextsmall_chv1_add`。
- 使用 `last.pth` 而非单纯 best.pth 的若干提交。
- TTA：flip、rotation、多尺度，尤其 `aug-last`、`aug-last-ufloss` 备注。
- soft voting ensemble。
- test/phase2 统计对齐变体：v22/v28ms 这类记录在 ensemble 后较高。

更可能只是整理代码或辅助试验：

- README 改写、requirements 补齐、loss 包目录整理。
- 删除数据统计/修复/可视化脚本。
- 默认推理路径回到 phase 1 dataset。
- 只保留少量 dataset infer 变体，丢失了 V21-V29 的完整消融链。

## 4. 比赛来源与 test 是否 public

### 4.1 本地证据

本地路径反复出现：

- `Mars_LSc_2025_dataset_1st_phase_updateB2`
- `Mars_LSc_2025_test_data_2nd_phase_updateB2`

`tools/data/data_prepare.py` 对 test 的处理只组织 images；`data_merge.py` 也对 `test/masks` 做可选处理。这说明比赛期间 test 很可能是“图像公开、label 隐藏”。

### 4.2 公开网页证据

可确认来源：

- PBVS 2026 challenge 页面列出 “1st Mars Landslide Segmentation Challenge (MARS-LS)” 和 Codabench 链接：`https://www.codabench.org/competitions/12305/`。PBVS 页面明确说该挑战鼓励 multimodal fusion、attention、transfer/domain adaptation，并提供 Codabench 页面。
- CVF challenge report 摘要写明：该挑战基于 MMLSv2，分两阶段，分别衡量 in-domain segmentation 和 isolated test region 的 spatial generalization；共有 94 teams、1072 submissions；冠军 Phase 1 mIoU 0.9023，Phase 2 mIoU 0.7958。
- MMLSv2 GitHub 当前是 public repository，README 写明数据可通过 Google Drive 下载；数据包含 664 张 train/val/test，以及 276 张 geographically disjoint isolated test；还列出了 Test 和 Isolated test 的 FG 统计与 benchmark 结果。
- MMLSv2 arXiv 摘要也写明：664 images distributed across train/validation/test splits，另有 276 images isolated test set released for spatial generalization。

参考链接：

- PBVS challenge page: https://pbvs-workshop.github.io/challenge.html
- Codabench competition: https://www.codabench.org/competitions/12305/
- MMLSv2 GitHub: https://github.com/MAIN-Lab/MMLS_v2
- Challenge report: https://openaccess.thecvf.com/content/CVPR2026W/PBVS/html/Ramos_1st_Mars_Landslide_Segmentation_Challenge_-_PBVS_2026_CVPRW_2026_paper.html
- MMLSv2 arXiv: https://arxiv.org/abs/2602.08112

### 4.3 结论

- 当前 test 是否公开：是。MMLSv2 GitHub 当前公开数据下载，并列出 test / isolated test 的 label-dependent 统计和 benchmark，说明公开研究版至少已经释放 test 信息。
- 是否能下载：当前能从 GitHub README 指向的 Google Drive 下载。
- 是否有 label：当前公开研究版很可能有 test/isolated test labels，至少 README 提供 test mask 统计和 benchmark；但我未能直接下载 Google Drive 内容确认压缩包内部结构。
- 比赛期间 test 是否公开：图像公开，label 隐藏。这个结论由本地 `Mars_LSc_2025_test_data_2nd_phase` 路径和提交脚本推断。
- 是否区分 public/private leaderboard：可访问到的静态页面确认“两阶段 Phase 1 / Phase 2”，但没有确认传统 public/private leaderboard split。不能把 Phase 1/2 等同于 public/private。
- 是否允许使用 test 做训练/伪标签/后处理调参：未能从静态 Codabench 页面确认 rules。保守建议：比赛期不应使用 test label；test images 只能用于生成提交，是否允许 pseudo-label/domain adaptation/统计归一化必须以 Codabench rules 为准。若规则未明确允许，不建议把 test 用于训练、阈值调参或后处理选择。赛后研究可以做 transductive/pseudo-label 实验，但必须单独标注，不能与比赛可比成绩混报。

## 5. 线上与本地分数不一致分析

### 5.1 相关性

对 `training_metrics.csv` 中同时有 `Best mIoU` 和 `online score` 的 141 行：

- Pearson = 0.3999
- Spearman = 0.2331

这说明线上/线下只有弱到中等线性相关，排序相关性很弱。用本地 mIoU 直接选模型，容易选错。

把 `val_metrics.csv` 的 10 个模型与 `training_metrics.csv` 中同名模型的最佳线上分数匹配，只有 8 个能匹配：

- Pearson = 0.7036
- Spearman = 0.3425

Pearson 看起来较高，主要来自 base/large dual 模型的线上分数很低；排序相关仍弱。

### 5.2 本地高但线上低

典型例子：

- `mask2former`：本地 Best mIoU 0.9899，但线上 0.25，是极端 outlier。
- `oneformer`：本地 Best mIoU 0.9927，但线上 0.33，也是极端 outlier。
- 早期多个模型本地 0.99 左右，但线上只有 0.82-0.85。
- `m3lsnet` 本地 0.94-0.97，但线上约 0.80-0.82。

这表明本地验证集很可能与线上 test 分布不一致，或某些模型的推理/resize/输入预处理与训练不匹配。

### 5.3 线上高但本地不最高

最高线上单模型/单提交集中在：

- `dual_segformer_convnextsmall_chv1_add`：最高 0.8665。
- `segformer_convnexttiny`：最高 0.8644。
- `segformer_convnextsmall`：最高 0.8631。
- `dual_upernet_convnexttiny_chv2_add`：最高约 0.8628/0.8629。

这些本地 mIoU 多在 0.96-0.98，不是本地最高。反而本地最高的 0.9927 一批线上并不领先。

### 5.4 后续本地验证与线上排序冲突

`val_metrics.csv` 显示：

- m3lsnet、ocrnet、unet、upernet、segformer 几个模型本地都是 0.8634，完全打平。
- PMDFNet Tiny/Small/Base/Large 反而逐步降低：0.8515、0.8498、0.8392、0.8354。

但线上最佳同名模型中：

- `dual_segformer_convnextsmall_chv1_add` 最高 0.8665，高于 `segformer_convnexttiny` 0.8644。
- `dual_segformer_convnexttiny_chv1_add` 最高 0.8606，也不差。

所以这个本地验证集不能可靠反映线上排序，至少不能评估 dual branch 是否有效。

### 5.5 因素影响

能从 CSV 推断的趋势：

- M3LSNet baseline：线上约 0.81，padding infer 可到 0.82，但整体落后。
- 早期 SMP/HF 模型：本地虚高，线上 0.82-0.85；Mask2Former/OneFormer 有明显崩溃提交。
- ConvNeXt + SegFormer/UPerNet：线上显著更稳。
- TTA/last checkpoint/UF loss：`aug-last`、`aug-last-ufloss` 备注对应较高分。
- PMDFNet branch：不是所有双流变体都提升；`dual_segformer_convnextsmall_chv1_add` 最强，`chv2/chv3/cat/att/moe` 需要独立消融。
- 大 backbone：`dual_segformer_convnextbase/large_chv1_add` 在后续本地分数更低，线上记录也低，可能过拟合或训练/资源不足。
- Ensemble：soft voting 后期最高约 0.7839；这是 phase 2 / isolated test 场景，低于 phase 1 单模型 0.86 是正常的分布 shift。

### 5.6 可能原因

优先级较高的原因：

- **验证集不代表线上 test**：官方 challenge report 明确有 Phase 2 isolated region，MMLSv2 也强调 isolated test 是 geographically disjoint；本地 val 是固定 66 张，很可能和 train 空间相关。
- **本地 metric 与官方 metric 可能不完全一致**：本地是 batch-average mIoU；官方通常是按全测试集累计混淆矩阵或 per-image 再平均，需查 evaluator。
- **空间相关/patch 重复**：README 也提示 validation mIoU 可因 spatial correlation 很高。随机或普通 image split 容易高估泛化。
- **test 统计归一化和 band order 风险**：CSV 里后期大量 `v2x/ms` 与 testB 统计对齐；这类策略对线上敏感，但本地 val 未必能验证。
- **推理路径混用**：整理版 voting 默认 `ana=True`，会读 val；真实 test 路径被注释，容易造成线上/线下命令不可比。
- **小目标/resize/crop 影响**：随机 resized crop scale 下限 0.2，可能改变小滑坡比例；TTA 多尺度和 threshold/postprocess 对小目标影响很大。
- **val 合并训练**：原始 `data_merge.py` 有 val 合并进 train 的痕迹，若线上提交用了 train+val，后续本地 val 就不能再作为独立验证。

## 6. 后续实验设计建议

### 6.1 是否继续使用当前本地验证集

不建议把当前 `val` 作为唯一模型选择依据。它可以保留为 smoke / sanity validation，但不应作为决定线上提交的主指标。

### 6.2 是否重新划分

建议重新划分，并保留官方 val 作为一个独立对照。更合理的是：

- 若能从文件名或元数据恢复 scene/geographic group：优先 scene-level / geographic split。
- 若无法恢复 group：做 stratified K-fold，按 foreground ratio 和 landslide area 分层。
- 同时构建一个“hard validation”：高/低 FG、极小目标、DEM/视觉统计偏离 train 的样本单独跟踪。

不推荐单纯 random split，因为 patch 空间相关会继续高估。

### 6.3 test 数据使用

- 比赛可比实验：不要用 test label；不要用线上分数反复调阈值/后处理。
- 如果 Codabench rules 未明确允许，不要把 test images 用于训练或伪标签。
- 赛后研究实验可单独做 transductive 设置：只用 test images 做统计归一化/domain adaptation/pseudo-label，但要明确标注为不可与比赛 closed-test 设置直接比较。
- 阈值调参应该在本地 validation/fold 上完成，线上只做最终验证。

### 6.4 最小可靠实验矩阵

建议先固定：数据 split、seed、epoch、optimizer、scheduler、loss、augmentation、TTA，然后一次只改一个变量。

最小矩阵：

1. Baseline：`segformer_convnexttiny`，UnetFormerLoss，当前 MarsAugmentor，no TTA。
2. 主结构：`dual_segformer_convnexttiny_chv1_add` vs baseline。
3. Backbone：tiny vs small，在 single 和 dual 各做一次。
4. Fusion：add vs cat vs att，只用同一 backbone/同一 split。
5. Channel split：chv1 vs chv2 vs chv3。
6. Loss：UnetFormerLoss vs CombinedLoss vs CE/Dice 权重变化。
7. Augmentation：无 mosaic、当前 mosaic、去掉 resized crop、轻/重 resized crop。
8. Input/stat：当前 split mean/std vs train mean/std only vs test-stat alignment，必须单独标注 transductive。
9. TTA：none、flip、flip+rot、flip+rot+ms。
10. Postprocess/threshold：argmax/0.5、概率阈值 sweep、small component remove/fill holes。
11. Ensemble：只在单模型与 TTA 固定后做，不参与主结构消融。
12. Pseudo-label：最后单独做，且只作为赛后 transductive。

### 6.5 每个实验记录字段

必须记录：

- `experiment_id`、`experiment_name`、timestamp、git commit、dirty status。
- data root、split file、fold、seed、train/val image count。
- model name、arch、encoder/backbone、pretrain、channel split、fusion。
- input size、batch size、epochs、lr、weight decay、optimizer、scheduler。
- loss、augmentation、mosaic prob、crop scale。
- checkpoint selection：best/last、best epoch、best metric。
- inference dataset variant、TTA、threshold、postprocess、ensemble members/weights。
- local global confusion matrix、mIoU、IoU_FG、IoU_BG、F1、precision、recall。
- online score 只在提交后手动追加，不参与自动选择。

### 6.6 线上分数使用原则

线上只用于少量最终验证：

- 每个实验阶段只提交 1-2 个从本地 fold 选择出的候选。
- 禁止用线上结果反复调阈值、TTA 权重和后处理。
- 若某个线上结果推翻本地趋势，回到 split/metric/推理一致性排查，而不是继续刷 leaderboard。

## 7. 新训练脚本设计方案

建议新增 `train_ablation.py`，不改旧训练脚本。功能：

- CLI + JSON config。
- 支持单实验和 batch config。
- 支持参数：data root、split file、fold、seed、model_name、arch、encoder/backbone、pretrain、input size、batch size、epochs、lr、optimizer、scheduler、loss、augmentation、mixed precision、checkpoint path、resume、val interval、output dir、experiment name。
- 默认复用 `networks.get_model()` 和现有 `MarsSegDataset` / `MosaicCastDataset`。
- 当提供 split file 时，使用新脚本内部 file-list dataset，不修改旧 `dataset.py`。
- 每次运行保存：
  - `config.json`
  - `command.txt`
  - `git.json`
  - `logs/train.log`
  - `metrics.csv`
  - `checkpoints/best.pth`
  - `checkpoints/last.pth`
- Metric 改为全验证集累计 confusion matrix，另保留 batch 平均 loss。
- checkpoint 保存完整 dict：model、optimizer、scheduler、epoch、best metric、config。
- 支持 `--dry-run` 做数据集加载和配置解析，不训练。

推荐输出目录：

`outputs_ablation/<experiment_name>_<timestamp>/`

这样不会覆盖现有 `outputs/`、checkpoint、log、CSV。

