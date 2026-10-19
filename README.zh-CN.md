# PMDFNet：面向多模态滑坡分割的物理动机双流融合网络

[English README](README.md)

本仓库当前以 **PMDFNet** 在 **converted MMLSv2** 上的论文级训练与复现实验为主线。比赛提交、旧版训练、SKG 分支和历史分析代码已归档到 `archive/legacy/`，不参与当前论文训练主流程。

项目显示名和仓库目录均为 PMDFNet；Python 模块和历史结果中保留的 `DualSegFormer`/`dual_segformer_*` 仅作为兼容标识，不改变旧 checkpoint 和复现入口。

## 目录结构

```text
PMDFNet/
├── train_ablation.py                 # 配置驱动的论文实验训练器
├── exp_train.sh                      # 当前论文实验总入口
├── scripts/                          # 训练启动脚本
│   ├── train_paper_experiments.sh    # 论文协议入口
│   ├── train_paper_experiments.py
│   └── ...                           # 当前论文启动辅助脚本
├── configs/                          # 训练配置和描述子配置
├── networks/                         # 模型结构
├── losses/                           # 损失函数
├── dataset.py                        # 数据集定义
├── augmentations.py                  # 训练增强
├── reproducibility.py                # 随机种子和确定性设置
├── doc/                              # 当前实验计划
├── tools/                            # 当前数据、评估和验证工具
│   ├── data/                         # MMLSv2 映射
│   ├── evaluation/                   # 冻结 checkpoint 的 val/test 评估
│   └── validation/                   # 数据与显存冒烟检查
└── archive/legacy/                   # 旧版训练、SKG、提交和分析代码
```

## 安装

```bash
conda create -n pmdfnet python=3.10
conda activate pmdfnet
pip install -r requirements.txt
```

## 数据映射

映射脚本位于 `tools/data/map_mmlsv2_to_mars.py`。它将 MMLSv2 的通道顺序转换为旧比赛格式，保留原始数据并写出新的 mapped 数据集。

```bash
python tools/data/map_mmlsv2_to_mars.py \
  --source-root /path/to/mmlsv2 \
  --output-root /path/to/mmlsv2_mapped_mars_ls
```

## 冻结模型评估

测试评估只能在模型、超参数和 checkpoint 选择规则冻结后执行：

```bash
CHECKPOINT=/path/to/best.pth \
DATA_ROOT=/path/to/mmlsv2_mapped_mars_ls \
  bash tools/evaluation/evaluate_reproduction_test.sh
```

历史结果汇总和比赛提交工具位于 `archive/legacy/tools/`，不属于当前论文实验协议。



