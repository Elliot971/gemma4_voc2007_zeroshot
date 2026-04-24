# Gemma-4 E2B 目标检测三范式对比：有监督 vs 零样本 vs 微调

## 项目概述

本项目在 Pascal VOC 2007 test 集上，对三种目标检测范式进行系统对比评测：

1. **PLN-ResNet18**（有监督）：复现 Point Linking Network，Backbone 替换为 ResNet-18
2. **Gemma-4 E2B-it**（零样本）：2.3B 有效参数的 VLM，不做任何微调，直接通过自然语言 prompt 输出检测框
3. **Gemma-4 E2B-it + LoRA**（微调）：在 VOC trainval 上 LoRA 微调视觉编码器和 LLM，仅训练 3.2% 参数

三个范式使用**同一数据集**（VOC 2007 Test，4952 张）、**同一评测代码**（`voc_eval.py`）、**同一指标**（mAP@0.5），确保结果严格可比。

### 三范式对比结果

| 方法 | mAP@0.5 | mAP@0.75 | 训练数据 | 可训练参数 | 推理速度 |
|------|---------|----------|----------|-----------|---------|
| PLN-ResNet18 (有监督) | 68.74% | --- | VOC 07+12 trainval (~16k) | 165M | ~30 ms/张 |
| Gemma-4 E2B (零样本) | 51.51% | 30.05% | 无 | 0 | 4370 ms/张 |
| **Gemma-4 E2B + LoRA** | **67.58%** | **51.00%** | VOC 07+12 trainval (~25k) | 169M (3.2%) | 6620 ms/张 |

LoRA 微调将 mAP@0.5 从 51.51% 提升到 67.58%（+16.07 个百分点），与 PLN 的 68.74% 仅差 1.16 个点。在 8/20 个类别上超过 PLN。

详细分析见 `report.tex`。

### 微调关键发现

- **最大受益类别**：boat (+39.68), tvmonitor (+29.39), sofa (+26.65), diningtable (+23.42), aeroplane (+23.65)
- **已饱和类别提升有限**：cat (+1.63), dog (+0.62)
- **mAP@0.75 大幅提升**：30.05% → 51.00%（+20.95 点），定位精度显著改善
- **推理变慢**：6.62s/张（比零样本慢 51%），LoRA adapter 增加前向开销

---

## 环境配置

### 系统要求

- Python 3.12
- PyTorch 2.5.1, CUDA 12.4
- 正式推理/训练：AutoDL RTX 4090 (24GB)，BF16 精度
- 本地调试：RTX 4060 (8GB)，需 4-bit 量化

### 安装依赖

```bash
# AutoDL 预装镜像 PyTorch 2.5.1/ Python 3.12 / CUDA 12.4
pip install transformers accelerate bitsandbytes peft pillow tqdm numpy

# 本地开发（需根据实际 GPU 调整 PyTorch/CUDA 版本）
conda create -n gemma python=3.12 -y
conda activate gemma
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install transformers accelerate bitsandbytes peft pillow tqdm numpy
```

### 模型下载

Gemma-4 E2B-it 权重约 10GB，由 HuggingFace transformers 自动下载管理。国内环境建议使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

模型为 Apache 2.0 许可，无需申请权限，直接下载即可。

### 数据准备

VOC 2007 数据集不包含在仓库中。通过 `--voc-root` 参数指定路径即可：

```bash
# 下载 VOC 2007 test 集（约 1.5GB）
cd /root/autodl-tmp
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar && rm VOCtest_06-Nov-2007.tar

# 如需微调，还需 VOC 2007 + 2012 trainval（约 5GB）
# 从官网或 Kaggle 下载 VOCtrainval_06-Nov-2007.tar 和 VOCtrainval_11-May-2012.tar

# 确保目录结构如下：
# /root/autodl-tmp/VOCdevkit/VOC2007/
#   ├── JPEGImages/
#   ├── Annotations/
#   └── ImageSets/Main/test.txt
```

---

## 使用方法

### 1. 单张图片验证

先用一张图验证 prompt、JSON 解析和坐标转换是否正常：

```bash
# AutoDL 4090 (BF16)
export HF_ENDPOINT=https://hf-mirror.com
python scripts/test_single.py

# 本地 4060 (4-bit 量化)
python scripts/test_single.py --quantize-4bit

# 指定图片
python scripts/test_single.py --image path/to/image.jpg
```

输出包含模型原始文本、解析出的检测框列表，以及可视化图片。

### 2. 零样本全量 mAP 评测

在 VOC 2007 test 全部 4952 张图上运行零样本评测：

```bash
# AutoDL 4090 (推荐)
export HF_ENDPOINT=https://hf-mirror.com
python scripts/eval_voc.py --voc-root /root/autodl-tmp/VOCdevkit/VOC2007

# 中断后从断点恢复
python scripts/eval_voc.py --resume

# 本地 debug (仅跑 10 张)
python scripts/eval_voc.py --quantize-4bit --max-images 10
```

评测结果保存在 `eval_results/`：
- `eval_result.json`：mAP 和各类别 AP
- `raw_outputs.json`：每张图的模型原始输出（用于分析）
- `checkpoint.json`：断点续传数据

### 3. 对比可视化

生成 GT vs Gemma-4 的并排对比图（可选叠加 PLN 结果做跨范式对比）：

```bash
# Gemma vs GT（不需要 PLN 结果）
python scripts/compare_vis.py --img-id 000001 000004 000006

# 自动选出有代表性的 case
python scripts/compare_vis.py --auto-select --num 10

# 叠加 PLN 结果做三方对比（需要 PLN 导出的检测 JSON）
python scripts/compare_vis.py --pln-results pln_detections.json --auto-select
```

输出保存在 `compare_results/`，左侧为 GT（或 PLN+GT），右侧为 Gemma 检测（蓝框）+ GT（绿色虚线）。

### 4. 错误模式分析（可选）

从已有的 checkpoint.json 重新分析检测错误，无需重新运行模型：

```bash
# 需要 VOC 2007 test 数据集（Annotations）
python scripts/analyze_errors.py
```

输出:
- mAP@0.5, 0.6, 0.7, 0.75, 0.8, 0.9 的详细对比
- 检测错误分类：正确 / 定位误差 / 背景误检 / 类别混淆 / 重复检测
- GT 召回情况：检测到 vs 漏检
- 保存到 `eval_results/analysis_result.json`

### 5. LoRA 微调训练

#### Step 1: 准备训练数据

```bash
# VOC 2007 + 2012 trainval（16,551 张，翻转增强后 ~25k 条）
python scripts/prepare_finetune_data.py \
  --voc-root /root/autodl-tmp/VOCdevkit/VOC2007 \
  --voc12-root /root/autodl-tmp/VOCdevkit/VOC2012 \
  --split trainval --augment

# 输出: finetune_data/sft_train_trainval_aug.jsonl
```

#### Step 2: 训练

```bash
# AutoDL 4090 (LoRA bf16，完整训练)
export HF_ENDPOINT=https://hf-mirror.com
python scripts/finetune_gemma.py \
  --data finetune_data/sft_train_trainval_aug.jsonl \
  --epochs 3 --batch-size 1 --gradient-accumulation 16 \
  --lora-rank 64 --output-dir finetune_results/lora_voc

# 本地 4060 (QLoRA 4-bit，仅验证 loss 下降)
python scripts/finetune_gemma.py \
  --data finetune_data/test_demo.jsonl \
  --qlora --max-samples 20 --epochs 3 --batch-size 1
```

训练关键配置：
- LoRA rank/alpha = 64/64
- 可训练参数 169M（占总参数 5.3B 的 3.2%）
- batch size=1 + grad_accum=16（等效 batch=16）
- 学习率 2e-4，cosine scheduler + warmup
- 约 13 小时完成（3 epoch，4653 步）

#### Step 3: 评测微调后模型

```bash
python scripts/eval_voc.py \
  --adapter finetune_results/lora_voc/adapter \
  --output-dir eval_results_finetuned
```

---

## 微调训练中的关键工程问题

回顾整个微调过程，共遇到 6 个工程难题

### （1）Gemma4ClippableLinear 与 PEFT 不兼容

**问题**：Gemma 4 视觉编码器使用自定义层 `Gemma4ClippableLinear`（内含 `nn.Linear` + 输入/输出 clamp buffers），PEFT 不支持此类型，报错：
```
ValueError: Target module Gemma4ClippableLinear is not supported
```

**解决**：编写 `_patch_clippable_linear()` 函数，遍历模型找到所有 `Gemma4ClippableLinear` 实例，创建继承 `nn.Linear` 的 `PatchedClippableLinear` 子类，保留原始权重和 clamp 行为。关键细节是必须保持原始 dtype 和 device，否则推理时报 dtype mismatch。此外，使用全限定名（full dotted paths）作为 LoRA target modules，避免 PEFT 按 leaf name 误匹配。

### （2）视觉编码器 LoRA target 收集困难

**问题**：Gemma 4 视觉编码器的 Linear 层命名不规范，PEFT 的 `target_modules` 按 leaf name 匹配会误匹配到非目标层。

**解决**：编写 `_get_target_modules()` 函数，手动收集所有符合条件的模块的全限定路径（如 `vision_tower.vision_model.vision_encoder.layers.0.mlp.gate_proj`），直接传给 PEFT，避免模糊匹配。

### （3）训练数据格式与 Processor 批处理不兼容

**问题**：`Gemma4Processor` 的 `apply_chat_template` 对批处理时 `images` 参数要求嵌套列表 `[[img1], [img2]]`，单张图直接用 `[img1]` 会报错。

**解决**：在 `VLMDataCollator` 中构造正确的嵌套格式，并处理单样本情况。

### （4）坐标空间约定

**问题**：Gemma 4 原生使用 [0, 1000] 整数坐标空间，但 VOC XML 标注偶有 `xmax > img_w` 的标注误差。

**解决**：在 `prepare_finetune_data.py` 中对坐标进行 clamp 到 [0, 1000]，保持整数格式，不引入小数改变模型原生约定。

### （5）训练效率与显存平衡

**问题**：5B 参数模型在 24GB 显存上全量训练不可行。初始尝试 batch size=2 时在反向传播阶段触发 OOM。

**解决**：
- LoRA：bf16 精度，可训练参数 169M（3.2%）
- batch size=1 + grad_accum=16，等效 batch=16
- cosine scheduler + warmup
- 3 epoch，约 13 小时完成

### （6）推理时加载 LoRA adapter 的 device/dtype 一致性

**问题**：加载 adapter 后模型 dtype 变成 float32，推理时报错。

**解决**：加载 adapter 后显式调用 `model = model.to("cuda").to(torch.bfloat16)`。

---

## 代码结构

```text
gemma4-voc-zeroshot/
├── scripts/
│   ├── test_single.py             # 单张图验证：模型加载、prompt、解析、可视化
│   ├── eval_voc.py                # 全量 mAP 评测：支持零样本/微调模型 + 断点续传
│   ├── compare_vis.py             # GT vs Gemma (vs PLN) 对比可视化
│   ├── analyze_errors.py          # IoU 阈值扫描 + 错误模式分析
│   ├── prepare_finetune_data.py   # VOC XML -> SFT 训练 JSONL
│   └── finetune_gemma.py          # LoRA / QLoRA 微调训练
├── utils/
│   ├── __init__.py
│   └── voc_eval.py                # VOC mAP 计算（与 PLN 项目共用同一份）
├── finetune_data/                 # 训练数据（.gitignore 排除）
├── finetune_results/              # 微调结果（.gitignore 排除）
├── eval_results/                  # 零样本评测结果
├── eval_results_finetuned/        # 微调后评测结果
├── figures/                       # 报告配图（柱状图、loss 曲线等）
├── figures/comparisons/           # 检测框可视化对比图
├── report.tex                     # LaTeX 三范式对比报告
├── .gitignore
└── README.md
```

---

## 参考文献

- Gemma Team, Google (2025). *Gemma 4 Technical Report*.
- Everingham, M., et al. (2010). *The Pascal Visual Object Classes (VOC) Challenge*. IJCV.
- Wang, X., et al. (2017). *Point Linking Network for Object Detection*. arXiv:1706.03646.
- Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR.
