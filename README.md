# Gemma-4 E2B 零样本目标检测：VOC 2007 评测

## 项目概述

本项目使用 Google Gemma-4 最小模型（`google/gemma-4-E2B-it`，2.3B 有效参数）对 Pascal VOC 2007 test 集进行**零样本目标检测**评测。模型不经过任何微调，直接通过自然语言 prompt 让 VLM 输出 JSON 格式的检测框，解析后使用标准 VOC mAP@0.5 指标评估。

本项目与 [PLN-ResNet18](https://github.com/Elliot971/pln_resnet18_seg) 项目共用同一份 VOC 2007 数据和同一套 `voc_eval.py` 评测代码，确保两个范式（传统 CNN 检测 vs VLM 零样本检测）的结果可以直接对比。

### 评测结果

| 方法 | mAP@0.5 | mAP@0.75 | 训练数据 | 推理速度 |
|------|---------|----------|----------|---------|
| PLN-ResNet18 (有监督) | 68.74% | --- | VOC 07+12 trainval | ~30 ms/张 |
| **Gemma-4 E2B (零样本)** | **51.51%** | **30.05%** | 无 | 4370 ms/张 |

详细分析见 `report.tex`。

### 零样本 mAP 损失分析（Gemma-4 E2B）

- **召回率**: 70.3% (8,456 / 12,032 GT 被正确检测)
- **漏检率**: 29.7% (person 类漏检最严重，1,278 个)
- **误检率**: 22.9% (2,931 / 13,191 检测框为误检)
  - 背景误检（幻觉）: 16.8%
  - 定位误差（类别对但框不准）: 12.4%
- **mAP@0.5 → mAP@0.75** 下降 41.7%，说明定位精度是 VLM 核心短板

- **零样本推理**：不做任何微调，直接 prompt 检测 20 类 VOC 物体
- **原生检测能力**：Gemma 4 支持 `box_2d` 输出格式，坐标空间为 1000x1000 归一化
- **三级容错解析**：直接 JSON 解析 -> 正则提取数组 -> 逐个提取对象
- **VOC 类别模糊匹配**：别名映射 + 子串匹配，处理 VLM 输出的同义词（如 airplane -> aeroplane）
- **断点续传**：每 50 张保存 checkpoint，4952 张全量评测中断后可从断点恢复
- **IoU 敏感性分析**：从 checkpoint 重新计算任意 IoU 阈值下的 mAP（无需重跑模型）
- **错误模式分析**：将检测框分类为正确/定位误差/误检/漏检，定量分析 mAP 损失来源

---

## 环境配置

### 系统要求

- Python 3.10
- PyTorch 2.1+, CUDA 12.1
- 正式推理：AutoDL RTX 4090 (24GB)，BF16 精度
- 本地调试：RTX 4060 (8GB)，需 4-bit 量化

### 安装依赖

```bash
# AutoDL 预装镜像 PyTorch 2.1 / Python 3.10 / CUDA 12.1
pip install transformers accelerate bitsandbytes pillow tqdm numpy

# 本地开发
conda create -n gemma python=3.10 -y
conda activate gemma
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers accelerate bitsandbytes pillow tqdm numpy
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

### 2. 全量 mAP 评测

在 VOC 2007 test 全部 4952 张图上运行评测：

```bash
# AutoDL 4090 (推荐)
export HF_ENDPOINT=https://hf-mirror.com
python scripts/eval_voc.py --voc-root /root/autodl-tmp/PLN-ResNet18/data/VOCdevkit/VOC2007

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

---

## 代码结构

```text
gemma4-voc-zeroshot/
├── scripts/
│   ├── test_single.py       # 单张图验证：模型加载、prompt、解析、可视化
│   ├── eval_voc.py          # 全量 mAP 评测：逐张推理 + 断点续传 + voc_eval
│   └── compare_vis.py       # GT vs Gemma (vs PLN) 对比可视化
├── utils/
│   ├── __init__.py
│   └── voc_eval.py          # VOC mAP 计算（与 PLN 项目共用同一份）
├── figures/                 # 报告配图
│   ├── compare_000001.jpg   # person+dog 对比
│   ├── compare_000004.jpg   # 密集场景对比
│   └── compare_000542.jpg   # cat 对比
├── report.tex               # LaTeX 评测报告
├── .gitignore
└── README.md
```

---

## 参考文献

- Gemma Team, Google (2025). *Gemma 4 Technical Report*.
- Everingham, M., et al. (2010). *The Pascal Visual Object Classes (VOC) Challenge*. IJCV.
- Wang, X., et al. (2017). *Point Linking Network for Object Detection*. arXiv:1706.03646.
