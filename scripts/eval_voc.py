"""
eval_voc.py - Gemma-4 E2B 在 VOC 2007 test 上的零样本目标检测 mAP 评测

核心设计:
  1. 逐张推理 VOC 2007 test (4952 张)
  2. 解析 Gemma 输出为 VOC 格式检测结果
  3. 复用 PLN 项目的 voc_eval.py 计算 mAP (确保同一把尺子)
  4. 支持断点续传 (checkpoint), 避免推理中断后从头开始
  5. 保存逐张原始输出, 便于后续分析 prompt 质量

用法:
  # AutoDL 4090 (推荐)
  export HF_ENDPOINT=https://hf-mirror.com
  python eval_voc.py --voc-root /root/autodl-tmp/PLN-ResNet18/data/VOCdevkit/VOC2007

  # 本地 4060 (4-bit 量化, 仅用于 debug 少量图)
  python eval_voc.py --quantize-4bit --max-images 10
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from test_single import (
    DETECTION_PROMPT,
    VOC_CLASSES,
    load_model,
    match_voc_class,
    parse_detections,
    run_inference,
)
from utils.voc_eval import voc_eval


def load_voc_ground_truths(voc_root: Path, img_ids: list):
    """读取 VOC XML 标注, 返回 voc_eval 所需格式"""
    import xml.etree.ElementTree as ET

    class_to_idx = {name: i for i, name in enumerate(VOC_CLASSES)}
    all_gts = []

    for img_id in img_ids:
        ann_path = voc_root / "Annotations" / f"{img_id}.xml"
        boxes = []
        labels = []

        if ann_path.exists():
            tree = ET.parse(str(ann_path))
            root = tree.getroot()
            for obj in root.findall("object"):
                # 跳过 difficult 标注 (VOC 标准协议)
                difficult = obj.find("difficult")
                if difficult is not None and int(difficult.text) == 1:
                    continue

                name = obj.find("name").text.lower().strip()
                if name not in class_to_idx:
                    continue

                bbox = obj.find("bndbox")
                x1 = float(bbox.find("xmin").text)
                y1 = float(bbox.find("ymin").text)
                x2 = float(bbox.find("xmax").text)
                y2 = float(bbox.find("ymax").text)

                boxes.append([x1, y1, x2, y2])
                labels.append(class_to_idx[name])

        if boxes:
            all_gts.append(
                {
                    "boxes": np.array(boxes, dtype=np.float32),
                    "labels": np.array(labels, dtype=np.int64),
                }
            )
        else:
            all_gts.append(
                {
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "labels": np.zeros((0,), dtype=np.int64),
                }
            )

    return all_gts


def detections_to_voc_format(detections: list):
    """将 parse_detections 输出转换为 voc_eval 所需的 numpy 格式"""
    class_to_idx = {name: i for i, name in enumerate(VOC_CLASSES)}

    if not detections:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
        }

    boxes = []
    scores = []
    labels = []
    for det in detections:
        boxes.append(det["bbox"])
        scores.append(det["score"])
        labels.append(class_to_idx[det["label"]])

    return {
        "boxes": np.array(boxes, dtype=np.float32),
        "scores": np.array(scores, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
    }


def save_checkpoint(ckpt_path: str, results: dict):
    """保存断点续传数据"""
    with open(ckpt_path, "w") as f:
        json.dump(results, f, ensure_ascii=False)


def load_checkpoint(ckpt_path: str):
    """加载断点续传数据"""
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Gemma-4 E2B VOC 2007 零样本 mAP 评测")
    parser.add_argument("--voc-root", type=str, default=None, help="VOC2007 根目录路径")
    parser.add_argument(
        "--quantize-4bit", action="store_true", help="4-bit 量化 (8GB 显存)"
    )
    parser.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument(
        "--max-images", type=int, default=None, help="最多评测图片数 (debug 用)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="eval_results", help="评测结果输出目录"
    )
    parser.add_argument("--resume", action="store_true", help="从上次断点继续")
    args = parser.parse_args()

    # 查找 VOC 数据
    if args.voc_root:
        voc_root = Path(args.voc_root)
    else:
        from test_single import find_voc_root

        voc_root = find_voc_root()

    print(f"VOC 数据集: {voc_root}")

    # 读取 test 集图片列表
    test_file = voc_root / "ImageSets" / "Main" / "test.txt"
    with open(test_file) as f:
        img_ids = [line.strip() for line in f if line.strip()]

    if args.max_images:
        img_ids = img_ids[: args.max_images]
    print(f"评测图片数: {len(img_ids)}")

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(output_dir / "checkpoint.json")
    raw_outputs_path = str(output_dir / "raw_outputs.json")

    # 断点续传
    raw_outputs = {}  # img_id -> raw_text
    start_idx = 0
    if args.resume:
        ckpt = load_checkpoint(ckpt_path)
        if ckpt:
            raw_outputs = ckpt.get("raw_outputs", {})
            start_idx = len(raw_outputs)
            print(f"从断点恢复: 已完成 {start_idx}/{len(img_ids)} 张")

    # 加载模型
    print(f"加载模型 {args.model_id}...")
    t0 = time.time()
    model, processor = load_model(args.model_id, args.quantize_4bit)
    print(f"模型加载完成 ({time.time() - t0:.1f}s)")

    # 逐张推理
    total_time = 0
    parse_failures = 0

    for i in tqdm(range(len(img_ids)), desc="推理进度", initial=start_idx):
        img_id = img_ids[i]

        # 跳过已完成的
        if img_id in raw_outputs:
            continue

        img_path = voc_root / "JPEGImages" / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"警告: 图片不存在 {img_path}, 跳过")
            raw_outputs[img_id] = ""
            continue

        image = Image.open(img_path).convert("RGB")

        t0 = time.time()
        try:
            raw_text = run_inference(model, processor, image)
        except Exception as e:
            print(f"推理失败 {img_id}: {e}")
            raw_text = ""
        elapsed = time.time() - t0
        total_time += elapsed

        raw_outputs[img_id] = raw_text

        # 每 50 张保存一次断点
        if (i + 1) % 50 == 0:
            save_checkpoint(ckpt_path, {"raw_outputs": raw_outputs})
            avg_time = total_time / (i - start_idx + 1)
            remaining = avg_time * (len(img_ids) - i - 1)
            print(
                f"\n[{i + 1}/{len(img_ids)}] 平均 {avg_time:.2f}s/张, "
                f"预计剩余 {remaining / 60:.1f} 分钟"
            )

    # 保存所有原始输出
    save_checkpoint(ckpt_path, {"raw_outputs": raw_outputs})
    with open(raw_outputs_path, "w") as f:
        json.dump(raw_outputs, f, ensure_ascii=False, indent=2)
    print(f"原始输出已保存至 {raw_outputs_path}")

    # ============ 解析 + 计算 mAP ============
    print("\n解析检测结果...")
    all_detections = []
    for img_id in img_ids:
        img_path = voc_root / "JPEGImages" / f"{img_id}.jpg"
        if img_path.exists():
            image = Image.open(img_path)
            img_w, img_h = image.size
            image.close()
        else:
            img_w, img_h = 500, 375  # fallback

        raw_text = raw_outputs.get(img_id, "")
        detections = parse_detections(raw_text, img_w, img_h)
        if not detections and raw_text:
            parse_failures += 1

        all_detections.append(detections_to_voc_format(detections))

    print(
        f"解析失败率: {parse_failures}/{len(img_ids)} "
        f"({parse_failures / len(img_ids) * 100:.1f}%)"
    )

    # 加载 GT
    print("加载 Ground Truth...")
    all_gts = load_voc_ground_truths(voc_root, img_ids)

    # 计算 mAP
    print("计算 mAP...")
    result = voc_eval(all_detections, all_gts)

    # 输出结果
    print(f"\n{'=' * 50}")
    print(f"Gemma-4 E2B-it 零样本检测 VOC 2007 test")
    print(f"{'=' * 50}")
    print(f"mAP @ IoU=0.5: {result['mAP'] * 100:.2f}%")
    print(f"\n{'类别':<15} {'AP (%)':<10}")
    print("-" * 25)
    for cls_name, ap in result["ap_per_class"]:
        print(f"{cls_name:<15} {ap * 100:.2f}")
    print("-" * 25)
    print(f"{'mAP':<15} {result['mAP'] * 100:.2f}")

    # 统计信息
    total_dets = sum(len(d["labels"]) for d in all_detections)
    total_gts = sum(len(g["labels"]) for g in all_gts)
    print(f"\n总检测数: {total_dets}, 总GT数: {total_gts}")
    print(f"平均每张检测数: {total_dets / len(img_ids):.1f}")
    print(f"总推理时间: {total_time / 60:.1f} 分钟")

    # 保存结果
    result_path = str(output_dir / "eval_result.json")
    with open(result_path, "w") as f:
        json.dump(
            {
                "mAP": result["mAP"],
                "ap_per_class": result["ap_per_class"],
                "total_images": len(img_ids),
                "total_detections": total_dets,
                "total_ground_truths": total_gts,
                "parse_failures": parse_failures,
                "total_time_sec": total_time,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n结果已保存至 {result_path}")


if __name__ == "__main__":
    main()
