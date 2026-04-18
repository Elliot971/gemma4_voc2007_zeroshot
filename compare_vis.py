"""
compare_vis.py - PLN vs Gemma-4 检测结果对比可视化

输入同一张图片，左边显示 PLN 的检测框（红色），右边显示 Gemma-4 的检测框（蓝色），
同时叠加 GT（绿色虚线），便于 Case Study 分析。

用法:
  # 对比单张图
  python compare_vis.py --img-id 000001

  # 对比多张指定图
  python compare_vis.py --img-id 000001 000004 000006

  # 自动选出 PLN 失败但 Gemma 成功的 case (需先跑完两边的评测)
  python compare_vis.py --auto-select --num 10
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from test_single import VOC_CLASSES, match_voc_class, parse_detections


def load_gt_for_image(voc_root: Path, img_id: str):
    """从 VOC XML 读取单张图的 GT"""
    ann_path = voc_root / "Annotations" / f"{img_id}.xml"
    boxes, labels = [], []
    if ann_path.exists():
        tree = ET.parse(str(ann_path))
        for obj in tree.getroot().findall("object"):
            diff = obj.find("difficult")
            if diff is not None and int(diff.text) == 1:
                continue
            name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")
            boxes.append(
                [
                    float(bbox.find("xmin").text),
                    float(bbox.find("ymin").text),
                    float(bbox.find("xmax").text),
                    float(bbox.find("ymax").text),
                ]
            )
            labels.append(name)
    return boxes, labels


def draw_boxes(draw, boxes, labels, color, width=3, dash=False):
    """在图上画检测框"""
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        if dash:
            # 绘制虚线矩形（简单模拟）
            step = 10
            for start in range(int(x1), int(x2), step * 2):
                end = min(start + step, int(x2))
                draw.line([(start, y1), (end, y1)], fill=color, width=width)
                draw.line([(start, y2), (end, y2)], fill=color, width=width)
            for start in range(int(y1), int(y2), step * 2):
                end = min(start + step, int(y2))
                draw.line([(x1, start), (x1, end)], fill=color, width=width)
                draw.line([(x2, start), (x2, end)], fill=color, width=width)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw.text((x1, max(0, y1 - 15)), label, fill=color)


def create_comparison(
    voc_root: Path,
    img_id: str,
    gemma_raw: str,
    pln_dets: dict = None,
    output_dir: Path = None,
):
    """生成单张图的对比可视化"""
    img_path = voc_root / "JPEGImages" / f"{img_id}.jpg"
    if not img_path.exists():
        print(f"图片不存在: {img_path}")
        return

    image = Image.open(img_path).convert("RGB")
    img_w, img_h = image.size

    # GT (绿色虚线)
    gt_boxes, gt_labels = load_gt_for_image(voc_root, img_id)

    # Gemma 检测 (蓝色)
    gemma_dets = parse_detections(gemma_raw, img_w, img_h)
    gemma_boxes = [d["bbox"] for d in gemma_dets]
    gemma_labels = [d["label"] for d in gemma_dets]

    # 创建并排对比图 (左: PLN 红色, 右: Gemma 蓝色)
    canvas_w = img_w * 2 + 20  # 20px 间隔
    canvas = Image.new("RGB", (canvas_w, img_h + 40), "white")

    # 左图: PLN
    left = image.copy()
    draw_left = ImageDraw.Draw(left)
    draw_boxes(draw_left, gt_boxes, gt_labels, "green", width=2, dash=True)
    if pln_dets:
        pln_boxes = pln_dets.get("boxes", [])
        pln_labels = pln_dets.get("labels", [])
        draw_boxes(draw_left, pln_boxes, pln_labels, "red", width=3)
    canvas.paste(left, (0, 30))

    # 右图: Gemma
    right = image.copy()
    draw_right = ImageDraw.Draw(right)
    draw_boxes(draw_right, gt_boxes, gt_labels, "green", width=2, dash=True)
    draw_boxes(draw_right, gemma_boxes, gemma_labels, "blue", width=3)
    canvas.paste(right, (img_w + 20, 30))

    # 标题
    draw_canvas = ImageDraw.Draw(canvas)
    draw_canvas.text((img_w // 2 - 30, 5), "PLN (Red)", fill="red")
    draw_canvas.text((img_w + 20 + img_w // 2 - 40, 5), "Gemma-4 (Blue)", fill="blue")

    # 保存
    if output_dir is None:
        output_dir = Path(__file__).parent / "compare_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"compare_{img_id}.jpg"
    canvas.save(str(save_path))
    print(f"[{img_id}] GT={len(gt_boxes)}, Gemma={len(gemma_dets)}, 保存至 {save_path}")


def find_voc_root():
    """自动查找 VOC 数据集路径"""
    from test_single import find_voc_root as _find

    return _find()


def main():
    parser = argparse.ArgumentParser(description="PLN vs Gemma-4 对比可视化")
    parser.add_argument(
        "--img-id", nargs="+", type=str, default=None, help="指定图片 ID"
    )
    parser.add_argument("--voc-root", type=str, default=None)
    parser.add_argument(
        "--gemma-raw-outputs",
        type=str,
        default="eval_results/raw_outputs.json",
        help="Gemma 原始输出 JSON 文件",
    )
    parser.add_argument("--output-dir", type=str, default="compare_results")
    parser.add_argument(
        "--auto-select", action="store_true", help="自动选出有代表性的 case"
    )
    parser.add_argument("--num", type=int, default=10, help="自动选择的图片数量")
    args = parser.parse_args()

    voc_root = Path(args.voc_root) if args.voc_root else find_voc_root()
    output_dir = Path(args.output_dir)

    # 加载 Gemma 原始输出
    raw_path = Path(args.gemma_raw_outputs)
    if not raw_path.exists():
        print(f"Gemma 原始输出文件不存在: {raw_path}")
        print("请先运行 eval_voc.py 生成评测结果")
        sys.exit(1)

    with open(raw_path) as f:
        raw_outputs = json.load(f)

    if args.img_id:
        img_ids = args.img_id
    elif args.auto_select:
        # 选出检测数量差异大的图片（可能是有趣的 case）
        scored = []
        for img_id, raw_text in raw_outputs.items():
            img_path = voc_root / "JPEGImages" / f"{img_id}.jpg"
            if not img_path.exists():
                continue
            image = Image.open(img_path)
            dets = parse_detections(raw_text, image.size[0], image.size[1])
            image.close()
            gt_boxes, _ = load_gt_for_image(voc_root, img_id)
            scored.append((img_id, len(dets), len(gt_boxes)))

        # 选检测数和 GT 数差异大的 + 一些检测准确的
        scored.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
        img_ids = [s[0] for s in scored[: args.num]]
        print(f"自动选出 {len(img_ids)} 张有代表性的图片")
    else:
        # 默认取前 10 张
        img_ids = list(raw_outputs.keys())[:10]

    for img_id in img_ids:
        raw_text = raw_outputs.get(img_id, "")
        create_comparison(voc_root, img_id, raw_text, output_dir=output_dir)

    print(f"\n共生成 {len(img_ids)} 张对比图, 保存在 {output_dir}/")


if __name__ == "__main__":
    main()
