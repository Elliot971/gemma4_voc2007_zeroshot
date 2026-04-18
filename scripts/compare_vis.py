"""
compare_vis.py - Side-by-side visualization of PLN vs Gemma-4 detections.

For the same image, draws PLN boxes (red) on the left and Gemma-4 boxes
(blue) on the right, with ground truth overlaid (green dashed) on both
panels. Useful for qualitative case-study analysis.

Usage:
  # Compare specific images
  python compare_vis.py --img-id 000001

  # Compare multiple images
  python compare_vis.py --img-id 000001 000004 000006

  # Auto-select interesting cases (requires eval_voc.py results)
  python compare_vis.py --auto-select --num 10
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from test_single import VOC_CLASSES, match_voc_class, parse_detections


def load_gt_for_image(voc_root: Path, img_id: str):
    """Load ground truth boxes and labels from a VOC XML annotation."""
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
    """Draw detection boxes on an image (supports dashed outlines for GT)."""
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        if dash:
            # Approximate dashed rectangle
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
    """Generate a side-by-side comparison image for a single sample."""
    img_path = voc_root / "JPEGImages" / f"{img_id}.jpg"
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return

    image = Image.open(img_path).convert("RGB")
    img_w, img_h = image.size

    # GT (green dashed)
    gt_boxes, gt_labels = load_gt_for_image(voc_root, img_id)

    # Gemma detections (blue)
    gemma_dets = parse_detections(gemma_raw, img_w, img_h)
    gemma_boxes = [d["bbox"] for d in gemma_dets]
    gemma_labels = [d["label"] for d in gemma_dets]

    # Create side-by-side canvas (left: PLN red, right: Gemma blue)
    canvas_w = img_w * 2 + 20  # 20px gap
    canvas = Image.new("RGB", (canvas_w, img_h + 40), "white")

    # Left panel: PLN
    left = image.copy()
    draw_left = ImageDraw.Draw(left)
    draw_boxes(draw_left, gt_boxes, gt_labels, "green", width=2, dash=True)
    if pln_dets:
        pln_boxes = pln_dets.get("boxes", [])
        pln_labels = pln_dets.get("labels", [])
        draw_boxes(draw_left, pln_boxes, pln_labels, "red", width=3)
    canvas.paste(left, (0, 30))

    # Right panel: Gemma
    right = image.copy()
    draw_right = ImageDraw.Draw(right)
    draw_boxes(draw_right, gt_boxes, gt_labels, "green", width=2, dash=True)
    draw_boxes(draw_right, gemma_boxes, gemma_labels, "blue", width=3)
    canvas.paste(right, (img_w + 20, 30))

    # Panel titles
    draw_canvas = ImageDraw.Draw(canvas)
    draw_canvas.text((img_w // 2 - 30, 5), "PLN (Red)", fill="red")
    draw_canvas.text((img_w + 20 + img_w // 2 - 40, 5), "Gemma-4 (Blue)", fill="blue")

    # Save
    if output_dir is None:
        output_dir = Path(__file__).parent / "compare_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"compare_{img_id}.jpg"
    canvas.save(str(save_path))
    print(
        f"[{img_id}] GT={len(gt_boxes)}, Gemma={len(gemma_dets)}, saved to {save_path}"
    )


def find_voc_root():
    """Auto-detect VOC dataset root."""
    from test_single import find_voc_root as _find

    return _find()


def main():
    parser = argparse.ArgumentParser(
        description="PLN vs Gemma-4 side-by-side detection comparison"
    )
    parser.add_argument(
        "--img-id", nargs="+", type=str, default=None, help="Image ID(s) to compare"
    )
    parser.add_argument("--voc-root", type=str, default=None)
    parser.add_argument(
        "--gemma-raw-outputs",
        type=str,
        default="eval_results/raw_outputs.json",
        help="Gemma raw output JSON from eval_voc.py",
    )
    parser.add_argument("--output-dir", type=str, default="compare_results")
    parser.add_argument(
        "--auto-select", action="store_true", help="Auto-select interesting cases"
    )
    parser.add_argument(
        "--num", type=int, default=10, help="Number of auto-selected images"
    )
    args = parser.parse_args()

    voc_root = Path(args.voc_root) if args.voc_root else find_voc_root()
    output_dir = Path(args.output_dir)

    # Load Gemma raw outputs
    raw_path = Path(args.gemma_raw_outputs)
    if not raw_path.exists():
        print(f"Gemma raw output file not found: {raw_path}")
        print("Run eval_voc.py first to generate evaluation results.")
        sys.exit(1)

    with open(raw_path) as f:
        raw_outputs = json.load(f)

    if args.img_id:
        img_ids = args.img_id
    elif args.auto_select:
        # Select images with large detection-count discrepancies (interesting cases)
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

        scored.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
        img_ids = [s[0] for s in scored[: args.num]]
        print(f"Auto-selected {len(img_ids)} representative images")
    else:
        # Default: first 10 images
        img_ids = list(raw_outputs.keys())[:10]

    for img_id in img_ids:
        raw_text = raw_outputs.get(img_id, "")
        create_comparison(voc_root, img_id, raw_text, output_dir=output_dir)

    print(f"\nGenerated {len(img_ids)} comparison images in {output_dir}/")


if __name__ == "__main__":
    main()
