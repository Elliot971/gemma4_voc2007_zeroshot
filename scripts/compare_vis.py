"""
compare_vis.py - Side-by-side detection visualization: GT vs Gemma-4 (vs PLN).

Generates comparison images with two or three panels:
  - Without PLN results: Left = GT (green), Right = Gemma (blue) + GT overlay
  - With PLN results:    Left = PLN (red) + GT, Right = Gemma (blue) + GT

PLN results JSON format (exported by PLN project's eval.py):
  {
    "000001": {"boxes": [[x1,y1,x2,y2], ...], "labels": ["dog", ...], "scores": [0.9, ...]},
    ...
  }

Usage:
  # Gemma vs GT only (no PLN needed)
  python compare_vis.py --img-id 000001 000004 000006

  # With PLN results for cross-paradigm comparison
  python compare_vis.py --pln-results pln_detections.json --img-id 000001

  # Auto-select interesting cases
  python compare_vis.py --auto-select --num 10
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

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
    pln_results: dict = None,
    output_dir: Path = None,
):
    """Generate a comparison image for a single sample.

    Two modes:
      - pln_results is None: two-panel (GT | Gemma+GT)
      - pln_results provided: two-panel (PLN+GT | Gemma+GT)
    """
    img_path = voc_root / "JPEGImages" / f"{img_id}.jpg"
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return

    image = Image.open(img_path).convert("RGB")
    img_w, img_h = image.size

    # Ground truth (green dashed on both panels)
    gt_boxes, gt_labels = load_gt_for_image(voc_root, img_id)

    # Gemma detections (blue)
    gemma_dets = parse_detections(gemma_raw, img_w, img_h)
    gemma_boxes = [d["bbox"] for d in gemma_dets]
    gemma_labels = [d["label"] for d in gemma_dets]

    # Canvas: two panels side by side
    gap = 20
    canvas_w = img_w * 2 + gap
    header_h = 30
    canvas = Image.new("RGB", (canvas_w, img_h + header_h), "white")

    # Determine left panel mode
    has_pln = pln_results is not None and img_id in pln_results
    if has_pln:
        pln_entry = pln_results[img_id]
        pln_boxes = pln_entry.get("boxes", [])
        pln_labels = pln_entry.get("labels", [])
        left_title = "PLN (Red) + GT (Green)"
    else:
        left_title = "GT (Green)"

    # Left panel
    left = image.copy()
    draw_left = ImageDraw.Draw(left)
    draw_boxes(draw_left, gt_boxes, gt_labels, "green", width=2, dash=not has_pln)
    if has_pln:
        draw_boxes(draw_left, pln_boxes, pln_labels, "red", width=3)
        draw_boxes(draw_left, gt_boxes, gt_labels, "green", width=2, dash=True)
    canvas.paste(left, (0, header_h))

    # Right panel: Gemma + GT
    right = image.copy()
    draw_right = ImageDraw.Draw(right)
    draw_boxes(draw_right, gt_boxes, gt_labels, "green", width=2, dash=True)
    draw_boxes(draw_right, gemma_boxes, gemma_labels, "blue", width=3)
    canvas.paste(right, (img_w + gap, header_h))

    # Panel titles
    draw_canvas = ImageDraw.Draw(canvas)
    draw_canvas.text((10, 8), left_title, fill="green" if not has_pln else "red")
    draw_canvas.text((img_w + gap + 10, 8), "Gemma-4 (Blue) + GT (Green)", fill="blue")

    # Save
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "compare_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"compare_{img_id}.jpg"
    canvas.save(str(save_path))

    pln_count = len(pln_boxes) if has_pln else "-"
    print(
        f"[{img_id}] GT={len(gt_boxes)}, PLN={pln_count}, "
        f"Gemma={len(gemma_dets)}, saved to {save_path}"
    )


def find_voc_root():
    """Auto-detect VOC dataset root."""
    from test_single import find_voc_root as _find

    return _find()


def main():
    parser = argparse.ArgumentParser(
        description="GT vs Gemma-4 (vs PLN) detection comparison visualization"
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
    parser.add_argument(
        "--pln-results",
        type=str,
        default=None,
        help="PLN detection results JSON (optional, for cross-paradigm comparison)",
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

    # Load PLN results (optional)
    pln_results = None
    if args.pln_results:
        pln_path = Path(args.pln_results)
        if not pln_path.exists():
            print(
                f"WARNING: PLN results file not found: {pln_path}, skipping PLN overlay"
            )
        else:
            with open(pln_path) as f:
                pln_results = json.load(f)
            print(f"Loaded PLN results for {len(pln_results)} images")

    # Select images
    if args.img_id:
        img_ids = args.img_id
    elif args.auto_select:
        scored = []
        for img_id, raw_text in raw_outputs.items():
            img_path = voc_root / "JPEGImages" / f"{img_id}.jpg"
            if not img_path.exists():
                continue
            image = Image.open(img_path)
            dets = parse_detections(raw_text, image.size[0], image.size[1])
            image.close()
            gt_boxes, _ = load_gt_for_image(voc_root, img_id)
            # Prioritize images with detection vs GT count mismatch
            scored.append((img_id, len(dets), len(gt_boxes)))

        scored.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
        img_ids = [s[0] for s in scored[: args.num]]
        print(f"Auto-selected {len(img_ids)} representative images")
    else:
        img_ids = list(raw_outputs.keys())[:10]

    for img_id in img_ids:
        raw_text = raw_outputs.get(img_id, "")
        create_comparison(
            voc_root,
            img_id,
            raw_text,
            pln_results=pln_results,
            output_dir=output_dir,
        )

    print(f"\nGenerated {len(img_ids)} comparison images in {output_dir}/")


if __name__ == "__main__":
    main()
