"""
eval_voc.py - Zero-shot object detection mAP evaluation on VOC 2007 test.

Core design:
  1. Run Gemma-4 E2B inference on every VOC 2007 test image (4952 total)
  2. Parse model output into VOC-format detections
  3. Reuse voc_eval.py from the PLN project (same evaluation ruler)
  4. Checkpoint support to resume after interruption
  5. Save per-image raw outputs for prompt quality analysis

Usage:
  # AutoDL 4090 (recommended)
  export HF_ENDPOINT=https://hf-mirror.com
  python eval_voc.py --voc-root /root/autodl-tmp/PLN-ResNet18/data/VOCdevkit/VOC2007

  # Local 4060 (4-bit quantized, debug only)
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

# Add project root to sys.path
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
    """Load VOC XML annotations into the format expected by voc_eval."""
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
                # Skip difficult annotations (standard VOC protocol)
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
    """Convert parse_detections output to numpy format for voc_eval."""
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
    """Save checkpoint for resumable inference."""
    with open(ckpt_path, "w") as f:
        json.dump(results, f, ensure_ascii=False)


def load_checkpoint(ckpt_path: str):
    """Load checkpoint from a previous interrupted run."""
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Gemma-4 E2B zero-shot mAP evaluation on VOC 2007 test"
    )
    parser.add_argument("--voc-root", type=str, default=None, help="VOC2007 root path")
    parser.add_argument(
        "--quantize-4bit", action="store_true", help="4-bit quantization (8GB VRAM)"
    )
    parser.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument(
        "--max-images", type=int, default=None, help="Max images to evaluate (debug)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="eval_results", help="Output directory"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # Locate VOC dataset
    if args.voc_root:
        voc_root = Path(args.voc_root)
    else:
        from test_single import find_voc_root

        voc_root = find_voc_root()

    print(f"VOC dataset: {voc_root}")

    # Load test split image list
    test_file = voc_root / "ImageSets" / "Main" / "test.txt"
    with open(test_file) as f:
        img_ids = [line.strip() for line in f if line.strip()]

    if args.max_images:
        img_ids = img_ids[: args.max_images]
    print(f"Number of images: {len(img_ids)}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(output_dir / "checkpoint.json")
    raw_outputs_path = str(output_dir / "raw_outputs.json")

    # Resume from checkpoint
    raw_outputs = {}  # img_id -> raw_text
    start_idx = 0
    if args.resume:
        ckpt = load_checkpoint(ckpt_path)
        if ckpt:
            raw_outputs = ckpt.get("raw_outputs", {})
            start_idx = len(raw_outputs)
            print(f"Resuming from checkpoint: {start_idx}/{len(img_ids)} done")

    # Load model
    print(f"Loading model {args.model_id}...")
    t0 = time.time()
    model, processor = load_model(args.model_id, args.quantize_4bit)
    print(f"Model loaded ({time.time() - t0:.1f}s)")

    # Per-image inference
    total_time = 0
    parse_failures = 0

    for i in tqdm(range(len(img_ids)), desc="Inference", initial=start_idx):
        img_id = img_ids[i]

        # Skip already-processed images
        if img_id in raw_outputs:
            continue

        img_path = voc_root / "JPEGImages" / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"WARNING: Image not found {img_path}, skipping")
            raw_outputs[img_id] = ""
            continue

        image = Image.open(img_path).convert("RGB")

        t0 = time.time()
        try:
            raw_text = run_inference(model, processor, image)
        except Exception as e:
            print(f"Inference failed {img_id}: {e}")
            raw_text = ""
        elapsed = time.time() - t0
        total_time += elapsed

        raw_outputs[img_id] = raw_text

        # Save checkpoint every 50 images
        if (i + 1) % 50 == 0:
            save_checkpoint(ckpt_path, {"raw_outputs": raw_outputs})
            avg_time = total_time / (i - start_idx + 1)
            remaining = avg_time * (len(img_ids) - i - 1)
            print(
                f"\n[{i + 1}/{len(img_ids)}] avg {avg_time:.2f}s/img, "
                f"ETA {remaining / 60:.1f} min"
            )

    # Save all raw outputs
    save_checkpoint(ckpt_path, {"raw_outputs": raw_outputs})
    with open(raw_outputs_path, "w") as f:
        json.dump(raw_outputs, f, ensure_ascii=False, indent=2)
    print(f"Raw outputs saved to {raw_outputs_path}")

    # ============ Parse detections + compute mAP ============
    print("\nParsing detections...")
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
        f"Parse failure rate: {parse_failures}/{len(img_ids)} "
        f"({parse_failures / len(img_ids) * 100:.1f}%)"
    )

    # Load ground truth
    print("Loading ground truth...")
    all_gts = load_voc_ground_truths(voc_root, img_ids)

    # Compute mAP
    print("Computing mAP...")
    result = voc_eval(all_detections, all_gts)

    # Print results
    print(f"\n{'=' * 50}")
    print(f"Gemma-4 E2B-it Zero-Shot Detection - VOC 2007 test")
    print(f"{'=' * 50}")
    print(f"mAP @ IoU=0.5: {result['mAP'] * 100:.2f}%")
    print(f"\n{'Class':<15} {'AP (%)':<10}")
    print("-" * 25)
    for cls_name, ap in result["ap_per_class"]:
        print(f"{cls_name:<15} {ap * 100:.2f}")
    print("-" * 25)
    print(f"{'mAP':<15} {result['mAP'] * 100:.2f}")

    # Summary statistics
    total_dets = sum(len(d["labels"]) for d in all_detections)
    total_gts = sum(len(g["labels"]) for g in all_gts)
    print(f"\nTotal detections: {total_dets}, Total GTs: {total_gts}")
    print(f"Avg detections per image: {total_dets / len(img_ids):.1f}")
    print(f"Total inference time: {total_time / 60:.1f} min")

    # Save results
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
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
