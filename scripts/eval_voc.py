"""
eval_voc.py - Zero-shot object detection mAP evaluation on VOC 2007 test.

Core design:
  1. Run Gemma-4 E2B inference on every VOC 2007 test image (4952 total)
  2. Parse model output into VOC-format detections
  3. Reuse voc_eval.py from the PLN project (same evaluation ruler)
  4. Checkpoint support to resume after interruption
  5. Save per-image raw outputs and timing for analysis
  6. Use token log-prob as confidence score (not fixed 1.0)
  7. Track failure types: empty_output / parse_failure / no_detection

Usage:
  # AutoDL 4090 (recommended)
  export HF_ENDPOINT=https://hf-mirror.com
  python eval_voc.py --voc-root /root/autodl-tmp/VOCdevkit/VOC2007

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
    classify_output,
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


def _patch_clippable_linear_eval(model):
    """Replace Gemma4ClippableLinear with nn.Linear-compatible modules for PEFT."""
    import torch.nn as nn

    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except ImportError:
        return

    class PatchedClippableLinear(nn.Linear):
        def __init__(self, in_features, out_features, use_clipped, clamp_buffers):
            super().__init__(in_features, out_features, bias=False)
            self.use_clipped = use_clipped
            if use_clipped:
                self.register_buffer("input_min", clamp_buffers["input_min"])
                self.register_buffer("input_max", clamp_buffers["input_max"])
                self.register_buffer("output_min", clamp_buffers["output_min"])
                self.register_buffer("output_max", clamp_buffers["output_max"])

        def forward(self, x):
            if self.use_clipped:
                x = torch.clamp(x, self.input_min, self.input_max)
            x = super().forward(x)
            if self.use_clipped:
                x = torch.clamp(x, self.output_min, self.output_max)
            return x

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, Gemma4ClippableLinear):
            continue
        parent_path = name.rsplit(".", 1)
        if len(parent_path) == 2:
            parent = dict(model.named_modules())[parent_path[0]]
            attr = parent_path[1]
        else:
            parent = model
            attr = name
        linear = module.linear
        clamp_bufs = None
        if module.use_clipped_linears:
            clamp_bufs = {
                "input_min": module.input_min.data.clone(),
                "input_max": module.input_max.data.clone(),
                "output_min": module.output_min.data.clone(),
                "output_max": module.output_max.data.clone(),
            }
        new_mod = PatchedClippableLinear(
            linear.in_features,
            linear.out_features,
            use_clipped=module.use_clipped_linears,
            clamp_buffers=clamp_bufs,
        )
        new_mod.weight.data.copy_(linear.weight.data)
        new_mod = new_mod.to(device=linear.weight.device, dtype=linear.weight.dtype)
        setattr(parent, attr, new_mod)
        replaced += 1

    print(
        f"[patch] Replaced {replaced} Gemma4ClippableLinear → PatchedClippableLinear(nn.Linear)"
    )


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
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (for fine-tuned model evaluation)",
    )
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
    confidence_map = {}  # img_id -> confidence score
    timing_map = {}  # img_id -> inference time (seconds)
    start_idx = 0
    if args.resume:
        ckpt = load_checkpoint(ckpt_path)
        if ckpt:
            raw_outputs = ckpt.get("raw_outputs", {})
            confidence_map = ckpt.get("confidence_map", {})
            timing_map = ckpt.get("timing_map", {})
            start_idx = len(raw_outputs)
            print(f"Resuming from checkpoint: {start_idx}/{len(img_ids)} done")

    # Load model
    print(f"Loading model {args.model_id}...")
    t0 = time.time()
    model, processor = load_model(args.model_id, args.quantize_4bit)

    # Load LoRA adapter if specified
    if args.adapter:
        from peft import PeftModel

        print("Patching Gemma4ClippableLinear for PEFT compatibility...")
        _patch_clippable_linear_eval(model)
        print(f"Loading LoRA adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.to("cuda").to(torch.bfloat16)
        print("Model moved to cuda (bf16) after loading LoRA adapter.")

    print(f"Model loaded ({time.time() - t0:.1f}s)")

    # Per-image inference
    total_time = 0
    inferred_count = 0

    for i in tqdm(range(len(img_ids)), desc="Inference", initial=start_idx):
        img_id = img_ids[i]

        # Skip already-processed images
        if img_id in raw_outputs:
            continue

        img_path = voc_root / "JPEGImages" / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"WARNING: Image not found {img_path}, skipping")
            raw_outputs[img_id] = ""
            confidence_map[img_id] = 0.0
            timing_map[img_id] = 0.0
            continue

        image = Image.open(img_path).convert("RGB")

        t0 = time.time()
        try:
            raw_text, confidence = run_inference(model, processor, image)
        except Exception as e:
            print(f"Inference failed {img_id}: {e}")
            raw_text = ""
            confidence = 0.0
        elapsed = time.time() - t0
        total_time += elapsed
        inferred_count += 1

        raw_outputs[img_id] = raw_text
        confidence_map[img_id] = confidence
        timing_map[img_id] = round(elapsed, 2)

        # Save checkpoint every 50 images
        if (i + 1) % 50 == 0:
            save_checkpoint(
                ckpt_path,
                {
                    "raw_outputs": raw_outputs,
                    "confidence_map": confidence_map,
                    "timing_map": timing_map,
                },
            )
            avg_time = total_time / inferred_count
            remaining = avg_time * (len(img_ids) - i - 1)
            print(
                f"\n[{i + 1}/{len(img_ids)}] avg {avg_time:.2f}s/img, "
                f"ETA {remaining / 60:.1f} min"
            )

    # Save all raw outputs
    save_checkpoint(
        ckpt_path,
        {
            "raw_outputs": raw_outputs,
            "confidence_map": confidence_map,
            "timing_map": timing_map,
        },
    )
    with open(raw_outputs_path, "w") as f:
        json.dump(raw_outputs, f, ensure_ascii=False, indent=2)
    print(f"Raw outputs saved to {raw_outputs_path}")

    # Save timing data
    timing_path = str(output_dir / "timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing_map, f, indent=2)
    print(f"Per-image timing saved to {timing_path}")

    # ============ Parse detections + compute mAP ============
    print("\nParsing detections...")
    all_detections = []
    status_counts = defaultdict(int)

    for img_id in img_ids:
        img_path = voc_root / "JPEGImages" / f"{img_id}.jpg"
        if img_path.exists():
            image = Image.open(img_path)
            img_w, img_h = image.size
            image.close()
        else:
            img_w, img_h = 500, 375  # fallback

        raw_text = raw_outputs.get(img_id, "")
        score = confidence_map.get(img_id, 1.0)
        detections = parse_detections(raw_text, img_w, img_h, score=score)

        # Classify outcome
        status = classify_output(raw_text, detections)
        status_counts[status] += 1

        all_detections.append(detections_to_voc_format(detections))

    # Print failure analysis
    print(f"\n--- Output Classification ---")
    for status_type in ["success", "no_detection", "parse_failure", "empty_output"]:
        count = status_counts.get(status_type, 0)
        pct = count / len(img_ids) * 100
        print(f"  {status_type:<16} {count:>5} ({pct:.1f}%)")

    # Load ground truth
    print("\nLoading ground truth...")
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
    timing_values = [v for v in timing_map.values() if v > 0]
    avg_time_per_img = np.mean(timing_values) if timing_values else 0

    print(f"\nTotal detections: {total_dets}, Total GTs: {total_gts}")
    print(f"Avg detections per image: {total_dets / len(img_ids):.1f}")
    print(f"Avg inference time: {avg_time_per_img:.2f}s/img")
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
                "status_counts": dict(status_counts),
                "avg_inference_time_sec": round(avg_time_per_img, 2),
                "total_time_sec": round(total_time, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
