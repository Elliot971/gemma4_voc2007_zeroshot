"""
analyze_errors.py - IoU sensitivity analysis and error mode breakdown.

Recomputes mAP at multiple IoU thresholds from saved checkpoint data
(no model inference needed) and categorizes detection errors into:
  - correct:      IoU >= 0.5 with matched GT (true positive at 0.5)
  - localization: correct class but 0.1 <= IoU < 0.5 (localization error)
  - classification: wrong class but IoU >= 0.5 with some GT (class confusion)
  - duplicate:    correct class, IoU >= 0.5 but GT already matched
  - background:   IoU < 0.1 with all GT (false positive / hallucination)
  - missed:       GT boxes with no matching detection (false negative)

Usage:
  python scripts/analyze_errors.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from test_single import VOC_CLASSES, parse_detections
from utils.voc_eval import voc_eval, compute_iou

VOC_ROOT = _PROJECT_ROOT / "VOCdevkit" / "VOC2007"


def load_ground_truths(img_ids):
    """Load GT annotations from VOC XML files."""
    import xml.etree.ElementTree as ET

    class_to_idx = {name: i for i, name in enumerate(VOC_CLASSES)}
    all_gts = []

    for img_id in img_ids:
        ann_path = VOC_ROOT / "Annotations" / f"{img_id}.xml"
        boxes, labels = [], []
        if ann_path.exists():
            tree = ET.parse(str(ann_path))
            for obj in tree.getroot().findall("object"):
                diff = obj.find("difficult")
                if diff is not None and int(diff.text) == 1:
                    continue
                name = obj.find("name").text.lower().strip()
                if name not in class_to_idx:
                    continue
                bbox = obj.find("bndbox")
                boxes.append(
                    [
                        float(bbox.find("xmin").text),
                        float(bbox.find("ymin").text),
                        float(bbox.find("xmax").text),
                        float(bbox.find("ymax").text),
                    ]
                )
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


def parse_all_detections(img_ids, raw_outputs, confidence_map):
    """Re-parse all detections from saved raw outputs."""
    from PIL import Image

    class_to_idx = {name: i for i, name in enumerate(VOC_CLASSES)}
    all_dets = []

    for img_id in img_ids:
        img_path = VOC_ROOT / "JPEGImages" / f"{img_id}.jpg"
        if img_path.exists():
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        else:
            img_w, img_h = 500, 375

        raw_text = raw_outputs.get(img_id, "")
        score = confidence_map.get(img_id, 1.0)
        detections = parse_detections(raw_text, img_w, img_h, score=score)

        if detections:
            boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
            scores = np.array([d["score"] for d in detections], dtype=np.float32)
            labels = np.array(
                [class_to_idx[d["label"]] for d in detections], dtype=np.int64
            )
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        all_dets.append({"boxes": boxes, "scores": scores, "labels": labels})

    return all_dets


def error_analysis(all_dets, all_gts):
    """Categorize every detection and every GT box into error types.

    Returns dict with per-class and overall error counts.
    """
    n_classes = len(VOC_CLASSES)

    # Per-class counters
    correct = defaultdict(int)  # TP at IoU >= 0.5
    localization = defaultdict(int)  # right class, 0.1 <= IoU < 0.5
    duplicate = defaultdict(int)  # right class, IoU >= 0.5 but GT taken
    background = defaultdict(int)  # IoU < 0.1 with all GT of same class
    other_fp = defaultdict(int)  # doesn't fit above categories cleanly
    missed = defaultdict(int)  # GT boxes not matched by any detection

    for img_idx in range(len(all_dets)):
        det = all_dets[img_idx]
        gt = all_gts[img_idx]

        n_det = len(det["labels"])
        n_gt = len(gt["labels"])

        # Track which GT boxes are matched
        gt_matched = (
            np.zeros(n_gt, dtype=bool) if n_gt > 0 else np.array([], dtype=bool)
        )

        # Sort detections by score descending (mimic voc_eval behavior)
        if n_det > 0:
            order = np.argsort(-det["scores"])
        else:
            order = []

        for d_idx in order:
            d_cls = int(det["labels"][d_idx])
            d_box = det["boxes"][d_idx]

            if n_gt == 0:
                background[d_cls] += 1
                continue

            # Compute IoU with all GT boxes
            ious = compute_iou(d_box[np.newaxis, :], gt["boxes"])[0]  # (n_gt,)

            # Find best IoU with same-class GT
            same_class_mask = gt["labels"] == d_cls
            if same_class_mask.any():
                same_ious = ious.copy()
                same_ious[~same_class_mask] = -1
                best_idx = np.argmax(same_ious)
                best_iou = same_ious[best_idx]

                if best_iou >= 0.5:
                    if not gt_matched[best_idx]:
                        correct[d_cls] += 1
                        gt_matched[best_idx] = True
                    else:
                        duplicate[d_cls] += 1
                elif best_iou >= 0.1:
                    localization[d_cls] += 1
                else:
                    background[d_cls] += 1
            else:
                # No same-class GT; check if IoU >= 0.5 with any GT (class confusion)
                if ious.max() >= 0.1:
                    other_fp[d_cls] += 1
                else:
                    background[d_cls] += 1

        # Count missed GT
        for g_idx in range(n_gt):
            if not gt_matched[g_idx]:
                missed[int(gt["labels"][g_idx])] += 1

    return {
        "correct": dict(correct),
        "localization": dict(localization),
        "duplicate": dict(duplicate),
        "background": dict(background),
        "other_fp": dict(other_fp),
        "missed": dict(missed),
    }


def main():
    # Load checkpoint
    ckpt_path = _PROJECT_ROOT / "eval_results" / "checkpoint.json"
    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path) as f:
        ckpt = json.load(f)

    raw_outputs = ckpt["raw_outputs"]
    confidence_map = ckpt["confidence_map"]

    # Load image IDs
    test_file = VOC_ROOT / "ImageSets" / "Main" / "test.txt"
    with open(test_file) as f:
        img_ids = [line.strip() for line in f if line.strip()]
    print(f"Images: {len(img_ids)}")

    # Parse detections and load GT
    print("Parsing detections...")
    all_dets = parse_all_detections(img_ids, raw_outputs, confidence_map)
    print("Loading ground truths...")
    all_gts = load_ground_truths(img_ids)

    # ===== P1: mAP at multiple IoU thresholds =====
    print("\n" + "=" * 60)
    print("IoU Threshold Sensitivity Analysis")
    print("=" * 60)

    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    iou_results = {}

    for thresh in thresholds:
        result = voc_eval(all_dets, all_gts, iou_thresh=thresh)
        iou_results[thresh] = result
        print(f"\nmAP@{thresh:.2f} = {result['mAP'] * 100:.2f}%")
        if thresh in [0.5, 0.75]:
            print(f"  {'Class':<15} {'AP (%)':<10}")
            print(f"  {'-' * 25}")
            for cls_name, ap in result["ap_per_class"]:
                print(f"  {cls_name:<15} {ap * 100:.2f}")

    # ===== P2: Error analysis =====
    print("\n" + "=" * 60)
    print("Error Mode Analysis (at IoU=0.5)")
    print("=" * 60)

    errors = error_analysis(all_dets, all_gts)

    # Overall summary
    total_correct = sum(errors["correct"].values())
    total_loc = sum(errors["localization"].values())
    total_dup = sum(errors["duplicate"].values())
    total_bg = sum(errors["background"].values())
    total_other = sum(errors["other_fp"].values())
    total_missed = sum(errors["missed"].values())
    total_dets = total_correct + total_loc + total_dup + total_bg + total_other
    total_gt = total_correct + total_missed  # matched + unmatched GT

    print(f"\nOverall Detection Breakdown ({total_dets} total detections):")
    print(
        f"  Correct (TP, IoU>=0.5):     {total_correct:>5} ({total_correct / total_dets * 100:.1f}%)"
    )
    print(
        f"  Localization error:         {total_loc:>5} ({total_loc / total_dets * 100:.1f}%)"
    )
    print(
        f"  Duplicate (GT taken):       {total_dup:>5} ({total_dup / total_dets * 100:.1f}%)"
    )
    print(
        f"  Background (IoU<0.1):       {total_bg:>5} ({total_bg / total_dets * 100:.1f}%)"
    )
    print(
        f"  Other FP:                   {total_other:>5} ({total_other / total_dets * 100:.1f}%)"
    )

    print(
        f"\nGround Truth Coverage ({total_gt + total_missed - total_correct} missed / {total_gt} total GT):"
    )
    # Actually: total_gt = total_correct + total_missed by construction
    print(
        f"  Matched (recalled):         {total_correct:>5} ({total_correct / (total_correct + total_missed) * 100:.1f}%)"
    )
    print(
        f"  Missed (FN):                {total_missed:>5} ({total_missed / (total_correct + total_missed) * 100:.1f}%)"
    )

    # Per-class breakdown
    print(
        f"\n{'Class':<15} {'Correct':>8} {'Loc.Err':>8} {'Dup':>5} {'BG/FP':>6} {'Missed':>7} {'Recall%':>8}"
    )
    print("-" * 62)
    for c, cls_name in enumerate(VOC_CLASSES):
        cor = errors["correct"].get(c, 0)
        loc = errors["localization"].get(c, 0)
        dup = errors["duplicate"].get(c, 0)
        bg = errors["background"].get(c, 0) + errors["other_fp"].get(c, 0)
        mis = errors["missed"].get(c, 0)
        total_cls_gt = cor + mis
        recall = cor / total_cls_gt * 100 if total_cls_gt > 0 else 0
        print(
            f"  {cls_name:<13} {cor:>8} {loc:>8} {dup:>5} {bg:>6} {mis:>7} {recall:>7.1f}%"
        )

    # Save results
    output = {
        "iou_sensitivity": {
            str(t): {"mAP": r["mAP"], "ap_per_class": r["ap_per_class"]}
            for t, r in iou_results.items()
        },
        "error_analysis": {
            "overall": {
                "total_detections": total_dets,
                "correct": total_correct,
                "localization_error": total_loc,
                "duplicate": total_dup,
                "background_fp": total_bg,
                "other_fp": total_other,
                "missed_gt": total_missed,
            },
            "per_class": {
                VOC_CLASSES[c]: {
                    "correct": errors["correct"].get(c, 0),
                    "localization": errors["localization"].get(c, 0),
                    "duplicate": errors["duplicate"].get(c, 0),
                    "background": errors["background"].get(c, 0),
                    "other_fp": errors["other_fp"].get(c, 0),
                    "missed": errors["missed"].get(c, 0),
                }
                for c in range(len(VOC_CLASSES))
            },
        },
    }

    out_path = _PROJECT_ROOT / "eval_results" / "analysis_result.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
