"""
VOC-style mAP evaluation (AP @ IoU=0.5).

Implements the standard PASCAL VOC evaluation protocol:
  1. For each class, sort detections by confidence descending.
  2. Match detections to ground-truth boxes by IoU ≥ 0.5 (greedy, one-to-one).
  3. Compute precision–recall curve.
  4. Compute AP using the 11-point interpolation (VOC2007) or all-point interpolation (VOC2010+).
  5. mAP = mean of per-class APs.

We use the all-point interpolation method (standard since VOC2010).
"""

import numpy as np
from collections import defaultdict


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes.

    Args:
        box_a: (Na, 4) [x1, y1, x2, y2]
        box_b: (Nb, 4)
    Returns:
        (Na, Nb) IoU matrix.
    """
    Na = box_a.shape[0]
    Nb = box_b.shape[0]

    # Intersection
    max_xy = np.minimum(
        box_a[:, 2:4][:, np.newaxis, :],  # (Na, 1, 2)
        box_b[:, 2:4][np.newaxis, :, :],  # (1, Nb, 2)
    )
    min_xy = np.maximum(
        box_a[:, 0:2][:, np.newaxis, :],
        box_b[:, 0:2][np.newaxis, :, :],
    )
    inter = np.clip(max_xy - min_xy, 0.0, None)
    inter_area = inter[:, :, 0] * inter[:, :, 1]  # (Na, Nb)

    # Union
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # (Na,)
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])  # (Nb,)
    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter_area

    return inter_area / np.maximum(union, 1e-10)


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """Compute AP using all-point interpolation (VOC2010+ style).

    Inserts sentinel values at beginning/end, then computes area under
    the precision-recall curve using the envelope (monotone decreasing) trick.
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Make precision monotonically decreasing (right to left)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum areas of rectangles
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def voc_eval(
    all_detections: list,
    all_ground_truths: list,
    num_classes: int = 20,
    iou_thresh: float = 0.5,
) -> dict:
    """Evaluate detections against ground truth using VOC protocol.

    Args:
        all_detections: list of dicts (one per image), each with:
            'boxes':  (D, 4) np.ndarray [x1, y1, x2, y2]
            'scores': (D,) np.ndarray
            'labels': (D,) np.ndarray (int class indices)
        all_ground_truths: list of dicts (one per image), each with:
            'boxes':  (G, 4) np.ndarray
            'labels': (G,) np.ndarray
        num_classes: number of categories.
        iou_thresh: IoU threshold for positive match.

    Returns:
        dict with:
            'mAP': mean AP over all classes.
            'ap_per_class': list of (class_name, AP) tuples.
    """
    # Organise by class
    # For each class: collect all detections and ground truths
    class_dets = defaultdict(list)  # class_id -> [(img_idx, score, box), ...]
    class_gts = defaultdict(list)  # class_id -> [(img_idx, box), ...]
    n_gt_per_class = defaultdict(int)

    for img_idx, gt in enumerate(all_ground_truths):
        for j in range(len(gt["labels"])):
            c = int(gt["labels"][j])
            class_gts[c].append((img_idx, gt["boxes"][j]))
            n_gt_per_class[c] += 1

    for img_idx, det in enumerate(all_detections):
        for j in range(len(det["labels"])):
            c = int(det["labels"][j])
            class_dets[c].append((img_idx, float(det["scores"][j]), det["boxes"][j]))

    # Compute per-class AP
    aps = []
    ap_per_class = []

    for c in range(num_classes):
        n_gt = n_gt_per_class[c]
        dets = class_dets[c]

        if n_gt == 0:
            ap_per_class.append((VOC_CLASSES[c], 0.0))
            aps.append(0.0)
            continue

        if len(dets) == 0:
            ap_per_class.append((VOC_CLASSES[c], 0.0))
            aps.append(0.0)
            continue

        # Sort by score descending
        dets.sort(key=lambda x: -x[1])

        # Organise GT by image for fast lookup
        gt_by_img = defaultdict(list)
        for img_idx, box in class_gts[c]:
            gt_by_img[img_idx].append(box)

        # Convert to arrays and track matched status
        gt_matched = {}
        for img_idx in gt_by_img:
            boxes = np.array(gt_by_img[img_idx])
            gt_by_img[img_idx] = boxes
            gt_matched[img_idx] = np.zeros(len(boxes), dtype=bool)

        # Evaluate each detection
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))

        for d_idx, (img_idx, score, box) in enumerate(dets):
            if img_idx not in gt_by_img:
                fp[d_idx] = 1
                continue

            gt_boxes = gt_by_img[img_idx]
            matched = gt_matched[img_idx]

            # Compute IoU with all GT boxes of this class in this image
            ious = compute_iou(box[np.newaxis, :], gt_boxes)[0]  # (num_gt,)
            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]

            if best_iou >= iou_thresh and not matched[best_iou_idx]:
                tp[d_idx] = 1
                matched[best_iou_idx] = True
            else:
                fp[d_idx] = 1

        # Cumulative sums
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / n_gt
        precision = cum_tp / (cum_tp + cum_fp)

        ap = voc_ap(recall, precision)
        ap_per_class.append((VOC_CLASSES[c], ap))
        aps.append(ap)

    mAP = float(np.mean(aps)) if len(aps) > 0 else 0.0

    return {
        "mAP": mAP,
        "ap_per_class": ap_per_class,
    }


def compute_map(
    all_detections: list,
    all_ground_truths: list,
    num_classes: int = 20,
    iou_thresh: float = 0.5,
) -> float:
    """Convenience wrapper returning just the mAP value."""
    result = voc_eval(all_detections, all_ground_truths, num_classes, iou_thresh)
    return result["mAP"]
