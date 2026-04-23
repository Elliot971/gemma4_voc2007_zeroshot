"""
prepare_finetune_data.py - Convert VOC XML annotations to SFT training JSONL.

Reads VOC XML annotations and converts bounding boxes to Gemma-4's native
[0, 1000] coordinate space. Applies horizontal flip augmentation (50% chance)
with synchronized coordinate transformation. Uses diverse prompt variants to
prevent overfitting to a single instruction format.

Usage:
  # VOC 2007 trainval only
  python scripts/prepare_finetune_data.py --voc-root /path/to/VOCdevkit/VOC2007

  # VOC 2007 + 2012 combined
  python scripts/prepare_finetune_data.py \
    --voc-root /path/to/VOCdevkit/VOC2007 \
    --extra-voc-root /path/to/VOCdevkit/VOC2012 \
    --split trainval

  # With horizontal flip augmentation (doubles dataset size)
  python scripts/prepare_finetune_data.py --augment --flip-prob 0.5
"""

import argparse
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

CLASS_TO_IDX = {name: i for i, name in enumerate(VOC_CLASSES)}

# Diverse prompts prevent the model from memorizing a single instruction.
# Each template conveys the same task in different wording.
PROMPT_TEMPLATES = [
    "Detect all objects in this image that belong to these categories: "
    + ", ".join(VOC_CLASSES)
    + ". For each detected object, output a JSON list where each element has "
    + "box_2d (as [y_min, x_min, y_max, x_max] in range 0-1000) and label "
    + "(the category name). Only output the JSON list, nothing else.",
    "Find all instances of "
    + ", ".join(VOC_CLASSES)
    + " in this image. Output a JSON array where each element contains "
    + "box_2d ([y_min, x_min, y_max, x_max], 0-1000) and label (class name).",
    "Please detect and localize the following categories in the image: "
    + ", ".join(VOC_CLASSES)
    + ". Return your answer as a JSON list of objects, each with box_2d "
    + "([y_min, x_min, y_max, x_max] normalized to 0-1000) and label.",
    "Identify all objects belonging to these classes: "
    + ", ".join(VOC_CLASSES)
    + '. For each object, provide its bounding box as "box_2d" '
    + "([y_min, x_min, y_max, x_max] in 0-1000) and class label in JSON format.",
    "Look at this image and find all objects from these categories: "
    + ", ".join(VOC_CLASSES)
    + '. Output a JSON list with "box_2d" ([y_min, x_min, y_max, x_max], 0-1000) '
    + 'and "label" for each detection.',
    "Given this image, detect all instances of "
    + ", ".join(VOC_CLASSES)
    + '. For each detection, output {"box_2d": [y_min, x_min, y_max, x_max], "label": class_name} '
    + "in a JSON array. Coordinates should be in 0-1000 range.",
    "What objects can you find in this image from these categories: "
    + ", ".join(VOC_CLASSES)
    + '? Return a JSON array of detections with "box_2d" and "label" fields. '
    + "Use [y_min, x_min, y_max, x_max] format in 0-1000 scale.",
    "Locate and label all objects from the following categories: "
    + ", ".join(VOC_CLASSES)
    + '. Respond with a JSON list, each entry having "box_2d" '
    + "([y_min, x_min, y_max, x_max]) and label. Use 0-1000 coordinate normalization.",
    "Analyze this image and detect every instance of "
    + ", ".join(VOC_CLASSES)
    + '. Format your response as a JSON array with "box_2d" (0-1000 normalized '
    + "[y_min, x_min, y_max, x_max]) and label for each object found.",
    "Enumerate all detectable objects from these classes: "
    + ", ".join(VOC_CLASSES)
    + '. Output a JSON array where each element is {"box_2d": [y_min, x_min, y_max, x_max], '
    + '"label": "class_name"}. Use 0-1000 coordinates.',
]


def convert_box(box, img_w, img_h):
    """Convert VOC pixel bounding box to Gemma-4 box_2d format.

    VOC format: [xmin, ymin, xmax, ymax] in pixel coordinates
    Gemma format: [y_min, x_min, y_max, x_max] in [0, 1000] range

    Args:
        box: dict with xmin, ymin, xmax, ymax (pixel coords).
        img_w: image width in pixels.
        img_h: image height in pixels.

    Returns:
        list [y_min, x_min, y_max, x_max] with integers in [0, 1000].
    """
    clamp = lambda v: max(0, min(1000, round(v)))
    return [
        clamp(box["ymin"] / img_h * 1000),
        clamp(box["xmin"] / img_w * 1000),
        clamp(box["ymax"] / img_h * 1000),
        clamp(box["xmax"] / img_w * 1000),
    ]


def parse_voc_xml(xml_path):
    """Parse a VOC XML annotation file.

    Skips difficult annotations (difficult=1). Returns empty list for images
    with no valid objects (these are valid training examples with empty outputs).

    Args:
        xml_path: path to the XML annotation file.

    Returns:
        tuple of (img_w, img_h, detections) where each detection is a dict
        with 'label', 'xmin', 'ymin', 'xmax', 'ymax'.
    """
    if not xml_path.exists():
        return 0, 0, []

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    detections = []
    for obj in root.findall("object"):
        difficult = obj.find("difficult")
        if difficult is not None and int(difficult.text) == 1:
            continue

        name = obj.find("name").text.lower().strip()
        if name not in CLASS_TO_IDX:
            continue

        bbox = obj.find("bndbox")
        detections.append(
            {
                "label": name,
                "xmin": float(bbox.find("xmin").text),
                "ymin": float(bbox.find("ymin").text),
                "xmax": float(bbox.find("xmax").text),
                "ymax": float(bbox.find("ymax").text),
            }
        )

    return img_w, img_h, detections


def load_voc_ids(voc_root, split="trainval"):
    """Load image IDs from a VOC dataset split.

    Args:
        voc_root: VOC dataset root directory (e.g. VOCdevkit/VOC2007).
        split: dataset split name (train, val, trainval, test).

    Returns:
        list of image ID strings (e.g. ["000001", "000002", ...]).
    """
    split_file = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    if not split_file.exists():
        print(f"WARNING: Split file not found: {split_file}")
        return []

    with open(split_file) as f:
        ids = [line.strip() for line in f if line.strip()]
    print(f"  {voc_root.name}/{split}: {len(ids)} images")
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Convert VOC XML annotations to SFT training JSONL"
    )
    parser.add_argument(
        "--voc-root",
        type=str,
        default=None,
        help="Primary VOC dataset root (e.g. VOCdevkit/VOC2007)",
    )
    parser.add_argument(
        "--extra-voc-root",
        type=str,
        default=None,
        help="Additional VOC dataset root (e.g. VOCdevkit/VOC2012)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="trainval",
        help="Dataset split to use (train, val, trainval)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: finetune_data/sft_train_<split>.jsonl)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply horizontal flip augmentation (doubles dataset size)",
    )
    parser.add_argument(
        "--flip-prob",
        type=float,
        default=0.5,
        help="Probability of horizontal flip when --augment is set (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Locate VOC dataset
    if args.voc_root:
        voc_roots = [Path(args.voc_root)]
    else:
        default = Path(__file__).parent.parent / "VOCdevkit" / "VOC2007"
        voc_roots = [default] if default.exists() else []

    if args.extra_voc_root:
        voc_roots.append(Path(args.extra_voc_root))

    if not voc_roots:
        print("ERROR: No VOC dataset found. Use --voc-root to specify.")
        sys.exit(1)

    print(f"VOC datasets: {[str(r) for r in voc_roots]}")

    # Load image IDs from all datasets
    all_ids = []
    for root in voc_roots:
        ids = load_voc_ids(root, args.split)
        all_ids.extend([(root, img_id) for img_id in ids])

    if not all_ids:
        print("ERROR: No images found.")
        sys.exit(1)

    print(f"Total images: {len(all_ids)}")

    # Output path
    output_dir = _PROJECT_ROOT / "finetune_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f"sft_train_{args.split}.jsonl"

    # Process images and write JSONL
    count = 0
    augmented_count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for voc_root, img_id in all_ids:
            xml_path = voc_root / "Annotations" / f"{img_id}.xml"
            img_w, img_h, detections = parse_voc_xml(xml_path)

            if img_w == 0:
                continue

            img_path = voc_root / "JPEGImages" / f"{img_id}.jpg"
            if not img_path.exists():
                continue

            img_rel = str(img_path)

            # Original sample
            prompt = random.choice(PROMPT_TEMPLATES)
            json_output = json.dumps(
                [
                    dict(box_2d=convert_box(d, img_w, img_h), label=d["label"])
                    for d in detections
                ],
                ensure_ascii=False,
            )
            out.write(
                json.dumps(
                    {
                        "image": img_rel,
                        "text": f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
                        f"<start_of_turn>model\n{json_output}<end_of_turn>",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

            # === Flipped augmentation sample ===
            if args.augment and random.random() < args.flip_prob:
                flipped_detections = []
                for d in detections:
                    flipped_detections.append(
                        {
                            "label": d["label"],
                            "xmin": img_w - d["xmax"],
                            "ymin": d["ymin"],
                            "xmax": img_w - d["xmin"],
                            "ymax": d["ymax"],
                        }
                    )

                prompt2 = random.choice(PROMPT_TEMPLATES)
                json_output2 = json.dumps(
                    [
                        dict(box_2d=convert_box(d, img_w, img_h), label=d["label"])
                        for d in flipped_detections
                    ],
                    ensure_ascii=False,
                )
                out.write(
                    json.dumps(
                        {
                            "image": img_rel,
                            "text": f"<start_of_turn>user\n{prompt2}<end_of_turn>\n"
                            f"<start_of_turn>model\n{json_output2}<end_of_turn>",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                augmented_count += 1

    print(f"\nOutput: {output_path}")
    print(f"  Original samples: {count}")
    print(f"  Augmented (flipped): {augmented_count}")
    print(f"  Total: {count + augmented_count}")
    print(f"  Prompt templates: {len(PROMPT_TEMPLATES)}")


if __name__ == "__main__":
    main()
