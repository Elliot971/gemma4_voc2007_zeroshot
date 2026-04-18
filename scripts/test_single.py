"""
test_single.py - Single-image zero-shot detection sanity check.

Validates the full pipeline: prompt design, JSON parsing, coordinate
conversion (1000-scale -> pixel), and label matching against VOC classes.

Usage:
  # AutoDL 4090 (24GB) - BF16 precision
  export HF_ENDPOINT=https://hf-mirror.com
  python scripts/test_single.py

  # Local RTX 4060 (8GB) - 4-bit quantization
  python scripts/test_single.py --quantize-4bit
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

# ==================== VOC 20 Classes ====================
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

# ==================== Detection Prompt ====================
DETECTION_PROMPT = (
    "Detect all objects in this image that belong to these categories: "
    + ", ".join(VOC_CLASSES)
    + ". For each detected object, output a JSON list where each element has "
    '"box_2d" (as [y_min, x_min, y_max, x_max] in range 0-1000) and "label" '
    "(the category name). Only output the JSON list, nothing else."
)

# ==================== Label Alias Mapping ====================
# Maps common VLM output synonyms to canonical VOC class names.
LABEL_ALIASES = {
    "airplane": "aeroplane",
    "plane": "aeroplane",
    "bike": "bicycle",
    "ship": "boat",
    "automobile": "car",
    "vehicle": "car",
    "kitten": "cat",
    "cattle": "cow",
    "dining table": "diningtable",
    "table": "diningtable",
    "puppy": "dog",
    "motorcycle": "motorbike",
    "people": "person",
    "man": "person",
    "woman": "person",
    "child": "person",
    "boy": "person",
    "girl": "person",
    "human": "person",
    "potted plant": "pottedplant",
    "plant": "pottedplant",
    "houseplant": "pottedplant",
    "flower": "pottedplant",
    "lamb": "sheep",
    "couch": "sofa",
    "locomotive": "train",
    "tv": "tvmonitor",
    "television": "tvmonitor",
    "monitor": "tvmonitor",
    "tv monitor": "tvmonitor",
    "screen": "tvmonitor",
}


def match_voc_class(label: str):
    """Fuzzy-match a model-predicted label to one of the 20 VOC classes.

    Matching hierarchy: exact match -> alias lookup -> substring match.
    Returns None if no match is found (detection will be discarded).
    """
    label = label.lower().strip()

    # Exact match
    if label in VOC_CLASSES:
        return label

    # Alias match
    if label in LABEL_ALIASES:
        return LABEL_ALIASES[label]

    # Substring fallback
    for alias, voc_cls in LABEL_ALIASES.items():
        if alias in label or label in alias:
            return voc_cls

    return None


def convert_detection(item: dict, img_w: int, img_h: int):
    """Convert a single Gemma detection from 1000-scale to pixel coordinates.

    Gemma 4 output format: box_2d = [y_min, x_min, y_max, x_max] (0-1000)
    VOC format:            [x_min, y_min, x_max, y_max] (pixel coordinates)

    Returns:
        dict with 'label', 'bbox', 'score', or None if invalid.
    """
    if "box_2d" not in item or "label" not in item:
        return None

    box = item["box_2d"]
    if not isinstance(box, list) or len(box) != 4:
        return None

    try:
        y1, x1, y2, x2 = [float(v) for v in box]
    except (ValueError, TypeError):
        return None

    # 1000-scale -> pixel coordinates
    x_min = x1 / 1000.0 * img_w
    y_min = y1 / 1000.0 * img_h
    x_max = x2 / 1000.0 * img_w
    y_max = y2 / 1000.0 * img_h

    # Sanity check: box must have positive area
    if x_max <= x_min or y_max <= y_min:
        return None

    matched = match_voc_class(item["label"])
    if matched is None:
        return None

    return {"label": matched, "bbox": [x_min, y_min, x_max, y_max], "score": 1.0}


def parse_detections(text: str, img_w: int, img_h: int):
    """Three-level fault-tolerant parser for Gemma's text output.

    Strategy:
        1. Direct JSON parse of the entire output
        2. Regex extraction of JSON arrays [...]
        3. Per-object extraction of JSON objects {...}

    Returns:
        list of dicts, each with 'label', 'bbox' [x1,y1,x2,y2], 'score'.
    """
    detections = []

    # Level 1: Direct JSON parse
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list):
            for item in parsed:
                det = convert_detection(item, img_w, img_h)
                if det:
                    detections.append(det)
            if detections:
                return detections
    except (json.JSONDecodeError, ValueError):
        pass

    # Level 2: Regex extract JSON arrays
    json_arrays = re.findall(r"\[[\s\S]*?\{[\s\S]*?\}[\s\S]*?\]", text)
    for arr_str in json_arrays:
        try:
            parsed = json.loads(arr_str)
            if isinstance(parsed, list):
                for item in parsed:
                    det = convert_detection(item, img_w, img_h)
                    if det:
                        detections.append(det)
                if detections:
                    return detections
        except (json.JSONDecodeError, ValueError):
            continue

    # Level 3: Extract individual JSON objects
    json_objects = re.findall(r"\{[^{}]+\}", text)
    for obj_str in json_objects:
        try:
            item = json.loads(obj_str)
            det = convert_detection(item, img_w, img_h)
            if det:
                detections.append(det)
        except (json.JSONDecodeError, ValueError):
            continue

    return detections


def visualize(image: Image.Image, detections: list, save_path: str):
    """Draw detection boxes on the image and save to disk."""
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 15)), label, fill="red")
    image.save(save_path)
    print(f"Visualization saved to {save_path}")


def load_model(model_id: str, quantize_4bit: bool = False):
    """Load Gemma-4 model with appropriate precision for available VRAM.

    Args:
        model_id: HuggingFace model identifier.
        quantize_4bit: If True, use bitsandbytes 4-bit quantization (~3GB VRAM).
                       If False, use BF16 (~10GB VRAM, requires 24GB GPU).
    """
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        print("Loading mode: 4-bit quantized (for 8GB VRAM)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        print("Loading mode: BF16 (for 24GB VRAM)")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def run_inference(model, processor, image: Image.Image):
    """Run zero-shot detection on a single image, return raw text output."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": DETECTION_PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output[0][input_len:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)
    return raw_text


def find_voc_root():
    """Auto-detect VOC2007 dataset root (compatible with local and AutoDL)."""
    candidates = [
        # Local: scripts/ -> gemma4-voc-zeroshot/ -> wxg_tasks2/
        Path(__file__).parent.parent.parent
        / "PLN-ResNet18"
        / "data"
        / "VOCdevkit"
        / "VOC2007",
        # Local: symlink in project data/
        Path(__file__).parent.parent / "data" / "VOCdevkit" / "VOC2007",
        # Local: absolute path
        Path(
            "/mnt/c/Users/Elliot/Desktop/wxg_tasks2/PLN-ResNet18/data/VOCdevkit/VOC2007"
        ),
        # AutoDL
        Path("/root/autodl-tmp/PLN-ResNet18/data/VOCdevkit/VOC2007"),
        Path("/root/autodl-tmp/VOCdevkit/VOC2007"),
    ]
    for p in candidates:
        if p.exists():
            return p
    print("ERROR: VOC2007 dataset not found. Searched paths:")
    for p in candidates:
        print(f"  {p}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Gemma-4 E2B zero-shot detection - single image sanity check"
    )
    parser.add_argument(
        "--quantize-4bit", action="store_true", help="Use 4-bit quantization (8GB VRAM)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image (default: first VOC test image)",
    )
    parser.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it")
    args = parser.parse_args()

    # Select test image
    if args.image:
        img_path = Path(args.image)
        img_id = img_path.stem
    else:
        voc_root = find_voc_root()
        test_file = voc_root / "ImageSets" / "Main" / "test.txt"
        if test_file.exists():
            with open(test_file) as f:
                img_ids = [line.strip() for line in f if line.strip()]
            img_id = img_ids[0]
        else:
            img_id = "000001"
        img_path = voc_root / "JPEGImages" / f"{img_id}.jpg"

    print(f"Test image: {img_path}")
    if not img_path.exists():
        print(f"ERROR: Image not found: {img_path}")
        sys.exit(1)

    image = Image.open(img_path).convert("RGB")
    img_w, img_h = image.size
    print(f"Image size: {img_w} x {img_h}")

    # Load model
    print(f"Loading {args.model_id}...")
    t0 = time.time()
    model, processor = load_model(args.model_id, args.quantize_4bit)
    print(f"Model loaded ({time.time() - t0:.1f}s)")

    # Inference
    print(f"Prompt: {DETECTION_PROMPT[:80]}...")
    t0 = time.time()
    raw_text = run_inference(model, processor, image)
    print(f"Inference done ({time.time() - t0:.1f}s)")

    print(f"\n===== Raw Model Output =====")
    print(raw_text)
    print(f"============================\n")

    # Parse
    detections = parse_detections(raw_text, img_w, img_h)
    print(f"Parsed {len(detections)} detection(s):")
    for det in detections:
        print(f"  {det['label']}: bbox={[round(v, 1) for v in det['bbox']]}")

    # Visualize
    output_dir = Path(__file__).parent.parent / "visualize"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"test_result_{img_id}.jpg"
    visualize(image.copy(), detections, str(output_path))


if __name__ == "__main__":
    main()
