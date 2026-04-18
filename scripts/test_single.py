"""
test_single.py - 单张图零样本检测验证
验证 Gemma-4 E2B 的 prompt 设计、JSON 解析、坐标转换是否正确

用法:
  # AutoDL 4090 (24GB) - BF16 精度
  export HF_ENDPOINT=https://hf-mirror.com
  python test_single.py

  # 本地 4060 (8GB) - 4-bit 量化
  export HF_ENDPOINT=https://hf-mirror.com
  python test_single.py --quantize-4bit
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

# ============ VOC 20 类 ============
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

# ============ Prompt 设计 ============
DETECTION_PROMPT = (
    "Detect all objects in this image that belong to these categories: "
    + ", ".join(VOC_CLASSES)
    + ". For each detected object, output a JSON list where each element has "
    '"box_2d" (as [y_min, x_min, y_max, x_max] in range 0-1000) and "label" '
    "(the category name). Only output the JSON list, nothing else."
)

# ============ VOC 类别别名映射 ============
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
    """模糊匹配模型输出的 label 到 VOC 20 类"""
    label = label.lower().strip()

    # 精确匹配
    if label in VOC_CLASSES:
        return label

    # 别名匹配
    if label in LABEL_ALIASES:
        return LABEL_ALIASES[label]

    # 子串匹配（兜底）
    for alias, voc_cls in LABEL_ALIASES.items():
        if alias in label or label in alias:
            return voc_cls

    return None


def convert_detection(item: dict, img_w: int, img_h: int):
    """
    将单个检测结果从 1000x1000 归一化空间转换为像素坐标。
    Gemma 4 输出: box_2d = [y_min, x_min, y_max, x_max] (0-1000)
    返回 VOC 格式: [x_min, y_min, x_max, y_max] (像素坐标)
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

    # 1000-scale -> 像素坐标
    x_min = x1 / 1000.0 * img_w
    y_min = y1 / 1000.0 * img_h
    x_max = x2 / 1000.0 * img_w
    y_max = y2 / 1000.0 * img_h

    if x_max <= x_min or y_max <= y_min:
        return None

    matched = match_voc_class(item["label"])
    if matched is None:
        return None

    return {"label": matched, "bbox": [x_min, y_min, x_max, y_max], "score": 1.0}


def parse_detections(text: str, img_w: int, img_h: int):
    """
    三级容错解析器：
    1. 直接 JSON 解析
    2. 正则提取 JSON 数组
    3. 逐个提取 JSON 对象
    """
    detections = []

    # 策略 1: 直接解析
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

    # 策略 2: 正则提取 JSON 数组
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

    # 策略 3: 逐个提取 JSON 对象
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
    """可视化检测结果"""
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 15)), label, fill="red")
    image.save(save_path)
    print(f"可视化已保存至 {save_path}")


def load_model(model_id: str, quantize_4bit: bool = False):
    """加载模型，根据显存自动选择精度"""
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        print("加载模式: 4-bit 量化 (适配 8GB 显存)")
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
        print("加载模式: BF16 (适配 24GB 显存)")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def run_inference(model, processor, image: Image.Image):
    """单张图推理，返回原始文本输出"""
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
    """自动查找 VOC 数据集路径（兼容本地和 AutoDL）"""
    candidates = [
        # 本地 (相对路径: scripts/ -> gemma4-voc-zeroshot/ -> wxg_tasks2/)
        Path(__file__).parent.parent.parent
        / "PLN-ResNet18"
        / "data"
        / "VOCdevkit"
        / "VOC2007",
        # 本地 (绝对路径)
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
    print("找不到 VOC 数据集，尝试过以下路径:")
    for p in candidates:
        print(f"  {p}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Gemma-4 E2B 单张图零样本检测验证")
    parser.add_argument(
        "--quantize-4bit", action="store_true", help="使用 4-bit 量化 (8GB 显存)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="指定测试图片路径，默认用 VOC test 第一张",
    )
    parser.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it")
    args = parser.parse_args()

    # 选择测试图片
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

    print(f"测试图片: {img_path}")
    if not img_path.exists():
        print(f"图片不存在: {img_path}")
        sys.exit(1)

    image = Image.open(img_path).convert("RGB")
    img_w, img_h = image.size
    print(f"图片尺寸: {img_w} x {img_h}")

    # 加载模型
    print(f"加载 {args.model_id}...")
    t0 = time.time()
    model, processor = load_model(args.model_id, args.quantize_4bit)
    print(f"模型加载完成 ({time.time() - t0:.1f}s)")

    # 推理
    print(f"Prompt: {DETECTION_PROMPT[:80]}...")
    t0 = time.time()
    raw_text = run_inference(model, processor, image)
    print(f"推理完成 ({time.time() - t0:.1f}s)")

    print(f"\n===== 模型原始输出 =====")
    print(raw_text)
    print(f"========================\n")

    # 解析
    detections = parse_detections(raw_text, img_w, img_h)
    print(f"解析到 {len(detections)} 个检测结果:")
    for det in detections:
        print(f"  {det['label']}: bbox={[round(v, 1) for v in det['bbox']]}")

    # 可视化
    output_path = Path(__file__).parent / f"test_result_{img_id}.jpg"
    visualize(image.copy(), detections, str(output_path))


if __name__ == "__main__":
    main()
