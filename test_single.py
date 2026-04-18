"""
test_single.py - 单张图零样本检测验证
验证 Gemma-4 E2B 的 prompt 设计、JSON 解析、坐标转换是否正确
"""

import torch
import json
import re
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
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
    '"box_2d" (as [y_min, x_min, y_max, x_max] in range 0-1000) and "label" (the category name). '
    "Only output the JSON list, nothing else."
)


def parse_detections(text: str, img_w: int, img_h: int):
    """
    鲁棒地从模型文本输出中解析检测结果。

    策略：
    1. 尝试直接 JSON 解析
    2. 用正则提取 JSON 数组片段
    3. 逐个提取 {box_2d, label} 对象

    返回: list of dict, 每个 dict 包含:
        - label: str (类别名)
        - bbox: [x_min, y_min, x_max, y_max] (像素坐标, VOC 格式)
        - score: float (零样本没有真正的 score, 固定为 1.0)
    """
    detections = []

    # 策略 1: 直接解析整段文本为 JSON
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list):
            for item in parsed:
                det = _convert_item(item, img_w, img_h)
                if det:
                    detections.append(det)
            if detections:
                return detections
    except (json.JSONDecodeError, ValueError):
        pass

    # 策略 2: 正则提取 JSON 数组 [...]
    json_arrays = re.findall(r"\[[\s\S]*?\{[\s\S]*?\}[\s\S]*?\]", text)
    for arr_str in json_arrays:
        try:
            parsed = json.loads(arr_str)
            if isinstance(parsed, list):
                for item in parsed:
                    det = _convert_item(item, img_w, img_h)
                    if det:
                        detections.append(det)
                if detections:
                    return detections
        except (json.JSONDecodeError, ValueError):
            continue

    # 策略 3: 逐个提取 JSON 对象 {...}
    json_objects = re.findall(r"\{[^{}]+\}", text)
    for obj_str in json_objects:
        try:
            item = json.loads(obj_str)
            det = _convert_item(item, img_w, img_h)
            if det:
                detections.append(det)
        except (json.JSONDecodeError, ValueError):
            continue

    return detections


def _convert_item(item: dict, img_w: int, img_h: int):
    """
    将单个检测结果从 1000x1000 归一化空间转换为像素坐标。
    Gemma 4 输出格式: box_2d = [y_min, x_min, y_max, x_max] (0-1000)
    VOC 格式: [x_min, y_min, x_max, y_max] (像素坐标)
    """
    if "box_2d" not in item or "label" not in item:
        return None

    box = item["box_2d"]
    label = item["label"].lower().strip()

    if not isinstance(box, list) or len(box) != 4:
        return None

    try:
        y1, x1, y2, x2 = [float(v) for v in box]
    except (ValueError, TypeError):
        return None

    # 归一化到 [0, 1] 再乘以实际尺寸
    x_min = x1 / 1000.0 * img_w
    y_min = y1 / 1000.0 * img_h
    x_max = x2 / 1000.0 * img_w
    y_max = y2 / 1000.0 * img_h

    # 基本合理性检查
    if x_max <= x_min or y_max <= y_min:
        return None

    # 匹配到 VOC 类别（模糊匹配）
    matched_label = _match_voc_class(label)
    if matched_label is None:
        return None

    return {
        "label": matched_label,
        "bbox": [x_min, y_min, x_max, y_max],
        "score": 1.0,
    }


def _match_voc_class(label: str):
    """模糊匹配模型输出的 label 到 VOC 20 类"""
    label = label.lower().strip()

    # 精确匹配
    if label in VOC_CLASSES:
        return label

    # 常见别名映射
    aliases = {
        "airplane": "aeroplane",
        "plane": "aeroplane",
        "aeroplane": "aeroplane",
        "bike": "bicycle",
        "bicycle": "bicycle",
        "bird": "bird",
        "boat": "boat",
        "ship": "boat",
        "bottle": "bottle",
        "bus": "bus",
        "car": "car",
        "automobile": "car",
        "vehicle": "car",
        "cat": "cat",
        "kitten": "cat",
        "chair": "chair",
        "cow": "cow",
        "cattle": "cow",
        "dining table": "diningtable",
        "table": "diningtable",
        "diningtable": "diningtable",
        "dog": "dog",
        "puppy": "dog",
        "horse": "horse",
        "motorbike": "motorbike",
        "motorcycle": "motorbike",
        "person": "person",
        "people": "person",
        "man": "person",
        "woman": "person",
        "child": "person",
        "boy": "person",
        "girl": "person",
        "human": "person",
        "potted plant": "pottedplant",
        "plant": "pottedplant",
        "pottedplant": "pottedplant",
        "houseplant": "pottedplant",
        "flower": "pottedplant",
        "sheep": "sheep",
        "lamb": "sheep",
        "sofa": "sofa",
        "couch": "sofa",
        "train": "train",
        "locomotive": "train",
        "tv": "tvmonitor",
        "television": "tvmonitor",
        "monitor": "tvmonitor",
        "tv monitor": "tvmonitor",
        "tvmonitor": "tvmonitor",
        "screen": "tvmonitor",
    }

    if label in aliases:
        return aliases[label]

    # 子串匹配
    for alias, voc_cls in aliases.items():
        if alias in label or label in alias:
            return voc_cls

    return None


def visualize(image: Image.Image, detections: list, save_path: str):
    """可视化检测结果"""
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 15)), f"{label}", fill="red")
    image.save(save_path)
    print(f"可视化已保存至 {save_path}")


def main():
    # 选择测试图片
    voc_root = (
        Path(__file__).parent.parent / "PLN-ResNet18" / "data" / "VOCdevkit" / "VOC2007"
    )
    if not voc_root.exists():
        # 尝试绝对路径
        voc_root = Path(
            "/mnt/c/Users/Elliot/Desktop/wxg_tasks2/PLN-ResNet18/data/VOCdevkit/VOC2007"
        )

    test_file = voc_root / "ImageSets" / "Main" / "test.txt"
    if test_file.exists():
        with open(test_file) as f:
            img_ids = [line.strip() for line in f if line.strip()]
        img_id = img_ids[0]  # 取第一张
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

    # 加载模型 (4-bit 量化以适配 8GB 显存)
    print("加载 Gemma-4 E2B-it (4-bit 量化)...")
    model_id = "google/gemma-4-E2B-it"

    try:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    except ImportError:
        print("bitsandbytes 未安装, 尝试 FP16 加载...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(model_id)
    print("模型加载完成")

    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": DETECTION_PROMPT},
            ],
        }
    ]

    # 推理
    print(f"Prompt: {DETECTION_PROMPT[:100]}...")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    print("开始推理...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output[0][input_len:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)

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
