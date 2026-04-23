"""
finetune_gemma.py - Fine-tune Gemma-4 with LoRA for VOC object detection.

Supports two modes:
  - QLoRA 4-bit: for local testing on RTX 4060 (8GB)
  - LoRA bf16: for full training on RTX 4090 (24GB)

Usage:
  # Local QLoRA test (10-20 samples, verify loss decreases)
  python scripts/finetune_gemma.py \
    --data finetune_data/test_demo.jsonl \
    --qlora --max-samples 20 --epochs 3 --batch-size 1

  # AutoDL full training
  python scripts/finetune_gemma.py \
    --data finetune_data/sft_train_trainval_aug.jsonl \
    --epochs 5 --batch-size 4 --gradient-accumulation 4 \
    --lora-rank 64 --output-dir finetune_results/lora_voc
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


MODEL_ID = "google/gemma-4-E2B-it"

LLM_TARGET_MODULES = ["q_proj", "v_proj"]
VISION_TARGET_MODULES = ["q_proj", "v_proj"]


class VOCDetectionDataset(Dataset):
    """Dataset for VOC detection SFT training.

    Each sample: {"image": "/path/to/img.jpg", "text": "<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>"}
    """

    def __init__(self, jsonl_path, processor, max_samples=None, image_base=None):
        self.processor = processor
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line)
                if image_base:
                    sample["image"] = str(Path(image_base) / Path(sample["image"]).name)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        text = sample["text"]
        return {"image": image, "text": text}


class VLMDataCollator:
    """Collator for vision-language model training.

    Processes (image, text) pairs into model inputs with proper labels.
    Masks the prompt tokens (user turn) from the loss.
    """

    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id

        # <start_of_turn>model\n is tokenized as 9 subword tokens
        self.model_marker_ids = processor.tokenizer(
            "<start_of_turn>model\n", add_special_tokens=False
        ).input_ids

    def _find_model_response_start(self, input_ids):
        """Find the token position where the model response begins.

        Searches for the sequence <start_of_turn>model\n in the token IDs
        and returns the position right after it (start of actual response).
        """
        seq = self.model_marker_ids
        seq_len = len(seq)
        for i in range(len(input_ids) - seq_len + 1):
            if input_ids[i : i + seq_len].tolist() == seq:
                return i + seq_len  # Position after "model\n"
        return len(input_ids)  # Fallback: mask everything

    def __call__(self, features):
        images = [f["image"] for f in features]
        texts = [f["text"] for f in features]

        # Insert image token at the beginning of each text
        texts_with_image = [f"<|image|>\n{t}" for t in texts]

        # Process with the processor
        # Images must be wrapped in nested lists: [[img1], [img2], ...]
        batch = self.processor(
            text=texts_with_image,
            images=[[img] for img in images],
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )

        # Create labels: mask everything before the model response
        labels = batch["input_ids"].clone()

        for i in range(len(features)):
            # Find where model response content starts (after <start_of_turn>model\n)
            resp_start = self._find_model_response_start(batch["input_ids"][i])
            labels[i, :resp_start] = -100

        # Mask padding tokens
        labels[labels == self.pad_token_id] = -100

        batch["labels"] = labels
        return batch


def load_model(model_id, qlora=False):
    """Load Gemma-4 model with optional QLoRA quantization.

    Args:
        model_id: HuggingFace model identifier.
        qlora: if True, load in 4-bit for memory-constrained GPUs.

    Returns:
        (model, processor) tuple.
    """
    hf_token = os.environ.get("HF_TOKEN", None)

    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

    if qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            attn_implementation="eager",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # Use flash_attention_2 if available (30-50% faster on 4090)
        try:
            import flash_attn

            attn_impl = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            attn_impl = "eager"
            print("Flash Attention 2 not available, using eager")

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
            attn_implementation=attn_impl,
        )

    return model, processor


def apply_lora(model, rank=64, alpha=64, vision_lora=True):
    """Apply LoRA adapters to the model.

    Args:
        model: the loaded Gemma-4 model.
        rank: LoRA rank.
        alpha: LoRA alpha (should be >= rank).
        vision_lora: if True, also apply LoRA to the vision encoder.

    Returns:
        PEFT-wrapped model.
    """
    target_modules = []

    # LLM modules
    llm_prefix = "language_model"
    for name, module in model.named_modules():
        for target in LLM_TARGET_MODULES:
            if name.endswith(target) and llm_prefix in name:
                target_modules.append(name)
                break

    # Vision encoder modules
    if vision_lora:
        vision_prefix = "vision_tower"
        for name, module in model.named_modules():
            for target in VISION_TARGET_MODULES:
                if name.endswith(target) and vision_prefix in name:
                    target_modules.append(name)
                    break

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-4 with LoRA")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to SFT JSONL file"
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="Model ID")
    parser.add_argument("--output-dir", type=str, default="finetune_results/lora_voc")
    parser.add_argument(
        "--qlora", action="store_true", help="Use QLoRA 4-bit (for 8GB GPU)"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument(
        "--no-vision-lora", action="store_true", help="Skip vision encoder LoRA"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit training samples"
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (0=single process, 4-8 for multi-GPU)",
    )
    parser.add_argument(
        "--image-base",
        type=str,
        default=None,
        help="Base path prefix for images (for AutoDL path mapping)",
    )
    args = parser.parse_args()

    model_id = args.model_id

    print(f"Model: {model_id}")
    print(f"Data: {args.data}")
    print(f"QLoRA: {args.qlora}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"Vision LoRA: {not args.no_vision_lora}")
    print(f"Output: {args.output_dir}")

    # Load model
    print("\nLoading model...")
    model, processor = load_model(model_id, qlora=args.qlora)

    # Apply LoRA
    print("\nApplying LoRA...")
    model = apply_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        vision_lora=not args.no_vision_lora,
    )

    # Load dataset
    print("\nLoading dataset...")
    dataset = VOCDetectionDataset(
        args.data,
        processor,
        max_samples=args.max_samples,
        image_base=args.image_base,
    )
    print(f"Dataset size: {len(dataset)} samples")

    # Data collator
    collator = VLMDataCollator(processor)

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save adapter
    print(f"\nSaving LoRA adapter to {output_dir / 'adapter'}")
    model.save_pretrained(str(output_dir / "adapter"))
    processor.save_pretrained(str(output_dir / "adapter"))

    print("Done!")


if __name__ == "__main__":
    main()
