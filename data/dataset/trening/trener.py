# train_whisper_hr_simple_4gb.py
# Minimal Whisper fine-tune (soundfile). Tuned for 4GB VRAM.

import os
import csv
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate


# -----------------------
# Config (4GB-safe)
# -----------------------
# If medium OOMs (likely), change to: "openai/whisper-small"
MODEL_NAME = "openai/whisper-medium"

LANGUAGE = "croatian"
TASK = "transcribe"

METADATA_CSV = "data/metadata.csv"
OUTPUT_DIR = "outputs/whisper-hr-finetune-4gb"

SEED = 42
TEST_SIZE = 0.05

LR = 5e-6                 # smaller LR helps small data + stability
EPOCHS = 2                # start small

PER_DEVICE_BATCH = 1      # MUST for 4GB
GRAD_ACCUM = 16           # effective batch = 16 (slow but stable)
WARMUP_STEPS = 50

FP16 = True               # helps memory on GPU

MAX_LABEL_LENGTH = 192    # lower = less memory
EVAL_STEPS = 50
SAVE_STEPS = 50
LOGGING_STEPS = 10


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(csv_path: str) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "path" not in r.fieldnames or "text" not in r.fieldnames:
            raise ValueError("metadata.csv must have columns: path,text")
        for row in r:
            path = (row["path"] or "").strip()
            text = (row["text"] or "").strip()
            if not path or not text:
                continue
            if not os.path.exists(path):
                print(f"⚠ missing audio, skipping: {path}")
                continue
            rows.append({"path": path, "text": text})
    if not rows:
        raise ValueError("No valid rows found in metadata.csv (check paths/text).")
    return rows


class CsvAudioDataset(Dataset):
    def __init__(self, items: List[Dict[str, str]], processor: WhisperProcessor):
        self.items = items
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        audio_path = item["path"]
        text = item["text"]

        audio, sr = sf.read(audio_path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if sr != 16000:
            raise ValueError(
                f"{audio_path} has sample rate {sr}. Convert all WAVs to 16kHz mono first."
            )

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]

        labels = self.processor.tokenizer(text).input_ids[:MAX_LABEL_LENGTH]

        return {"input_features": input_features, "labels": labels}


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def main():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, VRAM: {props.total_memory/1024**3:.2f} GB")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    items = load_metadata(METADATA_CSV)
    random.shuffle(items)
    split = max(1, int(len(items) * (1.0 - TEST_SIZE)))
    train_items = items[:split]
    eval_items = items[split:] if split < len(items) else items[:1]

    train_ds = CsvAudioDataset(train_items, processor)
    eval_ds = CsvAudioDataset(eval_items, processor)

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.use_cache = False
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    model.config.suppress_tokens = []

    # Memory saver (slower but helps on small VRAM)
    #model.gradient_checkpointing_enable()

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=EPOCHS,
        fp16=bool(FP16 and torch.cuda.is_available()),
        predict_with_generate=True,
        generation_max_length=MAX_LABEL_LENGTH,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,

        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        report_to="none",
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=WhisperDataCollator(processor),
        compute_metrics=compute_metrics,
    )

    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ CUDA OOM (4GB VRAM is usually not enough for whisper-medium training).")
            print("✅ Fix options (choose one):")
            print("  1) Change MODEL_NAME to 'openai/whisper-small' (recommended).")
            print("  2) Run on CPU (set CUDA unavailable or uninstall GPU torch).")
            print("  3) Reduce MAX_LABEL_LENGTH further (e.g. 128).")
            print("  4) Keep batch=1 and increase GRAD_ACCUM if needed.")
        raise

    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✅ Saved fine-tuned model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()