#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: Voxtral-mini-3b PEFT Fine-Tuning (LoRA/DoRA) on ESD Fold Splits
=======================================================================

Abstract:
    This script implements a Parameter-Efficient Fine-Tuning (PEFT) pipeline for the
    Voxtral-mini-3b multimodal model. It is designed for Speech Emotion Recognition (SER)
    tasks using the Emotional Speech Dataset (ESD).

    Methodology:
    1.  **Data Ingestion**: Loads a specific fold from the ESD dataset metadata.
    2.  **Stratified Partitioning**: Splits the fold into training and validation sets,
        ensuring a fixed number of samples per class (Stratified Sampling) to rigorously
        evaluate generalization.
    3.  **Prompt Engineering**: Implements a novel "User-Only" chat template strategy
        to bypass role-validation constraints in the base model, enabling direct supervised
        learning on the generated tokens.
    4.  **Optimization**: Utilizes QLoRA (4-bit quantization) or standard LoRA for
        memory-efficient adaptation of linear projection layers (q, k, v, o).


"""

import os
# Disable tokenizer parallelism to prevent deadlocks in multi-process DataLoaders.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import random
import argparse
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset

from transformers import (
    AutoProcessor,
    VoxtralForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

# Define the target taxonomy for emotion classification.
EMOS = ["Angry", "Happy", "Sad", "Neutral"]
KEEP = set(EMOS)

def to_abs(p: str) -> str:
    """Converts relative paths to absolute system paths."""
    return os.path.abspath(os.path.expanduser(p))

def safe_makedirs(p: str):
    """Ensures the existence of the output directory structure."""
    os.makedirs(p, exist_ok=True)

def set_seed(seed: int):
    """
    Enforces deterministic behavior across random number generators (Python, NumPy, PyTorch)
    to ensure reproducibility of experimental results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def speaker_is_english(wav_rel: str) -> bool:
    """
    Heuristic filter to isolate English speakers (IDs 11-20) from the ESD dataset,
    ensuring linguistic consistency in the training data.
    """
    m = re.search(r"downloads/esd/(\d{4})/", (wav_rel or "").replace("\\", "/"))
    if not m:
        return False
    spk = int(m.group(1))
    return 11 <= spk <= 20

def norm_emo(e: str) -> str:
    """Normalizes emotion labels to Capitalized format."""
    return (e or "").strip().capitalize()

class SimpleLogCallback(TrainerCallback):
    """
    Custom callback for logging training metrics (Loss, Eval Loss) to stdout
    in a concise format for real-time monitoring.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            print(f"step={state.global_step} | loss={logs['loss']:.4f}", flush=True)
        if "eval_loss" in logs:
            print(f"epoch={state.epoch:.2f} | step={state.global_step} | eval_loss={logs['eval_loss']:.4f}", flush=True)

# -------------------------
# Class: VoxtralChatAudioTransform
# -------------------------
class VoxtralChatAudioTransform:
    """
    Data Transformation Pipeline.

    This class handles the tokenization and formatting of audio-text pairs.
    Crucially, it implements a workaround for the 'assistant-role' constraint
    in the mistral_common library.

    Mechanism:
    1.  Constructs a Single-Turn conversation consisting ONLY of a USER message.
    2.  Manually appends the target label tokens to the sequence.
    3.  Generates a loss mask to ensure backpropagation occurs only on the label tokens.
    """
    def __init__(self, processor, prompt_text: str, max_new_tokens: int = 8, debug_once: bool = True):
        self.proc = processor
        self.tok = processor.tokenizer
        self.prompt_text = prompt_text
        self.max_new_tokens = int(max_new_tokens)
        self.debug_once = bool(debug_once)
        self._printed = False

        # Ensure the tokenizer has a valid padding token.
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def _user_content(self, wav_path: str):
        # Formats the input payload as expected by the multimodal processor.
        return [
            {"type": "audio", "path": wav_path},
            {"type": "text", "text": self.prompt_text},
        ]

    def _encode_one(self, wav_path: str, label_text: str) -> Dict[str, torch.Tensor]:
        # Step 1: Encode the USER prefix (Audio + Instruction).
        # We deliberately omit the 'assistant' role to avoid validation errors.
        msgs = [
            {"role": "user", "content": self._user_content(wav_path)},
        ]
        enc = self.proc.apply_chat_template(
            msgs,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        prefix_ids = enc["input_ids"][0]
        prefix_attn = enc["attention_mask"][0]
        prefix_len = int(prefix_ids.numel())

        # Step 2: Manually tokenize and append the Target Label.
        # A leading space is added for tokenization consistency.
        lab_ids = self.tok.encode(" " + label_text, add_special_tokens=False)
        if self.tok.eos_token_id is not None:
            lab_ids = lab_ids + [self.tok.eos_token_id]

        # Truncate label to max_new_tokens to prevent unbounded generation during training.
        lab_ids = lab_ids[: max(1, self.max_new_tokens)]
        lab = torch.tensor(lab_ids, dtype=torch.long)

        # Concatenate Prefix and Label to form the full input sequence.
        input_ids = torch.cat([prefix_ids, lab], dim=0)
        attention_mask = torch.cat([prefix_attn, torch.ones_like(lab)], dim=0)

        # Step 3: Construct the Labels Tensor with Masking.
        # We set indices corresponding to the prefix to -100, which is the
        # default ignore_index for CrossEntropyLoss in PyTorch.
        # This ensures the model is not penalized for the instruction prompts.
        labels = input_ids.clone()
        labels[:prefix_len] = -100

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Step 4: Propagate multimodal features (audio tensors) to the output dict.
        for k, v in enc.items():
            if k in out:
                continue
            if torch.is_tensor(v):
                out[k] = v[0]

        # Debug logging for the first sample to verify tensor shapes.
        if self.debug_once and (not self._printed):
            self._printed = True
            print("DEBUG out keys:", sorted(list(out.keys())), flush=True)
            if "input_features" in out:
                print("DEBUG input_features:", tuple(out["input_features"].shape), flush=True)
            print("DEBUG prefix_len:", prefix_len, "| total_len:", int(input_ids.numel()), flush=True)
            print("DEBUG label_text:", label_text, flush=True)

        return out

    def __call__(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        """Dataset map function supporting both batched and unbatched inputs."""
        is_batched = isinstance(ex["audio_path"], list)
        def as_list(x): return x if isinstance(x, list) else [x]

        audio_paths = as_list(ex["audio_path"])
        labels_txt = as_list(ex["label"])

        outs = [self._encode_one(wp, lb) for wp, lb in zip(audio_paths, labels_txt)]
        if not outs:
            return {}

        keys = outs[0].keys()
        packed = {k: [o[k] for o in outs] for k in keys}
        return packed if is_batched else {k: v[0] for k, v in packed.items()}

# -------------------------
# Collator
# -------------------------
def _pad_right_to_shape(t: torch.Tensor, target_shape: List[int], pad_value: float = 0.0) -> torch.Tensor:
    """Helper function to pad tensors to a unified shape for batching."""
    if list(t.shape) == target_shape:
        return t
    pad = []
    # Calculate padding required for each dimension.
    for cur, tgt in zip(reversed(t.shape), reversed(target_shape)):
        pad.extend([0, int(tgt) - int(cur)])
    return F.pad(t, pad, mode="constant", value=pad_value)

@dataclass
class VoxtralPaddingCollator:
    """
    Custom Data Collator.
    Dynamically pads input sequences and multimodal features to the maximum length
    present in the current batch (dynamic padding), which is more memory efficient
    than padding to a fixed global maximum.
    """
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}

        # Pad text-based tensors (IDs, Masks, Labels)
        for key in ["input_ids", "attention_mask", "labels"]:
            tensors = [f[key] for f in features]
            max_len = max(int(t.size(0)) for t in tensors)
            if key == "labels":
                pv = -100  # Ignore index for loss
            elif key == "attention_mask":
                pv = 0     # Masked out
            else:
                pv = self.pad_token_id
            padded = []
            for t in tensors:
                diff = max_len - int(t.size(0))
                if diff > 0:
                    pad = torch.full((diff,), pv, dtype=t.dtype)
                    padded.append(torch.cat([t, pad], dim=0))
                else:
                    padded.append(t)
            batch[key] = torch.stack(padded)

        # Pad multimodal features (e.g., 'input_features' for audio)
        extra_keys = set(features[0].keys()) - {"input_ids", "attention_mask", "labels"}
        for key in sorted(extra_keys):
            v0 = features[0][key]
            if not torch.is_tensor(v0):
                continue
            tensors = [f[key] for f in features]
            nd = tensors[0].dim()
            if any(t.dim() != nd for t in tensors):
                continue
            # Determine maximum shape across all dimensions
            max_shape = [max(int(t.shape[d]) for t in tensors) for d in range(nd)]
            pv = 0.0 if tensors[0].dtype.is_floating_point else 0
            batch[key] = torch.stack([_pad_right_to_shape(t, max_shape, pv) for t in tensors])

        return batch

# -------------------------
# Main Execution Block
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Voxtral-mini-3b Fine-Tuning Pipeline")
    # Path Arguments
    ap.add_argument("--meta_dir", required=True, help="Directory containing ESD metadata JSONLs")
    ap.add_argument("--audio_root", required=True, help="Root directory for audio files")
    ap.add_argument("--model_id", required=True, help="HuggingFace model ID or path")
    ap.add_argument("--output_dir", required=True, help="Directory to save checkpoints")
    
    # Experiment Arguments
    ap.add_argument("--fold", type=int, default=2, help="ESD Fold index to use for training")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--val_per_class", type=int, default=100, help="Samples per class reserved for validation")

    # Training Hyperparameters
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=1, help="Early stopping patience")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    ap.add_argument("--train_bs", type=int, default=1, help="Batch size per device")
    ap.add_argument("--eval_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=2)

    # PEFT / DoRA Arguments
    ap.add_argument("--lora_dropout", type=float, default=0.1)

    # Quantization Arguments
    ap.add_argument("--load_in_4bit", action="store_true", help="Enable QLoRA (4-bit quantization)")

    args = ap.parse_args()
    set_seed(args.seed)
    safe_makedirs(args.output_dir)

    prompt_text = (
        "You are an expert at recognizing emotions from speech.\n"
        "Listen to the audio and output only ONE label from:\n"
        "Angry, Happy, Sad, Neutral."
    )

    # ---------------------------------------------------------
    # Data Loading Phase
    # ---------------------------------------------------------
    jsonl_path = os.path.join(args.meta_dir, "esd", f"fold_{args.fold}", f"esd_train_fold_{args.fold}.jsonl")
    with open(jsonl_path, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    recs = []
    for r in rows:
        emo = norm_emo(r.get("emo", r.get("label", "")))
        wav_rel = (r.get("wav") or "").replace("\\", "/")

        # Filtering logic: Keep specific emotions and English speakers only.
        if emo not in KEEP:
            continue
        if not speaker_is_english(wav_rel):
            continue

        wav_abs = to_abs(os.path.join(args.audio_root, wav_rel))
        if not os.path.isfile(wav_abs):
            continue

        recs.append({"audio_path": wav_abs, "label": emo})

    if not recs:
        raise RuntimeError("No training samples found. Check meta_dir/audio_root and speaker filter.")

    # ---------------------------------------------------------
    # Stratified Splitting (Train vs Validation)
    # ---------------------------------------------------------
    rng = random.Random(args.seed)
    by = {e: [] for e in EMOS}
    for r in recs:
        by[r["label"]].append(r)

    train_l, val_l = [], []
    for e in EMOS:
        lst = by[e]
        rng.shuffle(lst)
        # Reserve strictly 'val_per_class' samples for validation
        v = min(int(args.val_per_class), len(lst))
        val_l.extend(lst[:v])
        train_l.extend(lst[v:])

    if not train_l:
        raise RuntimeError("Train set became empty after carving validation. Reduce val_per_class.")

    # Optional: Downsample training data to ensure balanced classes (undersampling majority class)
    tcnt = Counter([r["label"] for r in train_l])
    min_c = min(tcnt.values())
    t_by = {e: [] for e in EMOS}
    for r in train_l:
        t_by[r["label"]].append(r)
    final_train = []
    for e in EMOS:
        final_train.extend(t_by[e][:min_c])
    rng.shuffle(final_train)

    print("Train size:", len(final_train), " | Val size:", len(val_l), flush=True)
    print("Train class counts:", Counter([r["label"] for r in final_train]), flush=True)
    print("Val class counts:", Counter([r["label"] for r in val_l]), flush=True)

    train_ds = Dataset.from_list(final_train)
    val_ds = Dataset.from_list(val_l)

    # ---------------------------------------------------------
    # Model Initialization & Configuration
    # ---------------------------------------------------------
    proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    transform = VoxtralChatAudioTransform(
        proc,
        prompt_text=prompt_text,
        max_new_tokens=8,
        debug_once=True,
    )
    train_ds.set_transform(transform)
    val_ds.set_transform(transform)

    # Quantization Config (if enabled)
    if args.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Normalized Float 4
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = VoxtralForConditionalGeneration.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation="sdpa", # Scaled Dot-Product Attention (FlashAttention compatible)
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = VoxtralForConditionalGeneration.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        # Enable Gradient Checkpointing to save memory
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

    # ---------------------------------------------------------
    # PEFT Adapter Injection (DoRA only: r=16, alpha=32)
    # ---------------------------------------------------------
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        use_dora=True,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    pad_id = proc.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = proc.tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")

    # ---------------------------------------------------------
    # Training Loop Definition
    # ---------------------------------------------------------
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,

        # Scheduler: Warmup -> Cosine Decay
        warmup_steps=50,
        lr_scheduler_type="cosine",

        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        fp16=True, # Mixed Precision Training

        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        remove_unused_columns=False, # Essential for custom multimodal datasets
        dataloader_num_workers=args.num_workers,
        report_to="none",
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
    )

    callbacks = [SimpleLogCallback(), EarlyStoppingCallback(early_stopping_patience=args.patience)]

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=VoxtralPaddingCollator(pad_token_id=pad_id),
        callbacks=callbacks,
    )

    print("\n=== START TRAIN ===", flush=True)
    trainer.train()

    # Save final adapter weights
    final_dir = os.path.join(args.output_dir, "final_adapter")
    safe_makedirs(final_dir)
    trainer.model.save_pretrained(final_dir)
    proc.save_pretrained(final_dir)
    print("Saved adapter:", final_dir, flush=True)
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()



r'''
 # Voxtral-Mini-3B PEFT Fine-Tuning for Speech Emotion Recognition (SER)

##
This repository contains a specialized training pipeline for fine-tuning the **Voxtral-mini-3b** multimodal model on the **Emotional Speech Dataset (ESD)**. The implementation utilizes Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically **Low-Rank Adaptation (LoRA)** and **Weight-Decomposed Low-Rank Adaptation (DoRA)**, to adapt the model for downstream classification tasks while minimizing computational overhead.

The training strategy employs a **QLoRA** approach (optional 4-bit quantization) to optimize memory usage, enabling training on consumer-grade hardware. A critical contribution of this script is the implementation of a **User-Only Chat Template** strategy, which bypasses standard "assistant" role constraints in `mistral_common` to allow for direct, supervised label generation from audio inputs.

## Methodology

### 1. Data Processing & Stratified Splitting
* **Source Data:** The pipeline operates on pre-defined ESD fold splits (JSONL format).
* **Dynamic Partitioning:** Unlike standard splits, this script dynamically partitions a single training fold into a **Training Set** and a **Validation Set** at runtime.
* **Class Stratification:** To ensure robust evaluation metrics, the validation set is strictly stratified, reserving exactly $N$ samples per emotion class (Angry, Happy, Sad, Neutral).
* **Speaker Filtration:** The data loading routine implements a heuristic filter to isolate English speakers (Speaker IDs 11–20), ensuring linguistic consistency during adaptation.

### 2. Model Architecture & Adaptation
* **Base Model:** Voxtral-mini-3b (a Mistral-based architecture extended for audio modality).
* **Quantization:** The pipeline supports **4-bit NormalFloat (NF4)** quantization via `bitsandbytes`, significantly reducing the VRAM footprint during the fine-tuning process.
* **Adapter Configuration:** * **Target Modules:** Linear layers `q_proj`, `k_proj`, `v_proj`, `o_proj`.
    * **Hyperparameters:** Configurable Rank ($r$) and Scaling Factor ($\alpha$).
    * **DoRA:** Optional integration of Weight-Decomposed Low-Rank Adaptation for enhanced learning stability compared to standard LoRA.

### 3. Novel Prompt Engineering (The "User-Only" Transform)
A key technical challenge in fine-tuning chat-based models for classification is the strict role enforcement (User/Assistant). This repository implements a custom `VoxtralChatAudioTransform` class to resolve this:
* **Single-Turn Construction:** The input sequence is formatted strictly as a `USER` message containing the audio tensor and the text prompt.
* **Manual Label Injection:** Instead of relying on the chat template to format an "assistant" response (which can trigger validation errors in `mistral_common`), the script manually tokenizes the target label (e.g., " Angry") and appends it to the sequence.
* **Loss Masking:** The `labels` tensor is masked with indices of `-100` for all prefix tokens (system instruction + audio tokens), ensuring that the Cross-Entropy loss is calculated **only** on the generated emotion label tokens.

## Key Features

* **Robust Fold Handling:** Automatically parses standard ESD directory structures and handles train/validation splitting internally without data leakage.
* **Memory Optimization:** Seamlessly integrates `bitsandbytes` for 4-bit loading and `peft` for adapter-based training.
* **Concurrency Control:** Manages tokenizer parallelism (`TOKENIZERS_PARALLELISM=false`) to prevent deadlocks in multi-worker DataLoaders.
* **Scheduler:** Implements a warm-up phase (50 steps) followed by a cosine decay scheduler to prevent early overfitting.
'''
