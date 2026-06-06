#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Localized Counterfactual XAI for Voxtral SER using PREVIOUS predictions
======================================================================

This script does NOT run full prediction/evaluation again.
It reads an existing predictions.jsonl / predictions.csv file produced by your
previous evaluation, selects samples, and runs only the explainability stage.

Main idea / research gap:
1) Use previous prediction as the target class.
2) Find temporally influential speech regions by occlusion.
3) On the top influential regions only, apply localized acoustic counterfactuals:
   energy, pitch, speed, noise, spectral dark/bright, smoothing.
4) Compare local counterfactual effects with:
   - random same-duration regions
   - global whole-utterance perturbation
5) Report probability drop, contrastive probability margin, and optional
   contrastive log-prob margin.

Expected previous prediction columns/keys:
- audio_path OR audio_id/path/wav/filepath
- ground_truth OR label
- prediction
- confidence
- probs OR p_angry,p_happy,p_sad,p_neutral
- correct optional

Example:
python3 code/explain_from_predictions_local_cf.py \
  --pred_jsonl eval_reports/.../predictions.jsonl \
  --base_model /path/to/voxtral-mini-3b \
  --adapter_dir /path/to/final_adapter \
  --load_in_4bit \
  --select_mode wrong_then_correct \
  --max_samples 100 \
  --wrong_samples 40 \
  --segment_sec 0.2 \
  --max_segments 36 \
  --top_k 3 \
  --random_n 5 \
  --mask_mode attenuate \
  --out_dir eval_reports/local_cf_xai
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import csv
import json
import math
import html
import argparse
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

import librosa
import soundfile as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoProcessor, VoxtralForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

EMOS = ["Angry", "Happy", "Sad", "Neutral"]
PROMPT_TEXT = (
    "You are an expert at recognizing emotions from speech.\n"
    "Listen to the audio and output only ONE label from:\n"
    "Angry, Happy, Sad, Neutral."
)

# ============================================================
# Basic utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def to_abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


def fail_if_missing_file(path: Optional[str], label: str):
    """Fail early with a clear message for required user-provided files."""
    if not path:
        raise ValueError(f"Missing required path for {label}.")
    if not os.path.isfile(to_abs(path)):
        raise FileNotFoundError(f"{label} not found: {path}")


def fail_if_missing_dir(path: Optional[str], label: str):
    """Fail early with a clear message for required user-provided directories."""
    if not path:
        raise ValueError(f"Missing required path for {label}.")
    if not os.path.isdir(to_abs(path)):
        raise NotADirectoryError(f"{label} not found: {path}")


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def softmax_np(xs: List[float]) -> List[float]:
    arr = np.array(xs, dtype=np.float64)
    arr = arr - np.max(arr)
    ex = np.exp(arr)
    return (ex / max(float(np.sum(ex)), 1e-12)).tolist()


def get_model_device(model) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ["1", "true", "yes", "y"]

# ============================================================
# Previous prediction loading
# ============================================================

def normalize_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    mapping = {
        "ang": "Angry", "anger": "Angry", "angry": "Angry",
        "hap": "Happy", "happy": "Happy", "happiness": "Happy", "exc": "Happy", "excited": "Happy",
        "sad": "Sad", "sadness": "Sad",
        "neu": "Neutral", "neutral": "Neutral",
    }
    if s in mapping:
        return mapping[s]
    s2 = str(x).strip().capitalize()
    return s2 if s2 in EMOS else None


def parse_probs(row: Dict[str, Any]) -> Dict[str, float]:
    if "probs" in row and row["probs"] not in [None, ""]:
        p = row["probs"]
        if isinstance(p, str):
            p = json.loads(p)
        out = {e: float(p.get(e, p.get(e.lower(), 0.0))) for e in EMOS}
        s = sum(out.values())
        if s > 0:
            out = {k: float(v / s) for k, v in out.items()}
        return out

    keymap = {
        "Angry": ["p_angry", "angry", "prob_angry"],
        "Happy": ["p_happy", "happy", "prob_happy"],
        "Sad": ["p_sad", "sad", "prob_sad"],
        "Neutral": ["p_neutral", "neutral", "prob_neutral"],
    }
    probs = {}
    for e, keys in keymap.items():
        val = None
        for k in keys:
            if k in row and str(row[k]).strip() != "":
                val = row[k]
                break
        probs[e] = float(val) if val is not None else 0.0
    s = sum(probs.values())
    if s > 0:
        probs = {k: float(v / s) for k, v in probs.items()}
    return probs


def get_audio_path(row: Dict[str, Any], audio_root: Optional[str]) -> str:
    for k in ["audio_path", "audio", "path", "wav", "file", "filepath", "audio_id"]:
        if k in row and str(row[k]).strip():
            p = str(row[k]).strip()
            break
    else:
        raise ValueError("Could not find audio path field in prediction row.")

    if audio_root and not os.path.isabs(p):
        p = os.path.join(audio_root, p)
    return to_abs(p)


def load_prediction_rows(args) -> List[Dict[str, Any]]:
    path = args.pred_jsonl or args.pred_csv
    if not path:
        raise ValueError("Provide --pred_jsonl or --pred_csv")
    path = to_abs(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    raw_rows = []
    if args.pred_jsonl or path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_rows.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            raw_rows = list(csv.DictReader(f))

    rows = []
    for i, r in enumerate(raw_rows):
        try:
            audio_path = get_audio_path(r, args.audio_root)
        except Exception as e:
            print(f"[WARN] row {i}: skipped because audio path failed: {e}")
            continue
        if not os.path.isfile(audio_path):
            print(f"[WARN] row {i}: audio not found: {audio_path}")
            continue

        gt = normalize_label(r.get("ground_truth", r.get("label", r.get("true_label", None))))
        pred = normalize_label(r.get("prediction", r.get("pred", r.get("predicted", None))))
        if pred is None:
            print(f"[WARN] row {i}: skipped because prediction label is missing/invalid")
            continue
        probs = parse_probs(r)
        if sum(probs.values()) <= 0:
            probs = {e: 1.0 / len(EMOS) for e in EMOS}
        conf = float(r.get("confidence", probs.get(pred, 0.0)))
        correct = parse_bool(r.get("correct", gt == pred if gt else False)) if ("correct" in r or gt) else None

        rows.append({
            "source_index": int(r.get("index", i)) if str(r.get("index", i)).strip() != "" else i,
            "dataset": str(r.get("dataset", args.dataset_name or "unknown")),
            "audio_id": str(r.get("audio_id", r.get("audio_path", audio_path))),
            "audio_path": audio_path,
            "ground_truth": gt,
            "prediction": pred,
            "confidence": conf,
            "probs": probs,
            "correct": correct,
            "raw": r,
        })
    return select_rows(rows, args)


def select_rows(rows: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    rng = random.Random(args.seed)
    rows = list(rows)

    if args.only_wrong:
        rows = [r for r in rows if r.get("correct") is False]
    if args.only_correct:
        rows = [r for r in rows if r.get("correct") is True]

    if args.select_mode == "first":
        selected = rows
    elif args.select_mode == "random":
        rng.shuffle(rows)
        selected = rows
    elif args.select_mode == "wrong_then_correct":
        wrong = [r for r in rows if r.get("correct") is False]
        correct = [r for r in rows if r.get("correct") is True]
        unknown = [r for r in rows if r.get("correct") is None]
        rng.shuffle(wrong)
        rng.shuffle(correct)
        rng.shuffle(unknown)
        selected = wrong[:max(0, args.wrong_samples)]
        remaining = max(0, args.max_samples - len(selected)) if args.max_samples > 0 else len(correct) + len(unknown)
        selected += correct[:remaining]
        if args.max_samples <= 0:
            selected += unknown
        elif len(selected) < args.max_samples:
            selected += unknown[:args.max_samples - len(selected)]
        rng.shuffle(selected)
    else:
        raise ValueError(args.select_mode)

    if args.max_samples and args.max_samples > 0:
        selected = selected[:args.max_samples]
    if not selected:
        raise RuntimeError("No prediction rows selected.")
    return selected

# ============================================================
# Model loading and scoring
# ============================================================

def load_processor(base_model: str, adapter_dir: str):
    try:
        return AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)
    except Exception:
        return AutoProcessor.from_pretrained(base_model, trust_remote_code=True)


def load_model_and_processor(args):
    processor = load_processor(args.base_model, args.adapter_dir)
    if args.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = VoxtralForConditionalGeneration.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation="sdpa",
        )
    else:
        base = VoxtralForConditionalGeneration.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation="sdpa",
        )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor


def build_user_prefix(processor, wav_path: str):
    msgs = [{
        "role": "user",
        "content": [
            {"type": "audio", "path": wav_path},
            {"type": "text", "text": PROMPT_TEXT},
        ],
    }]
    return processor.apply_chat_template(
        msgs,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


@torch.inference_mode()
def score_audio_labels(model, processor, wav_path: str) -> Dict[str, Any]:
    device = get_model_device(model)
    tok = processor.tokenizer
    enc = build_user_prefix(processor, wav_path)
    prefix_ids = enc["input_ids"][0]
    prefix_attn = enc["attention_mask"][0]
    prefix_len = int(prefix_ids.numel())

    mean_logprobs = {}
    sum_logprobs = {}
    for lab in EMOS:
        lab_ids = tok.encode(" " + lab, add_special_tokens=False)
        lab_t = torch.tensor(lab_ids, dtype=torch.long)
        full_ids = torch.cat([prefix_ids, lab_t], dim=0)
        full_attn = torch.cat([prefix_attn, torch.ones_like(lab_t)], dim=0)

        batch = {}
        for k, v in enc.items():
            if not torch.is_tensor(v):
                continue
            if k == "input_ids":
                batch[k] = full_ids.unsqueeze(0)
            elif k == "attention_mask":
                batch[k] = full_attn.unsqueeze(0)
            else:
                batch[k] = v
            batch[k] = batch[k].to(device)

        out = model(**batch)
        logits = out.logits[0]
        token_lps = []
        for j, tid in enumerate(lab_ids):
            abs_pos = prefix_len + j
            pred_pos = abs_pos - 1
            lp = F.log_softmax(logits[pred_pos].float(), dim=-1)[int(tid)]
            token_lps.append(float(lp.detach().cpu()))
        mean_logprobs[lab] = float(np.mean(token_lps))
        sum_logprobs[lab] = float(np.sum(token_lps))
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    probs_list = softmax_np([mean_logprobs[e] for e in EMOS])
    probs = {e: float(p) for e, p in zip(EMOS, probs_list)}
    pred = max(probs, key=probs.get)
    return {
        "prediction": pred,
        "confidence": float(probs[pred]),
        "probs": probs,
        "mean_logprobs": mean_logprobs,
        "sum_logprobs": sum_logprobs,
    }

# ============================================================
# Audio helpers
# ============================================================

def load_audio(wav_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    return y.astype(np.float32), sr


def write_wav(path: str, y: np.ndarray, sr: int):
    y = np.nan_to_num(y).astype(np.float32)
    y = np.clip(y, -1.0, 1.0)
    sf.write(path, y, sr)


def fade_window(n: int, fade_len: int) -> np.ndarray:
    w = np.ones(n, dtype=np.float32)
    if n <= 1 or fade_len <= 1:
        return w
    f = min(fade_len, n // 2)
    ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
    w[:f] = ramp
    w[-f:] = ramp[::-1]
    return w


def replace_region(y: np.ndarray, sr: int, start: float, end: float, new_seg: np.ndarray, crossfade_ms: float = 10.0) -> np.ndarray:
    y2 = y.copy()
    s = max(0, int(round(start * sr)))
    e = min(len(y2), int(round(end * sr)))
    if e <= s:
        return y2
    target_len = e - s
    new_seg = np.asarray(new_seg, dtype=np.float32)
    if len(new_seg) == 0:
        new_seg = np.zeros(target_len, dtype=np.float32)
    if len(new_seg) != target_len:
        xp = np.linspace(0, 1, len(new_seg))
        xq = np.linspace(0, 1, target_len)
        new_seg = np.interp(xq, xp, new_seg).astype(np.float32)
    fade_len = int(round((crossfade_ms / 1000.0) * sr))
    w = fade_window(target_len, fade_len)
    y2[s:e] = (1.0 - w) * y2[s:e] + w * new_seg
    return y2


def mask_region(y, sr, start, end, mode="attenuate", attenuate_factor=0.2):
    s = max(0, int(round(start * sr)))
    e = min(len(y), int(round(end * sr)))
    if e <= s:
        return y.copy()
    seg = y[s:e].copy()
    if mode == "zero":
        new_seg = np.zeros_like(seg)
    elif mode == "attenuate":
        new_seg = seg * float(attenuate_factor)
    elif mode == "mean":
        new_seg = np.ones_like(seg) * float(np.mean(seg))
    else:
        raise ValueError(mode)
    return replace_region(y, sr, start, end, new_seg)


def mask_many_regions(y, sr, regions, mode="attenuate", attenuate_factor=0.2):
    y2 = y.copy()
    for r in regions:
        y2 = mask_region(y2, sr, float(r["start"]), float(r["end"]), mode=mode, attenuate_factor=attenuate_factor)
    return y2


def keep_only_regions(y, sr, regions):
    y2 = np.zeros_like(y)
    for r in regions:
        s = max(0, int(round(float(r["start"]) * sr)))
        e = min(len(y), int(round(float(r["end"]) * sr)))
        if e > s:
            y2[s:e] = y[s:e]
    return y2

# ============================================================
# Segmentation and acoustic features
# ============================================================

def make_speech_aware_segments(y, sr, segment_sec=0.2, max_segments=36, min_speech_ratio=0.25):
    duration = len(y) / sr if sr else 0.0
    if duration <= 0:
        return []
    frame_length = 1024
    hop_length = 256
    try:
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        thr = max(float(np.percentile(rms, 30)), float(0.08 * np.max(rms)), 1e-8)
        active = rms > thr
        if np.any(active):
            first_t = max(0.0, float(times[np.argmax(active)]) - 0.05)
            last_t = min(duration, float(times[len(active) - 1 - np.argmax(active[::-1])]) + 0.05)
        else:
            first_t, last_t = 0.0, duration
    except Exception:
        first_t, last_t = 0.0, duration

    segments = []
    t = first_t
    full_rms = None
    try:
        full_rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        thr2 = max(float(np.percentile(full_rms, 30)), float(0.08 * np.max(full_rms)), 1e-8)
    except Exception:
        thr2 = None

    while t < last_t:
        end = min(last_t, t + segment_sec)
        sidx = max(0, int(round(t * sr)))
        eidx = min(len(y), int(round(end * sr)))
        if eidx > sidx:
            if thr2 is not None:
                try:
                    seg_rms = librosa.feature.rms(y=y[sidx:eidx], frame_length=1024, hop_length=256)[0]
                    speech_ratio = float(np.mean(seg_rms > thr2))
                except Exception:
                    speech_ratio = 1.0
            else:
                speech_ratio = 1.0
            if speech_ratio >= min_speech_ratio:
                segments.append({"start": float(t), "end": float(end), "speech_ratio": speech_ratio})
        t += segment_sec

    if not segments:
        nseg = int(math.ceil(duration / segment_sec))
        segments = [{"start": float(i * segment_sec), "end": float(min(duration, (i + 1) * segment_sec)), "speech_ratio": 1.0} for i in range(nseg)]

    if max_segments and len(segments) > max_segments:
        idxs = np.linspace(0, len(segments) - 1, max_segments).round().astype(int)
        segments = [segments[int(i)] for i in idxs]
    return segments


def audio_features(y: np.ndarray, sr: int, start=None, end=None) -> Dict[str, Any]:
    if start is not None and end is not None:
        s = max(0, int(round(float(start) * sr)))
        e = min(len(y), int(round(float(end) * sr)))
        seg = y[s:e]
    else:
        seg = y
    seg = np.asarray(seg, dtype=np.float32)
    dur = len(seg) / sr if sr else 0.0
    out = {"duration_sec": float(dur)}
    if len(seg) < max(256, int(0.03 * sr)):
        out.update({"rms_db": None, "f0_mean_hz": None, "f0_std_hz": None, "f0_range_hz": None,
                    "voiced_ratio": None, "spectral_centroid_hz": None, "zcr": None, "onset_rate_per_sec": None})
        return out

    rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
    out["rms_db"] = float(20.0 * np.log10(rms + 1e-12))
    try:
        f0, _, _ = librosa.pyin(seg, fmin=50, fmax=500, sr=sr)
        voiced = np.isfinite(f0)
        f0v = f0[voiced]
    except Exception:
        voiced = np.array([])
        f0v = np.array([])
    out["voiced_ratio"] = float(np.mean(voiced)) if len(voiced) else None
    if len(f0v) > 0:
        out["f0_mean_hz"] = float(np.mean(f0v))
        out["f0_std_hz"] = float(np.std(f0v))
        out["f0_range_hz"] = float(np.max(f0v) - np.min(f0v))
    else:
        out["f0_mean_hz"] = out["f0_std_hz"] = out["f0_range_hz"] = None
    try:
        out["spectral_centroid_hz"] = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
    except Exception:
        out["spectral_centroid_hz"] = None
    try:
        out["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(seg)))
    except Exception:
        out["zcr"] = None
    try:
        onset_env = librosa.onset.onset_strength(y=seg, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        out["onset_rate_per_sec"] = float(len(onsets) / max(dur, 1e-6))
    except Exception:
        out["onset_rate_per_sec"] = None
    return out


def rel_change(local, full):
    if local is None or full is None or abs(float(full)) < 1e-9:
        return None
    return float((float(local) - float(full)) / abs(float(full)))


def cue_summary(local_feat, full_feat):
    cues = []
    if local_feat.get("rms_db") is not None and full_feat.get("rms_db") is not None:
        diff = local_feat["rms_db"] - full_feat["rms_db"]
        if diff > 2.0: cues.append("higher loudness/energy")
        elif diff < -2.0: cues.append("lower loudness/energy")
    d = rel_change(local_feat.get("f0_mean_hz"), full_feat.get("f0_mean_hz"))
    if d is not None:
        if d > 0.10: cues.append("higher pitch")
        elif d < -0.10: cues.append("lower pitch")
    d = rel_change(local_feat.get("f0_std_hz"), full_feat.get("f0_std_hz"))
    if d is not None:
        if d > 0.15: cues.append("larger pitch variation")
        elif d < -0.15: cues.append("smaller pitch variation")
    d = rel_change(local_feat.get("spectral_centroid_hz"), full_feat.get("spectral_centroid_hz"))
    if d is not None:
        if d > 0.10: cues.append("brighter high-frequency content")
        elif d < -0.10: cues.append("darker low-frequency content")
    d = rel_change(local_feat.get("onset_rate_per_sec"), full_feat.get("onset_rate_per_sec"))
    if d is not None:
        if d > 0.15: cues.append("faster local acoustic changes")
        elif d < -0.15: cues.append("slower local acoustic changes")
    if not cues:
        cues.append("no strong simple acoustic cue difference")
    return cues

# ============================================================
# Acoustic transforms
# ============================================================

def fit_length(seg: np.ndarray, target_len: int) -> np.ndarray:
    seg = np.asarray(seg, dtype=np.float32)
    if len(seg) == target_len:
        return seg
    if len(seg) < 2:
        return np.zeros(target_len, dtype=np.float32)
    xp = np.linspace(0, 1, len(seg))
    xq = np.linspace(0, 1, target_len)
    return np.interp(xq, xp, seg).astype(np.float32)


def spectral_tilt(seg: np.ndarray, sr: int, mode: str) -> np.ndarray:
    n = len(seg)
    if n < 16:
        return seg.copy()
    spec = np.fft.rfft(seg)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    fmax = max(freqs[-1], 1.0)
    norm = freqs / fmax
    if mode == "darken":
        gain = 1.0 - 0.65 * norm
    elif mode == "brighten":
        gain = 0.55 + 0.85 * norm
    else:
        gain = np.ones_like(norm)
    out = np.fft.irfft(spec * gain, n=n).astype(np.float32)
    peak = max(np.max(np.abs(out)), 1e-6)
    orig_peak = max(np.max(np.abs(seg)), 1e-6)
    return out * min(1.0, orig_peak / peak)


def smooth_dynamics(seg: np.ndarray, sr: int, win_ms: float = 35.0) -> np.ndarray:
    if len(seg) < 8:
        return seg.copy()
    win = max(3, int(round(sr * win_ms / 1000.0)))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(seg, kernel, mode="same").astype(np.float32)


def add_noise_at_snr(seg: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = float(np.mean(seg ** 2) + 1e-12)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, math.sqrt(noise_power), size=len(seg)).astype(np.float32)
    return (seg + noise).astype(np.float32)


def transform_segment(seg: np.ndarray, sr: int, transform: str, rng: np.random.Generator) -> np.ndarray:
    seg = np.asarray(seg, dtype=np.float32)
    target_len = len(seg)
    if target_len == 0:
        return seg
    if transform == "energy_down":
        return seg * 0.35
    if transform == "energy_up":
        return np.clip(seg * 1.6, -1.0, 1.0).astype(np.float32)
    if transform == "pitch_up":
        try:
            return librosa.effects.pitch_shift(y=seg, sr=sr, n_steps=2.0).astype(np.float32)
        except Exception:
            return seg.copy()
    if transform == "pitch_down":
        try:
            return librosa.effects.pitch_shift(y=seg, sr=sr, n_steps=-2.0).astype(np.float32)
        except Exception:
            return seg.copy()
    if transform == "time_faster":
        try:
            return fit_length(librosa.effects.time_stretch(seg, rate=1.25), target_len)
        except Exception:
            return seg.copy()
    if transform == "time_slower":
        try:
            return fit_length(librosa.effects.time_stretch(seg, rate=0.80), target_len)
        except Exception:
            return seg.copy()
    if transform == "noise_local":
        return add_noise_at_snr(seg, snr_db=10.0, rng=rng)
    if transform == "spectral_darken":
        return spectral_tilt(seg, sr, "darken")
    if transform == "spectral_brighten":
        return spectral_tilt(seg, sr, "brighten")
    if transform == "smooth_dynamics":
        return smooth_dynamics(seg, sr)
    raise ValueError(f"Unknown transform: {transform}")


def apply_transform_region(y: np.ndarray, sr: int, start: float, end: float, transform: str, rng: np.random.Generator) -> np.ndarray:
    s = max(0, int(round(start * sr)))
    e = min(len(y), int(round(end * sr)))
    if e <= s:
        return y.copy()
    new_seg = transform_segment(y[s:e], sr, transform, rng)
    return replace_region(y, sr, start, end, new_seg)

# ============================================================
# XAI methods
# ============================================================

def make_original_result_from_prediction(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prediction": row["prediction"],
        "confidence": float(row["confidence"]),
        "probs": {e: float(row["probs"].get(e, 0.0)) for e in EMOS},
        "mean_logprobs": row.get("mean_logprobs", None),
        "sum_logprobs": row.get("sum_logprobs", None),
    }



def choose_contrast_class(original_result: Dict[str, Any], pred: str, true_label: Optional[str] = None) -> Optional[str]:
    """
    Choose a contrast class for contrastive explanations.
    Priority:
      1) ground truth if prediction is wrong
      2) second-highest probability class from the original distribution
    This fixes the old behavior where correct predictions had no contrast margin.
    """
    if true_label in EMOS and true_label != pred:
        return true_label
    probs = original_result.get("probs", {}) or {}
    candidates = [(e, float(probs.get(e, 0.0))) for e in EMOS if e != pred]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]


def compute_prob_margin(result: Dict[str, Any], pred: str, contrast: Optional[str]) -> Optional[float]:
    if not contrast or contrast not in EMOS:
        return None
    probs = result.get("probs", {}) or {}
    return float(probs.get(pred, 0.0) - probs.get(contrast, 0.0))


def compute_logprob_margin(result: Dict[str, Any], pred: str, contrast: Optional[str]) -> Optional[float]:
    if not contrast or contrast not in EMOS:
        return None
    mlp = result.get("mean_logprobs")
    if not isinstance(mlp, dict):
        return None
    if pred not in mlp or contrast not in mlp:
        return None
    return float(mlp[pred] - mlp[contrast])


def safe_z(value: Optional[float], mean: Optional[float], std: Optional[float]) -> Optional[float]:
    if value is None or mean is None or std is None or abs(float(std)) < 1e-9:
        return None
    return float((float(value) - float(mean)) / float(std))


def safe_same_direction_ratio(local_effect: Optional[float], global_effect: Optional[float]) -> Optional[float]:
    """
    Ratio is only meaningful if local and global effects have the same direction
    and the global denominator is not tiny. Otherwise return None instead of a
    misleading huge/negative ratio.
    """
    if local_effect is None or global_effect is None:
        return None
    local_effect = float(local_effect)
    global_effect = float(global_effect)
    if abs(global_effect) < 1e-3:
        return None
    if local_effect == 0.0:
        return 0.0
    if np.sign(local_effect) != np.sign(global_effect):
        return None
    return float(local_effect / global_effect)


def score_target_effect(original_result, perturbed_result, pred, true_label=None, contrast_label=None):
    """
    Compute predicted-class drop and contrastive margin drops.
    Fixed issues:
      - margins are now computed also for correct predictions using top-2 contrast
      - log-prob margins are computed whenever original and perturbed logprobs exist
      - output schema is consistent across local, random, global and occlusion tests
    """
    contrast = contrast_label or choose_contrast_class(original_result, pred, true_label)
    p_before = float(original_result.get("probs", {}).get(pred, 0.0))
    p_after = float(perturbed_result.get("probs", {}).get(pred, 0.0))

    out = {
        "pred_class": pred,
        "true_class": true_label,
        "contrast_class": contrast,
        "prob_before": p_before,
        "prob_after": p_after,
        "prob_drop": float(p_before - p_after),
        "new_prediction": perturbed_result.get("prediction"),
        "new_probs": perturbed_result.get("probs", {}),
    }

    if contrast in EMOS:
        cb = float(original_result.get("probs", {}).get(contrast, 0.0))
        ca = float(perturbed_result.get("probs", {}).get(contrast, 0.0))
        mb = compute_prob_margin(original_result, pred, contrast)
        ma = compute_prob_margin(perturbed_result, pred, contrast)
        out.update({
            "contrast_prob_before": cb,
            "contrast_prob_after": ca,
            "contrast_prob_change": float(ca - cb),
            "prob_margin_before": mb,
            "prob_margin_after": ma,
            "prob_margin_drop": float(mb - ma) if mb is not None and ma is not None else None,
        })

        lmb = compute_logprob_margin(original_result, pred, contrast)
        lma = compute_logprob_margin(perturbed_result, pred, contrast)
        out.update({
            "logprob_margin_before": lmb,
            "logprob_margin_after": lma,
            "logprob_margin_drop": float(lmb - lma) if lmb is not None and lma is not None else None,
        })
    else:
        out.update({
            "contrast_prob_before": None,
            "contrast_prob_after": None,
            "contrast_prob_change": None,
            "prob_margin_before": None,
            "prob_margin_after": None,
            "prob_margin_drop": None,
            "logprob_margin_before": None,
            "logprob_margin_after": None,
            "logprob_margin_drop": None,
        })
    return out


def run_temporal_occlusion(model, processor, y, sr, original_result, tmp_dir, segment_sec, max_segments, mask_mode, attenuate_factor, true_label=None):
    pred = original_result["prediction"]
    orig_prob = float(original_result["probs"].get(pred, 0.0))
    contrast = choose_contrast_class(original_result, pred, true_label)
    candidate_segments = make_speech_aware_segments(y, sr, segment_sec=segment_sec, max_segments=max_segments)
    segments = []
    for i, cand in enumerate(candidate_segments):
        start, end = float(cand["start"]), float(cand["end"])
        y_masked = mask_region(y, sr, start, end, mode=mask_mode, attenuate_factor=attenuate_factor)
        tmp_wav = os.path.join(tmp_dir, f"masked_{i:03d}.wav")
        write_wav(tmp_wav, y_masked, sr)
        r = score_audio_labels(model, processor, tmp_wav)
        eff = score_target_effect(original_result, r, pred, true_label=true_label, contrast_label=contrast)
        masked_prob = float(r["probs"].get(pred, 0.0))
        segments.append({
            "index": i,
            "start": start,
            "end": end,
            "duration": float(end - start),
            "speech_ratio": float(cand.get("speech_ratio", 1.0)),
            "contrast_class": contrast,
            "original_prob": orig_prob,
            "masked_prob": masked_prob,
            "importance_prob_drop": float(eff["prob_drop"]),
            "importance_prob_margin_drop": eff.get("prob_margin_drop"),
            "importance_logprob_margin_drop": eff.get("logprob_margin_drop"),
            "prob_margin_before": eff.get("prob_margin_before"),
            "prob_margin_after": eff.get("prob_margin_after"),
            "logprob_margin_before": eff.get("logprob_margin_before"),
            "logprob_margin_after": eff.get("logprob_margin_after"),
            "masked_prediction": r["prediction"],
            "masked_probs": r["probs"],
        })
    return segments

def run_deletion_curve(model, processor, y, sr, original_result, ranked_segments, tmp_dir, max_k, mask_mode, attenuate_factor):
    pred = original_result["prediction"]
    curve = [{"k": 0, "prob": float(original_result["probs"][pred]), "prediction": original_result["prediction"]}]
    for k in range(1, min(max_k, len(ranked_segments)) + 1):
        y_del = mask_many_regions(y, sr, ranked_segments[:k], mode=mask_mode, attenuate_factor=attenuate_factor)
        p = os.path.join(tmp_dir, f"deletion_top{k}.wav")
        write_wav(p, y_del, sr)
        r = score_audio_labels(model, processor, p)
        curve.append({"k": k, "prob": float(r["probs"][pred]), "prediction": r["prediction"]})
    xs = np.array([c["k"] for c in curve], dtype=float)
    ys = np.array([c["prob"] for c in curve], dtype=float)
    auc = float(np.trapz(ys, xs) / max(xs[-1], 1.0)) if len(xs) > 1 else float(ys[0])
    return {"pred_class": pred, "curve": curve, "deletion_auc": auc}


def run_insertion_curve(model, processor, y, sr, original_result, ranked_segments, tmp_dir, max_k):
    pred = original_result["prediction"]
    curve = []
    for k in range(1, min(max_k, len(ranked_segments)) + 1):
        y_keep = keep_only_regions(y, sr, ranked_segments[:k])
        p = os.path.join(tmp_dir, f"insertion_top{k}.wav")
        write_wav(p, y_keep, sr)
        r = score_audio_labels(model, processor, p)
        curve.append({"k": k, "prob": float(r["probs"][pred]), "prediction": r["prediction"]})
    xs = np.array([c["k"] for c in curve], dtype=float)
    ys = np.array([c["prob"] for c in curve], dtype=float)
    auc = float(np.trapz(ys, xs) / max(xs[-1], 1.0)) if len(xs) > 1 else (float(ys[0]) if len(ys) else 0.0)
    return {"pred_class": pred, "curve": curve, "insertion_auc": auc}


def compute_comp_suff(original_result, deletion, insertion):
    pred = original_result["prediction"]
    p0 = float(original_result["probs"][pred])
    deletion_probs = [float(c["prob"]) for c in deletion.get("curve", []) if c.get("k", 0) > 0]
    insertion_probs = [float(c["prob"]) for c in insertion.get("curve", [])]
    comprehensiveness = float(np.mean([p0 - p for p in deletion_probs])) if deletion_probs else None
    sufficiency = float(np.mean([p0 - p for p in insertion_probs])) if insertion_probs else None
    return {"comprehensiveness_avg_drop": comprehensiveness, "sufficiency_avg_gap": sufficiency}


def sample_random_regions(y, sr, duration, n, seed):
    rng = random.Random(seed)
    total_dur = len(y) / sr
    duration = min(float(duration), total_dur)
    if total_dur <= duration:
        return [{"start": 0.0, "end": total_dur}]
    regions = []
    for _ in range(n):
        st = rng.uniform(0.0, max(0.0, total_dur - duration))
        regions.append({"start": float(st), "end": float(st + duration)})
    return regions



def run_local_paralinguistic_counterfactuals(model, processor, y, sr, original_result, top_segments, tmp_dir, ground_truth,
                                             transforms, random_n=5, seed=42):
    """
    Run localized acoustic counterfactuals and return BOTH schemas:
      - items: backward-compatible flat list
      - top_segment_results + best_local_counterfactual: professional report schema
    Fixed issues:
      - professional HTML/plots previously expected keys that were never produced
      - ratio now avoids misleading values when global effect is tiny or opposite sign
      - margins are computed for correct samples using top-2 contrast class
    """
    pred = original_result["prediction"]
    contrast = choose_contrast_class(original_result, pred, ground_truth)
    rng = np.random.default_rng(seed)
    results = {
        "pred_class": pred,
        "true_class": ground_truth,
        "contrast_class": contrast,
        "items": [],
        "top_segment_results": [],
        "best_local_counterfactual": None,
    }
    total_dur = len(y) / sr

    flat_best = None
    flat_best_score = -1e18

    for rank, seg in enumerate(top_segments, start=1):
        start, end = float(seg["start"]), float(seg["end"])
        dur = float(end - start)
        random_regions = sample_random_regions(y, sr, dur, random_n, seed + 1000 * rank)
        seg_result = {
            "rank": rank,
            "start": start,
            "end": end,
            "duration": dur,
            "transform_results": [],
        }

        for tr in transforms:
            # Local top-region perturbation
            y_local = apply_transform_region(y, sr, start, end, tr, rng)
            p_local = os.path.join(tmp_dir, f"local_r{rank}_{tr}.wav")
            write_wav(p_local, y_local, sr)
            r_local = score_audio_labels(model, processor, p_local)
            local_eff = score_target_effect(original_result, r_local, pred, true_label=ground_truth, contrast_label=contrast)

            # Global whole-utterance perturbation with same transform
            y_global = apply_transform_region(y, sr, 0.0, total_dur, tr, rng)
            p_global = os.path.join(tmp_dir, f"global_r{rank}_{tr}.wav")
            write_wav(p_global, y_global, sr)
            r_global = score_audio_labels(model, processor, p_global)
            global_eff = score_target_effect(original_result, r_global, pred, true_label=ground_truth, contrast_label=contrast)

            # Random same-duration regions
            rand_effects = []
            for j, rr in enumerate(random_regions):
                y_rand = apply_transform_region(y, sr, rr["start"], rr["end"], tr, rng)
                p_rand = os.path.join(tmp_dir, f"random_r{rank}_{tr}_{j}.wav")
                write_wav(p_rand, y_rand, sr)
                r_rand = score_audio_labels(model, processor, p_rand)
                rand_effects.append(score_target_effect(original_result, r_rand, pred, true_label=ground_truth, contrast_label=contrast))

            rand_drops = np.array([x.get("prob_drop", np.nan) for x in rand_effects], dtype=float) if rand_effects else np.array([])
            rand_margin_drops = np.array([x.get("prob_margin_drop", np.nan) for x in rand_effects], dtype=float) if rand_effects else np.array([])
            rand_logmargin_drops = np.array([x.get("logprob_margin_drop", np.nan) for x in rand_effects], dtype=float) if rand_effects else np.array([])

            random_summary = {
                "random_n": int(len(rand_effects)),
                "random_prob_drop_mean": float(np.nanmean(rand_drops)) if np.any(np.isfinite(rand_drops)) else None,
                "random_prob_drop_std": float(np.nanstd(rand_drops)) if np.any(np.isfinite(rand_drops)) else None,
                "random_prob_margin_drop_mean": float(np.nanmean(rand_margin_drops)) if np.any(np.isfinite(rand_margin_drops)) else None,
                "random_prob_margin_drop_std": float(np.nanstd(rand_margin_drops)) if np.any(np.isfinite(rand_margin_drops)) else None,
                "random_logprob_margin_drop_mean": float(np.nanmean(rand_logmargin_drops)) if np.any(np.isfinite(rand_logmargin_drops)) else None,
                "random_logprob_margin_drop_std": float(np.nanstd(rand_logmargin_drops)) if np.any(np.isfinite(rand_logmargin_drops)) else None,
            }
            random_summary["local_vs_random_prob_z"] = safe_z(
                local_eff.get("prob_drop"), random_summary["random_prob_drop_mean"], random_summary["random_prob_drop_std"]
            )
            random_summary["local_vs_random_margin_z"] = safe_z(
                local_eff.get("prob_margin_drop"), random_summary["random_prob_margin_drop_mean"], random_summary["random_prob_margin_drop_std"]
            )
            random_summary["local_vs_random_z_logmargin"] = safe_z(
                local_eff.get("logprob_margin_drop"), random_summary["random_logprob_margin_drop_mean"], random_summary["random_logprob_margin_drop_std"]
            )

            local_drop = local_eff.get("prob_drop")
            global_drop = global_eff.get("prob_drop")
            local_margin_drop = local_eff.get("prob_margin_drop")
            global_margin_drop = global_eff.get("prob_margin_drop")
            local_logmargin_drop = local_eff.get("logprob_margin_drop")
            global_logmargin_drop = global_eff.get("logprob_margin_drop")

            item = {
                "top_region_rank": rank,
                "transform": tr,
                "region": {"start": start, "end": end, "duration": dur},
                "local_effect": local_eff,
                "global_effect": global_eff,
                "random_baseline": random_summary,
                "local_global_ratio_prob_drop": safe_same_direction_ratio(local_drop, global_drop),
                "local_global_ratio_margin_drop": safe_same_direction_ratio(local_margin_drop, global_margin_drop),
                "local_global_ratio_logmargin": safe_same_direction_ratio(local_logmargin_drop, global_logmargin_drop),
            }
            results["items"].append(item)

            tr_result = {
                "transform": tr,
                "prob_before": local_eff.get("prob_before"),
                "prob_after": local_eff.get("prob_after"),
                "prob_drop": local_eff.get("prob_drop"),
                "contrast_class": contrast,
                "prob_margin_before": local_eff.get("prob_margin_before"),
                "prob_margin_after": local_eff.get("prob_margin_after"),
                "prob_margin_drop": local_eff.get("prob_margin_drop"),
                "logprob_margin_before": local_eff.get("logprob_margin_before"),
                "logprob_margin_after": local_eff.get("logprob_margin_after"),
                "logprob_margin_drop": local_eff.get("logprob_margin_drop"),
                "random_prob_drop_mean": random_summary.get("random_prob_drop_mean"),
                "random_prob_drop_std": random_summary.get("random_prob_drop_std"),
                "random_prob_margin_drop_mean": random_summary.get("random_prob_margin_drop_mean"),
                "random_prob_margin_drop_std": random_summary.get("random_prob_margin_drop_std"),
                "random_logprob_margin_drop_mean": random_summary.get("random_logprob_margin_drop_mean"),
                "random_logprob_margin_drop_std": random_summary.get("random_logprob_margin_drop_std"),
                "local_vs_random_z_prob": random_summary.get("local_vs_random_prob_z"),
                "local_vs_random_z_margin": random_summary.get("local_vs_random_margin_z"),
                "local_vs_random_z_logmargin": random_summary.get("local_vs_random_z_logmargin"),
                "global_prob_drop": global_eff.get("prob_drop"),
                "global_prob_margin_drop": global_eff.get("prob_margin_drop"),
                "global_logprob_margin_drop": global_eff.get("logprob_margin_drop"),
                "local_global_ratio_prob_drop": item["local_global_ratio_prob_drop"],
                "local_global_ratio_margin_drop": item["local_global_ratio_margin_drop"],
                "local_global_ratio_logmargin": item["local_global_ratio_logmargin"],
            }
            seg_result["transform_results"].append(tr_result)

            # Best score priority: log-margin if available, then prob-margin, then prob drop.
            score_candidates = [
                local_eff.get("logprob_margin_drop"),
                local_eff.get("prob_margin_drop"),
                local_eff.get("prob_drop"),
            ]
            best_score = next((float(v) for v in score_candidates if v is not None and np.isfinite(float(v))), -1e18)
            if best_score > flat_best_score:
                flat_best_score = best_score
                flat_best = {
                    "top_region_rank": rank,
                    "start": start,
                    "end": end,
                    "duration": dur,
                    "transform": tr,
                    **tr_result,
                }

        results["top_segment_results"].append(seg_result)

    results["best_local_counterfactual"] = flat_best
    return results

# ============================================================
# Plots and reports

# ============================================================
# Plots and reports
# ============================================================

def plot_waveform_importance(y, sr, segments, pred, conf, out_png):
    t = np.arange(len(y)) / sr
    plt.figure(figsize=(14, 4))
    plt.plot(t, y, linewidth=0.7)
    if segments:
        max_imp = max(abs(float(s["importance_prob_drop"])) for s in segments) + 1e-9
        for s in segments:
            imp = max(0.0, float(s["importance_prob_drop"]))
            alpha = min(0.75, 0.12 + 0.60 * imp / max_imp)
            plt.axvspan(float(s["start"]), float(s["end"]), alpha=alpha)
    plt.title(f"Waveform with important regions | prediction={pred} | confidence={conf:.3f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_segment_importance(segments, out_png):
    xs = [f"{s['start']:.1f}-{s['end']:.1f}s" for s in segments]
    ys = [float(s["importance_prob_drop"]) for s in segments]
    plt.figure(figsize=(12, 4))
    plt.bar(xs, ys)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Probability drop")
    plt.title("Temporal occlusion importance by segment")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_curve(curve_obj, key, title, ylabel, out_png):
    curve = curve_obj.get("curve", [])
    xs = [c["k"] for c in curve]
    ys = [c["prob"] for c in curve]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Number of top segments")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_acoustic_cues(full_feat, top_feat, out_png):
    labels, vals = [], []
    # dB uses difference, not ratio
    if full_feat.get("rms_db") is not None and top_feat.get("rms_db") is not None:
        labels.append("rms_db_diff")
        vals.append(float(top_feat["rms_db"] - full_feat["rms_db"]))
    for k in ["f0_mean_hz", "f0_std_hz", "spectral_centroid_hz", "onset_rate_per_sec"]:
        f, t = full_feat.get(k), top_feat.get(k)
        if f is not None and t is not None and abs(float(f)) > 1e-9:
            labels.append(k + "_rel")
            vals.append(float((t - f) / abs(f)))
    if not labels:
        labels, vals = ["no_valid_cues"], [0.0]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, vals)
    plt.axhline(0.0, linestyle="--")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Difference / relative change")
    plt.title("Top segment acoustic cue comparison")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_local_counterfactuals(local_cf, out_png):
    items = local_cf.get("items", [])
    top1 = [it for it in items if int(it.get("top_region_rank", 0)) == 1]
    if not top1:
        return
    labels = [it["transform"] for it in top1]
    local = [it["local_effect"]["prob_drop"] for it in top1]
    random_mean = [it["random_baseline"].get("random_prob_drop_mean") or 0.0 for it in top1]
    global_drop = [it["global_effect"]["prob_drop"] for it in top1]

    x = np.arange(len(labels))
    w = 0.25
    plt.figure(figsize=(13, 4.5))
    plt.bar(x - w, local, width=w, label="local top segment")
    plt.bar(x, random_mean, width=w, label="random region mean")
    plt.bar(x + w, global_drop, width=w, label="global utterance")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Predicted-class probability drop")
    plt.title("Localized acoustic counterfactuals vs random/global baselines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def best_local_cf_text(local_cf):
    items = local_cf.get("items", []) if local_cf else []
    if not items:
        return ""
    def score(it):
        return float(it["local_effect"].get("prob_margin_drop", it["local_effect"].get("prob_drop", 0.0)))
    best = max(items, key=score)
    le = best["local_effect"]
    rb = best["random_baseline"]
    region = best["region"]
    txt = (
        f" Localized acoustic counterfactual analysis shows the strongest tested cue is '{best['transform']}' "
        f"inside {region['start']:.2f}s-{region['end']:.2f}s. "
        f"Changing only this region changes the predicted-class probability from "
        f"{le['prob_before']:.3f} to {le['prob_after']:.3f} "
        f"(drop={le['prob_drop']:.3f})."
    )
    if le.get("prob_margin_drop") is not None:
        txt += f" The contrastive probability margin drops by {le['prob_margin_drop']:.3f}."
    if rb.get("random_prob_drop_mean") is not None:
        z = rb.get("local_vs_random_prob_z")
        ztxt = f", z={z:.2f}" if z is not None else ""
        txt += f" Random same-duration regions have average drop {rb['random_prob_drop_mean']:.3f}{ztxt}."
    return txt


def make_human_explanation(evidence):
    pred = evidence["prediction"]
    conf = float(evidence["confidence"])
    gt = evidence.get("ground_truth")
    correct = evidence.get("correct")
    top = evidence.get("top_segments", [])
    gt_text = f" The ground-truth label is {gt}, so this prediction is {'correct' if correct else 'wrong'}." if gt else ""
    if not top:
        return f"The model predicts {pred} with confidence {conf:.3f}.{gt_text} No segment-level explanation was produced."
    s0 = top[0]
    cues = "; ".join(s0.get("cue_summary", []))
    local_text = best_local_cf_text(evidence.get("localized_acoustic_counterfactuals", {}))
    faith = evidence.get("faithfulness", {})
    faith_txt = ""
    if faith.get("comprehensiveness_avg_drop") is not None:
        faith_txt = f" Average comprehensiveness drop is {faith['comprehensiveness_avg_drop']:.3f}, and sufficiency gap is {faith.get('sufficiency_avg_gap', 0.0):.3f}."
    return (
        f"The model predicts {pred} with confidence {conf:.3f}.{gt_text} "
        f"The most influential region is {s0['start']:.2f}s-{s0['end']:.2f}s. "
        f"Masking/attenuating this region reduces the predicted-class probability by {s0['importance_prob_drop']:.3f}. "
        f"Its main acoustic descriptors are: {cues}."
        f"{local_text}"
        f"{faith_txt} "
        f"This is a post-hoc perturbation-based explanation, not a claim about hidden internal reasoning."
    )


def write_sample_html(evidence, sample_dir):
    explanation = html.escape(evidence["human_explanation"])
    rows = ""
    for e in EMOS:
        rows += f"<tr><td>{e}</td><td>{evidence['probs'].get(e, 0.0):.4f}</td></tr>\n"
    top_rows = ""
    for s in evidence.get("top_segments", []):
        top_rows += f"<tr><td>{s['start']:.2f}-{s['end']:.2f}s</td><td>{s['importance_prob_drop']:.4f}</td><td>{html.escape('; '.join(s.get('cue_summary', [])))}</td></tr>\n"

    cf_rows = ""
    for it in evidence.get("localized_acoustic_counterfactuals", {}).get("items", []):
        le = it["local_effect"]
        rb = it["random_baseline"]
        ge = it["global_effect"]
        cf_rows += (
            f"<tr><td>{it['top_region_rank']}</td><td>{html.escape(it['transform'])}</td>"
            f"<td>{le['prob_drop']:.4f}</td><td>{le.get('prob_margin_drop', '')}</td>"
            f"<td>{rb.get('random_prob_drop_mean', '')}</td><td>{rb.get('local_vs_random_prob_z', '')}</td>"
            f"<td>{ge['prob_drop']:.4f}</td><td>{it.get('local_global_ratio_prob_drop', ''):.4f}</td></tr>\n"
        )

    body = f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Localized XAI Sample</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 28px; line-height: 1.45; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 18px; }}
img {{ max-width: 100%; border: 1px solid #eee; border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ddd; padding: 8px; }}
th {{ background: #f3f3f3; }}
</style></head><body>
<h1>Sample {evidence['sample_index']} | pred={html.escape(evidence['prediction'])} | gt={html.escape(str(evidence.get('ground_truth')))}</h1>
<div class="card"><h2>Human-readable explanation</h2><p>{explanation}</p></div>
<div class="card"><h2>Prediction from previous file</h2><p><b>Audio:</b> {html.escape(evidence['audio_id'])}<br><b>Correct:</b> {evidence.get('correct')}</p><table><tr><th>Class</th><th>Probability</th></tr>{rows}</table></div>
<div class="card"><h2>Top temporal regions</h2><table><tr><th>Time</th><th>Drop</th><th>Acoustic cues</th></tr>{top_rows}</table></div>
<div class="card"><h2>Localized acoustic counterfactuals</h2><table><tr><th>Top region</th><th>Transform</th><th>Local prob drop</th><th>Local margin drop</th><th>Random mean drop</th><th>Local-vs-random z</th><th>Global prob drop</th><th>Local/global ratio</th></tr>{cf_rows}</table></div>
<div class="card"><h2>Plots</h2>
<h3>Waveform importance</h3><img src="waveform_importance.png">
<h3>Segment importance</h3><img src="segment_importance.png">
<h3>Deletion curve</h3><img src="deletion_curve.png">
<h3>Insertion curve</h3><img src="insertion_curve.png">
<h3>Acoustic cue comparison</h3><img src="acoustic_cues.png">
<h3>Localized counterfactuals</h3><img src="localized_counterfactuals.png">
</div>
<div class="card"><h2>Note</h2><p>This report uses previous predictions as input and only re-scores perturbed audio for explanation.</p></div>
</body></html>
"""
    with open(os.path.join(sample_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(body)


def write_index_html(out_dir, explained_items, aggregate):
    rows = ""
    for item in explained_items:
        rel = os.path.relpath(item["report_path"], out_dir)
        rows += f"<tr><td>{item['sample_index']}</td><td>{html.escape(str(item['ground_truth']))}</td><td>{html.escape(item['prediction'])}</td><td>{item['confidence']:.4f}</td><td>{item['correct']}</td><td>{item.get('top1_drop')}</td><td><a href='{html.escape(rel)}'>open report</a></td></tr>\n"
    body = f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Localized Voxtral SER XAI</title>
<style>body{{font-family:Arial,sans-serif;margin:28px;line-height:1.45}}.card{{border:1px solid #ddd;border-radius:10px;padding:16px;margin-bottom:18px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px}}th{{background:#f3f3f3}}</style></head><body>
<h1>Localized Acoustic Counterfactual XAI Report</h1>
<div class="card"><h2>Aggregate summary</h2><pre>{html.escape(json.dumps(jsonable(aggregate), indent=2))}</pre></div>
<div class="card"><h2>Samples</h2><table><tr><th>Sample</th><th>Ground truth</th><th>Prediction</th><th>Confidence</th><th>Correct</th><th>Top1 drop</th><th>Report</th></tr>{rows}</table></div>
</body></html>
"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(body)

# ============================================================
# Explanation diagnosis and classification
# ============================================================

def classify_evidence_strength(value: Optional[float]) -> str:
    """Classify a probability drop into evidence strength categories."""
    if value is None or not np.isfinite(float(value)):
        return "unknown"
    v = abs(float(value))
    if v >= 0.15:
        return "strong"
    if v >= 0.05:
        return "moderate"
    if v >= 0.01:
        return "weak"
    return "negligible"


def compute_explanation_diagnosis(top_segments, local_cf, deletion, insertion):
    """
    Analyze and classify the type of explanation produced.
    
    Returns a diagnosis dict with:
    - scenario: type of evidence (localized, global, mixed, weak, robust, etc.)
    - local_occlusion_strength: evidence from temporal occlusion
    - best_counterfactual_strength: evidence from localized interventions
    - global_sensitivity_strength: evidence from global perturbations
    - max_global_transform: which global transform had strongest effect
    - max_global_prob_drop: magnitude of best global effect
    """
    diagnosis = {
        "scenario": "unknown",
        "local_occlusion_strength": None,
        "best_counterfactual_strength": None,
        "global_sensitivity_strength": None,
        "max_global_transform": None,
        "max_global_prob_drop": None,
    }
    
    # Extract local occlusion strength from top segment
    local_occlusion_drop = None
    if top_segments:
        local_occlusion_drop = float(top_segments[0].get("importance_prob_drop", 0.0))
        diagnosis["local_occlusion_strength"] = classify_evidence_strength(local_occlusion_drop)
    
    # Extract best local counterfactual effect
    best_local_drop = None
    items = local_cf.get("items", []) if local_cf else []
    if items:
        def cf_score(it):
            le = it.get("local_effect", {})
            v = le.get("prob_margin_drop") or le.get("prob_drop", 0.0)
            return float(v) if v is not None else -1e18
        best_item = max(items, key=cf_score)
        best_local_drop = float(best_item["local_effect"].get("prob_margin_drop", best_item["local_effect"].get("prob_drop", 0.0)))
        diagnosis["best_counterfactual_strength"] = classify_evidence_strength(best_local_drop)
    
    # Extract best global perturbation effect
    best_global_drop = None
    best_global_transform = None
    if items:
        for item in items:
            ge = item.get("global_effect", {})
            global_drop = ge.get("prob_drop")
            if global_drop is not None:
                global_drop = float(global_drop)
                if best_global_drop is None or global_drop > best_global_drop:
                    best_global_drop = global_drop
                    best_global_transform = item.get("transform")
    
    if best_global_drop is not None:
        diagnosis["global_sensitivity_strength"] = classify_evidence_strength(best_global_drop)
        diagnosis["max_global_transform"] = best_global_transform
        diagnosis["max_global_prob_drop"] = float(best_global_drop)
    
    # Classify scenario based on relative magnitudes
    local_str = diagnosis["local_occlusion_strength"]
    cf_str = diagnosis["best_counterfactual_strength"]
    global_str = diagnosis["global_sensitivity_strength"]
    
    strong_categories = {"strong", "moderate"}
    weak_categories = {"weak", "negligible"}
    
    if local_str in strong_categories and cf_str in strong_categories and global_str not in strong_categories:
        diagnosis["scenario"] = "localized_counterfactual_evidence"
    elif local_str in strong_categories and global_str not in strong_categories:
        diagnosis["scenario"] = "localized_evidence"
    elif global_str in strong_categories and local_str not in strong_categories:
        diagnosis["scenario"] = "global_sensitivity"
    elif local_str in strong_categories and global_str in strong_categories:
        diagnosis["scenario"] = "mixed_evidence"
    elif local_str in weak_categories and global_str in weak_categories:
        diagnosis["scenario"] = "weak_evidence"
    else:
        diagnosis["scenario"] = "inconclusive"
    
    return diagnosis

# ============================================================
# Main sample explanation
# ============================================================

def explain_one_sample(model, processor, row, sample_index, args, out_dir):
    sample_dir = os.path.join(out_dir, "explain", f"sample_{sample_index:05d}")
    safe_mkdir(sample_dir)
    original_result = make_original_result_from_prediction(row)

    with tempfile.TemporaryDirectory() as tmp_dir:
        y, sr = load_audio(row["audio_path"], args.sample_rate)
        full_feat = audio_features(y, sr)

        # Optional: rescore original only for log-prob margins, not for evaluation.
        if args.rescore_original:
            rescored = score_audio_labels(model, processor, row["audio_path"])
            original_result["mean_logprobs"] = rescored.get("mean_logprobs")
            original_result["sum_logprobs"] = rescored.get("sum_logprobs")
            if args.use_rescored_original_probs:
                original_result["probs"] = rescored["probs"]
                original_result["confidence"] = float(rescored["probs"].get(original_result["prediction"], row["confidence"]))

        segments = run_temporal_occlusion(
            model, processor, y, sr, original_result, tmp_dir,
            segment_sec=args.segment_sec,
            max_segments=args.max_segments,
            mask_mode=args.mask_mode,
            attenuate_factor=args.attenuate_factor,
            true_label=row.get("ground_truth"),
        )
        def _rank_score(seg):
            for key in ("importance_logprob_margin_drop", "importance_prob_margin_drop", "importance_prob_drop"):
                v = seg.get(key)
                if v is not None:
                    try:
                        v = float(v)
                        if np.isfinite(v):
                            return v
                    except Exception:
                        pass
            return -1e18
        ranked = sorted(segments, key=_rank_score, reverse=True)
        top_segments = ranked[:args.top_k]

        for s in top_segments:
            feat = audio_features(y, sr, s["start"], s["end"])
            s["acoustic_features"] = feat
            s["cue_summary"] = cue_summary(feat, full_feat)

        deletion = run_deletion_curve(model, processor, y, sr, original_result, ranked, tmp_dir, args.deletion_max_k, args.mask_mode, args.attenuate_factor)
        insertion = run_insertion_curve(model, processor, y, sr, original_result, ranked, tmp_dir, args.deletion_max_k)
        faithfulness = compute_comp_suff(original_result, deletion, insertion)

        transforms = [x.strip() for x in args.local_cf_transforms.split(",") if x.strip()]
        local_cf = run_local_paralinguistic_counterfactuals(
            model, processor, y, sr, original_result, top_segments, tmp_dir,
            ground_truth=row.get("ground_truth"),
            transforms=transforms,
            random_n=args.random_n,
            seed=args.seed + sample_index,
        ) if top_segments else {}

    diagnosis = compute_explanation_diagnosis(top_segments, local_cf, deletion, insertion)
    
    evidence = {
        "sample_index": sample_index,
        "source_index": row["source_index"],
        "dataset": row["dataset"],
        "audio_id": row["audio_id"],
        "audio_path": row["audio_path"],
        "ground_truth": row.get("ground_truth"),
        "prediction": row["prediction"],
        "confidence": float(original_result.get("confidence", row["confidence"])),
        "correct": row.get("correct"),
        "probs": original_result["probs"],
        "full_acoustic_features": full_feat,
        "all_segments": segments,
        "top_segments": top_segments,
        "deletion": deletion,
        "insertion": insertion,
        "faithfulness": faithfulness,
        "localized_acoustic_counterfactuals": local_cf,
        "explanation_diagnosis": diagnosis,
    }
    evidence["human_explanation"] = make_human_explanation(evidence)

    with open(os.path.join(sample_dir, "evidence.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(evidence), f, indent=2, ensure_ascii=False)
    with open(os.path.join(sample_dir, "explanation.txt"), "w", encoding="utf-8") as f:
        f.write(evidence["human_explanation"] + "\n")

    plot_waveform_importance(y, sr, top_segments, original_result["prediction"], original_result.get("confidence", row["confidence"]), os.path.join(sample_dir, "waveform_importance.png"))
    plot_segment_importance(ranked, os.path.join(sample_dir, "segment_importance.png"))
    plot_curve(deletion, "prob", f"Deletion curve | AUC={deletion['deletion_auc']:.3f}", f"P({original_result['prediction']})", os.path.join(sample_dir, "deletion_curve.png"))
    plot_curve(insertion, "prob", f"Insertion curve | AUC={insertion['insertion_auc']:.3f}", f"P({original_result['prediction']})", os.path.join(sample_dir, "insertion_curve.png"))
    if top_segments:
        plot_acoustic_cues(full_feat, top_segments[0]["acoustic_features"], os.path.join(sample_dir, "acoustic_cues.png"))
    if local_cf:
        plot_local_counterfactuals(local_cf, os.path.join(sample_dir, "localized_counterfactuals.png"))
    write_sample_html(evidence, sample_dir)

    best_cf = None
    items = local_cf.get("items", []) if local_cf else []
    if items:
        def _cf_score(it):
            le = it.get("local_effect", {})
            for key in ("logprob_margin_drop", "prob_margin_drop", "prob_drop"):
                v = le.get(key)
                if v is not None:
                    try:
                        v = float(v)
                        if np.isfinite(v):
                            return v
                    except Exception:
                        pass
            return -1e18
        best_cf = max(items, key=_cf_score)

    diagnosis = evidence.get("explanation_diagnosis", {})
    return {
        "sample_index": sample_index,
        "source_index": row["source_index"],
        "ground_truth": row.get("ground_truth"),
        "prediction": original_result["prediction"],
        "confidence": float(original_result.get("confidence", row["confidence"])),
        "correct": row.get("correct"),
        "report_path": os.path.join(sample_dir, "report.html"),
        "top1_drop": float(top_segments[0]["importance_prob_drop"]) if top_segments else None,
        "top1_margin_drop": top_segments[0].get("importance_prob_margin_drop") if top_segments else None,
        "top1_logmargin_drop": top_segments[0].get("importance_logprob_margin_drop") if top_segments else None,
        "best_local_transform": best_cf["transform"] if best_cf else None,
        "best_local_prob_drop": best_cf["local_effect"]["prob_drop"] if best_cf else None,
        "best_local_margin_drop": best_cf["local_effect"].get("prob_margin_drop") if best_cf else None,
        "best_local_logmargin_drop": best_cf["local_effect"].get("logprob_margin_drop") if best_cf else None,
        "best_local_vs_random_z": best_cf["random_baseline"].get("local_vs_random_prob_z") if best_cf else None,
        "explanation_type": diagnosis.get("scenario"),
        "local_occlusion_strength": diagnosis.get("local_occlusion_strength"),
        "best_counterfactual_strength": diagnosis.get("best_counterfactual_strength"),
        "global_sensitivity_strength": diagnosis.get("global_sensitivity_strength"),
        "max_global_transform": diagnosis.get("max_global_transform"),
        "max_global_prob_drop": diagnosis.get("max_global_prob_drop"),
    }



def aggregate_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    drops = [x["top1_drop"] for x in items if x.get("top1_drop") is not None]
    margin_drops = [x.get("top1_margin_drop") for x in items if x.get("top1_margin_drop") is not None]
    logmargin_drops = [x.get("top1_logmargin_drop") for x in items if x.get("top1_logmargin_drop") is not None]
    local_drops = [x["best_local_prob_drop"] for x in items if x.get("best_local_prob_drop") is not None]
    local_margins = [x["best_local_margin_drop"] for x in items if x.get("best_local_margin_drop") is not None]
    zs = [x["best_local_vs_random_z"] for x in items if x.get("best_local_vs_random_z") is not None]
    transform_counts = {}
    explanation_type_counts = {}
    for x in items:
        tr = x.get("best_local_transform")
        if tr:
            transform_counts[tr] = transform_counts.get(tr, 0) + 1
        et = x.get("explanation_type")
        if et:
            explanation_type_counts[et] = explanation_type_counts.get(et, 0) + 1

    def mean_or_none(vals):
        vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
        return float(np.mean(vals)) if vals else None

    def median_or_none(vals):
        vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
        return float(np.median(vals)) if vals else None

    correct_items = [x for x in items if x.get("correct") is True]
    wrong_items = [x for x in items if x.get("correct") is False]

    summary = {
        "n_explained": len(items),
        "explained_samples": len(items),
        "n_correct": len(correct_items),
        "n_wrong": len(wrong_items),
        "avg_top1_temporal_drop": mean_or_none(drops),
        "median_top1_temporal_drop": median_or_none(drops),
        "avg_top1_prob_drop": mean_or_none(drops),
        "median_top1_prob_drop": median_or_none(drops),
        "avg_top1_margin_drop": mean_or_none(margin_drops),
        "avg_top1_logmargin_drop": mean_or_none(logmargin_drops),
        "avg_best_local_cf_drop": mean_or_none(local_drops),
        "avg_best_contrastive_margin_drop": mean_or_none(local_margins),
        "avg_best_local_vs_random_z": mean_or_none(zs),
        "best_transform_counts": transform_counts,
        "explanation_type_counts": explanation_type_counts,
        "correct_avg_top1_drop": mean_or_none([x.get("top1_drop") for x in correct_items]),
        "wrong_avg_top1_drop": mean_or_none([x.get("top1_drop") for x in wrong_items]),
    }
    return summary



def validate_evidence_schema(evidence: Dict[str, Any]) -> List[str]:
    """Return schema issues for one sample evidence dictionary."""
    required = [
        "sample_index", "source_index", "dataset", "audio_id", "audio_path",
        "prediction", "confidence", "probs", "all_segments", "top_segments",
        "deletion", "insertion", "faithfulness", "localized_acoustic_counterfactuals",
        "explanation_diagnosis", "human_explanation",
    ]
    issues = ["missing key: " + k for k in required if k not in evidence]
    probs = evidence.get("probs", {})
    if not isinstance(probs, dict):
        issues.append("probs is not a dictionary")
    else:
        missing_labels = [e for e in EMOS if e not in probs]
        if missing_labels:
            issues.append("missing probability labels: " + ",".join(missing_labels))
        total = sum(float(probs.get(e, 0.0) or 0.0) for e in EMOS)
        if not np.isfinite(total) or abs(total - 1.0) > 1e-3:
            issues.append("probabilities do not sum to 1 within tolerance: %.6f" % total)
    if evidence.get("top_segments"):
        top = evidence["top_segments"][0]
        for k in ["start", "end", "importance_prob_drop", "contrast_class"]:
            if k not in top:
                issues.append("top segment missing key: " + k)
    if not isinstance(evidence.get("localized_acoustic_counterfactuals", {}), dict):
        issues.append("localized_acoustic_counterfactuals is not a dictionary")
    return issues


def expected_sample_artifacts() -> List[str]:
    return [
        "evidence.json", "explanation.txt", "report.html",
        "waveform_importance.png", "segment_importance.png", "deletion_curve.png",
        "insertion_curve.png", "removal_keep_summary.png", "acoustic_cues.png",
        "localized_counterfactuals.png", "topk_counterfactual_summary.png",
    ]


def run_quality_checks(out_dir: str, explained_items: List[Dict[str, Any]], aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """Check output consistency after a run and write a compact audit object."""
    checks = []
    def add(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})
    add("has_explained_samples", len(explained_items) > 0, "n=%d" % len(explained_items))
    add("aggregate_count_matches_items", aggregate.get("n_explained") == len(explained_items), "aggregate=%s items=%d" % (aggregate.get("n_explained"), len(explained_items)))
    missing_files = []
    schema_issues = {}
    csv_json_mismatches = []
    for item in explained_items:
        sample_index = int(item["sample_index"])
        sample_dir = os.path.join(out_dir, "explain", "sample_%05d" % sample_index)
        for fn in expected_sample_artifacts():
            if not os.path.isfile(os.path.join(sample_dir, fn)):
                missing_files.append(os.path.join(sample_dir, fn))
        evidence_path = os.path.join(sample_dir, "evidence.json")
        if os.path.isfile(evidence_path):
            with open(evidence_path, "r", encoding="utf-8") as f:
                evidence = json.load(f)
            issues = validate_evidence_schema(evidence)
            if issues:
                schema_issues[str(sample_index)] = issues
            top = evidence.get("top_segments") or []
            if top and item.get("top1_drop") is not None:
                if abs(float(top[0].get("importance_prob_drop", 0.0)) - float(item["top1_drop"])) > 1e-9:
                    csv_json_mismatches.append(sample_index)
    add("all_sample_artifacts_present", not missing_files, "missing=%d" % len(missing_files))
    add("evidence_schema_valid", not schema_issues, "samples_with_issues=%d" % len(schema_issues))
    add("summary_matches_evidence_top1", not csv_json_mismatches, "mismatches=%s" % csv_json_mismatches[:10])
    add("diagnosis_counts_available", bool(aggregate.get("explanation_type_counts")), str(aggregate.get("explanation_type_counts", {})))
    quality = {"overall_passed": all(c["passed"] for c in checks), "checks": checks, "missing_files": missing_files[:50], "schema_issues": schema_issues, "notes": ["Quality checks validate output consistency, not guaranteed explanation faithfulness.", "Very small probability drops should be interpreted as weak or negligible diagnostic evidence."]}
    with open(os.path.join(out_dir, "quality_checks.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(quality), f, indent=2, ensure_ascii=False)
    return quality


def write_paper_summary(out_dir: str, aggregate: Dict[str, Any], quality: Dict[str, Any], args):
    """Write a concise paper/thesis summary for the completed XAI run."""
    lines = [
        "# Local-vs-Global Counterfactual XAI Summary", "",
        "## Method",
        "This analysis uses a post-hoc perturbation-based explanation pipeline for Voxtral-based speech emotion recognition. It reads previous prediction files, treats the previous predicted class as the explanation target, and re-scores only perturbed audio with the Voxtral base model and PEFT adapter.", "",
        "Temporal occlusion identifies short speech regions whose attenuation changes the predicted-class probability or contrastive margin. Localized acoustic counterfactuals are applied only to top-ranked regions and compared against random same-duration regions and whole-utterance global perturbations.", "",
        "## Metrics",
        "Probability drop measures the decrease in the previously predicted class after perturbation. Contrastive margin drop measures how much the target class loses separation from a contrast class. Log-probability margin drop uses mean label-token log-probabilities and should be treated as a diagnostic scoring proxy, not as a calibrated posterior. Local-vs-random z-scores compare the top-region effect with random same-duration controls. Local/global ratios are reported only when the global denominator is non-tiny and has the same sign.", "",
        "## Results",
        "Explained samples: %s" % aggregate.get("n_explained"),
        "Correct predictions: %s; wrong predictions: %s" % (aggregate.get("n_correct"), aggregate.get("n_wrong")),
        "Average top-1 temporal drop: %s" % _fmt(aggregate.get("avg_top1_prob_drop")),
        "Median top-1 temporal drop: %s" % _fmt(aggregate.get("median_top1_prob_drop")),
        "Average best localized counterfactual drop: %s" % _fmt(aggregate.get("avg_best_local_cf_drop")),
        "Average contrastive margin drop: %s" % _fmt(aggregate.get("avg_best_contrastive_margin_drop")),
        "Average local-vs-random z-score: %s" % _fmt(aggregate.get("avg_best_local_vs_random_z")),
        "Best transformation counts: %s" % json.dumps(aggregate.get("best_transform_counts", {}), sort_keys=True),
        "Diagnosis counts: %s" % json.dumps(aggregate.get("explanation_type_counts", {}), sort_keys=True), "",
        "## Interpretation",
        "The outputs should be interpreted as input-output sensitivity evidence. Stronger local drops suggest that the highlighted region is diagnostically important for the model output under the chosen perturbation. Weak or negligible drops indicate that evidence may be distributed, robust to the perturbation, or not captured by the tested transformations.", "",
        "## Limitations",
        "The method does not reveal internal reasoning and does not guarantee faithfulness. Perturbations can create out-of-distribution audio, label-token probabilities may be imperfectly calibrated, and previous predictions are used as targets to avoid full reevaluation. Statistical claims require repeated runs, larger sample sizes, and formal tests.", "",
        "## Quality Checks", "Overall passed: %s" % quality.get("overall_passed"),
    ]
    with open(os.path.join(out_dir, "paper_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_novelty_assessment(out_dir: str):
    """Write a strict novelty assessment for the method."""
    text = """# Novelty Assessment\n\nVerdict: useful applied XAI pipeline, with potential to be framed as a moderate research contribution if supported by stronger experiments and ablations.\n\nThe method is valuable because it adapts perturbation-based XAI to a Voxtral-style audio-language SER pipeline, combines temporal occlusion with localized acoustic counterfactuals, and compares local effects against random same-duration and global whole-utterance controls. The report generation is also unusually complete for research diagnostics.\n\nThe method should not be claimed as a fundamentally new explanation theory. Most components are careful integrations of known ideas: occlusion, counterfactual perturbation, random baselines, global sensitivity checks, deletion/insertion curves, and contrastive margins.\n\nSafe claim: this is a professional post-hoc perturbation-based diagnostic pipeline for measuring localized and global input-output sensitivity in Voxtral SER.\n\nAvoid claiming: guaranteed faithfulness, causal discovery of internal reasoning, or proof that highlighted regions are the only basis for the model decision.\n\nTo strengthen publication-level novelty, add cross-dataset statistical analysis, perturbation ablations, segment-length sensitivity, human/acoustic validation, calibration checks, and comparisons with at least one established audio explanation baseline.\n"""
    with open(os.path.join(out_dir, "novelty_assessment.md"), "w", encoding="utf-8") as f:
        f.write(text)

def main():
    ap = argparse.ArgumentParser()

    # Existing predictions
    ap.add_argument("--pred_jsonl", default=None)
    ap.add_argument("--pred_csv", default=None)
    ap.add_argument("--audio_root", default=None)
    ap.add_argument("--dataset_name", default=None)

    # Model needed only for perturbed re-scoring
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--load_in_4bit", action="store_true")

    # Selection
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--select_mode", choices=["first", "random", "wrong_then_correct"], default="wrong_then_correct")
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--wrong_samples", type=int, default=40)
    ap.add_argument("--only_wrong", action="store_true")
    ap.add_argument("--only_correct", action="store_true")

    # XAI
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--segment_sec", type=float, default=0.2)
    ap.add_argument("--max_segments", type=int, default=36)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--deletion_max_k", type=int, default=5)
    ap.add_argument("--mask_mode", choices=["zero", "attenuate", "mean"], default="attenuate")
    ap.add_argument("--attenuate_factor", type=float, default=0.2)
    ap.add_argument("--random_n", type=int, default=10)
    ap.add_argument("--rescore_original", action="store_true", help="Rescore original selected samples to obtain log-probability margins.")
    ap.add_argument("--use_rescored_original_probs", action="store_true", help="Use rescored original probabilities instead of previous file probabilities.")
    ap.add_argument(
        "--local_cf_transforms",
        default="energy_down,energy_up,pitch_up,pitch_down,time_faster,time_slower,noise_local,spectral_darken,spectral_brighten,smooth_dynamics",
    )

    # Output
    ap.add_argument("--out_dir", required=True)

    args = ap.parse_args()
    if args.pred_jsonl:
        fail_if_missing_file(args.pred_jsonl, "prediction JSONL")
    if args.pred_csv:
        fail_if_missing_file(args.pred_csv, "prediction CSV")
    fail_if_missing_dir(args.base_model, "base model directory")
    fail_if_missing_dir(args.adapter_dir, "PEFT adapter directory")
    if args.audio_root:
        fail_if_missing_dir(args.audio_root, "audio root directory")

    set_seed(args.seed)
    safe_mkdir(args.out_dir)
    safe_mkdir(os.path.join(args.out_dir, "explain"))

    print("Loading previous predictions...")
    rows = load_prediction_rows(args)
    print(f"Selected rows for XAI: {len(rows)}")

    print("Loading Voxtral model for perturbed re-scoring...")
    model, processor = load_model_and_processor(args)
    print("Model loaded.")

    explained_items = []
    summary_csv = os.path.join(args.out_dir, "localized_xai_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=[
            "sample_index", "source_index", "ground_truth", "prediction", "confidence", "correct",
            "top1_drop", "top1_margin_drop", "top1_logmargin_drop",
            "best_local_transform", "best_local_prob_drop", "best_local_margin_drop", "best_local_logmargin_drop",
            "best_local_vs_random_z",
            "explanation_type", "local_occlusion_strength", "best_counterfactual_strength",
            "global_sensitivity_strength", "max_global_transform", "max_global_prob_drop",
            "report_path"
        ], extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows):
            print(f"[{i+1}/{len(rows)}] XAI only: {row['audio_id']}", flush=True)
            try:
                item = explain_one_sample(model, processor, row, i, args, args.out_dir)
                explained_items.append(item)
                writer.writerow(item)
                fcsv.flush()
            except Exception as e:
                print(f"[ERROR] explaining row {i}: {e}", flush=True)

    aggregate = aggregate_summary(explained_items)
    with open(os.path.join(args.out_dir, "localized_xai_aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(aggregate), f, indent=2, ensure_ascii=False)
    write_index_html(args.out_dir, explained_items, aggregate)
    quality = run_quality_checks(args.out_dir, explained_items, aggregate)
    write_paper_summary(args.out_dir, aggregate, quality, args)
    write_novelty_assessment(args.out_dir)

    print("\nDONE.")
    print("Summary CSV:", summary_csv)
    print("Aggregate JSON:", os.path.join(args.out_dir, "localized_xai_aggregate.json"))
    print("Quality checks:", os.path.join(args.out_dir, "quality_checks.json"))
    print("Paper summary:", os.path.join(args.out_dir, "paper_summary.md"))
    print("Novelty assessment:", os.path.join(args.out_dir, "novelty_assessment.md"))
    print("Open report:", os.path.join(args.out_dir, "index.html"))




# ===== FIXED PROFESSIONAL LOCAL-CF REPORT PATCH START =====
# This patch intentionally overrides plotting and reporting functions above.
# It fixes data-schema mismatches, misleading ratios, missing log-margin fields,
# overclaiming on weak evidence, and title/subtitle overlap in figures.

PLOT_DPI = 230
PALETTE = {
    "blue": "#2563EB",
    "blue_dark": "#1E3A8A",
    "orange": "#F97316",
    "green": "#16A34A",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "light_gray": "#E5E7EB",
    "dark": "#111827",
}


def _pretty_transform_name(name):
    mapping = {
        "energy_down": "Energy ↓",
        "energy_up": "Energy ↑",
        "pitch_up": "Pitch ↑",
        "pitch_down": "Pitch ↓",
        "time_faster": "Faster",
        "time_slower": "Slower",
        "noise_local": "Local noise",
        "spectral_darken": "Darker spectrum",
        "spectral_brighten": "Brighter spectrum",
        "smooth_dynamics": "Smooth dynamics",
    }
    return mapping.get(str(name), str(name).replace("_", " ").title())


def _fmt(x, nd=3, empty="—"):
    try:
        if x is None:
            return empty
        x = float(x)
        if not np.isfinite(x):
            return empty
        return f"{x:.{nd}f}"
    except Exception:
        return empty


def _fmt_small(x, empty="—"):
    try:
        if x is None:
            return empty
        x = float(x)
        if not np.isfinite(x):
            return empty
        return f"{x:.4f}" if abs(x) < 0.01 else f"{x:.3f}"
    except Exception:
        return empty


def evidence_strength(prob_drop):
    try:
        d = abs(float(prob_drop))
    except Exception:
        return "unknown"
    if d >= 0.15:
        return "strong"
    if d >= 0.05:
        return "moderate"
    if d >= 0.01:
        return "weak"
    return "negligible"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def _savefig(out_png):
    plt.tight_layout()
    plt.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def _title_with_subtitle(fig, ax, title, subtitle=None):
    fig.suptitle(title, fontsize=13.5, weight="bold", y=0.98)
    if subtitle:
        ax.set_title(subtitle, fontsize=9.5, color=PALETTE["gray"], pad=12)
    fig.subplots_adjust(top=0.82)


def _items_to_professional_schema(local_cf):
    """Compatibility layer for old outputs that only contain local_cf['items']."""
    if local_cf.get("top_segment_results"):
        return local_cf
    items = local_cf.get("items", []) if local_cf else []
    grouped = {}
    for it in items:
        rank = int(it.get("top_region_rank", 0))
        reg = it.get("region", {})
        grouped.setdefault(rank, {
            "rank": rank,
            "start": reg.get("start", 0.0),
            "end": reg.get("end", 0.0),
            "duration": reg.get("duration", 0.0),
            "transform_results": [],
        })
        le = it.get("local_effect", {})
        rb = it.get("random_baseline", {})
        ge = it.get("global_effect", {})
        grouped[rank]["transform_results"].append({
            "transform": it.get("transform"),
            "prob_drop": le.get("prob_drop"),
            "prob_margin_drop": le.get("prob_margin_drop"),
            "logprob_margin_drop": le.get("logprob_margin_drop"),
            "random_prob_drop_mean": rb.get("random_prob_drop_mean"),
            "random_prob_drop_std": rb.get("random_prob_drop_std"),
            "local_vs_random_z_prob": rb.get("local_vs_random_prob_z"),
            "local_vs_random_z_logmargin": rb.get("local_vs_random_z_logmargin"),
            "global_prob_drop": ge.get("prob_drop"),
            "local_global_ratio_prob_drop": it.get("local_global_ratio_prob_drop"),
        })
    local_cf = dict(local_cf or {})
    local_cf["top_segment_results"] = [grouped[k] for k in sorted(grouped)]
    if not local_cf.get("best_local_counterfactual") and items:
        def sc(it):
            le = it.get("local_effect", {})
            for key in ("logprob_margin_drop", "prob_margin_drop", "prob_drop"):
                v = le.get(key)
                if v is not None:
                    return float(v)
            return -1e18
        best = max(items, key=sc)
        le = best.get("local_effect", {})
        reg = best.get("region", {})
        local_cf["best_local_counterfactual"] = {
            "top_region_rank": best.get("top_region_rank"),
            "start": reg.get("start"),
            "end": reg.get("end"),
            "transform": best.get("transform"),
            "prob_drop": le.get("prob_drop"),
            "prob_margin_drop": le.get("prob_margin_drop"),
            "logprob_margin_drop": le.get("logprob_margin_drop"),
            "local_vs_random_z_prob": best.get("random_baseline", {}).get("local_vs_random_prob_z"),
        }
    return local_cf


def plot_waveform_importance(y, sr, segments, pred, conf, out_png):
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.plot(t, y, linewidth=0.65, color=PALETTE["dark"], alpha=0.75)

    if segments:
        max_imp = max(abs(float(s.get("importance_prob_drop", 0.0))) for s in segments) + 1e-9
        for rank, s in enumerate(segments, start=1):
            imp = max(0.0, float(s.get("importance_prob_drop", 0.0)))
            alpha = min(0.50, 0.12 + 0.38 * imp / max_imp)
            color = PALETTE["red"] if rank == 1 else PALETTE["orange"]
            ax.axvspan(float(s["start"]), float(s["end"]), alpha=alpha, color=color)
            ax.text(
                (float(s["start"]) + float(s["end"])) / 2,
                ax.get_ylim()[1] * 0.88,
                f"Top-{rank}\nΔ={_fmt_small(s.get('importance_prob_drop'))}",
                ha="center",
                va="top",
                fontsize=8.5,
                color=PALETTE["dark"],
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.85),
            )

    _title_with_subtitle(fig, ax, f"Waveform evidence map — prediction: {pred} | confidence: {float(conf):.3f}")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    _style_axes(ax)
    _savefig(out_png)


def plot_segment_importance(segments, out_png):
    if not segments:
        return
    shown = segments[:min(12, len(segments))]
    xs = [f"{i+1}\n{s['start']:.2f}-{s['end']:.2f}s" for i, s in enumerate(shown)]
    ys = [float(s.get("importance_prob_drop", 0.0)) for s in shown]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    bars = ax.bar(xs, ys, color=PALETTE["blue"], alpha=0.88)
    if bars:
        bars[0].set_color(PALETTE["red"])
    y_span = max(abs(min(ys)), abs(max(ys)), 0.01)
    offset = 0.04 * y_span
    for b, yv in zip(bars, ys):
        va = "bottom" if yv >= 0 else "top"
        ytxt = yv + offset if yv >= 0 else yv - offset
        ax.text(b.get_x() + b.get_width() / 2, ytxt, _fmt_small(yv), ha="center", va=va, fontsize=8)

    # Evidence thresholds make weak/negligible cases visually clear.
    ax.axhline(0.01, linestyle="--", linewidth=1.0, color=PALETTE["gray"], alpha=0.65)
    ax.text(0.01, 0.01, "weak threshold Δ=0.01", transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=8, color=PALETTE["gray"])
    _title_with_subtitle(
        fig, ax,
        "Temporal occlusion — effect of removing each region independently",
        "Bars show probability drop for the predicted class; tiny values mean weak local evidence."
    )
    ax.set_ylabel("Drop in predicted-class probability")
    ax.set_xlabel("Ranked temporal region")
    low = min(0, min(ys) - 2 * offset)
    high = max(0.03, max(ys) + 3 * offset)
    ax.set_ylim(low, high)
    _style_axes(ax)
    _savefig(out_png)


def plot_curve(curve_obj, *args):
    if len(args) == 4:
        _key, title, ylabel, out_png = args
    elif len(args) == 3:
        out_png, title, ylabel = args
    else:
        raise TypeError(f"plot_curve expected either 3 or 4 arguments after curve_obj, got {len(args)}")

    curve = curve_obj.get("curve", [])
    if not curve:
        return

    xs = [int(c["k"]) for c in curve]
    ys = [float(c["prob"]) for c in curve]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(xs, ys, marker="o", linewidth=2.4, color=PALETTE["blue"])
    ax.scatter(xs, ys, s=58, color=PALETTE["blue_dark"], zorder=3)

    for x, yv in zip(xs, ys):
        ax.text(x, min(1.03, yv + 0.035), _fmt(yv, 2), ha="center", va="bottom", fontsize=9)

    if "deletion" in str(title).lower():
        xlabel = "Top regions removed cumulatively"
        plot_title = "Comprehensiveness test — removing top regions"
        subtitle = "A stronger explanation should reduce the predicted-class probability when top regions are removed."
    elif "insertion" in str(title).lower():
        xlabel = "Top regions kept only"
        plot_title = "Sufficiency test — keeping only top regions"
        subtitle = "A sufficient explanation should preserve the prediction when only top regions are kept."
    else:
        xlabel = "Number of top regions"
        plot_title = title
        subtitle = None

    _title_with_subtitle(fig, ax, plot_title, subtitle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xs)
    _style_axes(ax)
    _savefig(out_png)


def plot_effect_summary(evidence, out_png):
    pred = evidence["prediction"]
    original_prob = float(evidence["probs"].get(pred, 0.0))

    labels = ["Original"]
    vals = [original_prob]
    colors = [PALETTE["gray"]]

    for c in evidence.get("deletion", {}).get("curve", []):
        k = int(c.get("k", 0))
        if k > 0:
            labels.append(f"Remove\nTop-{k}")
            vals.append(float(c.get("prob", 0.0)))
            colors.append(PALETTE["red"])

    for c in evidence.get("insertion", {}).get("curve", []):
        k = int(c.get("k", 0))
        if k > 0:
            labels.append(f"Keep only\nTop-{k}")
            vals.append(float(c.get("prob", 0.0)))
            colors.append(PALETTE["green"])

    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    bars = ax.bar(labels, vals, color=colors, alpha=0.88)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, min(1.03, v + 0.025), _fmt(v, 2), ha="center", va="bottom", fontsize=9)

    _title_with_subtitle(
        fig, ax,
        "Decision evidence test — removing vs keeping top regions",
        "Removal tests necessity/comprehensiveness; keep-only tests sufficiency of highlighted regions."
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"P({pred})")
    _style_axes(ax)
    _savefig(out_png)


def plot_acoustic_cues(full_feat, top_feat, out_png):
    labels, values = [], []

    if full_feat.get("rms_db") is not None and top_feat.get("rms_db") is not None:
        labels.append("Energy\n(dB diff)")
        values.append(float(top_feat["rms_db"] - full_feat["rms_db"]))

    feature_map = [
        ("f0_mean_hz", "Pitch mean\n(relative)"),
        ("f0_std_hz", "Pitch variation\n(relative)"),
        ("spectral_centroid_hz", "Spectral centroid\n(relative)"),
        ("onset_rate_per_sec", "Local changes\n(relative)"),
    ]
    for k, label in feature_map:
        f, t = full_feat.get(k), top_feat.get(k)
        if f is None or t is None or abs(float(f)) < 1e-9:
            continue
        values.append(float((t - f) / abs(f)))
        labels.append(label)

    if not labels:
        labels, values = ["No valid\nnumeric cues"], [0.0]

    colors = [PALETTE["green"] if v >= 0 else PALETTE["red"] for v in values]
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.axhline(0.0, linestyle="--", linewidth=1.2, color=PALETTE["dark"], alpha=0.65)

    y_span = max(abs(min(values)), abs(max(values)), 0.1)
    for b, v in zip(bars, values):
        y = v + 0.03 * y_span if v >= 0 else v - 0.04 * y_span
        va = "bottom" if v >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, y, _fmt(v, 2), ha="center", va=va, fontsize=9)

    _title_with_subtitle(
        fig, ax,
        "Acoustic profile of the top influential region",
        "Energy is absolute dB difference; the other features are relative changes from the full utterance."
    )
    ax.set_ylabel("Feature difference (mixed scale)")
    _style_axes(ax)
    _savefig(out_png)


def plot_local_counterfactuals(local_cf, out_png):
    local_cf = _items_to_professional_schema(local_cf)
    if not local_cf.get("top_segment_results"):
        return
    top_seg = local_cf["top_segment_results"][0]
    trs = top_seg.get("transform_results", [])
    if not trs:
        return

    def sc(x):
        for k in ("logprob_margin_drop", "prob_margin_drop", "prob_drop"):
            v = x.get(k)
            if v is not None:
                return float(v)
        return -1e18

    trs = sorted(trs, key=sc, reverse=True)
    names = [_pretty_transform_name(x.get("transform")) for x in trs]
    prob_drops = [float(x.get("prob_drop") or 0.0) for x in trs]
    random_means = [float(x.get("random_prob_drop_mean") or 0.0) for x in trs]
    global_drops = [float(x.get("global_prob_drop") or 0.0) for x in trs]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 5.0))
    b1 = ax.bar(x - width, prob_drops, width, label="Top region", color=PALETTE["blue"], alpha=0.9)
    ax.bar(x, random_means, width, label="Random regions", color=PALETTE["orange"], alpha=0.82)
    ax.bar(x + width, global_drops, width, label="Global utterance", color=PALETTE["purple"], alpha=0.75)

    y_all = prob_drops + random_means + global_drops
    y_span = max(abs(min(y_all)), abs(max(y_all)), 0.03)
    for b, v in zip(b1, prob_drops):
        va = "bottom" if v >= 0 else "top"
        ytxt = v + 0.03 * y_span if v >= 0 else v - 0.04 * y_span
        ax.text(b.get_x() + b.get_width() / 2, ytxt, _fmt_small(v), ha="center", va=va, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Drop in predicted-class probability")
    _title_with_subtitle(
        fig, ax,
        "Localized acoustic counterfactuals — top region vs controls",
        "Top-region effects should be compared with random and global perturbation controls."
    )
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(min(0, min(y_all) - 0.12 * y_span), max(0.05, max(y_all) + 0.18 * y_span))
    _style_axes(ax)
    _savefig(out_png)


def plot_topk_counterfactual_summary(local_cf, out_png):
    local_cf = _items_to_professional_schema(local_cf)
    rows = []
    for seg in local_cf.get("top_segment_results", []):
        trs = seg.get("transform_results", [])
        if not trs:
            continue
        def sc(x):
            for k in ("logprob_margin_drop", "prob_margin_drop", "prob_drop"):
                v = x.get(k)
                if v is not None:
                    return float(v)
            return -1e18
        best = max(trs, key=sc)
        rows.append({
            "segment": f"Top-{seg['rank']}\n{float(seg['start']):.2f}-{float(seg['end']):.2f}s",
            "transform": _pretty_transform_name(best.get("transform")),
            "prob_drop": float(best.get("prob_drop") or 0.0),
            "z": best.get("local_vs_random_z_logmargin") or best.get("local_vs_random_z_prob"),
        })

    if not rows:
        return

    labels = [r["segment"] + "\n" + r["transform"] for r in rows]
    vals = [r["prob_drop"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    bars = ax.bar(labels, vals, color=PALETTE["purple"], alpha=0.86)
    y_span = max(abs(min(vals)), abs(max(vals)), 0.03)
    for b, r in zip(bars, rows):
        z_text = "" if r["z"] is None else f"\nz={float(r['z']):.2f}"
        va = "bottom" if r["prob_drop"] >= 0 else "top"
        ytxt = r["prob_drop"] + 0.04 * y_span if r["prob_drop"] >= 0 else r["prob_drop"] - 0.04 * y_span
        ax.text(b.get_x() + b.get_width() / 2, ytxt, f"{_fmt_small(r['prob_drop'])}{z_text}", ha="center", va=va, fontsize=9)
    _title_with_subtitle(fig, ax, "Best localized counterfactual per top region")
    ax.set_ylabel("Drop in predicted-class probability")
    ax.set_ylim(min(0, min(vals) - 0.15 * y_span), max(0.05, max(vals) + 0.25 * y_span))
    _style_axes(ax)
    _savefig(out_png)


def _interpret_ratio(ratio):
    if ratio is None:
        return " The local/global ratio is not reported because the global effect is tiny or in the opposite direction."
    try:
        r = float(ratio)
    except Exception:
        return ""
    if abs(r) >= 1.0:
        return f" The local/global sensitivity ratio is {r:.2f}, meaning the local intervention is at least as influential as applying the same transformation globally."
    return f" The local/global sensitivity ratio is {r:.2f}, suggesting that the transformation also has distributed/global effects."


def local_cf_sentence(local_cf):
    local_cf = _items_to_professional_schema(local_cf)
    best = local_cf.get("best_local_counterfactual")
    if not best:
        return ""

    transform_pretty = _pretty_transform_name(best.get("transform"))
    pdrop = best.get("prob_drop")
    ldrop = best.get("logprob_margin_drop")
    z = best.get("local_vs_random_z_logmargin") or best.get("local_vs_random_z_prob")
    ratio = best.get("local_global_ratio_logmargin") or best.get("local_global_ratio_prob_drop")
    strength = evidence_strength(pdrop)

    parts = [
        f" The strongest localized counterfactual among the tested interventions is '{transform_pretty}'."
    ]
    if pdrop is not None:
        parts.append(f" It changes the predicted-class probability by {_fmt_small(pdrop)} ({strength} effect).")
    if ldrop is not None:
        parts.append(f" The contrastive log-probability margin changes by {_fmt_small(ldrop)}.")
    if z is not None:
        parts.append(f" Compared with random regions of the same duration, its z-score is {float(z):.2f}.")
    parts.append(_interpret_ratio(ratio))
    parts.append(" This should be read as localized input-output sensitivity evidence, not proof that this cue alone causally determines the emotion.")
    return "".join(parts)


def make_human_explanation(evidence):
    pred = evidence["prediction"]
    conf = float(evidence["confidence"])
    gt = evidence.get("ground_truth")
    correct = evidence.get("correct")
    top = evidence.get("top_segments", [])

    gt_text = ""
    if gt:
        gt_text = f" The ground-truth label is {gt}, so this prediction is {'correct' if correct else 'wrong'}."

    if not top:
        return f"The model predicts {pred} with confidence {conf:.3f}.{gt_text} No temporal explanation was produced."

    s0 = top[0]
    start, end = float(s0["start"]), float(s0["end"])
    drop = float(s0.get("importance_prob_drop", 0.0))
    pm_drop = s0.get("importance_prob_margin_drop")
    lm_drop = s0.get("importance_logprob_margin_drop")
    strength = evidence_strength(drop)
    cues = "; ".join(s0.get("cue_summary", []))
    contrast = s0.get("contrast_class")

    deletion_curve = evidence.get("deletion", {}).get("curve", [])
    insertion_curve = evidence.get("insertion", {}).get("curve", [])

    deletion_text = ""
    if len(deletion_curve) > 1:
        p0 = float(deletion_curve[0].get("prob", conf))
        p1 = float(deletion_curve[1].get("prob", 0.0))
        deletion_text = f" In the cumulative deletion test, removing the top region changes P({pred}) from {p0:.3f} to {p1:.3f}."

    insertion_text = ""
    if len(insertion_curve) > 0:
        p_keep1 = float(insertion_curve[0].get("prob", 0.0))
        insertion_text = f" In the sufficiency test, keeping only the top region gives P({pred})={p_keep1:.3f}."

    if strength in ["negligible", "weak"]:
        faith_sentence = (
            f"The largest local occlusion effect is {strength}: masking this region changes P({pred}) by only {_fmt_small(drop)}. "
            f"Therefore, this sample provides {strength} localized evidence; the decision may be robust, distributed across the utterance, or driven by broader context. "
        )
    else:
        faith_sentence = (
            f"Masking this region changes P({pred}) by {_fmt_small(drop)}, which is a {strength} local sensitivity effect. "
        )

    margin_text = ""
    if contrast:
        if lm_drop is not None:
            margin_text = f" The contrast class is {contrast}; the contrastive log-probability margin changes by {_fmt_small(lm_drop)}."
        elif pm_drop is not None:
            margin_text = f" The contrast class is {contrast}; the contrastive probability margin changes by {_fmt_small(pm_drop)}."

    cf_text = local_cf_sentence(evidence.get("localized_acoustic_counterfactuals", {}))

    return (
        f"The model predicts {pred} with confidence {conf:.3f}.{gt_text} "
        f"The highest-ranked audio region is from {start:.2f}s to {end:.2f}s. "
        f"{faith_sentence}"
        f"Acoustically, this region is characterized by: {cues}."
        f"{margin_text}"
        f"{deletion_text}"
        f"{insertion_text}"
        f"{cf_text} "
        f"This explanation is post-hoc and perturbation-based; it describes input-output sensitivity, not direct access to the model's hidden reasoning."
    )


def _html_metric_card(title, value, subtitle=""):
    return f"""
    <div class="metric">
      <div class="metric-title">{html.escape(str(title))}</div>
      <div class="metric-value">{html.escape(str(value))}</div>
      <div class="metric-subtitle">{html.escape(str(subtitle))}</div>
    </div>
    """


def _html_img(src, caption):
    return f"""
    <figure>
      <img src="{html.escape(src)}" alt="{html.escape(caption)}">
      <figcaption>{html.escape(caption)}</figcaption>
    </figure>
    """


def write_sample_html(evidence, sample_dir):
    try:
        plot_effect_summary(evidence, os.path.join(sample_dir, "removal_keep_summary.png"))
    except Exception as e:
        print(f"[WARN] could not create removal_keep_summary.png: {e}")

    local_cf = _items_to_professional_schema(evidence.get("localized_acoustic_counterfactuals", {}))
    evidence["localized_acoustic_counterfactuals"] = local_cf

    try:
        plot_topk_counterfactual_summary(local_cf, os.path.join(sample_dir, "topk_counterfactual_summary.png"))
    except Exception as e:
        print(f"[WARN] could not create topk_counterfactual_summary.png: {e}")

    explanation = html.escape(evidence["human_explanation"])
    pred, gt = evidence["prediction"], evidence.get("ground_truth")
    title = f"Sample {evidence['sample_index']} | pred={pred} | gt={gt}"

    prob_rows = ""
    for e in EMOS:
        p = float(evidence["probs"].get(e, 0.0))
        cls = "pred-row" if e == pred else ""
        prob_rows += f"<tr class='{cls}'><td>{html.escape(e)}</td><td>{p:.4f}</td></tr>\n"

    top_rows = ""
    for rank, s in enumerate(evidence.get("top_segments", []), start=1):
        top_rows += (
            "<tr>"
            f"<td>Top-{rank}</td>"
            f"<td>{s['start']:.2f}–{s['end']:.2f}s</td>"
            f"<td>{_fmt_small(s.get('importance_prob_drop'))}</td>"
            f"<td>{_fmt_small(s.get('importance_prob_margin_drop'))}</td>"
            f"<td>{_fmt_small(s.get('importance_logprob_margin_drop'))}</td>"
            f"<td>{html.escape(str(s.get('contrast_class', '—')))}</td>"
            f"<td>{html.escape('; '.join(s.get('cue_summary', [])))}</td>"
            "</tr>\n"
        )

    cf_rows = ""
    for seg in local_cf.get("top_segment_results", []):
        for tr in seg.get("transform_results", []):
            z_val = tr.get("local_vs_random_z_logmargin") or tr.get("local_vs_random_z_prob")
            ratio_val = tr.get("local_global_ratio_logmargin") or tr.get("local_global_ratio_prob_drop")
            z_txt = "—" if z_val is None else f"{float(z_val):.3f}"
            ratio_txt = "—" if ratio_val is None else f"{float(ratio_val):.3f}"
            cf_rows += (
                "<tr>"
                f"<td>Top-{seg['rank']}</td>"
                f"<td>{html.escape(_pretty_transform_name(tr['transform']))}</td>"
                f"<td>{_fmt_small(tr.get('prob_drop'))}</td>"
                f"<td>{_fmt_small(tr.get('prob_margin_drop'))}</td>"
                f"<td>{_fmt_small(tr.get('logprob_margin_drop'))}</td>"
                f"<td>{_fmt_small(tr.get('random_prob_drop_mean'))}</td>"
                f"<td>{z_txt}</td>"
                f"<td>{ratio_txt}</td>"
                "</tr>\n"
            )

    top1_drop = "—"
    top1_lm = "—"
    strength = "—"
    if evidence.get("top_segments"):
        d = evidence["top_segments"][0].get("importance_prob_drop")
        top1_drop = _fmt_small(d)
        top1_lm = _fmt_small(evidence["top_segments"][0].get("importance_logprob_margin_drop"))
        strength = evidence_strength(d)

    best = local_cf.get("best_local_counterfactual") or {}
    best_transform = _pretty_transform_name(best.get("transform", "—")) if best else "—"

    body = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#f8fafc; --card:#fff; --text:#111827; --muted:#64748b; --line:#e5e7eb; --blue:#2563eb; }}
body {{ font-family: Inter, Arial, sans-serif; margin:0; background:var(--bg); color:var(--text); line-height:1.55; }}
.header {{ background:linear-gradient(135deg,#111827,#1e3a8a); color:white; padding:28px 42px; }}
.header h1 {{ margin:0 0 8px 0; font-size:28px; }}
.header p {{ margin:0; color:#dbeafe; }}
.container {{ max-width:1180px; margin:26px auto; padding:0 22px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:20px 22px; margin-bottom:20px; box-shadow:0 8px 24px rgba(15,23,42,.06); }}
.metrics {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:14px; margin-bottom:20px; }}
.metric {{ background:white; border:1px solid var(--line); border-radius:14px; padding:16px; box-shadow:0 8px 20px rgba(15,23,42,.05); }}
.metric-title {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
.metric-value {{ font-size:22px; font-weight:750; margin-top:4px; }}
.metric-subtitle {{ color:var(--muted); font-size:12px; margin-top:2px; }}
.explanation {{ font-size:15px; background:#f8fbff; border-left:5px solid var(--blue); }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
td,th {{ border-bottom:1px solid var(--line); padding:10px; text-align:left; }}
th {{ color:var(--muted); font-weight:700; background:#f9fafb; }}
.pred-row {{ background:#eff6ff; font-weight:700; }}
.grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
figure {{ margin:0; }}
img {{ width:100%; border:1px solid var(--line); border-radius:12px; background:white; }}
figcaption {{ color:var(--muted); font-size:12px; margin-top:8px; }}
.note {{ color:var(--muted); font-size:13px; }}
@media(max-width:1000px) {{ .metrics,.grid-2 {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="header"><h1>Localized Counterfactual XAI Report</h1><p>{html.escape(title)}</p></div>
<div class="container">

<div class="metrics">
  {_html_metric_card("Prediction", pred, f"ground truth: {gt}")}
  {_html_metric_card("Confidence", f"{float(evidence['confidence']):.3f}", "target prediction")}
  {_html_metric_card("Top-1 prob. drop", top1_drop, "occlusion effect")}
  {_html_metric_card("Evidence strength", strength, "based on top-1 drop")}
  {_html_metric_card("Best local intervention", best_transform, "counterfactual")}
</div>

<div class="card explanation">
  <h2>Human-readable interpretation</h2>
  <p>{explanation}</p>
  <p class="note">Deletion/removal tests necessity; keep-only tests sufficiency; localized counterfactuals test which acoustic intervention inside the region changes the output. Very small drops should be interpreted as weak or negligible local evidence.</p>
</div>

<div class="card"><h2>Class probabilities</h2><table><tr><th>Class</th><th>Probability</th></tr>{prob_rows}</table></div>

<div class="card"><h2>Top temporal regions</h2>
<table><tr><th>Rank</th><th>Time</th><th>Prob drop</th><th>Prob-margin drop</th><th>Log-margin drop</th><th>Contrast</th><th>Acoustic profile</th></tr>{top_rows}</table>
</div>

<div class="card">
  <h2>Decision evidence plots</h2>
  {_html_img("waveform_importance.png", "Waveform with highlighted top evidence regions.")}
  <div class="grid-2" style="margin-top:18px;">
    {_html_img("removal_keep_summary.png", "Original vs removing top regions vs keeping only top regions.")}
    {_html_img("segment_importance.png", "Independent temporal occlusion importance for ranked regions.")}
  </div>
</div>

<div class="card">
  <h2>Deletion and insertion faithfulness tests</h2>
  <div class="grid-2">
    {_html_img("deletion_curve.png", "Comprehensiveness: cumulative effect of removing top regions.")}
    {_html_img("insertion_curve.png", "Sufficiency: prediction when only top regions are kept.")}
  </div>
</div>

<div class="card">
  <h2>Localized acoustic counterfactuals</h2>
  <p class="note">Ratios are hidden when the global effect is tiny or has the opposite sign, because such ratios are not scientifically meaningful.</p>
  <div class="grid-2">
    {_html_img("localized_counterfactuals.png", "Top-1 localized acoustic interventions vs random/global controls.")}
    {_html_img("topk_counterfactual_summary.png", "Best localized acoustic intervention for each top region.")}
  </div>
  <table style="margin-top:18px;">
    <tr><th>Segment</th><th>Intervention</th><th>Prob drop</th><th>Prob-margin drop</th><th>Log-margin drop</th><th>Random mean prob drop</th><th>Z vs random</th><th>Local/global ratio</th></tr>
    {cf_rows}
  </table>
</div>

<div class="card"><h2>Acoustic descriptor profile</h2>{_html_img("acoustic_cues.png", "Acoustic descriptors of the top region relative to the full utterance.")}</div>

</div>
</body>
</html>
"""
    with open(os.path.join(sample_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(body)


def write_index_html(out_dir, explained_items, aggregate):
    rows = ""
    for item in explained_items:
        rel_path = os.path.relpath(item["report_path"], out_dir)
        top1 = "—" if item.get("top1_drop") is None else _fmt_small(item.get("top1_drop"))
        strength = evidence_strength(item.get("top1_drop")) if item.get("top1_drop") is not None else "—"
        best = html.escape(str(item.get("best_local_transform") or "—"))
        rows += (
            "<tr>"
            f"<td>{item['sample_index']}</td>"
            f"<td>{html.escape(str(item['ground_truth']))}</td>"
            f"<td>{html.escape(str(item['prediction']))}</td>"
            f"<td>{item['confidence']:.4f}</td>"
            f"<td>{item['correct']}</td>"
            f"<td>{top1}</td>"
            f"<td>{strength}</td>"
            f"<td>{best}</td>"
            f"<td><a href='{html.escape(rel_path)}'>Open report</a></td>"
            "</tr>\n"
        )

    body = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Localized Counterfactual XAI Report</title>
<style>
body {{ font-family: Inter, Arial, sans-serif; margin:0; background:#f8fafc; color:#111827; }}
.header {{ background:linear-gradient(135deg,#111827,#1e3a8a); color:white; padding:30px 44px; }}
.container {{ max-width:1180px; margin:26px auto; padding:0 22px; }}
.card {{ background:white; border:1px solid #e5e7eb; border-radius:16px; padding:20px; box-shadow:0 8px 24px rgba(15,23,42,.06); }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
td,th {{ border-bottom:1px solid #e5e7eb; padding:10px; text-align:left; }}
th {{ background:#f9fafb; color:#64748b; }}
a {{ color:#2563eb; font-weight:700; text-decoration:none; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:18px; }}
.metric {{ background:white; border:1px solid #e5e7eb; border-radius:14px; padding:16px; }}
.metric-title {{ color:#64748b; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
.metric-value {{ font-size:24px; font-weight:750; margin-top:4px; }}
.metric-subtitle {{ color:#64748b; font-size:12px; margin-top:2px; }}
</style>
</head>
<body>
<div class="header"><h1>Localized Counterfactual XAI Report</h1><p>Dataset-level overview of selected explanations</p></div>
<div class="container">
  <div class="metrics">
    {_html_metric_card("Explained samples", aggregate.get('explained_samples', aggregate.get('n_explained')), "")}
    {_html_metric_card("Average top-1 prob drop", _fmt(aggregate.get('avg_top1_prob_drop')), "occlusion")}
    {_html_metric_card("Average top-1 log-margin drop", _fmt(aggregate.get('avg_top1_logmargin_drop')), "contrastive")}
    {_html_metric_card("Average best local CF drop", _fmt(aggregate.get('avg_best_local_cf_drop')), "counterfactual")}
  </div>
  <div class="card">
    <h2>Samples</h2>
    <table>
      <tr><th>Sample</th><th>Ground truth</th><th>Prediction</th><th>Confidence</th><th>Correct</th><th>Top-1 drop</th><th>Evidence strength</th><th>Best local intervention</th><th>Report</th></tr>
      {rows}
    </table>
  </div>
</div>
</body>
</html>
"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(body)

# ===== FIXED PROFESSIONAL LOCAL-CF REPORT PATCH END =====


if __name__ == "__main__":
    main()
