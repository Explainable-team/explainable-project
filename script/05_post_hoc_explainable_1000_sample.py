#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LG-ProXAI-Stats: Local/Random/Global Statistical Diagnosis for Voxtral SER.

This is a compact class-level statistical XAI experiment. It reads previous
prediction files, selects up to 200 class-aware samples, scores temporal
attenuation, Top-2 local evidence, random same-duration controls, and global
prosody perturbations, then writes aggregate statistics and paper-ready reports.
It intentionally does not create per-sample HTML reports or per-sample plots.
"""

import argparse
import csv
import json
import math
import os
import platform
import random
import socket
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    import transformers
except Exception:
    transformers = None

try:
    import librosa
except Exception:
    librosa = None

EMOS = ["Angry", "Happy", "Sad", "Neutral"]

# Heavy Voxtral/audio helpers are loaded after argparse has handled --help.
fail_if_missing_dir = None
fail_if_missing_file = None
fit_length = None
get_audio_path = None
jsonable = None
load_audio = None
load_model_and_processor = None
make_original_result_from_prediction = None
make_speech_aware_segments = None
mask_many_regions = None
mask_region = None
normalize_label = None
parse_bool = None
parse_probs = None
safe_z = None
score_audio_labels = None
score_target_effect = None
set_seed = None
spectral_tilt = None
to_abs = None
write_wav = None


def load_professional_primitives():
    global fail_if_missing_dir, fail_if_missing_file, fit_length, get_audio_path, jsonable
    global load_audio, load_model_and_processor, make_original_result_from_prediction
    global make_speech_aware_segments, mask_many_regions, mask_region, normalize_label
    global parse_bool, parse_probs, safe_z, score_audio_labels, score_target_effect
    global set_seed, spectral_tilt, to_abs, write_wav

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from explain_from_predictions_local_global_diagnosis_05s_PROFESSIONAL import (
        fail_if_missing_dir as _fail_if_missing_dir,
        fail_if_missing_file as _fail_if_missing_file,
        fit_length as _fit_length,
        get_audio_path as _get_audio_path,
        jsonable as _jsonable,
        load_audio as _load_audio,
        load_model_and_processor as _load_model_and_processor,
        make_original_result_from_prediction as _make_original_result_from_prediction,
        make_speech_aware_segments as _make_speech_aware_segments,
        mask_many_regions as _mask_many_regions,
        mask_region as _mask_region,
        normalize_label as _normalize_label,
        parse_bool as _parse_bool,
        parse_probs as _parse_probs,
        safe_z as _safe_z,
        score_audio_labels as _score_audio_labels,
        score_target_effect as _score_target_effect,
        set_seed as _set_seed,
        spectral_tilt as _spectral_tilt,
        to_abs as _to_abs,
        write_wav as _write_wav,
    )
    fail_if_missing_dir = _fail_if_missing_dir
    fail_if_missing_file = _fail_if_missing_file
    fit_length = _fit_length
    get_audio_path = _get_audio_path
    jsonable = _jsonable
    load_audio = _load_audio
    load_model_and_processor = _load_model_and_processor
    make_original_result_from_prediction = _make_original_result_from_prediction
    make_speech_aware_segments = _make_speech_aware_segments
    mask_many_regions = _mask_many_regions
    mask_region = _mask_region
    normalize_label = _normalize_label
    parse_bool = _parse_bool
    parse_probs = _parse_probs
    safe_z = _safe_z
    score_audio_labels = _score_audio_labels
    score_target_effect = _score_target_effect
    set_seed = _set_seed
    spectral_tilt = _spectral_tilt
    to_abs = _to_abs
    write_wav = _write_wav


BEHAVIORS = [
    "strong_localized_evidence",
    "global_prosody_evidence",
    "mixed_local_global_evidence",
    "random_like_evidence",
    "contrastive_shift_evidence",
    "conflicting_evidence",
    "robust_evidence",
    "weak_inconclusive_evidence",
]

DEFAULT_GLOBAL_TRANSFORMS = [
    "global_energy_down",
    "global_energy_up",
    "global_pitch_down",
    "global_pitch_up",
    "global_time_faster",
    "global_time_slower",
    "global_spectral_darken",
    "global_spectral_brighten",
]

PROSODY_TYPE = {
    "global_energy_down": "energy_sensitive",
    "global_energy_up": "energy_sensitive",
    "global_pitch_down": "pitch_sensitive",
    "global_pitch_up": "pitch_sensitive",
    "global_time_faster": "speaking_rate_sensitive",
    "global_time_slower": "speaking_rate_sensitive",
    "global_spectral_darken": "spectral_sensitive",
    "global_spectral_brighten": "spectral_sensitive",
}

SAMPLE_FIELDS = [
    "sample_index", "source_index", "dataset", "audio_id", "ground_truth",
    "prediction", "correct", "confidence", "contrast_class",
    "top1_start", "top1_end", "top1_duration", "top1_local_prob_drop",
    "top1_local_margin_drop", "top1_local_logmargin_drop",
    "top2_start", "top2_end", "top2_duration", "top2_local_prob_drop",
    "top2_local_margin_drop", "top2_local_logmargin_drop",
    "top2_combined_prob_drop", "top2_combined_margin_drop",
    "top2_combined_logmargin_drop", "best_local_source",
    "best_local_prob_drop", "best_local_margin_drop",
    "best_local_logmargin_drop", "local_evidence_strength", "random_n",
    "top1_random_prob_drop_mean", "top1_random_prob_drop_std",
    "top2_random_prob_drop_mean", "top2_random_prob_drop_std",
    "random_margin_drop_mean", "random_margin_drop_std",
    "top1_local_minus_random", "top2_local_minus_random",
    "top1_local_vs_random_z", "top2_local_vs_random_z",
    "top1_greater_than_random", "top2_greater_than_random",
    "best_global_transform", "best_global_prob_drop",
    "best_global_margin_drop", "best_global_logmargin_drop",
    "best_global_new_prediction", "best_global_flip",
    "global_evidence_strength", "global_prosody_type",
    "local_global_ratio", "local_vs_global_label",
    "dominant_evidence_source", "behavior_type", "behavior_confidence",
    "dominant_pattern_short_text", "n_variants_scored", "elapsed_sec",
    "sec_per_variant", "batch_size_used", "batching_fallback_used",
]

VARIANT_FIELDS = [
    "sample_index", "source_index", "dataset", "prediction", "ground_truth",
    "correct", "variant_type", "variant_name", "transform", "segment_rank",
    "start", "end", "prob_before", "prob_after", "prob_drop",
    "margin_before", "margin_after", "margin_drop", "logmargin_before",
    "logmargin_after", "logmargin_drop", "new_prediction", "flip",
]


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def finite_float(x, default=None):
    try:
        if x is None:
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def mean_or_none(values):
    vals = [finite_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def std_or_none(values):
    vals = [finite_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else None)


def median_or_none(values):
    vals = [finite_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def mean_std_median_ci95(values, seed: int = 42) -> Dict[str, Any]:
    vals = np.array([finite_float(v) for v in values if finite_float(v) is not None], dtype=float)
    if vals.size == 0:
        return {"mean": None, "std": None, "median": None, "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    med = float(np.median(vals))
    if vals.size >= 5:
        rng = np.random.default_rng(seed)
        boots = [float(np.mean(rng.choice(vals, size=vals.size, replace=True))) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
    elif vals.size > 1:
        half = 1.96 * std / math.sqrt(vals.size)
        lo, hi = mean - half, mean + half
    else:
        lo = hi = None
    return {"mean": mean, "std": std, "median": med, "ci95_low": None if lo is None else float(lo), "ci95_high": None if hi is None else float(hi)}


def strength_label(prob_drop=None, margin_drop=None) -> str:
    p = abs(finite_float(prob_drop, 0.0))
    m = abs(finite_float(margin_drop, 0.0))
    if p >= 0.15 or m >= 0.25:
        return "strong"
    if p >= 0.05 or m >= 0.10:
        return "moderate"
    if p >= 0.01 or m >= 0.03:
        return "weak"
    return "negligible"


def effect_strength_counts(values, metric_type="prob") -> Dict[str, float]:
    vals = [finite_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    n = len(vals)
    counts = {"strong": 0, "moderate": 0, "weak": 0, "negligible": 0}
    for v in vals:
        if metric_type == "margin":
            lab = strength_label(None, v)
        else:
            lab = strength_label(v, None)
        counts[lab] += 1
    return {k + "_percent": float(100.0 * c / n) if n else None for k, c in counts.items()}


def cliff_delta(x, y) -> Optional[float]:
    x = [finite_float(v) for v in x]
    y = [finite_float(v) for v in y]
    x = [v for v in x if v is not None]
    y = [v for v in y if v is not None]
    if not x or not y:
        return None
    gt = lt = 0
    for a in x:
        for b in y:
            gt += a > b
            lt += a < b
    return float((gt - lt) / (len(x) * len(y)))


def load_prediction_rows_for_stats(args) -> List[Dict[str, Any]]:
    path = args.pred_jsonl or args.pred_csv
    if not path:
        raise ValueError("Provide --pred_jsonl or --pred_csv")
    path = to_abs(path)
    raw_rows = []
    if args.pred_jsonl or path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            raw_rows = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            raw_rows = list(csv.DictReader(f))

    rows = []
    for i, r in enumerate(raw_rows):
        try:
            audio_path = get_audio_path(r, args.audio_root)
            if not os.path.isfile(audio_path):
                print(f"[WARN] row {i}: missing audio: {audio_path}")
                continue
        except Exception as e:
            print(f"[WARN] row {i}: skipped: {e}")
            continue
        gt = normalize_label(r.get("ground_truth", r.get("label", r.get("true_label"))))
        pred = normalize_label(r.get("prediction", r.get("pred", r.get("predicted"))))
        if pred is None:
            continue
        probs = parse_probs(r)
        if sum(probs.values()) <= 0:
            probs = {e: 1.0 / len(EMOS) for e in EMOS}
        conf = finite_float(r.get("confidence"), probs.get(pred, 0.0))
        correct = parse_bool(r.get("correct", gt == pred if gt else False)) if ("correct" in r or gt) else None
        rows.append({
            "source_index": int(r.get("index", i)) if str(r.get("index", i)).strip() else i,
            "dataset": str(r.get("dataset", args.dataset_name or "unknown")),
            "audio_id": str(r.get("audio_id", r.get("audio_path", audio_path))),
            "audio_path": audio_path,
            "ground_truth": gt,
            "prediction": pred,
            "confidence": float(conf),
            "probs": probs,
            "correct": correct,
            "raw": r,
        })
    return rows


def select_rows_class_aware(rows: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    rng = random.Random(args.seed)
    if args.only_wrong:
        rows = [r for r in rows if r.get("correct") is False]
    if args.only_correct:
        rows = [r for r in rows if r.get("correct") is True]
    if args.class_balance_mode == "none":
        if args.select_mode == "wrong_then_correct":
            wrong = [r for r in rows if r.get("correct") is False]
            correct = [r for r in rows if r.get("correct") is True]
            unknown = [r for r in rows if r.get("correct") is None]
            for part in (wrong, correct, unknown):
                rng.shuffle(part)
            selected = wrong[:max(0, args.wrong_samples)]
            selected += correct[:max(0, args.max_samples - len(selected))]
            selected += unknown[:max(0, args.max_samples - len(selected))]
            rng.shuffle(selected)
            return selected[:args.max_samples]
        rows = list(rows)
        rng.shuffle(rows)
        return rows[:args.max_samples]

    by_class = {e: [r for r in rows if r["prediction"] == e] for e in EMOS}
    for part in by_class.values():
        rng.shuffle(part)
    target_per_class = int(math.ceil(args.max_samples / len(EMOS))) if args.max_samples > 0 else 0
    selected = []
    reasons = {}

    if args.class_balance_mode == "predicted_class_fixed_correct_wrong":
        target_correct = max(0, int(args.correct_per_class))
        target_wrong = max(0, int(args.wrong_per_class))
        target_n = target_correct + target_wrong
        for cls in EMOS:
            cls_rows = by_class[cls]
            correct = [r for r in cls_rows if r.get("correct") is True]
            wrong = [r for r in cls_rows if r.get("correct") is False]
            chosen = []
            for r in correct[:target_correct]:
                reasons[id(r)] = f"fixed_balance:{cls}:correct"
                chosen.append(r)
            for r in wrong[:target_wrong]:
                reasons[id(r)] = f"fixed_balance:{cls}:wrong"
                chosen.append(r)
            used_cls = {id(r) for r in chosen}
            fill = [r for r in cls_rows if id(r) not in used_cls][:max(0, target_n - len(chosen))]
            for r in fill:
                reasons[id(r)] = f"fixed_balance:{cls}:same_class_fill"
            selected.extend(chosen + fill)
        if args.max_samples and len(selected) < args.max_samples:
            used = {id(r) for r in selected}
            leftovers = [r for r in rows if id(r) not in used]
            rng.shuffle(leftovers)
            fill = leftovers[:args.max_samples - len(selected)]
            for r in fill:
                reasons[id(r)] = "fixed_balance:global_fill"
            selected.extend(fill)
        selected = selected[:args.max_samples]
        for r in selected:
            r["selection_reason"] = reasons.get(id(r), "selected")
        rng.shuffle(selected)
        return selected

    wrong_left = max(0, args.wrong_samples)
    for cls in EMOS:
        cls_rows = by_class[cls]
        if args.class_balance_mode == "predicted_class_and_correctness_balanced":
            wrong = [r for r in cls_rows if r.get("correct") is False]
            correct = [r for r in cls_rows if r.get("correct") is True]
            unknown = [r for r in cls_rows if r.get("correct") is None]
            take_wrong = min(len(wrong), max(0, target_per_class // 2), wrong_left)
            chosen = wrong[:take_wrong]
            wrong_left -= take_wrong
            chosen += correct[:max(0, target_per_class - len(chosen))]
            chosen += unknown[:max(0, target_per_class - len(chosen))]
            if len(chosen) < target_per_class:
                used = {id(r) for r in chosen}
                chosen += [r for r in cls_rows if id(r) not in used][:target_per_class - len(chosen)]
        else:
            wrong = [r for r in cls_rows if r.get("correct") is False]
            correct_or_unknown = [r for r in cls_rows if r.get("correct") is not False]
            take_wrong = min(len(wrong), wrong_left, max(1, target_per_class // 4))
            chosen = wrong[:take_wrong]
            wrong_left -= take_wrong
            chosen += correct_or_unknown[:max(0, target_per_class - len(chosen))]
            if len(chosen) < target_per_class:
                used = {id(r) for r in chosen}
                chosen += [r for r in cls_rows if id(r) not in used][:target_per_class - len(chosen)]
        for r in chosen:
            reasons[id(r)] = f"class_balance:{cls}"
        selected.extend(chosen)

    used = {id(r) for r in selected}
    leftovers = [r for r in rows if id(r) not in used]
    rng.shuffle(leftovers)
    if args.wrong_samples > sum(1 for r in selected if r.get("correct") is False):
        need = args.wrong_samples - sum(1 for r in selected if r.get("correct") is False)
        extra_wrong = [r for r in leftovers if r.get("correct") is False][:need]
        for r in extra_wrong:
            reasons[id(r)] = "wrong_sample_fill"
        selected.extend(extra_wrong)
        used = {id(r) for r in selected}
        leftovers = [r for r in leftovers if id(r) not in used]
    if len(selected) < args.max_samples:
        fill = leftovers[:args.max_samples - len(selected)]
        for r in fill:
            reasons[id(r)] = "remaining_slot_fill"
        selected.extend(fill)
    selected = selected[:args.max_samples]
    for r in selected:
        r["selection_reason"] = reasons.get(id(r), "selected")
    return selected


def write_manifest(rows: List[Dict[str, Any]], out_dir: str):
    path = os.path.join(out_dir, "selected_samples_manifest.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_index", "source_index", "dataset", "audio_id",
            "ground_truth", "prediction", "correct", "confidence", "selection_reason",
        ])
        writer.writeheader()
        for i, r in enumerate(rows):
            writer.writerow({
                "sample_index": i,
                "source_index": r["source_index"],
                "dataset": r["dataset"],
                "audio_id": r["audio_id"],
                "ground_truth": r.get("ground_truth"),
                "prediction": r["prediction"],
                "correct": r.get("correct"),
                "confidence": r.get("confidence"),
                "selection_reason": r.get("selection_reason", "selected"),
            })


def choose_contrast(original_result: Dict[str, Any], pred: str, gt: Optional[str]) -> Optional[str]:
    if gt in EMOS and gt != pred:
        return gt
    probs = original_result.get("probs", {}) or {}
    candidates = [(e, finite_float(probs.get(e), 0.0)) for e in EMOS if e != pred]
    return max(candidates, key=lambda x: x[1])[0] if candidates else None


def rank_score(effect: Dict[str, Any]) -> float:
    for key in ("logprob_margin_drop", "prob_margin_drop", "prob_drop"):
        v = finite_float(effect.get(key))
        if v is not None:
            return v
    return -1e18


def transform_global_audio(y: np.ndarray, sr: int, transform: str) -> np.ndarray:
    if transform == "global_energy_down":
        return (y * 0.50).astype(np.float32)
    if transform == "global_energy_up":
        return np.clip(y * 1.40, -1.0, 1.0).astype(np.float32)
    if transform == "global_pitch_down":
        try:
            return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-2.0).astype(np.float32)
        except Exception:
            return y.copy()
    if transform == "global_pitch_up":
        try:
            return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2.0).astype(np.float32)
        except Exception:
            return y.copy()
    if transform == "global_time_faster":
        try:
            return fit_length(librosa.effects.time_stretch(y, rate=1.20), len(y))
        except Exception:
            return y.copy()
    if transform == "global_time_slower":
        try:
            return fit_length(librosa.effects.time_stretch(y, rate=0.85), len(y))
        except Exception:
            return y.copy()
    if transform == "global_spectral_darken":
        return spectral_tilt(y, sr, "darken")
    if transform == "global_spectral_brighten":
        return spectral_tilt(y, sr, "brighten")
    if transform == "global_noise_mild":
        rng = np.random.default_rng(123)
        power = float(np.mean(y ** 2) + 1e-12)
        noise = rng.normal(0.0, math.sqrt(power / (10 ** (20 / 10))), size=len(y)).astype(np.float32)
        return (y + noise).astype(np.float32)
    raise ValueError(f"Unknown global transform: {transform}")


class BatchScorer:
    def __init__(self, model, processor, batch_size: int = 2):
        self.model = model
        self.processor = processor
        self.batch_size = max(1, int(batch_size))
        self.fallback_logged = False
        self.used_fallback = False

    def score(self, wav_paths: List[str]) -> List[Dict[str, Any]]:
        results = []
        idx = 0
        while idx < len(wav_paths):
            chunk = wav_paths[idx:idx + self.batch_size]
            if len(chunk) > 1:
                if not self.fallback_logged:
                    print("[WARN] Voxtral scoring is label-conditioned; using reliable sequential scoring", flush=True)
                    self.fallback_logged = True
                self.used_fallback = True
            for p in chunk:
                while True:
                    try:
                        results.append(score_audio_labels(self.model, self.processor, p))
                        break
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and torch is not None and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            self.batch_size = 1
                            print("[WARN] CUDA OOM during scoring; retrying with batch_size=1", flush=True)
                            continue
                        raise
            idx += len(chunk)
        return results


def sample_random_regions(y, sr, duration, n, seed, avoid=None):
    rng = random.Random(seed)
    total_dur = len(y) / sr if sr else 0.0
    duration = min(float(duration), total_dur)
    if total_dur <= 0 or duration <= 0:
        return []
    if total_dur <= duration:
        return [{"start": 0.0, "end": float(total_dur)}]
    avoid = avoid or []
    regions = []
    for _ in range(n):
        best = None
        for _try in range(40):
            st = rng.uniform(0.0, max(0.0, total_dur - duration))
            cand = {"start": float(st), "end": float(st + duration)}
            overlap = False
            for a in avoid:
                if max(cand["start"], a["start"]) < min(cand["end"], a["end"]):
                    overlap = True
                    break
            if not overlap:
                best = cand
                break
            best = cand
        regions.append(best)
    return regions


def variant_row_base(row, sample_index, variant_type, variant_name, transform=None, segment_rank=None, start=None, end=None):
    return {
        "sample_index": sample_index,
        "source_index": row["source_index"],
        "dataset": row["dataset"],
        "prediction": row["prediction"],
        "ground_truth": row.get("ground_truth"),
        "correct": row.get("correct"),
        "variant_type": variant_type,
        "variant_name": variant_name,
        "transform": transform,
        "segment_rank": segment_rank,
        "start": start,
        "end": end,
    }


def fill_variant_effect(base, effect):
    out = dict(base)
    out.update({
        "prob_before": effect.get("prob_before"),
        "prob_after": effect.get("prob_after"),
        "prob_drop": effect.get("prob_drop"),
        "margin_before": effect.get("prob_margin_before"),
        "margin_after": effect.get("prob_margin_after"),
        "margin_drop": effect.get("prob_margin_drop"),
        "logmargin_before": effect.get("logprob_margin_before"),
        "logmargin_after": effect.get("logprob_margin_after"),
        "logmargin_drop": effect.get("logprob_margin_drop"),
        "new_prediction": effect.get("new_prediction"),
        "flip": effect.get("new_prediction") != effect.get("pred_class"),
    })
    return out


def assign_behavior(sample: Dict[str, Any]) -> Tuple[str, float, str]:
    local_p = finite_float(sample.get("best_local_prob_drop"), 0.0)
    local_m = finite_float(sample.get("best_local_margin_drop"), 0.0)
    global_p = finite_float(sample.get("best_global_prob_drop"), 0.0)
    global_m = finite_float(sample.get("best_global_margin_drop"), 0.0)
    local_strength = strength_label(local_p, local_m)
    global_strength = strength_label(global_p, global_m)
    local_mod = local_strength in {"strong", "moderate"}
    global_mod = global_strength in {"strong", "moderate"}
    top1_z = finite_float(sample.get("top1_local_vs_random_z"))
    top2_z = finite_float(sample.get("top2_local_vs_random_z"))
    z_best = max([z for z in [top1_z, top2_z] if z is not None], default=None)
    clearly_random_better = z_best is not None and z_best >= 0.75
    margin_mod = strength_label(None, max(abs(local_m), abs(global_m))) in {"strong", "moderate"}
    prob_weak = strength_label(max(abs(local_p), abs(global_p)), None) in {"weak", "negligible"}

    if local_p < -0.05 or global_p < -0.05 or local_m < -0.10 or global_m < -0.10:
        return "conflicting_evidence", 0.90, "Perturbation increased target confidence or margin, indicating ambiguous competing evidence."
    if local_mod and global_mod:
        return "mixed_local_global_evidence", 0.85, "Both Top-2 temporal attenuation and global prosody perturbations produced moderate or strong sensitivity."
    if local_mod and clearly_random_better and local_p >= global_p:
        return "strong_localized_evidence", 0.85, "Top-2 temporal evidence was stronger than random same-duration controls and global perturbations."
    if global_mod and (not local_mod or global_p >= local_p):
        return "global_prosody_evidence", 0.82, "Whole-utterance prosody perturbations dominated the local temporal evidence."
    if prob_weak and margin_mod:
        return "contrastive_shift_evidence", 0.78, "Target probability changed little, but contrastive emotion margin shifted substantially."
    if not clearly_random_better and local_strength in {"weak", "moderate"}:
        return "random_like_evidence", 0.70, "Top temporal regions were not clearly stronger than random same-duration controls."
    if local_strength == "negligible" and global_strength == "negligible" and not sample.get("best_global_flip"):
        return "robust_evidence", 0.75, "Prediction was robust to tested local, random, and global perturbations."
    return "weak_inconclusive_evidence", 0.55, "Perturbations did not produce a dominant diagnostic pattern."


def local_global_label(local_drop, global_drop):
    lp = finite_float(local_drop, 0.0)
    gp = finite_float(global_drop, 0.0)
    if abs(lp) < 0.01 and abs(gp) < 0.01:
        return "both_negligible", "robust_or_weak"
    if lp >= gp * 1.25 and lp >= 0.01:
        return "local_greater", "local"
    if gp >= lp * 1.25 and gp >= 0.01:
        return "global_greater", "global"
    return "similar", "mixed"


def explain_one_sample_stats(model, processor, row, sample_index, args, scorer: BatchScorer):
    t0 = time.time()
    global_transforms = [x.strip() for x in args.global_transforms.split(",") if x.strip()]
    original_result = make_original_result_from_prediction(row)
    variant_rows = []
    evidence = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        y, sr = load_audio(row["audio_path"], args.sample_rate)
        if args.rescore_original:
            rescored = scorer.score([row["audio_path"]])[0]
            original_result["mean_logprobs"] = rescored.get("mean_logprobs")
            original_result["sum_logprobs"] = rescored.get("sum_logprobs")
            if args.use_rescored_original_probs:
                original_result["probs"] = rescored["probs"]
                original_result["confidence"] = float(rescored["probs"].get(row["prediction"], row["confidence"]))
        pred = row["prediction"]
        contrast = choose_contrast(original_result, pred, row.get("ground_truth"))

        variants = []
        segments = make_speech_aware_segments(y, sr, args.segment_sec, args.max_segments)
        for i, seg in enumerate(segments):
            p = os.path.join(tmp_dir, f"occlusion_{i:03d}.wav")
            write_wav(p, mask_region(y, sr, seg["start"], seg["end"], args.mask_mode, args.attenuate_factor), sr)
            variants.append(("occlusion", f"occlusion_{i:03d}", None, None, seg["start"], seg["end"], p, seg))

        scored = scorer.score([v[6] for v in variants])
        segment_effects = []
        for v, res in zip(variants, scored):
            eff = score_target_effect(original_result, res, pred, row.get("ground_truth"), contrast)
            seg = dict(v[7])
            seg.update({
                "rank_input_index": len(segment_effects),
                "effect": eff,
                "importance_prob_drop": eff.get("prob_drop"),
                "importance_prob_margin_drop": eff.get("prob_margin_drop"),
                "importance_logprob_margin_drop": eff.get("logprob_margin_drop"),
            })
            segment_effects.append(seg)
        ranked = sorted(segment_effects, key=lambda s: rank_score(s["effect"]), reverse=True)
        top_segments = ranked[:max(1, args.top_k)]

        local_variants = []
        for rank, seg in enumerate(top_segments[:2], start=1):
            p = os.path.join(tmp_dir, f"local_top{rank}.wav")
            write_wav(p, mask_region(y, sr, seg["start"], seg["end"], args.mask_mode, args.attenuate_factor), sr)
            local_variants.append(("local_top%d" % rank, f"local_top{rank}", None, rank, seg["start"], seg["end"], p, seg))
        if top_segments:
            p = os.path.join(tmp_dir, "local_top1_top2_combined.wav")
            write_wav(p, mask_many_regions(y, sr, top_segments[:2], args.mask_mode, args.attenuate_factor), sr)
            local_variants.append(("local_top1_top2_combined", "local_top1_top2_combined", None, 12, None, None, p, None))

        random_variants = []
        for rank, seg in enumerate(top_segments[:2], start=1):
            regions = sample_random_regions(
                y, sr, float(seg["end"] - seg["start"]), args.random_n,
                args.seed + sample_index * 1009 + rank * 137,
                avoid=[{"start": float(seg["start"]), "end": float(seg["end"])}],
            )
            for j, rr in enumerate(regions):
                p = os.path.join(tmp_dir, f"random_top{rank}_{j}.wav")
                write_wav(p, mask_region(y, sr, rr["start"], rr["end"], args.mask_mode, args.attenuate_factor), sr)
                random_variants.append((f"random_top{rank}", f"random_top{rank}_{j}", None, rank, rr["start"], rr["end"], p, rr))

        global_variants = []
        for tr in global_transforms:
            p = os.path.join(tmp_dir, f"{tr}.wav")
            write_wav(p, transform_global_audio(y, sr, tr), sr)
            global_variants.append(("global_prosody", tr, tr, None, 0.0, len(y) / sr, p, None))

        all_variants = local_variants + random_variants + global_variants
        all_scores = scorer.score([v[6] for v in all_variants])
        local_effects, random_effects, global_effects = {}, defaultdict(list), {}
        for v, res in zip(all_variants, all_scores):
            vtype, name, transform, rank, start, end, _path, _meta = v
            eff = score_target_effect(original_result, res, pred, row.get("ground_truth"), contrast)
            if vtype.startswith("local_top"):
                local_effects[vtype] = eff
            elif vtype.startswith("random_top"):
                random_effects[vtype].append(eff)
            elif vtype == "global_prosody":
                global_effects[name] = eff
            variant_rows.append(fill_variant_effect(
                variant_row_base(row, sample_index, vtype, name, transform, rank, start, end), eff
            ))

        top1_seg = top_segments[0] if len(top_segments) >= 1 else {}
        top2_seg = top_segments[1] if len(top_segments) >= 2 else {}
        top1_eff = local_effects.get("local_top1", top1_seg.get("effect", {}))
        top2_eff = local_effects.get("local_top2", top2_seg.get("effect", {}))
        combined_eff = local_effects.get("local_top1_top2_combined", {})
        top1_r = random_effects.get("random_top1", [])
        top2_r = random_effects.get("random_top2", [])
        top1_rand = [e.get("prob_drop") for e in top1_r]
        top2_rand = [e.get("prob_drop") for e in top2_r]
        rand_margin = [e.get("prob_margin_drop") for vals in random_effects.values() for e in vals]
        best_local_source, best_local_eff = max(
            [("top1", top1_eff), ("top2", top2_eff), ("top2_combined", combined_eff)],
            key=lambda kv: rank_score(kv[1]),
        )
        best_global_transform, best_global_eff = (None, {})
        if global_effects:
            best_global_transform, best_global_eff = max(global_effects.items(), key=lambda kv: rank_score(kv[1]))
        top1_z = safe_z(top1_eff.get("prob_drop"), mean_or_none(top1_rand), std_or_none(top1_rand))
        top2_z = safe_z(top2_eff.get("prob_drop"), mean_or_none(top2_rand), std_or_none(top2_rand))
        ratio = None
        if abs(finite_float(best_global_eff.get("prob_drop"), 0.0)) >= 1e-3:
            ratio = finite_float(best_local_eff.get("prob_drop"), 0.0) / finite_float(best_global_eff.get("prob_drop"), 1.0)
        lg_label, dominant_source = local_global_label(best_local_eff.get("prob_drop"), best_global_eff.get("prob_drop"))

        sample_row = {
            "sample_index": sample_index,
            "source_index": row["source_index"],
            "dataset": row["dataset"],
            "audio_id": row["audio_id"],
            "ground_truth": row.get("ground_truth"),
            "prediction": pred,
            "correct": row.get("correct"),
            "confidence": float(original_result.get("confidence", row["confidence"])),
            "contrast_class": contrast,
            "top1_start": top1_seg.get("start"), "top1_end": top1_seg.get("end"),
            "top1_duration": (top1_seg.get("end") - top1_seg.get("start")) if top1_seg else None,
            "top1_local_prob_drop": top1_eff.get("prob_drop"),
            "top1_local_margin_drop": top1_eff.get("prob_margin_drop"),
            "top1_local_logmargin_drop": top1_eff.get("logprob_margin_drop"),
            "top2_start": top2_seg.get("start"), "top2_end": top2_seg.get("end"),
            "top2_duration": (top2_seg.get("end") - top2_seg.get("start")) if top2_seg else None,
            "top2_local_prob_drop": top2_eff.get("prob_drop"),
            "top2_local_margin_drop": top2_eff.get("prob_margin_drop"),
            "top2_local_logmargin_drop": top2_eff.get("logprob_margin_drop"),
            "top2_combined_prob_drop": combined_eff.get("prob_drop"),
            "top2_combined_margin_drop": combined_eff.get("prob_margin_drop"),
            "top2_combined_logmargin_drop": combined_eff.get("logprob_margin_drop"),
            "best_local_source": best_local_source,
            "best_local_prob_drop": best_local_eff.get("prob_drop"),
            "best_local_margin_drop": best_local_eff.get("prob_margin_drop"),
            "best_local_logmargin_drop": best_local_eff.get("logprob_margin_drop"),
            "local_evidence_strength": strength_label(best_local_eff.get("prob_drop"), best_local_eff.get("prob_margin_drop")),
            "random_n": args.random_n,
            "top1_random_prob_drop_mean": mean_or_none(top1_rand),
            "top1_random_prob_drop_std": std_or_none(top1_rand),
            "top2_random_prob_drop_mean": mean_or_none(top2_rand),
            "top2_random_prob_drop_std": std_or_none(top2_rand),
            "random_margin_drop_mean": mean_or_none(rand_margin),
            "random_margin_drop_std": std_or_none(rand_margin),
            "top1_local_minus_random": finite_float(top1_eff.get("prob_drop"), 0.0) - finite_float(mean_or_none(top1_rand), 0.0),
            "top2_local_minus_random": finite_float(top2_eff.get("prob_drop"), 0.0) - finite_float(mean_or_none(top2_rand), 0.0),
            "top1_local_vs_random_z": top1_z,
            "top2_local_vs_random_z": top2_z,
            "top1_greater_than_random": bool(top1_z is not None and top1_z > 0.75),
            "top2_greater_than_random": bool(top2_z is not None and top2_z > 0.75),
            "best_global_transform": best_global_transform,
            "best_global_prob_drop": best_global_eff.get("prob_drop"),
            "best_global_margin_drop": best_global_eff.get("prob_margin_drop"),
            "best_global_logmargin_drop": best_global_eff.get("logprob_margin_drop"),
            "best_global_new_prediction": best_global_eff.get("new_prediction"),
            "best_global_flip": best_global_eff.get("new_prediction") != pred if best_global_eff else False,
            "global_evidence_strength": strength_label(best_global_eff.get("prob_drop"), best_global_eff.get("prob_margin_drop")),
            "global_prosody_type": PROSODY_TYPE.get(best_global_transform),
            "local_global_ratio": ratio,
            "local_vs_global_label": lg_label,
            "dominant_evidence_source": dominant_source,
        }
        behavior, bconf, btxt = assign_behavior(sample_row)
        sample_row.update({
            "behavior_type": behavior,
            "behavior_confidence": bconf,
            "dominant_pattern_short_text": btxt,
            "n_variants_scored": len(all_variants) + len(variants) + (1 if args.rescore_original else 0),
            "elapsed_sec": time.time() - t0,
            "sec_per_variant": (time.time() - t0) / max(1, len(all_variants) + len(variants)),
            "batch_size_used": scorer.batch_size,
            "batching_fallback_used": scorer.used_fallback,
        })
        evidence = {
            "sample_row": sample_row,
            "variant_rows": variant_rows,
            "top_segments": top_segments[:2],
            "global_effects": global_effects,
            "random_effects": dict(random_effects),
            "note": "Post-hoc perturbation-based input-output sensitivity evidence, not causal proof or hidden reasoning access.",
        }
    return sample_row, variant_rows, evidence


def read_done_keys(sample_csv: str) -> set:
    if not os.path.isfile(sample_csv):
        return set()
    with open(sample_csv, "r", encoding="utf-8", newline="") as f:
        return {(int(r["sample_index"]), int(r["source_index"])) for r in csv.DictReader(f) if r.get("sample_index") and r.get("source_index")}


def append_dict_row(path, fields, row):
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(obj), ensure_ascii=False) + "\n")
        f.flush()


def read_csv_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def build_confusion_matrix(rows):
    matrix = {gt: {pred: 0 for pred in EMOS} for gt in EMOS}
    unknown = Counter()
    for r in rows:
        gt = r.get("ground_truth")
        pred = r.get("prediction")
        if gt in EMOS and pred in EMOS:
            matrix[gt][pred] += 1
        else:
            unknown[(gt, pred)] += 1
    return matrix, unknown


def write_confusion_outputs(out_dir, sample_rows, selected_rows):
    rows = sample_rows if sample_rows else selected_rows
    matrix, unknown = build_confusion_matrix(rows)
    cm_rows = []
    for gt in EMOS:
        row = {"ground_truth": gt}
        total = sum(matrix[gt].values())
        for pred in EMOS:
            row[f"pred_{pred}"] = matrix[gt][pred]
        row["row_total"] = total
        row["row_accuracy_percent"] = float(100.0 * matrix[gt][gt] / total) if total else None
        cm_rows.append(row)
    write_csv(
        os.path.join(out_dir, "confusion_matrix_selected_samples.csv"),
        cm_rows,
        ["ground_truth"] + [f"pred_{e}" for e in EMOS] + ["row_total", "row_accuracy_percent"],
    )

    by_pred_rows = []
    for pred in EMOS:
        part = [r for r in rows if r.get("prediction") == pred]
        correct = sum(str(r.get("correct")).lower() == "true" for r in part)
        wrong = sum(str(r.get("correct")).lower() == "false" for r in part)
        by_gt = Counter(r.get("ground_truth") for r in part)
        row = {
            "predicted_class": pred,
            "n": len(part),
            "correct": correct,
            "wrong": wrong,
            "correct_percent": float(100.0 * correct / len(part)) if part else None,
        }
        for gt in EMOS:
            row[f"gt_{gt}"] = by_gt.get(gt, 0)
        by_pred_rows.append(row)
    write_csv(
        os.path.join(out_dir, "selected_sampling_balance_summary.csv"),
        by_pred_rows,
        ["predicted_class", "n", "correct", "wrong", "correct_percent"] + [f"gt_{e}" for e in EMOS],
    )
    unknown_out = {f"gt={k[0]}|pred={k[1]}": v for k, v in unknown.items()}
    return matrix, unknown_out


def grouped(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r.get(key)].append(r)
    return out


def compute_aggregate_outputs(out_dir: str, args, selected_rows: List[Dict[str, Any]]):
    sample_rows = read_csv_rows(os.path.join(out_dir, "xai_200_sample_rows.csv"))
    variant_rows = read_csv_rows(os.path.join(out_dir, "xai_200_variant_rows.csv"))
    for r in sample_rows:
        for k, v in list(r.items()):
            if v == "":
                r[k] = None

    confusion_matrix, confusion_unknown = write_confusion_outputs(out_dir, sample_rows, selected_rows)

    class_local = []
    for cls in EMOS:
        part = [r for r in sample_rows if r.get("prediction") == cls]
        s1 = mean_std_median_ci95([r.get("top1_local_prob_drop") for r in part], args.seed)
        s2 = mean_std_median_ci95([r.get("top2_local_prob_drop") for r in part], args.seed)
        row = {
            "predicted_class": cls, "n": len(part),
            "mean_top1_local_drop": s1["mean"], "std_top1_local_drop": s1["std"],
            "median_top1_local_drop": s1["median"], "ci95_top1_local_drop": ci_text(s1),
            "mean_top2_local_drop": s2["mean"], "std_top2_local_drop": s2["std"],
            "median_top2_local_drop": s2["median"], "ci95_top2_local_drop": ci_text(s2),
            "mean_top2_combined_drop": mean_or_none([r.get("top2_combined_prob_drop") for r in part]),
            "std_top2_combined_drop": std_or_none([r.get("top2_combined_prob_drop") for r in part]),
            "mean_local_margin_drop": mean_or_none([r.get("best_local_margin_drop") for r in part]),
        }
        row.update(effect_strength_counts([r.get("best_local_prob_drop") for r in part], "prob"))
        class_local.append(row)
    write_csv(os.path.join(out_dir, "class_local_statistics.csv"), class_local, list(class_local[0].keys()) if class_local else [])

    class_random = []
    for cls in EMOS:
        part = [r for r in sample_rows if r.get("prediction") == cls]
        rand_vals = [r.get("top1_random_prob_drop_mean") for r in part] + [r.get("top2_random_prob_drop_mean") for r in part]
        class_random.append({
            "predicted_class": cls, "n": len(part),
            "mean_random_drop": mean_or_none(rand_vals),
            "std_random_drop": std_or_none(rand_vals),
            "mean_local_minus_random": mean_or_none([r.get("top1_local_minus_random") for r in part] + [r.get("top2_local_minus_random") for r in part]),
            "mean_local_vs_random_z": mean_or_none([r.get("top1_local_vs_random_z") for r in part] + [r.get("top2_local_vs_random_z") for r in part]),
            "percent_local_greater_than_random": percent_bool([r.get("top1_greater_than_random") for r in part] + [r.get("top2_greater_than_random") for r in part]),
        })
    write_csv(os.path.join(out_dir, "class_random_statistics.csv"), class_random, list(class_random[0].keys()) if class_random else [])

    global_stats = []
    for cls in EMOS:
        for tr in [x.strip() for x in args.global_transforms.split(",") if x.strip()]:
            part = [v for v in variant_rows if v.get("prediction") == cls and v.get("variant_type") == "global_prosody" and v.get("transform") == tr]
            st = mean_std_median_ci95([r.get("prob_drop") for r in part], args.seed)
            row = {
                "predicted_class": cls, "global_transform": tr, "prosody_type": PROSODY_TYPE.get(tr),
                "n": len(part), "mean_global_prob_drop": st["mean"],
                "std_global_prob_drop": st["std"], "median_global_prob_drop": st["median"],
                "ci95_global_prob_drop": ci_text(st),
                "mean_global_margin_drop": mean_or_none([r.get("margin_drop") for r in part]),
                "std_global_margin_drop": std_or_none([r.get("margin_drop") for r in part]),
                "flip_rate": percent_bool([r.get("flip") for r in part]),
            }
            row.update(effect_strength_counts([r.get("prob_drop") for r in part], "prob"))
            global_stats.append(row)
    write_csv(os.path.join(out_dir, "class_global_statistics.csv"), global_stats, list(global_stats[0].keys()) if global_stats else [])

    behavior_rows = []
    for cls in EMOS:
        part = [r for r in sample_rows if r.get("prediction") == cls]
        cnt = Counter(r.get("behavior_type") for r in part)
        row = {"predicted_class": cls, "n": len(part)}
        for b in BEHAVIORS:
            row[b + "_count"] = cnt.get(b, 0)
            row[b + "_percent"] = float(100.0 * cnt.get(b, 0) / len(part)) if part else None
        behavior_rows.append(row)
    write_csv(os.path.join(out_dir, "class_behavior_distribution.csv"), behavior_rows, list(behavior_rows[0].keys()) if behavior_rows else [])

    cww = []
    for label, part in [("correct", [r for r in sample_rows if str(r.get("correct")).lower() == "true"]), ("wrong", [r for r in sample_rows if str(r.get("correct")).lower() == "false"])]:
        cnt = Counter(r.get("behavior_type") for r in part)
        cww.append({
            "group": label, "n": len(part),
            "mean_local_drop": mean_or_none([r.get("best_local_prob_drop") for r in part]),
            "std_local_drop": std_or_none([r.get("best_local_prob_drop") for r in part]),
            "median_local_drop": median_or_none([r.get("best_local_prob_drop") for r in part]),
            "mean_global_drop": mean_or_none([r.get("best_global_prob_drop") for r in part]),
            "std_global_drop": std_or_none([r.get("best_global_prob_drop") for r in part]),
            "median_global_drop": median_or_none([r.get("best_global_prob_drop") for r in part]),
            "mean_margin_drop": mean_or_none([r.get("best_local_margin_drop") for r in part]),
            "flip_rate": percent_bool([r.get("best_global_flip") for r in part]),
            "dominant_behavior": cnt.most_common(1)[0][0] if cnt else None,
        })
    write_csv(os.path.join(out_dir, "correct_vs_wrong_statistics.csv"), cww, list(cww[0].keys()))

    aggregate = build_aggregate(sample_rows, selected_rows, confusion_matrix, confusion_unknown)
    with open(os.path.join(out_dir, "xai_200_aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(aggregate), f, indent=2, ensure_ascii=False)

    tests = statistical_tests(sample_rows, variant_rows)
    write_csv(os.path.join(out_dir, "statistical_tests.csv"), tests, list(tests[0].keys()) if tests else ["test", "statistic", "p_value", "effect_size", "p_fdr", "note"])

    if not args.no_plots:
        write_dataset_plots(out_dir, sample_rows, variant_rows)
    write_dominant_patterns(out_dir, sample_rows)
    write_output_file_index(out_dir, args)
    write_paper_ready(out_dir, aggregate, class_local, class_random, global_stats, behavior_rows, cww, tests)
    write_novelty_assessment(out_dir)
    write_reproducibility(out_dir, args)
    quality = write_quality_checks(out_dir, sample_rows, variant_rows, selected_rows)
    return aggregate, quality


def ci_text(stat):
    if stat.get("ci95_low") is None:
        return None
    return f"[{stat['ci95_low']:.6f}, {stat['ci95_high']:.6f}]"


def percent_bool(values):
    vals = []
    for v in values:
        if v is None or v == "":
            continue
        vals.append(str(v).lower() in {"true", "1", "yes"})
    return float(100.0 * sum(vals) / len(vals)) if vals else None


def build_aggregate(sample_rows, selected_rows, confusion_matrix=None, confusion_unknown=None):
    cnt_beh = Counter(r.get("behavior_type") for r in sample_rows)
    cnt_g = Counter(r.get("best_global_transform") for r in sample_rows)
    cnt_p = Counter(r.get("global_prosody_type") for r in sample_rows)
    n = len(sample_rows)
    return {
        "n_explained": n,
        "n_correct": sum(str(r.get("correct")).lower() == "true" for r in sample_rows),
        "n_wrong": sum(str(r.get("correct")).lower() == "false" for r in sample_rows),
        "selection_distribution": {
            "selected_n": len(selected_rows),
            "by_predicted_class": dict(Counter(r["prediction"] for r in selected_rows)),
            "by_correctness": dict(Counter(str(r.get("correct")) for r in selected_rows)),
            "by_predicted_class_and_correctness": {
                cls: {
                    "correct": sum(r["prediction"] == cls and r.get("correct") is True for r in selected_rows),
                    "wrong": sum(r["prediction"] == cls and r.get("correct") is False for r in selected_rows),
                    "unknown": sum(r["prediction"] == cls and r.get("correct") is None for r in selected_rows),
                }
                for cls in EMOS
            },
        },
        "mean_top1_local_drop": mean_or_none([r.get("top1_local_prob_drop") for r in sample_rows]),
        "mean_top2_local_drop": mean_or_none([r.get("top2_local_prob_drop") for r in sample_rows]),
        "mean_top2_combined_drop": mean_or_none([r.get("top2_combined_prob_drop") for r in sample_rows]),
        "mean_random_drop": mean_or_none([r.get("top1_random_prob_drop_mean") for r in sample_rows] + [r.get("top2_random_prob_drop_mean") for r in sample_rows]),
        "mean_best_global_drop": mean_or_none([r.get("best_global_prob_drop") for r in sample_rows]),
        "behavior_counts": dict(cnt_beh),
        "behavior_percentages": {k: 100.0 * v / n for k, v in cnt_beh.items()} if n else {},
        "best_global_transform_counts": dict(cnt_g),
        "global_prosody_type_counts": dict(cnt_p),
        "correct_vs_wrong_summary": {
            "correct_mean_local": mean_or_none([r.get("best_local_prob_drop") for r in sample_rows if str(r.get("correct")).lower() == "true"]),
            "wrong_mean_local": mean_or_none([r.get("best_local_prob_drop") for r in sample_rows if str(r.get("correct")).lower() == "false"]),
            "correct_mean_global": mean_or_none([r.get("best_global_prob_drop") for r in sample_rows if str(r.get("correct")).lower() == "true"]),
            "wrong_mean_global": mean_or_none([r.get("best_global_prob_drop") for r in sample_rows if str(r.get("correct")).lower() == "false"]),
        },
        "confusion_matrix_ground_truth_by_prediction": confusion_matrix or {},
        "confusion_unknown_pairs": confusion_unknown or {},
    }


def statistical_tests(sample_rows, variant_rows):
    rows = []
    try:
        from scipy import stats
    except Exception as e:
        return [{"test": "scipy_unavailable", "statistic": None, "p_value": None, "effect_size": None, "p_fdr": None, "note": str(e)}]
    tests_for_fdr = []
    groups = [[finite_float(r.get("best_local_prob_drop")) for r in sample_rows if r.get("prediction") == cls] for cls in EMOS]
    groups = [[v for v in g if v is not None] for g in groups]
    if sum(len(g) > 0 for g in groups) >= 2:
        stat, p = stats.kruskal(*[g for g in groups if g])
        rows.append({"test": "kruskal_local_drops_across_predicted_classes", "statistic": float(stat), "p_value": float(p), "effect_size": None, "p_fdr": None, "note": ""})
        tests_for_fdr.append(len(rows) - 1)
    for metric, name in [("best_local_prob_drop", "local"), ("best_global_prob_drop", "global")]:
        corr = [finite_float(r.get(metric)) for r in sample_rows if str(r.get("correct")).lower() == "true"]
        wrong = [finite_float(r.get(metric)) for r in sample_rows if str(r.get("correct")).lower() == "false"]
        corr = [v for v in corr if v is not None]
        wrong = [v for v in wrong if v is not None]
        if corr and wrong:
            stat, p = stats.mannwhitneyu(corr, wrong, alternative="two-sided")
            rows.append({"test": f"mann_whitney_correct_vs_wrong_{name}_drops", "statistic": float(stat), "p_value": float(p), "effect_size": cliff_delta(corr, wrong), "p_fdr": None, "note": "effect_size=Cliff_delta"})
            tests_for_fdr.append(len(rows) - 1)
    ggroups = [[finite_float(r.get("prob_drop")) for r in variant_rows if r.get("prediction") == cls and r.get("variant_type") == "global_prosody"] for cls in EMOS]
    ggroups = [[v for v in g if v is not None] for g in ggroups]
    if sum(len(g) > 0 for g in ggroups) >= 2:
        stat, p = stats.kruskal(*[g for g in ggroups if g])
        rows.append({"test": "kruskal_global_drops_across_predicted_classes", "statistic": float(stat), "p_value": float(p), "effect_size": None, "p_fdr": None, "note": ""})
        tests_for_fdr.append(len(rows) - 1)
    pvals = [rows[i]["p_value"] for i in tests_for_fdr]
    try:
        from statsmodels.stats.multitest import multipletests
        _, padj, _, _ = multipletests(pvals, method="fdr_bh")
    except Exception:
        padj = benjamini_hochberg(pvals)
    for idx, p in zip(tests_for_fdr, padj):
        rows[idx]["p_fdr"] = float(p)
    return rows


def benjamini_hochberg(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [None] * m
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        val = min(prev, pvals[i] * m / rank)
        adj[i] = val
        prev = val
    return adj


def write_dataset_plots(out_dir, sample_rows, variant_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    globals()["plt"] = plt
    plot_dir = os.path.join(out_dir, "plots")
    safe_mkdir(plot_dir)
    x = np.arange(len(EMOS))
    width = 0.28
    top1 = [mean_or_none([r.get("top1_local_prob_drop") for r in sample_rows if r.get("prediction") == c]) or 0 for c in EMOS]
    top2 = [mean_or_none([r.get("top2_local_prob_drop") for r in sample_rows if r.get("prediction") == c]) or 0 for c in EMOS]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, top1, width, label="Top-1")
    ax.bar(x + width / 2, top2, width, label="Top-2")
    ax.set_xticks(x, EMOS)
    ax.set_ylabel("Mean probability drop")
    ax.set_title("Average Top-1/Top-2 Local Drop by Predicted Class")
    ax.legend()
    save_plot(fig, os.path.join(plot_dir, "avg_top1_top2_local_drop_by_class.png"))

    transforms = sorted(set(v.get("transform") for v in variant_rows if v.get("variant_type") == "global_prosody" and v.get("transform")))
    if transforms:
        data = np.array([[mean_or_none([v.get("prob_drop") for v in variant_rows if v.get("prediction") == c and v.get("transform") == tr]) or 0 for tr in transforms] for c in EMOS])
        fig, ax = plt.subplots(figsize=(max(9, len(transforms) * 1.1), 4.8))
        im = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(transforms)), [t.replace("global_", "") for t in transforms], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(EMOS)), EMOS)
        ax.set_title("Average Global Drop by Class and Transform")
        fig.colorbar(im, ax=ax, label="Mean probability drop")
        save_plot(fig, os.path.join(plot_dir, "avg_global_drop_by_class_transform.png"))

    behavior_counts = np.array([[sum(r.get("prediction") == c and r.get("behavior_type") == b for r in sample_rows) for b in BEHAVIORS] for c in EMOS], dtype=float)
    denom = np.maximum(behavior_counts.sum(axis=1, keepdims=True), 1.0)
    behavior_pct = behavior_counts / denom * 100.0
    fig, ax = plt.subplots(figsize=(11, 5.4))
    bottom = np.zeros(len(EMOS))
    for j, b in enumerate(BEHAVIORS):
        ax.bar(EMOS, behavior_pct[:, j], bottom=bottom, label=b.replace("_evidence", "").replace("_", " "))
        bottom += behavior_pct[:, j]
    ax.set_ylabel("Percent")
    ax.set_title("Behavior Distribution by Predicted Class")
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_plot(fig, os.path.join(plot_dir, "behavior_distribution_by_class.png"))

    local = [mean_or_none([r.get("best_local_prob_drop") for r in sample_rows if r.get("prediction") == c]) or 0 for c in EMOS]
    rand = [mean_or_none([r.get("top1_random_prob_drop_mean") for r in sample_rows if r.get("prediction") == c] + [r.get("top2_random_prob_drop_mean") for r in sample_rows if r.get("prediction") == c]) or 0 for c in EMOS]
    glob = [mean_or_none([r.get("best_global_prob_drop") for r in sample_rows if r.get("prediction") == c]) or 0 for c in EMOS]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width, local, width, label="Best local")
    ax.bar(x, rand, width, label="Random")
    ax.bar(x + width, glob, width, label="Best global")
    ax.set_xticks(x, EMOS)
    ax.set_ylabel("Mean probability drop")
    ax.set_title("Local, Random, and Global Comparison by Class")
    ax.legend()
    save_plot(fig, os.path.join(plot_dir, "local_random_global_comparison_by_class.png"))

    best_counts = Counter((r.get("prediction"), r.get("best_global_transform")) for r in sample_rows if r.get("best_global_transform"))
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(EMOS))
    for tr in transforms:
        vals = np.array([best_counts.get((c, tr), 0) for c in EMOS])
        ax.bar(EMOS, vals, bottom=bottom, label=tr.replace("global_", ""))
        bottom += vals
    ax.set_title("Best Global Transform Counts by Class")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7, ncol=2)
    save_plot(fig, os.path.join(plot_dir, "best_global_transform_counts_by_class.png"))

    corr_local = [finite_float(r.get("best_local_prob_drop")) for r in sample_rows if str(r.get("correct")).lower() == "true"]
    wrong_local = [finite_float(r.get("best_local_prob_drop")) for r in sample_rows if str(r.get("correct")).lower() == "false"]
    corr_global = [finite_float(r.get("best_global_prob_drop")) for r in sample_rows if str(r.get("correct")).lower() == "true"]
    wrong_global = [finite_float(r.get("best_global_prob_drop")) for r in sample_rows if str(r.get("correct")).lower() == "false"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(["correct local", "wrong local", "correct global", "wrong global"], [
        mean_or_none(corr_local) or 0, mean_or_none(wrong_local) or 0,
        mean_or_none(corr_global) or 0, mean_or_none(wrong_global) or 0,
    ])
    ax.set_ylabel("Mean probability drop")
    ax.set_title("Correct vs Wrong Drops")
    save_plot(fig, os.path.join(plot_dir, "correct_vs_wrong_drops.png"))

    prosody_types = sorted(set(PROSODY_TYPE.values()))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(EMOS))
    for pt in prosody_types:
        vals = np.array([sum(r.get("prediction") == c and r.get("global_prosody_type") == pt for r in sample_rows) for c in EMOS])
        ax.bar(EMOS, vals, bottom=bottom, label=pt)
        bottom += vals
    ax.set_title("Global Prosody Type Distribution by Class")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    save_plot(fig, os.path.join(plot_dir, "global_prosody_type_distribution_by_class.png"))


    matrix, _unknown = build_confusion_matrix(sample_rows)
    cm = np.array([[matrix[gt][pred] for pred in EMOS] for gt in EMOS], dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(EMOS)), EMOS)
    ax.set_yticks(np.arange(len(EMOS)), EMOS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground truth")
    ax.set_title("Selected-Sample Confusion Matrix")
    for i in range(len(EMOS)):
        for j in range(len(EMOS)):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="Count")
    save_plot(fig, os.path.join(plot_dir, "confusion_matrix_selected_samples.png"))


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def write_dominant_patterns(out_dir, sample_rows):
    lines = ["# Class Dominant Patterns", ""]
    for cls in EMOS:
        part = [r for r in sample_rows if r.get("prediction") == cls]
        cnt = Counter(r.get("behavior_type") for r in part)
        dominant = cnt.most_common(1)[0][0] if cnt else "unavailable"
        strongest_global = Counter(r.get("best_global_transform") for r in part if r.get("best_global_transform")).most_common(1)
        prosody = Counter(r.get("global_prosody_type") for r in part if r.get("global_prosody_type")).most_common(1)
        orientation = Counter(r.get("dominant_evidence_source") for r in part).most_common(1)
        lines += [
            f"## {cls}",
            f"- n: {len(part)}",
            f"- Dominant behavior type: {dominant}",
            f"- Mean best local drop: {fmt(mean_or_none([r.get('best_local_prob_drop') for r in part]))}",
            f"- Strongest global transform: {strongest_global[0][0] if strongest_global else 'unavailable'}",
            f"- Dominant prosody type: {prosody[0][0] if prosody else 'unavailable'}",
            f"- Overall pattern: {orientation[0][0] if orientation else 'unavailable'}",
            "- Safe interpretation: these results describe perturbation-based input-output sensitivity, not internal causal reasoning.",
            "",
        ]
    Path(out_dir, "class_dominant_patterns.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_ready(out_dir, aggregate, class_local, class_random, global_stats, behavior_rows, cww, tests):
    selection = aggregate.get("selection_distribution", {})
    per_class_correctness = selection.get("by_predicted_class_and_correctness", {})
    lines = [
        "# Q1-Style Statistical XAI Results",
        "",
        "## Abstract-Style Summary",
        f"This run analyzed {aggregate.get('n_explained')} selected predictions with Top-2 temporal attenuation, random same-duration controls, and global prosody perturbations. The evidence is post-hoc input-output sensitivity evidence for Voxtral SER.",
        "",
        "## Research Gap",
        "Prior speech XAI work often focuses on sample-level explanations for encoder-based speech classifiers. This experiment shifts the emphasis to class-level statistical behavior for a Voxtral audio-language SER model.",
        "",
        "## Novelty",
        "LG-ProXAI-Stats combines Top-2 temporal evidence, random same-duration baselines, whole-utterance prosody perturbations, contrastive emotion margins, and an eight-category behavior taxonomy.",
        "",
        "## Sampling Design",
        f"Selected samples: {selection.get('selected_n', 'NA')}. Predicted-class distribution: `{json.dumps(selection.get('by_predicted_class', {}), sort_keys=True)}`.",
        f"Predicted-class/correctness distribution: `{json.dumps(per_class_correctness, sort_keys=True)}`.",
        "The files `selected_sampling_balance_summary.csv` and `confusion_matrix_selected_samples.csv` provide the reviewer-facing sample-balance and ground-truth/prediction tables.",
        "This design balances explanations by predicted class. Correct and wrong examples are intentionally sampled within each predicted class, so results describe behavior conditional on the model output class rather than the ground-truth class distribution.",
        "",
        "## Method",
        "For each selected sample, the previous prediction is treated as the target class. The contrast class is the ground truth for wrong predictions, otherwise the second-highest probability class. Temporal segments are attenuated, ranked by log-margin drop, probability-margin drop, then target probability drop. Top-2 segments are compared with random controls and global prosody transforms.",
        "",
        "## Perturbation Inventory",
        "Included in this class-level statistics run: speech-aware temporal segment attenuation for all candidate segments, re-scoring of Top-1 and Top-2 attenuated regions, combined Top-1+Top-2 attenuation, random same-duration attenuation controls, and whole-utterance global prosody transforms.",
        f"Not included in this compact {selection.get('selected_n', 'N')}-sample statistics run: per-sample deletion/insertion curves, keep-only sufficiency curves, localized acoustic counterfactual transform grids, and HTML/figure-heavy professional reports. Those are better suited to the professional per-sample pipeline because they multiply the perturbation budget substantially.",
        "The phrase `best global` means maximum sensitivity among the tested global transforms, not an unbiased average global effect. Per-transform global statistics should be treated as the primary global-prosody evidence.",
        "",
        "## Why This Sample Size",
        "The fixed-balance run allocates the requested sample budget evenly across predicted classes. Larger runs improve class-level mean estimates and correct-vs-wrong comparisons, but they remain perturbation-based diagnostics rather than causal explanations. Any class/correctness shortage is recorded in the sampling distribution above.",
        "",
        "## Why Top-2",
        "Top-2 captures distributed short-term evidence better than Top-1 while keeping the perturbation budget compact and interpretable.",
        "",
        "## Metrics",
        "Outputs include mean, standard deviation, median, bootstrap or small-sample 95% confidence intervals, flip rate, effect-strength percentages, behavior distributions, statistical tests, and FDR-adjusted p-values where available. Voxtral scores are forced-choice label scores derived from normalized label-token likelihoods, not guaranteed calibrated emotion probabilities.",
        "",
        "## Aggregate Results",
        f"- Mean Top-1 local drop: {fmt(aggregate.get('mean_top1_local_drop'))}",
        f"- Mean Top-2 local drop: {fmt(aggregate.get('mean_top2_local_drop'))}",
        f"- Mean combined Top-2 drop: {fmt(aggregate.get('mean_top2_combined_drop'))}",
        f"- Mean random drop: {fmt(aggregate.get('mean_random_drop'))}",
        f"- Mean best global drop: {fmt(aggregate.get('mean_best_global_drop'))}",
        f"- Behavior counts: `{json.dumps(aggregate.get('behavior_counts', {}), sort_keys=True)}`",
        f"- Selected-sample confusion matrix: `{json.dumps(aggregate.get('confusion_matrix_ground_truth_by_prediction', {}), sort_keys=True)}`",
        "",
        "## Statistical Tests",
    ]
    if tests:
        for t in tests:
            lines.append(f"- {t.get('test')}: statistic={fmt(t.get('statistic'))}, p={fmt(t.get('p_value'), 4)}, FDR={fmt(t.get('p_fdr'), 4)}, effect={fmt(t.get('effect_size'))}")
    else:
        lines.append("- No statistical tests were available.")
    lines += [
        "",
        "## Limitations",
        "This method does not prove internal reasoning, does not guarantee faithfulness, and can be affected by out-of-distribution perturbation artifacts or label-token calibration limitations. Top regions and best global transforms are selected by maximum observed effect, so local and best-global summaries can contain winner-selection bias; random controls and per-transform tables should be used to contextualize those maxima.",
        "",
        "## Safe Claims",
        "The method provides post-hoc perturbation-based diagnostic evidence about local temporal sensitivity, global prosody sensitivity, and contrastive margin shifts in Voxtral SER. It should not be described as causal proof.",
        "",
        "## Plot Captions",
        "- `plots/avg_top1_top2_local_drop_by_class.png`: class-wise Top-1 and Top-2 local sensitivity.",
        "- `plots/avg_global_drop_by_class_transform.png`: global prosody sensitivity by class and transform.",
        "- `plots/behavior_distribution_by_class.png`: eight-category behavior taxonomy distribution.",
        "- `plots/local_random_global_comparison_by_class.png`: local/random/global comparison by class.",
        "- `plots/correct_vs_wrong_drops.png`: fragility comparison for correct and wrong predictions.",
        "- `plots/confusion_matrix_selected_samples.png`: selected-sample ground-truth by predicted-class matrix.",
    ]
    Path(out_dir, "paper_ready_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_output_file_index(out_dir, args):
    lines = [
        "# Output File Index",
        "",
        "This run uses historical filenames containing `xai_200` for streaming/resume compatibility. In this final configuration, the intended selected sample budget is controlled by `--max_samples`, not by the filename.",
        "",
        "## Core Tables",
        "- `selected_samples_manifest.csv`: selected rows before perturbation scoring, including class and correct/wrong sampling reason.",
        "- `selected_sampling_balance_summary.csv`: reviewer-friendly predicted-class sample counts, correct/wrong counts, and ground-truth composition.",
        "- `confusion_matrix_selected_samples.csv`: ground-truth by predicted-class confusion matrix for the selected explained samples.",
        "- `xai_200_sample_rows.csv`: one completed explanation-statistics row per sample.",
        "- `xai_200_variant_rows.csv`: one row per scored perturbation variant.",
        "- `xai_200_evidence.jsonl`: per-sample evidence payloads for audit and reproducibility.",
        "",
        "## Aggregates And Reports",
        "- `paper_ready_results.md`: main report text with sampling, method, results, limitations, and plot captions.",
        "- `class_dominant_patterns.md`: per-predicted-class behavior-pattern interpretation.",
        "- `class_local_statistics.csv`: local temporal evidence statistics by predicted class.",
        "- `class_random_statistics.csv`: random-control comparison by predicted class.",
        "- `class_global_statistics.csv`: global prosody transform statistics by predicted class.",
        "- `class_behavior_distribution.csv`: behavior taxonomy counts and percentages by predicted class.",
        "- `correct_vs_wrong_statistics.csv`: correct/wrong fragility comparison.",
        "- `statistical_tests.csv`: nonparametric tests and FDR-adjusted p-values where available.",
        "- `quality_checks.json`: output completeness and numeric sanity checks.",
        "- `reproducibility.json`: model, adapter, perturbation, and environment metadata.",
        "",
        "## Main Plots",
        "- `plots/confusion_matrix_selected_samples.png`",
        "- `plots/avg_top1_top2_local_drop_by_class.png`",
        "- `plots/avg_global_drop_by_class_transform.png`",
        "- `plots/behavior_distribution_by_class.png`",
        "- `plots/local_random_global_comparison_by_class.png`",
        "- `plots/correct_vs_wrong_drops.png`",
        "- `plots/best_global_transform_counts_by_class.png`",
        "- `plots/global_prosody_type_distribution_by_class.png`",
        "",
        f"Configured max samples: {args.max_samples}",
        f"Requested correct/wrong per predicted class: {getattr(args, 'correct_per_class', None)}/{getattr(args, 'wrong_per_class', None)}",
    ]
    Path(out_dir, "output_file_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_novelty_assessment(out_dir):
    text = """# Novelty Assessment

This is not a new XAI theory. It is an applied XAI contribution built around a compact statistical diagnosis protocol for Voxtral-based speech emotion recognition.

The novelty is the local/random/global/contrastive diagnosis for Voxtral SER: Top-2 temporal attenuation, random same-duration controls, whole-utterance prosody transforms, contrastive emotion margins, class-level behavior statistics, and an eight-category behavior taxonomy.

Voxtral differs from Wav2Vec2/XLS-R encoder models because it is an audio-language generative model scored through label-token likelihoods. That makes calibration and contrastive margin reporting especially important.

Top-2 improves over Top-1 by allowing short distributed evidence to be measured without expanding into heavy per-sample storytelling. Class-level statistics improve over sample-only reports by supporting means, confidence intervals, flip rates, behavior proportions, and correct-vs-wrong comparisons.

This compact statistics pipeline deliberately does not run every professional per-sample perturbation. It omits deletion/insertion curves and localized acoustic counterfactual grids to keep larger class-level experiments computationally feasible. The omitted components remain valuable for qualitative case studies and can be reported separately as professional examples.

For Q1-level publication, inspect statistical stability, run ablations over segment duration and perturbation strength, compare against at least one established audio XAI baseline, and avoid causal claims.
"""
    Path(out_dir, "novelty_assessment.md").write_text(text, encoding="utf-8")


def write_reproducibility(out_dir, args):
    versions = {
        "Python": sys.version,
        "torch": getattr(torch, "__version__", None) if torch is not None else None,
        "transformers": getattr(transformers, "__version__", None) if transformers is not None else None,
        "librosa": getattr(librosa, "__version__", None) if librosa is not None else None,
        "numpy": np.__version__,
    }
    obj = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "command_line": sys.argv,
        "seed": args.seed,
        "model_path": args.base_model,
        "adapter_path": args.adapter_dir,
        "prediction_path": args.pred_jsonl or args.pred_csv,
        "global_transforms": [x.strip() for x in args.global_transforms.split(",") if x.strip()],
        "sampling": {
            "class_balance_mode": args.class_balance_mode,
            "max_samples": args.max_samples,
            "correct_per_class": getattr(args, "correct_per_class", None),
            "wrong_per_class": getattr(args, "wrong_per_class", None),
        },
        "perturbation_inventory": {
            "included": [
                "speech_aware_temporal_occlusion_all_candidate_segments",
                "top1_temporal_attenuation",
                "top2_temporal_attenuation",
                "combined_top1_top2_temporal_attenuation",
                "random_same_duration_temporal_attenuation_controls",
                "whole_utterance_global_prosody_transforms",
            ],
            "not_included_in_stats_run": [
                "deletion_curve",
                "insertion_or_keep_only_curve",
                "localized_acoustic_counterfactual_transform_grid",
                "per_sample_professional_html_report",
            ],
        },
        "perturbation_parameters": {
            "segment_sec": args.segment_sec,
            "max_segments": args.max_segments,
            "top_k": args.top_k,
            "random_n": args.random_n,
            "mask_mode": args.mask_mode,
            "attenuate_factor": args.attenuate_factor,
        },
        "thresholds": {"prob": {"strong": 0.15, "moderate": 0.05, "weak": 0.01}, "margin": {"strong": 0.25, "moderate": 0.10, "weak": 0.03}},
        "versions": versions,
    }
    Path(out_dir, "reproducibility.json").write_text(json.dumps(jsonable(obj), indent=2), encoding="utf-8")


def write_quality_checks(out_dir, sample_rows, variant_rows, selected_rows):
    expected = [
        "selected_samples_manifest.csv", "xai_200_sample_rows.csv", "xai_200_variant_rows.csv",
        "xai_200_evidence.jsonl", "class_local_statistics.csv", "class_random_statistics.csv",
        "class_global_statistics.csv", "class_behavior_distribution.csv",
        "correct_vs_wrong_statistics.csv", "class_dominant_patterns.md",
        "xai_200_aggregate.json", "paper_ready_results.md", "novelty_assessment.md",
        "statistical_tests.csv", "reproducibility.json", "selected_sampling_balance_summary.csv",
        "confusion_matrix_selected_samples.csv", "output_file_index.md",
    ]
    checks = []
    def add(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})
    add("sample_rows_exist", len(sample_rows) > 0, f"n={len(sample_rows)}")
    add("variant_rows_exist", len(variant_rows) > 0, f"n={len(variant_rows)}")
    add("jsonl_exists", os.path.isfile(os.path.join(out_dir, "xai_200_evidence.jsonl")))
    add("every_sample_has_behavior_type", all(r.get("behavior_type") for r in sample_rows))
    add("every_sample_has_best_fields", all(r.get("best_local_prob_drop") is not None and r.get("best_global_transform") for r in sample_rows))
    key_files = [os.path.join(out_dir, f) for f in expected]
    missing = [f for f in key_files if not os.path.isfile(f)]
    add("all_expected_files_exist", not missing, missing[:5])
    classes = Counter(r.get("prediction") for r in sample_rows)
    add("classes_represented_if_possible", bool(classes), dict(classes))
    add("selected_sample_count", len(sample_rows) <= len(selected_rows), f"explained={len(sample_rows)} selected={len(selected_rows)}")
    if selected_rows:
        selected_by_class = Counter(r.get("prediction") for r in selected_rows)
        add("selected_predicted_class_balance", len(set(selected_by_class.values())) <= 2, dict(selected_by_class))
    if sample_rows:
        sample_by_class = Counter(r.get("prediction") for r in sample_rows)
        add("explained_predicted_classes_present", all(sample_by_class.get(cls, 0) > 0 for cls in EMOS), dict(sample_by_class))
    add("resume_metadata_exists", os.path.isfile(os.path.join(out_dir, "progress_state.json")))
    nan_issues = []
    for r in sample_rows:
        for k in ["best_local_prob_drop", "best_global_prob_drop"]:
            v = finite_float(r.get(k))
            if r.get(k) is not None and v is None:
                nan_issues.append((r.get("sample_index"), k, r.get(k)))
    add("no_nan_inf_key_fields", not nan_issues, nan_issues[:5])
    quality = {
        "overall_passed": all(c["passed"] for c in checks),
        "checks": checks,
        "missing_files": missing,
        "notes": ["Checks validate output completeness and numeric sanity, not explanation faithfulness."],
    }
    Path(out_dir, "quality_checks.json").write_text(json.dumps(jsonable(quality), indent=2), encoding="utf-8")
    return quality


def fmt(x, nd=3):
    v = finite_float(x)
    if v is None:
        return "NA"
    return f"{v:.{nd}f}"


def parse_args():
    ap = argparse.ArgumentParser(description="Q1-style Top-2 local/random/global statistical XAI for Voxtral SER")
    ap.add_argument("--pred_jsonl", default=None)
    ap.add_argument("--pred_csv", default=None)
    ap.add_argument("--audio_root", default=None)
    ap.add_argument("--dataset_name", default=None)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--select_mode", choices=["first", "random", "wrong_then_correct"], default="wrong_then_correct")
    ap.add_argument("--class_balance_mode", choices=["none", "predicted_class_balanced", "predicted_class_and_correctness_balanced", "predicted_class_fixed_correct_wrong"], default="predicted_class_balanced")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--wrong_samples", type=int, default=50)
    ap.add_argument("--correct_per_class", type=int, default=40)
    ap.add_argument("--wrong_per_class", type=int, default=10)
    ap.add_argument("--only_wrong", action="store_true")
    ap.add_argument("--only_correct", action="store_true")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--segment_sec", type=float, default=0.5)
    ap.add_argument("--max_segments", type=int, default=10)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--random_n", type=int, default=3)
    ap.add_argument("--mask_mode", choices=["zero", "attenuate", "mean"], default="attenuate")
    ap.add_argument("--attenuate_factor", type=float, default=0.2)
    ap.add_argument("--global_transforms", default=",".join(DEFAULT_GLOBAL_TRANSFORMS))
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--rescore_original", action="store_true")
    ap.add_argument("--use_rescored_original_probs", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry_run_samples", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    load_professional_primitives()
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

    rows = load_prediction_rows_for_stats(args)
    selected = select_rows_class_aware(rows, args)
    if args.dry_run_samples and args.dry_run_samples > 0:
        selected = selected[:args.dry_run_samples]
    if not selected:
        raise RuntimeError("No samples selected.")
    write_manifest(selected, args.out_dir)
    print(f"Selected {len(selected)} samples. Distribution: {dict(Counter(r['prediction'] for r in selected))}", flush=True)

    sample_csv = os.path.join(args.out_dir, "xai_200_sample_rows.csv")
    variant_csv = os.path.join(args.out_dir, "xai_200_variant_rows.csv")
    evidence_jsonl = os.path.join(args.out_dir, "xai_200_evidence.jsonl")
    done = read_done_keys(sample_csv) if args.resume else set()

    print("Loading Voxtral model for perturbed re-scoring...", flush=True)
    model, processor = load_model_and_processor(args)
    scorer = BatchScorer(model, processor, args.batch_size)
    print("Model loaded.", flush=True)

    for i, row in enumerate(selected):
        key = (i, int(row["source_index"]))
        if (args.resume or args.skip_existing) and key in done:
            print(f"[SKIP] sample {i} source={row['source_index']} already completed", flush=True)
            continue
        print(f"[{i + 1}/{len(selected)}] {row['dataset']} source={row['source_index']} pred={row['prediction']} gt={row.get('ground_truth')}", flush=True)
        try:
            sample_row, variant_rows, evidence = explain_one_sample_stats(model, processor, row, i, args, scorer)
            append_dict_row(sample_csv, SAMPLE_FIELDS, sample_row)
            for vr in variant_rows:
                append_dict_row(variant_csv, VARIANT_FIELDS, vr)
            append_jsonl(evidence_jsonl, evidence)
            progress = {
                "last_completed_sample_index": i,
                "last_completed_source_index": row["source_index"],
                "completed_count": len(read_done_keys(sample_csv)),
                "timestamp": datetime.now().isoformat(),
            }
            Path(args.out_dir, "progress_state.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[ERROR] sample {i} failed: {type(e).__name__}: {e}", flush=True)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate, quality = compute_aggregate_outputs(args.out_dir, args, selected)
    print("\nDONE.", flush=True)
    print("Sample rows:", sample_csv)
    print("Variant rows:", variant_csv)
    print("Aggregate:", os.path.join(args.out_dir, "xai_200_aggregate.json"))
    print("Paper-ready report:", os.path.join(args.out_dir, "paper_ready_results.md"))
    print("Quality checks:", os.path.join(args.out_dir, "quality_checks.json"))
    print("Quality passed:", quality.get("overall_passed"))


if __name__ == "__main__":
    main()
