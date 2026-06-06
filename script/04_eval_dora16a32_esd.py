#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: Voxtral-mini-3b DoRA-16/32 Evaluation Framework (ESD Test Fold)
==========================================================================

Abstract:
    This script performs a rigorous evaluation of fine-tuned PEFT adapters on the 
    Emotional Speech Dataset (ESD) test splits. It goes beyond standard accuracy metrics 
    by incorporating uncertainty quantification and statistical significance testing.

    Key Methodologies:
    1.  **Multimodal Ablation**: Evaluates models in two regimes: 'Audio-Only' vs. 
        'Audio + Text Transcript' to quantify the marginal gain of linguistic context.
    2.  **Uncertainty Quantification**: Computes Expected Calibration Error (ECE) and 
        Selective Accuracy (Risk-Coverage curves) to measure model reliability.
    3.  **Statistical Significance**: Applies McNemar's Test (paired non-parametric) 
        to determine if performance differences between adapters (or modalities) are 
        statistically significant ($p < 0.05$).
    4.  **Operational Metrics**: Tracks inference latency percentiles (p50, p90, p99) 
        crucial for real-time deployment.

Usage:
    Run via command line pointing to the test set metadata and the directory containing 
    fine-tuned adapters.
"""

import os, re, json, time, argparse, glob
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoProcessor, VoxtralForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
)

# Disable parallelism to prevent deadlocks in high-throughput inference environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMOS = ["Angry", "Happy", "Sad", "Neutral"]
KEEP = set(EMOS)
EMOS_LOWER = set([e.lower() for e in EMOS])

# -------------------------
# Utility Functions
# -------------------------

def to_abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def norm_emo(e: str) -> str:
    return (e or "").strip().capitalize()

def resolve_wav_path(audio_root: str, wav_field: str) -> Optional[str]:
    """
    Robustly resolves relative audio paths against a root directory, handling
    potential OS path separator mismatches.
    """
    wav_field = (wav_field or "").replace("\\", "/")
    if not wav_field:
        return None
    if os.path.isabs(wav_field) and os.path.isfile(wav_field):
        return wav_field
    c1 = os.path.join(audio_root, wav_field)
    if os.path.isfile(c1):
        return c1
    parent = os.path.dirname(audio_root.rstrip("/"))
    c2 = os.path.join(parent, wav_field)
    if os.path.isfile(c2):
        return c2
    return None

def speaker_id_from_esd_rel(wav_rel_or_abs: str) -> str:
    """Extracts speaker ID (e.g., '0011') from standard ESD file paths."""
    s = (wav_rel_or_abs or "").replace("\\", "/")
    m = re.search(r"downloads/esd/(\d{4})/", s)
    if m:
        return m.group(1)
    base = os.path.basename(s)
    m2 = re.match(r"^(\d{4})_", base)
    return m2.group(1) if m2 else "Unknown"

def utt_id_from_path(wav_path: str) -> Optional[str]:
    """Extracts unique utterance ID for merging results later."""
    base = os.path.basename(wav_path)
    m = re.match(r"(\d{4}_\d{6})\.wav$", base)
    return m.group(1) if m else None

def extract_label_from_text(txt: str) -> Optional[str]:
    """
    Heuristic Parsing: Extracts the classification label from the model's 
    free-text generation.
    Strategy:
    1. Exact word boundary search (Regex).
    2. Fallback substring search (nearest position).
    3. Default to 'Neutral' if parsing fails.
    """
    if txt is None:
        return "Neutral"
    t = txt.strip().lower().replace("\n", " ").replace("\t", " ")
    if not t:
        return "Neutral"
    
    # 1. Precise word boundary check
    for emo in EMOS:
        if re.search(rf"\b{emo.lower()}\b", t):
            return emo
            
    # 2. Fallback substring search
    best, best_pos = "Neutral", 10**9
    for emo in EMOS:
        p = t.find(emo.lower())
        if p != -1 and p < best_pos:
            best, best_pos = emo, p
            
    return best if best_pos < 10**9 else "Neutral"

def safe_tag_from_path(p: str) -> str:
    p = os.path.normpath(p)
    if os.path.basename(p) == "final_adapter":
        return os.path.basename(os.path.dirname(p))
    return os.path.basename(p)

# -------------------------
# Transcript Parsing (ESD Specific)
# -------------------------
def read_esd_transcript(audio_root: str, wav_abs: str) -> Optional[str]:
    """
    Locates and parses the ground-truth text transcript for a given audio file
    within the ESD directory structure.
    """
    spk = speaker_id_from_esd_rel(wav_abs)
    utt = utt_id_from_path(wav_abs)
    if not spk or not utt:
        return None

    # ESD directory structures vary; try multiple common locations
    paths_to_try = [
        os.path.join(audio_root, "downloads", "esd", spk, f"{spk}.txt"),
        os.path.join(os.path.dirname(audio_root.rstrip("/")), "downloads", "esd", spk, f"{spk}.txt"),
        os.path.join(audio_root, spk, f"{spk}.txt"),
    ]

    txt_path = None
    for p in paths_to_try:
        if os.path.isfile(p):
            txt_path = p
            break
    if not txt_path:
        return None

    utt_re = re.compile(rf"^{re.escape(utt)}\b")

    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if not utt_re.match(line): continue

                parts = line.split()
                if len(parts) < 2: return None

                # ESD format: UTT_ID [Transcript Words] [EmotionLabel]
                # We extract the middle part as the transcript.
                last = parts[-1].strip().lower()
                if last in EMOS_LOWER and len(parts) >= 3:
                    content_parts = parts[1:-1]
                else:
                    content_parts = parts[1:]

                mid = " ".join(content_parts).strip()
                return mid if mid else None
    except Exception:
        return None
    return None

# -------------------------
# Data Loading
# -------------------------
def load_esd_test_rows(meta_dir: str, audio_root: str, fold: int) -> Tuple[str, List[dict], int]:
    """Loads the specific JSONL fold file designed for testing."""
    jsonl = os.path.join(meta_dir, "esd", f"fold_{fold}", f"esd_test_fold_{fold}.jsonl")
    rows, missing = [], 0

    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)

            emo = norm_emo(ex.get("emo", ex.get("label", "")))
            if emo not in KEEP: continue

            wav_rel = (ex.get("wav", "") or "").replace("\\", "/")
            wav_abs = resolve_wav_path(audio_root, wav_rel)
            if not wav_abs:
                missing += 1
                continue

            length = ex.get("length", None)
            spk = speaker_id_from_esd_rel(wav_rel)

            rows.append({
                "utt_id": utt_id_from_path(wav_abs),
                "wav": wav_abs,
                "label": emo,
                "length_sec": float(length) if length is not None else float("nan"),
                "speaker": str(spk),
            })

    return jsonl, rows, missing

# -------------------------
# Adapter Discovery & Model Loading
# -------------------------
def resolve_adapter_dir(p: str) -> Optional[str]:
    p = to_abs(p)
    if os.path.isfile(os.path.join(p, "adapter_config.json")):
        return p
    fa = os.path.join(p, "final_adapter")
    if os.path.isfile(os.path.join(fa, "adapter_config.json")):
        return fa
    return None

def discover_adapters(adapters_root: str) -> List[str]:
    adapters_root = to_abs(adapters_root)
    out = []
    for d in sorted(glob.glob(os.path.join(adapters_root, "*"))):
        if not os.path.isdir(d): continue
        rd = resolve_adapter_dir(d)
        if rd is not None:
            out.append(rd)
    return out

def load_base_and_processor(base_model: str, load_in_4bit: bool):
    """Loads the base Voxtral model (Mistral-Audio) and Processor."""
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = VoxtralForConditionalGeneration.from_pretrained(
            base_model,
            trust_remote_code=True,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation="sdpa",
        )
    else:
        try:
            base = VoxtralForConditionalGeneration.from_pretrained(
                base_model, trust_remote_code=True, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
            )
        except TypeError:
            base = VoxtralForConditionalGeneration.from_pretrained(
                base_model, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
            )
    return processor, base

def wrap_with_adapter(base, adapter_dir: str):
    """Injects LoRA/DoRA weights into the base model."""
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass
    device = next(model.parameters()).device
    return model, device

# -------------------------
# Inference Logic
# -------------------------
def build_user_only_conversation(wav_path: str, prompt_text: str):
    """
    Constructs a 'User-Only' single-turn chat payload.
    This avoids role validation errors in mistral_common by not forcing an 'assistant' role.
    """
    wav_path = os.path.abspath(wav_path)
    if not os.path.isfile(wav_path):
        raise RuntimeError(f"Missing wav: {wav_path}")
    audio_part = {
        "type": "audio", "audio_url": {"url": wav_path},
        "content": wav_path, "path": wav_path, "url": wav_path,
    }
    return [{"role": "user", "content": [audio_part, {"type": "text", "text": prompt_text}]}]

def infer_one(processor, model, device, wav_path, prompt_text, max_new_tokens=8, do_sample=False, temperature=0.2, top_p=0.95):
    conv = build_user_only_conversation(wav_path, prompt_text)
    inputs = processor.apply_chat_template(conv, tokenize=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, return_dict_in_generate=True, output_scores=True)
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

    t0 = time.time()
    with torch.no_grad():
        gen_out = model.generate(**inputs, **gen_kwargs)
    latency_ms = (time.time() - t0) * 1000.0

    in_len = int(inputs["input_ids"].shape[1])
    seq = gen_out.sequences
    new_tokens = seq[:, in_len:]
    txt = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    pred = extract_label_from_text(txt)
    
    # Extract Confidence (Softmax probability of the first generated token)
    conf = float("nan")
    try:
        if gen_out.scores and len(gen_out.scores) > 0:
            s0 = gen_out.scores[0][0] # Score of first token for first batch item
            p0 = torch.softmax(s0.float(), dim=-1)
            chosen_id = int(seq[0, in_len].item())
            conf = float(p0[chosen_id].item())
    except Exception:
        pass

    return pred, txt, conf, latency_ms

# -------------------------
# Advanced Metrics (The "Academic" Part)
# -------------------------

def reliability_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """
    Computes Expected Calibration Error (ECE).
    Measures how well the model's confidence scores align with its actual accuracy.
    Lower ECE = Better Calibration (more trustworthy probabilities).
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    m = np.isfinite(conf)
    conf, correct = conf[m], correct[m]
    if len(conf) == 0:
        return float("nan"), np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(conf, bins) - 1
    ids = np.clip(ids, 0, n_bins - 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_cnt = np.zeros(n_bins)

    for b in range(n_bins):
        mb = (ids == b)
        if mb.sum() > 0:
            bin_acc[b] = correct[mb].mean()
            bin_conf[b] = conf[mb].mean()
            bin_cnt[b] = mb.sum()

    N = max(1, len(conf))
    ece = float(np.sum((bin_cnt / N) * np.abs(bin_acc - bin_conf)))
    return ece, bin_acc, bin_conf, bin_cnt

def selective_accuracy_curve(conf: np.ndarray, correct: np.ndarray, n_points: int = 20):
    """
    Computes Selective Accuracy (Risk-Coverage).
    Simulates a scenario where the model abstains from prediction when confidence is low.
    'risk' (Area Under Risk-Coverage Curve) should be low.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    m = np.isfinite(conf)
    conf, correct = conf[m], correct[m]
    if len(conf) == 0:
        return [], [], float("nan")

    order = np.argsort(-conf)
    correct_sorted = correct[order]
    coverages = np.linspace(0.1, 1.0, n_points)
    cov_list, acc_list = [], []
    for c in coverages:
        k = max(1, int(round(c * len(correct_sorted))))
        acc = float(correct_sorted[:k].mean())
        cov_list.append(float(k / len(correct_sorted)))
        acc_list.append(acc)

    risk = np.trapz([1.0 - a for a in acc_list], x=cov_list)
    return cov_list, acc_list, float(risk)

def mcnemar_from_two_preds(y_true, y_pred_a, y_pred_b):
    """
    Performs McNemar's Test on paired nominal data.
    Used to determine if the disagreement between two models (Model A vs Model B) 
    is statistically significant.
    Result: 'chi_sq_stat' > 3.841 implies p < 0.05.
    """
    y_true = np.asarray(y_true)
    a = (np.asarray(y_pred_a) == y_true)
    b = (np.asarray(y_pred_b) == y_true)
    n01 = int((~a & b).sum()) # A wrong, B right
    n10 = int((a & ~b).sum()) # A right, B wrong
    denom = n01 + n10
    stat = ((abs(n01 - n10) - 1) ** 2) / denom if denom > 0 else 0.0
    return {
        "n01_a_wrong_b_right": n01, "n10_a_right_b_wrong": n10,
        "total_divergent": denom, "chi_sq_stat": stat,
        "significant_p05": stat > 3.841
    }

def compute_metrics_extended(y_true, y_pred, conf, latencies):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "num_samples": len(y_true),
    }
    
    # Latency Stats
    lat = np.array(latencies)
    lat = lat[np.isfinite(lat)]
    if len(lat) > 0:
        out["latency_ms_p50"] = float(np.percentile(lat, 50))
        out["latency_ms_p90"] = float(np.percentile(lat, 90))
        out["latency_ms_p99"] = float(np.percentile(lat, 99))

    # ECE & Risk
    correct = (np.array(y_true) == np.array(y_pred)).astype(int)
    ece, _, _, _ = reliability_ece(conf, correct)
    out["ece_10bins"] = ece
    _, _, risk = selective_accuracy_curve(conf, correct)
    out["selective_risk_area"] = risk
    return out

# -------------------------
# Plotting
# -------------------------
def plot_confusion(mat, labels, out_png, title, normalize=False):
    plt.figure(figsize=(7, 6))
    m = mat.astype(np.float64)
    if normalize:
        m = m / np.maximum(m.sum(axis=1, keepdims=True), 1.0)
    plt.imshow(m, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    thresh = float(np.max(m)) / 2.0 if m.size else 0.0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = f"{m[i, j]:.2f}" if normalize else f"{int(mat[i, j])}"
            col = "white" if m[i, j] > thresh else "black"
            plt.text(j, i, val, ha="center", va="center", color=col)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_bar_grouped(df, xcol, series, out_png, title, ylim=(0,1)):
    xvals = df[xcol].tolist()
    x = np.arange(len(xvals))
    w = 0.8 / max(1, len(series))
    plt.figure(figsize=(max(10, len(xvals)*0.7), 5))
    for i, s in enumerate(series):
        plt.bar(x - 0.4 + w/2 + i*w, df[s].values, width=w, label=s)
    plt.xticks(x, xvals, rotation=45, ha="right")
    plt.ylim(*ylim)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

# -------------------------
# Execution Logic
# -------------------------
def run_mode(args, mode_name, processor, model, device, rows, out_dir):
    ensure_dir(out_dir)

    base_prompt = "You are an expert at recognizing emotions from speech.\nListen to the audio and output only ONE label from:\nAngry, Happy, Sad, Neutral."
    text_prompt = "You are an expert at recognizing emotions from speech.\nListen to the audio.\nYou also have the transcript (may help disambiguate emotion):\nTRANSCRIPT: {transcript}\n\nOutput only ONE label from:\nAngry, Happy, Sad, Neutral."

    y_true, y_pred, raw_txt, conf1, lat_ms = [], [], [], [], []
    lengths, wavs, transcripts, speakers, utts = [], [], [], [], []
    found_t = 0

    for i, r in enumerate(rows, 1):
        transcript_val = None
        if mode_name == "audio_text":
            t = read_esd_transcript(args.audio_root, r["wav"])
            if t and t.strip():
                transcript_val = t.strip()
                found_t += 1

        prompt = text_prompt.format(transcript=transcript_val) if (mode_name == "audio_text" and transcript_val) else base_prompt

        pred, txt, conf, lat = infer_one(
            processor, model, device, r["wav"], prompt,
            max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
            temperature=args.temperature, top_p=args.top_p
        )

        y_true.append(r["label"])
        y_pred.append(pred)
        raw_txt.append(txt)
        conf1.append(conf)
        lat_ms.append(lat)
        lengths.append(r.get("length_sec", float("nan")))
        wavs.append(r["wav"])
        transcripts.append(transcript_val or "")
        speakers.append(r["speaker"])
        utts.append(r["utt_id"])

        if args.progress_every > 0 and (i % args.progress_every == 0):
            print(f"{mode_name}: {i}/{len(rows)} done", flush=True)

    df = pd.DataFrame({
        "utt_id": utts, "wav": wavs, "speaker": speakers,
        "true": y_true, "pred": y_pred, "raw_text": raw_txt,
        "confidence_first_token": conf1, "latency_ms": lat_ms,
        "length_sec": lengths, "transcript": transcripts,
    })
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    metrics = compute_metrics_extended(y_true, y_pred, conf1, lat_ms)
    if mode_name == "audio_text":
        metrics["transcripts_found_rate"] = float(found_t / max(1, len(rows)))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    save_helped_plots(df, out_dir, mode_name, metrics["ece_10bins"])
    return df, metrics

def save_helped_plots(df, out_dir, tag, ece):
    cm = confusion_matrix(df["true"], df["pred"], labels=EMOS)
    plot_confusion(cm, EMOS, os.path.join(out_dir, "confusion_counts.png"), f"{tag} Confusion (Counts)", normalize=False)
    plot_confusion(cm, EMOS, os.path.join(out_dir, "confusion_norm.png"), f"{tag} Confusion (Norm)", normalize=True)
    
    # Selective Curve
    correct = (df["true"] == df["pred"]).astype(int)
    cover, acc, risk = selective_accuracy_curve(df["confidence_first_token"].values, correct.values)
    if len(cover) > 0:
        plt.figure(figsize=(7, 5))
        plt.plot(cover, acc, "o-")
        plt.title(f"Selective Accuracy (Risk={risk:.3f}) - {tag}")
        plt.xlabel("Coverage"); plt.ylabel("Accuracy")
        plt.grid(alpha=0.3); plt.savefig(os.path.join(out_dir, "selective_curve.png"), dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser(description="Voxtral DoRA-16/32 Evaluation Framework - ESD")

    ap.add_argument("--meta_dir", required=True, help="Path to ESD metadata root")
    ap.add_argument("--audio_root", required=True, help="Path to audio files root")
    ap.add_argument("--base_model", required=True, help="Base model ID or local path")
    ap.add_argument("--adapter_dir", required=True, help="Specific DoRA r=16 alpha=32 adapter path or its run directory")
    ap.add_argument("--out_root", required=True, help="Output directory for this DoRA-16/32 evaluation")

    ap.add_argument("--fold", type=int, default=2, help="Test fold index")
    ap.add_argument("--load_in_4bit", action="store_true", help="Use QLoRA 4-bit loading")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--progress_every", type=int, default=200)

    args = ap.parse_args()

    # Normalize paths
    for k in ["meta_dir", "audio_root", "base_model", "adapter_dir", "out_root"]:
        setattr(args, k, to_abs(getattr(args, k)))

    adapter_dir = resolve_adapter_dir(args.adapter_dir)
    if adapter_dir is None:
        raise SystemExit(f"No valid DoRA-16/32 adapter found at: {args.adapter_dir}")

    adapter_tag = "dora16a32"
    fold_root = os.path.join(args.out_root, f"fold_{args.fold}")
    adapter_root = os.path.join(fold_root, adapter_tag)
    compare_root = os.path.join(adapter_root, "compare")
    ensure_dir(compare_root)

    # Load test data
    jsonl_path, rows, missing = load_esd_test_rows(args.meta_dir, args.audio_root, args.fold)
    print(f"=== ESD Fold{args.fold} | DoRA 16/32 | Rows: {len(rows)} | Missing: {missing} ===", flush=True)
    print(f"Adapter: {adapter_dir}", flush=True)

    # Load model once and evaluate only this adapter
    processor, base = load_base_and_processor(args.base_model, args.load_in_4bit)
    model, device = wrap_with_adapter(base, adapter_dir)

    df_audio, met_audio = run_mode(
        args, "audio_only", processor, model, device, rows,
        os.path.join(adapter_root, "audio_only")
    )
    df_text, met_text = run_mode(
        args, "audio_text", processor, model, device, rows,
        os.path.join(adapter_root, "audio_text")
    )

    # Audio-only vs audio+text comparison for the same DoRA-16/32 adapter
    mcn = mcnemar_from_two_preds(df_audio["true"], df_audio["pred"], df_text["pred"])
    with open(os.path.join(compare_root, "mcnemar_audio_vs_text.json"), "w") as f:
        json.dump(mcn, f, indent=2)

    merged = pd.merge(df_audio, df_text, on="utt_id", suffixes=("_audio", "_text"))
    merged[(merged["true_audio"] != merged["pred_audio"]) & (merged["true_text"] == merged["pred_text"])].to_csv(
        os.path.join(compare_root, "helped_by_text.csv"), index=False
    )
    merged[(merged["true_audio"] == merged["pred_audio"]) & (merged["true_text"] != merged["pred_text"])].to_csv(
        os.path.join(compare_root, "hurt_by_text.csv"), index=False
    )

    # Single-adapter leaderboard
    agg_rows = []
    for mode, met in [("audio_only", met_audio), ("audio_plus_text", met_text)]:
        row = {"adapter": adapter_tag, "mode": mode}
        row.update(met)
        if mode == "audio_plus_text":
            row.update({k: v for k, v in mcn.items() if k in ["chi_sq_stat", "significant_p05"]})
        agg_rows.append(row)

    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(adapter_root, "leaderboard_dora16a32.csv"), index=False)

    print(f"\n=== DONE: ESD DoRA 16/32 evaluation saved to {adapter_root} ===", flush=True)

if __name__ == "__main__":
    main()
