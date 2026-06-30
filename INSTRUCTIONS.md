# Project Instructions

This runbook describes how to reproduce the LG-ProXAI project workflow from raw
dataset placement through DoRA adaptation, evaluation, and perturbation
diagnostics.

## 1. Prepare the Environment

Install the Python dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use a CUDA GPU for realistic training and scoring time. If `--load_in_4bit` is
used, verify that PyTorch, CUDA, and `bitsandbytes` are compatible on the host.

## 2. Place Datasets Locally

Raw audio is excluded from Git. Place the datasets here:

```text
data/raw/ESD/
data/raw/IEMOCAP/
```

The scripts resolve audio through EmoBox metadata under `src/EmoBox/data/`.
Before running model commands, confirm that metadata `wav` paths resolve from
the chosen `--audio_root`.

Main label policy:

- ESD: English speakers `0011`-`0020`; use Angry, Happy, Sad, Neutral.
- IEMOCAP: use Angry, Happy, Sad, Neutral; merge Excited into Happy when
  required for four-class alignment.
- ESD Surprise is excluded from the adapted four-class setup.

## 3. Audit Data and Zero-Shot Baselines

Run the notebooks in order:

```text
script/01_data_exploration.ipynb
script/02_zero_shot_esd.ipynb
script/02_zero_shot_iemocap.ipynb
```

Record the fold index, dataset source, access date, label mapping, and any path
fixes made during notebook execution.

## 4. Train the DoRA Adapter

Reference command:

```bash
python script/03_train_dora16a32_voxtral_esd.py \
  --meta_dir src/EmoBox/data \
  --audio_root data/raw/ESD \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --output_dir outputs/esd_dora16a32_fold2 \
  --fold 2 \
  --seed 42 \
  --val_per_class 100 \
  --epochs 3 \
  --lr 1e-4 \
  --train_bs 1 \
  --eval_bs 1 \
  --grad_accum 16 \
  --load_in_4bit
```

Training configuration to preserve in the experiment log:

- DoRA rank 16, alpha 32, dropout 0.1
- target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`
- cosine schedule and mixed precision as configured by the script
- validation loss and early stopping decision
- final adapter directory, usually `outputs/esd_dora16a32_fold2/final_adapter/`

## 5. Evaluate Predictive Performance

ESD in-domain evaluation:

```bash
python script/04_eval_dora16a32_esd.py \
  --meta_dir src/EmoBox/data \
  --audio_root data/raw/ESD \
  --base_model mistralai/Voxtral-Mini-3B-2507 \
  --adapter_dir outputs/esd_dora16a32_fold2/final_adapter \
  --out_root eval_reports/esd \
  --fold 2 \
  --load_in_4bit
```

IEMOCAP cross-corpus evaluation:

```bash
python script/04_eval_dora16a32_iemocap.py \
  --meta_dir src/EmoBox/data \
  --audio_root data/raw/IEMOCAP \
  --base_model mistralai/Voxtral-Mini-3B-2507 \
  --adapter_dir outputs/esd_dora16a32_fold2/final_adapter \
  --out_root eval_reports/iemocap \
  --fold 1 \
  --load_in_4bit
```

Preserve prediction files because LG-ProXAI consumes them. Report at least
accuracy, balanced accuracy, macro-F1, MCC, Cohen's kappa, confusion matrices,
and calibration or selective-risk summaries when generated.

## 6. Run LG-ProXAI Diagnostics

Use audio-only prediction files for the main report interpretation so the
diagnostic targets acoustic evidence rather than transcript content.

Qualitative 20-sample analysis with per-sample reports:

```bash
python script/05_post_hoc_explainable_20_sample.py \
  --pred_jsonl eval_reports/esd/fold_2/dora16a32/audio_only/predictions.jsonl \
  --dataset_name ESD \
  --base_model mistralai/Voxtral-Mini-3B-2507 \
  --adapter_dir outputs/esd_dora16a32_fold2/final_adapter \
  --out_dir eval_reports/lg_proxai_20_esd \
  --load_in_4bit \
  --max_samples 20 \
  --segment_sec 0.2 \
  --top_k 3 \
  --random_n 10
```

Class-level 1,000-sample statistics:

```bash
python script/05_post_hoc_explainable_1000_sample.py \
  --pred_jsonl eval_reports/esd/fold_2/dora16a32/audio_only/predictions.jsonl \
  --dataset_name ESD \
  --base_model mistralai/Voxtral-Mini-3B-2507 \
  --adapter_dir outputs/esd_dora16a32_fold2/final_adapter \
  --out_dir eval_reports/lg_proxai_1000_esd \
  --load_in_4bit \
  --class_balance_mode predicted_class_balanced \
  --max_samples 1000 \
  --segment_sec 0.5 \
  --top_k 2 \
  --random_n 3 \
  --batch_size 2
```

Run the same 1,000-sample diagnostic for IEMOCAP by replacing the prediction
file, `--dataset_name`, and `--out_dir`.

## 7. Interpret Results Safely

Use these terms consistently:

- **forced-choice label score**: normalized score over fixed candidate labels;
  not a calibrated probability.
- **local score drop**: score decrease after attenuating selected temporal
  regions.
- **random control drop**: score decrease from same-duration random regions.
- **maximum observed global drop**: largest score decrease among the tested
  whole-utterance transformations.
- **mixed local-global sensitivity**: selected local evidence is stronger than
  random controls, while the strongest global perturbation is larger than the
  local effect.

Avoid these claims:

- the highlighted region is the only reason for the prediction
- the perturbation proves the model's internal causal mechanism
- the model identifies a speaker's true emotional state
- forced-choice scores are calibrated emotion probabilities

## 8. Keep Outputs Organized

Recommended generated directories:

```text
outputs/
eval_reports/
artifacts/
runs/
```

Do not commit raw audio, checkpoints, generated prediction files, temporary
perturbed audio, or large report exports unless they are intentionally part of a
submission package.

## 9. Minimum Experiment Log

For every number in a report, record:

```text
Experiment ID:
Date:
Repository commit:
Dataset and source:
Dataset access date:
Fold:
Label mapping:
Base model:
Adapter path:
Command:
Seed:
Hardware:
CUDA/PyTorch/transformers/peft versions:
Prediction file:
Metric file:
LG-ProXAI output directory:
Known limitations:
```
