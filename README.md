# Explainable Speech Emotion Recognition with Voxtral and DoRA

This repository contains an academic research workflow for speech emotion
recognition (SER) with a multimodal large language model. The project studies
how a Voxtral-mini-3B model can be adapted to emotion classification with
parameter-efficient fine-tuning, evaluated across audio-only and audio-plus-text
settings, and interpreted with post-hoc counterfactual explanations.

The implementation focuses on four target labels:

- Angry
- Happy
- Sad
- Neutral

## Research Objective

The main objective is to build and evaluate an explainable SER pipeline that:

1. Uses zero-shot baselines on ESD and IEMOCAP.
2. Fine-tunes Voxtral-mini-3B on ESD with DoRA/LoRA adapters.
3. Evaluates model performance, calibration, latency, and modality effects.
4. Produces localized counterfactual explanations for selected predictions.

This framing makes the repository suitable for a course project, thesis
experiment, or reproducible research submission.

## Repository Structure

```text
.
|-- data/
|   |-- README.md
|   `-- raw/
|       |-- ESD/
|       `-- IEMOCAP/
|-- docs/
|   `-- REPRODUCIBILITY.md
|-- script/
|   |-- README.md
|   |-- 01_data_exploration.ipynb
|   |-- 02_zero_shot_esd.ipynb
|   |-- 02_zero_shot_iemocap.ipynb
|   |-- 03_train_dora16a32_voxtral_esd.py
|   |-- 04_eval_dora16a32_esd.py
|   |-- 04_eval_dora16a32_iemocap.py
|   |-- 05_post_hoc_explainable_20_sample.py
|   `-- 05_post_hoc_explainable_1000_sample.py
|-- src/
|   |-- README.md
|   `-- EmoBox/
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Method Summary

### Data

The project uses two speech emotion datasets:

- ESD: Emotional Speech Dataset
- IEMOCAP: Interactive Emotional Dyadic Motion Capture dataset

Raw audio files are not included because of size and licensing constraints. The
expected local placement is documented in [data/README.md](data/README.md).

### Fine-Tuning

The training script adapts Voxtral-mini-3B using PEFT with DoRA enabled:

- adapter rank: 16
- LoRA alpha: 32
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- optional 4-bit loading through QLoRA
- stratified validation split by emotion class
- English-speaker filtering for ESD speakers 0011-0020

The script uses a user-only multimodal chat template and manually appends label
tokens so that loss is computed only on the target emotion label.

### Evaluation

The evaluation scripts report:

- accuracy and balanced accuracy
- macro and weighted F1
- Matthews correlation coefficient
- Cohen's kappa
- confusion matrices
- expected calibration error
- selective accuracy/risk coverage
- inference latency percentiles
- McNemar tests for paired modality comparisons

### Explainability

The post-hoc explainability scripts read previous prediction files and generate
localized counterfactual analyses. They identify influential temporal regions
with occlusion, perturb the most influential segments, and compare local
counterfactual effects with random-region and whole-utterance perturbations.

## Environment Setup

Create and activate a Python environment before installing dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU execution is strongly recommended. Quantized training and inference require
compatible CUDA, PyTorch, and `bitsandbytes` installations.

## Data Preparation

Place datasets under:

```text
data/raw/ESD/
data/raw/IEMOCAP/
```

The scripts expect EmoBox-style metadata under `src/EmoBox/data`. For example,
ESD fold metadata is expected at:

```text
src/EmoBox/data/esd/fold_2/esd_train_fold_2.jsonl
src/EmoBox/data/esd/fold_2/esd_test_fold_2.jsonl
```

Depending on your local dataset layout, `--audio_root` should point to the
directory from which the `wav` paths in the metadata can be resolved.

## Running the Workflow

### 1. Explore the Data

Open and run:

```text
script/01_data_exploration.ipynb
```

### 2. Run Zero-Shot Baselines

Open and run:

```text
script/02_zero_shot_esd.ipynb
script/02_zero_shot_iemocap.ipynb
```

### 3. Fine-Tune on ESD

```bash
python script/03_train_dora16a32_voxtral_esd.py \
  --meta_dir src/EmoBox/data \
  --audio_root data/raw/ESD \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --output_dir outputs/esd_dora16a32_fold2 \
  --fold 2 \
  --epochs 3 \
  --train_bs 1 \
  --eval_bs 1 \
  --grad_accum 16 \
  --load_in_4bit
```

The final adapter is saved under:

```text
outputs/esd_dora16a32_fold2/final_adapter/
```

### 4. Evaluate on ESD

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

### 5. Evaluate on IEMOCAP

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

### 6. Run Post-Hoc Explainability

Use a prediction file produced by an evaluation run:

```bash
python script/05_post_hoc_explainable_20_sample.py \
  --pred_jsonl eval_reports/esd/fold_2/dora16a32/audio_only/predictions.jsonl \
  --base_model mistralai/Voxtral-Mini-3B-2507 \
  --adapter_dir outputs/esd_dora16a32_fold2/final_adapter \
  --out_dir eval_reports/local_cf_xai \
  --load_in_4bit \
  --max_samples 20
```

If your evaluation output is CSV rather than JSONL, use the corresponding
argument supported by the explainability script.

## Expected Outputs

Typical generated outputs include:

- adapter checkpoints and final PEFT adapter weights
- prediction CSV/JSONL files
- metric summaries in JSON/CSV format
- confusion matrices
- selective accuracy curves
- modality comparison reports
- localized counterfactual explanation plots and HTML reports

Generated outputs are intentionally ignored by Git. Keep only code,
documentation, metadata, and small reproducibility artifacts in version control.

## Reproducibility Notes

For each reported experiment, record:

- dataset version and access date
- model identifier or local checkpoint hash
- random seed
- fold index
- command-line arguments
- GPU type and CUDA/PyTorch versions
- generated metrics and prediction files

More detail is provided in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## External Code

The `src/EmoBox` directory contains the EmoBox framework used for dataset
metadata and evaluation support. It is external source code included for
reproducibility. See [src/README.md](src/README.md) and the upstream EmoBox
README for details.

## Ethical and Academic Considerations

Speech emotion recognition is sensitive because predictions may be affected by
speaker identity, language, recording quality, acted versus natural emotion, and
annotation subjectivity. Results should be interpreted as dataset-conditioned
classification performance, not as reliable inference of a person's internal
state. Report limitations, failure cases, and subgroup analyses where possible.

## Citation

If this repository is used in a report or thesis, cite the datasets, Voxtral,
PEFT/LoRA/DoRA methods, and EmoBox according to their official citation
instructions. Add a project-specific citation once the final report metadata is
available.
