# Explainable Speech Emotion Recognition with Voxtral and DoRA

This repository contains an academic research workflow for speech emotion
recognition (SER) with a multimodal large language model. It studies whether
Voxtral-Mini-3B can be adapted to four-way emotion classification with
parameter-efficient fine-tuning, evaluated across audio-only and audio-plus-text
settings, and interpreted with localized post-hoc counterfactual explanations.

The target label set is:

- Angry
- Happy
- Sad
- Neutral

## Abstract

Speech emotion recognition is a high-variance task because labels depend on
speaker identity, language, acted versus conversational speech, recording
conditions, and annotator subjectivity. This project therefore treats SER as a
dataset-conditioned classification problem rather than as direct inference of a
speaker's internal emotional state. The workflow establishes zero-shot baselines
on ESD and IEMOCAP, fine-tunes Voxtral-Mini-3B on the English subset of ESD
with DoRA/LoRA adapters, evaluates cross-dataset generalization on IEMOCAP, and
uses perturbation-based counterfactual analysis to inspect which temporal audio
regions affect model predictions.

## Research Questions

1. How well does Voxtral-Mini-3B perform as a zero-shot SER classifier on ESD
   and IEMOCAP?
2. Does DoRA/LoRA fine-tuning on ESD improve performance and calibration over
   zero-shot inference?
3. How does performance change between audio-only and audio-plus-transcript
   settings?
4. Which local temporal regions most influence the model's emotion predictions,
   and how do those local perturbations compare with random-region and
   whole-utterance perturbations?

## Repository Structure

```text
.
|-- data/
|   |-- README.md
|   `-- raw/
|       |-- README.md
|       |-- ESD/
|       |   `-- README.md
|       `-- IEMOCAP/
|           `-- README.md
|-- docs/
|   |-- DATASETS.md
|   `-- REPRODUCIBILITY.md
|-- report/
|   |-- README.md
|   `-- LG-ProXAI_SER_Report.pdf
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
|-- CITATION.cff
|-- requirements.txt
`-- README.md
```

## Datasets

Raw audio files are not committed because of file size and dataset licensing.
Place local copies under `data/raw/ESD/` and `data/raw/IEMOCAP/`.

| Dataset | Role in this project | Access link |
| --- | --- | --- |
| [ESD](https://www.kaggle.com/datasets/nguyenthanhlim/emotional-speech-dataset-esd) | Primary fine-tuning dataset; English speakers `0011`-`0020`; four labels used from the corpus. | Kaggle mirror |
| [IEMOCAP](https://www.kaggle.com/datasets/dejolilandry/iemocapfullrelease) | Cross-dataset evaluation and generalization analysis after ESD fine-tuning. | Kaggle full-release mirror |

Verify that your use complies with the official dataset licenses and access
conditions before downloading or redistributing any material. Additional details
are provided in [docs/DATASETS.md](docs/DATASETS.md).

## Method Summary

### Data Preparation

The scripts use EmoBox-style metadata under `src/EmoBox/data/`. For example,
ESD fold metadata is expected at:

```text
src/EmoBox/data/esd/fold_2/esd_train_fold_2.jsonl
src/EmoBox/data/esd/fold_2/esd_test_fold_2.jsonl
```

The `--audio_root` argument should point to the local directory from which the
metadata `wav` paths can be resolved.

### Fine-Tuning

The training script adapts `mistralai/Voxtral-Mini-3B-2507` with PEFT:

- DoRA/LoRA adapter rank: 16
- LoRA alpha: 32
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- optional 4-bit loading through QLoRA
- stratified validation split by emotion class
- ESD English-speaker filtering for speakers `0011`-`0020`

The training prompt uses a user-only multimodal chat template and appends label
tokens manually so loss is computed only on the target emotion label.

### Evaluation

Evaluation scripts report accuracy, balanced accuracy, macro and weighted F1,
Matthews correlation coefficient, Cohen's kappa, confusion matrices, expected
calibration error, selective accuracy/risk coverage, inference latency
percentiles, and McNemar tests for paired modality comparisons.

### Explainability

The post-hoc explainability scripts consume saved prediction files and generate
localized counterfactual analyses. They identify influential temporal regions
with occlusion, perturb the most influential regions, and compare local effects
against random-region and whole-utterance perturbations. These reports explain
model behavior, not human emotional state.

## Environment Setup

Create and activate a Python environment before installing dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU execution is strongly recommended. Quantized training and inference require
compatible CUDA, PyTorch, and `bitsandbytes` installations.

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

If evaluation output is stored as CSV rather than JSONL, use `--pred_csv`
instead of `--pred_jsonl`.

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

## Reproducibility

For each reported experiment, record the dataset version and access date, model
identifier or local checkpoint hash, random seed, fold index, command-line
arguments, GPU type, CUDA/PyTorch versions, generated metrics, and prediction
files. Use [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) as the experiment
checklist.

## External Code

`src/EmoBox` contains the EmoBox framework used for dataset metadata,
preprocessing conventions, and evaluation support. It is external source code
included for reproducibility. Project-specific training, evaluation, and
explainability logic is implemented in `script/`.

## Ethical and Academic Considerations

SER systems can encode dataset bias and may perform differently across speaker
groups, languages, recording conditions, and acted versus natural speech.
Results should be reported as model behavior on specified datasets and splits.
Avoid claims that the model can reliably infer a person's true internal
emotional state. Reports should include limitations, failure cases, licensing
constraints, and subgroup or distribution-shift analyses where available.

## License

No project-level license file is currently included. Before reusing code,
models, reports, or derived artifacts, verify the licensing terms for this
repository, EmoBox, Voxtral, PEFT dependencies, and the underlying datasets. This
repository does not redistribute raw ESD or IEMOCAP files.

## Citation

If this repository is used in a report, thesis, or paper, cite this repository
using [CITATION.cff](CITATION.cff), and cite the datasets, Voxtral,
PEFT/LoRA/DoRA methods, and EmoBox according to their official citation
instructions.
