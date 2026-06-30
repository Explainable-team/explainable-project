# LG-ProXAI: Explainable Speech Emotion Recognition with Voxtral and DoRA

This repository contains the research workflow for **LG-ProXAI**, a
Local-Random-Global perturbation diagnostic protocol for speech emotion
recognition (SER). The project adapts `mistralai/Voxtral-Mini-3B-2507` with a
DoRA parameter-efficient adapter, evaluates the adapted model on ESD and
IEMOCAP, and audits prediction sensitivity with controlled perturbations.

The project studies model behavior on four emotion labels:

```text
Angry, Happy, Sad, Neutral
```

It does not claim to infer a speaker's true internal emotional state. Reported
explanations are post-hoc input-output sensitivity measurements under a fixed
scoring rule.

## Project Snapshot

| Area | Configuration |
| --- | --- |
| Base model | `mistralai/Voxtral-Mini-3B-2507` |
| Adaptation | DoRA/LoRA rank 16, alpha 32, dropout 0.1 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Training data | English subset of ESD, speakers `0011`-`0020` |
| In-domain evaluation | ESD four-class test setting |
| Cross-corpus evaluation | IEMOCAP, with excitement merged into Happy |
| XAI protocol | LG-ProXAI local attenuation, random controls, global prosody perturbations |
| Main report | [report/LG-ProXAI_SER_Report.pdf](report/LG-ProXAI_SER_Report.pdf) |
| Runbook | [INSTRUCTIONS.md](INSTRUCTIONS.md) |

## Research Questions

1. Which temporal regions and acoustic transformations produce the largest
   changes in the predicted-label forced-choice score?
2. Are selected local effects stronger than same-duration random controls?
3. Are local effects distinguishable from whole-utterance prosody sensitivity?
4. How do diagnostic patterns change between in-domain ESD evaluation and
   cross-corpus IEMOCAP evaluation?

## Main Results

Audio-only DoRA evaluation from the project report:

| Metric | ESD | IEMOCAP |
| --- | ---: | ---: |
| Accuracy | 82.64% | 61.75% |
| Balanced accuracy | 82.64% | 65.02% |
| Macro-F1 | 82.43% | 62.54% |
| Matthews correlation coefficient | 0.7729 | 0.4990 |
| Cohen's kappa | 0.7686 | 0.4894 |

LG-ProXAI diagnostics on 1,000 predicted-class-balanced samples per dataset:

| Diagnostic quantity | ESD | IEMOCAP |
| --- | ---: | ---: |
| Random control score drop | 0.058 | 0.029 |
| Top-1 local score drop | 0.187 | 0.158 |
| Top-1 + Top-2 score drop | 0.228 | 0.163 |
| Maximum observed global score drop | 0.486 | 0.427 |
| Top-1 / random ratio | 3.22 | 5.40 |
| Maximum global / Top-1 ratio | 2.60 | 2.71 |
| Dominant pattern | Mixed local-global | Mixed local-global |
| Dominant global sensitivity | Speaking rate | Speaking rate |
| Incorrect/correct local ratio | 1.62 | 1.47 |

These results support the report's main interpretation: selected temporal
regions matter more than random same-duration regions, but the strongest tested
whole-utterance prosody perturbations produce larger score drops than local
attenuation. The adapted model therefore shows mixed local-global sensitivity.

## Method Overview

The workflow has five stages:

1. **Data audit**: inspect ESD and IEMOCAP metadata, label mappings, transcript
   availability, and local audio path resolution.
2. **Zero-shot sanity checks**: evaluate Voxtral before adaptation on ESD and
   IEMOCAP notebooks.
3. **DoRA adaptation**: fine-tune Voxtral-Mini-3B on ESD English speakers with
   rank 16, alpha 32, dropout 0.1, cosine decay, mixed precision, gradient
   accumulation, and validation-based early stopping.
4. **Predictive evaluation**: compute in-domain ESD and cross-corpus IEMOCAP
   metrics in audio-only and audio-plus-text modes where available.
5. **LG-ProXAI diagnosis**: rescore original and perturbed audio variants with
   the same fixed forced-choice label scoring rule.

## LG-ProXAI Protocol

LG-ProXAI compares original and perturbed audio under a fixed label-token
scoring rule. For each candidate emotion label, the model produces a
length-normalized log-likelihood. Scores are normalized over the fixed label
set, yielding a forced-choice label score. This score is not a calibrated
emotion probability.

The diagnostic interventions are:

- **Local attenuation**: split speech-active audio into short windows and
  attenuate each selected region.
- **Random controls**: perturb same-duration random regions while avoiding the
  selected evidence region.
- **Global prosody perturbations**: modify the whole utterance with energy,
  pitch, time-scale, and spectral transformations.
- **Contrastive margins**: track score and margin changes against the strongest
  competing class.
- **Deletion and keep-only checks**: used in the smaller qualitative pipeline
  for comprehensiveness and sufficiency-style analysis.

Interpretation guardrail: LG-ProXAI measures controlled post-hoc perturbation
sensitivity. It should not be described as causal evidence of the model's
internal mechanism or as proof of the true emotional content of speech.

## Repository Structure

```text
.
|-- README.md
|-- INSTRUCTIONS.md
|-- CITATION.cff
|-- requirements.txt
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
`-- src/
    |-- README.md
    `-- EmoBox/
```

Generated outputs are intentionally excluded from Git. Use directories such as
`outputs/`, `eval_reports/`, `artifacts/`, or `runs/` for adapters, metrics,
prediction files, plots, and temporary perturbed audio.

## Environment Setup

Create a Python environment and install dependencies:

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

## Data Placement

Raw datasets are not committed. Place local copies under:

```text
data/raw/ESD/
data/raw/IEMOCAP/
```

The scripts use EmoBox metadata from `src/EmoBox/data/`; `--audio_root` should
point to the local directory from which the metadata `wav` paths can be
resolved. See [docs/DATASETS.md](docs/DATASETS.md).

## Quick Workflow

### 1. Explore data

```text
script/01_data_exploration.ipynb
```

### 2. Run zero-shot baselines

```text
script/02_zero_shot_esd.ipynb
script/02_zero_shot_iemocap.ipynb
```

### 3. Fine-tune the ESD DoRA adapter

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

### 6. Run qualitative LG-ProXAI analysis

```bash
python script/05_post_hoc_explainable_20_sample.py \
  --pred_jsonl eval_reports/esd/fold_2/dora16a32/audio_only/predictions.jsonl \
  --base_model mistralai/Voxtral-Mini-3B-2507 \
  --adapter_dir outputs/esd_dora16a32_fold2/final_adapter \
  --out_dir eval_reports/lg_proxai_20_esd \
  --load_in_4bit \
  --max_samples 20
```

### 7. Run class-level LG-ProXAI statistics

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
  --random_n 3
```

Use `--pred_csv` instead of `--pred_jsonl` if the evaluation output is stored as
CSV.

## Expected Outputs

Typical generated outputs include:

- PEFT adapter checkpoints and final adapter weights
- prediction CSV/JSONL files
- metrics JSON/CSV summaries
- confusion matrices and selective accuracy/risk outputs
- calibration and latency summaries
- LG-ProXAI per-sample qualitative reports
- LG-ProXAI class-level summary tables and plots
- temporary perturbed audio files created during scoring

## Reproducibility

For each reported run, record the dataset source and access date, model
identifier or local checkpoint hash, adapter path, random seed, fold index,
command-line arguments, GPU type, CUDA/PyTorch versions, generated metrics, and
prediction files. Use [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) as the
experiment checklist.

## External Code

`src/EmoBox` contains external EmoBox code and metadata conventions used for
dataset organization and evaluation support. The project-specific training,
evaluation, and LG-ProXAI logic is implemented in `script/`.

## Ethical and Academic Considerations

SER systems can encode dataset bias and may perform differently across speaker
groups, languages, recording conditions, and acted versus natural speech. Report
results as model behavior on specified datasets and splits. Include limitations,
failure cases, licensing constraints, and distribution-shift analysis where
available.

## License

No project-level license file is currently included. Before reusing code,
models, reports, or derived artifacts, verify the licensing terms for this
repository, EmoBox, Voxtral, PEFT dependencies, and the underlying datasets.
This repository does not redistribute raw ESD or IEMOCAP audio.

## Citation

If this repository is used in a report, thesis, or paper, cite this repository
using [CITATION.cff](CITATION.cff), and cite ESD, IEMOCAP, Voxtral, LoRA, DoRA,
and EmoBox according to their official citation instructions.
