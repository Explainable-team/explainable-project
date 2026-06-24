# Reproducibility Checklist

Use this checklist for every experiment reported from this repository. The goal
is to make each reported number traceable to a dataset version, code state,
model state, hardware configuration, and exact command.

## Environment

Record:

- operating system
- Python version
- CUDA version
- GPU model and VRAM
- PyTorch version
- `transformers`, `peft`, `datasets`, and `bitsandbytes` versions
- repository commit hash

Useful commands:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
python -c "import transformers, peft, datasets; print(transformers.__version__, peft.__version__, datasets.__version__)"
git rev-parse HEAD
```

## Data

Record:

- dataset name and version
- official source and any Kaggle mirror used
- access date
- local root path used for `--audio_root`
- metadata path used for `--meta_dir`
- fold index
- label set
- any speaker, language, duration, class, or transcript filtering

For this project, the default label set is:

```text
Angry, Happy, Sad, Neutral
```

## Model

Record:

- base model ID or local model path
- adapter directory
- whether 4-bit loading was enabled
- LoRA/DoRA rank, alpha, dropout, and target modules
- random seed
- checkpoint or adapter hash when available

## Training

Record the exact training command. At minimum, include:

- `--meta_dir`
- `--audio_root`
- `--model_id`
- `--output_dir`
- `--fold`
- `--seed`
- `--val_per_class`
- `--epochs`
- `--lr`
- batch size and gradient accumulation
- `--load_in_4bit`, if used

Save the final adapter, training logs, validation metrics, and any early
stopping decision.

## Evaluation

Record the exact evaluation command and preserve:

- predictions CSV or JSONL
- metrics JSON or CSV
- confusion matrices
- calibration metrics
- selective accuracy curves
- McNemar comparison files
- latency summaries

Report both audio-only and audio-plus-text results when available.

## Explainability

For counterfactual explanations, record:

- input prediction file
- sample selection strategy
- number of samples
- segment duration
- number of top segments
- perturbation types
- random seed
- output report directory

When discussing explanations, distinguish between evidence that changed the
model prediction and evidence that supports claims about human emotion. The
counterfactual analysis explains model behavior, not the speaker's internal
state.

## Recommended Experiment Log Template

```text
Experiment ID:
Date:
Research question:
Repository commit:
Dataset:
Dataset source:
Dataset access date:
Fold:
Base model:
Adapter:
Command:
Seed:
Hardware:
Key metrics:
Output directory:
Known limitations:
Notes:
```
