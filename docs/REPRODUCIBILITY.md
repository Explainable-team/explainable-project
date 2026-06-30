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

Project-specific label and split policy:

- ESD adaptation uses English speakers `0011`-`0020`.
- ESD Surprise is excluded from the adapted four-class setup.
- IEMOCAP Excited is merged into Happy when four-class alignment is required.
- The main LG-ProXAI report uses predicted-class-balanced diagnostic subsets.

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

Report both audio-only and audio-plus-text results when available. For the main
LG-ProXAI acoustic interpretation, preserve the audio-only prediction file used
as diagnostic input.

## Explainability

For LG-ProXAI perturbation diagnostics, record:

- input prediction file
- sample selection strategy
- number of samples
- segment duration
- number of top segments
- perturbation types
- forced-choice scoring configuration
- random-control count and exclusion policy
- global transformation list
- whether original samples were rescored
- random seed
- output report directory

When discussing explanations, distinguish between evidence that changed the model
score and evidence that supports claims about human emotion. LG-ProXAI explains
model input-output sensitivity under tested perturbations, not the speaker's
internal state and not the model's internal causal mechanism.

Use the report terminology consistently:

- `forced-choice label score`: normalized fixed-label score, not calibrated
  probability
- `local drop`: score decrease under selected temporal attenuation
- `random control drop`: score decrease under same-duration random perturbation
- `maximum observed global drop`: largest score decrease among the tested
  utterance-level transformations
- `mixed local-global sensitivity`: local evidence exceeds random controls but
  the strongest global perturbation exceeds local evidence

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
