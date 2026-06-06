# Data

This directory is reserved for dataset files used by the project. Raw audio is
not committed to the repository because of dataset licensing and file size.

## Expected Layout

```text
data/
|-- raw/
|   |-- ESD/
|   `-- IEMOCAP/
`-- README.md
```

## Datasets

### ESD

The Emotional Speech Dataset should be downloaded from its official source and
placed under:

```text
data/raw/ESD/
```

The training script currently filters ESD to English speakers with IDs
`0011`-`0020` and to the labels `Angry`, `Happy`, `Sad`, and `Neutral`.

### IEMOCAP

The Interactive Emotional Dyadic Motion Capture dataset should be obtained
through its official access process and placed under:

```text
data/raw/IEMOCAP/
```

IEMOCAP has restrictive licensing. Do not commit audio, video, transcripts, or
other protected files unless redistribution is explicitly permitted.

## Metadata

Experiment scripts use EmoBox-style metadata stored under:

```text
src/EmoBox/data/
```

Before running experiments, confirm that each metadata `wav` path can be
resolved from the `--audio_root` argument used in the command.

## Generated Data

Processed files, prediction outputs, plots, reports, checkpoints, and temporary
counterfactual audio should be written outside `data/raw`, for example under
`outputs/`, `eval_reports/`, or `artifacts/`.
