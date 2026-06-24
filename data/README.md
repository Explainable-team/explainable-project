# Data

This directory documents the local dataset layout used by the project. Raw
audio, transcripts, video files, checkpoints, generated reports, and other large
artifacts are not committed to Git because of licensing and file-size
constraints.

## Expected Layout

```text
data/
|-- README.md
`-- raw/
    |-- README.md
    |-- ESD/
    |   `-- README.md
    `-- IEMOCAP/
        `-- README.md
```

## Datasets

| Dataset | Local path | Access |
| --- | --- | --- |
| [ESD](https://www.kaggle.com/datasets/nguyenthanhlim/emotional-speech-dataset-esd) | `data/raw/ESD/` | Kaggle mirror; verify the official license and citation requirements before use. |
| [IEMOCAP](https://www.kaggle.com/datasets/dejolilandry/iemocapfullrelease) | `data/raw/IEMOCAP/` | Kaggle full-release mirror; verify official IEMOCAP access and redistribution restrictions before use. |

## Project-Specific Filtering

The ESD training workflow currently uses the English-speaking subset, speakers
`0011`-`0020`, and maps the task to `Angry`, `Happy`, `Sad`, and `Neutral`.

The IEMOCAP evaluation workflow maps the task to the same four labels, with
`Excited` treated as part of the `Happy` class when required by the experiment.

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
