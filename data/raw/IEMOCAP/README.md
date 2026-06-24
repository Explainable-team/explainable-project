# IEMOCAP Dataset

This folder is reserved for IEMOCAP, the Interactive Emotional Dyadic Motion
Capture dataset.

Kaggle full-release mirror: https://www.kaggle.com/datasets/dejolilandry/iemocapfullrelease

The actual dataset files are not included in this repository. IEMOCAP has
restrictive access and redistribution terms, so verify that your use complies
with the official license before downloading, storing, or sharing any files.

## Use in This Project

- Zero-shot evaluation dataset.
- Held-out cross-dataset evaluation after fine-tuning on ESD.
- Labels are mapped to `Angry`, `Happy`, `Sad`, and `Neutral`; `Excited` may be
  merged into `Happy` depending on the experiment.

## Expected Structure

```text
data/raw/IEMOCAP/
|-- Session1/
|-- Session2/
|-- Session3/
|-- Session4/
`-- Session5/
```
