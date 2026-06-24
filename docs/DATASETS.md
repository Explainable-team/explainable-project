# Dataset Documentation

This project uses ESD and IEMOCAP for speech emotion recognition experiments.
Raw files are excluded from Git. Only metadata, scripts, and documentation are
kept in the repository.

## Dataset Sources

| Dataset | Kaggle link | Project role | Notes |
| --- | --- | --- | --- |
| [ESD](https://www.kaggle.com/datasets/nguyenthanhlim/emotional-speech-dataset-esd) | `nguyenthanhlim/emotional-speech-dataset-esd` | Fine-tuning and in-domain evaluation. | Use the English-speaking subset for the main DoRA experiment. |
| [IEMOCAP](https://www.kaggle.com/datasets/dejolilandry/iemocapfullrelease) | `dejolilandry/iemocapfullrelease` | Zero-shot and cross-dataset evaluation. | Verify official access and redistribution restrictions before use. |

## Local Placement

```text
data/raw/ESD/
data/raw/IEMOCAP/
```

The experiment scripts use EmoBox metadata from `src/EmoBox/data/` and resolve
audio through the `--audio_root` command-line argument.

## Label Mapping

The main experiments use four labels:

```text
Angry, Happy, Sad, Neutral
```

For IEMOCAP, `Excited` may be merged into `Happy` to align with the four-label
evaluation setting. Record the exact mapping in every experiment log.

## Academic Use

For every reported result, cite the original dataset papers or official dataset
pages, record the access date, and document any filtering. If a Kaggle mirror is
used for convenience, still check the original license and access terms before
redistributing data or publishing derived materials.
