# Experiment Scripts

This directory contains the main experimental workflow. The numeric prefixes
indicate the recommended execution order.

## Files

| File | Purpose |
| --- | --- |
| `01_data_exploration.ipynb` | Inspect dataset metadata, class distributions, and audio availability. |
| `02_zero_shot_esd.ipynb` | Run or document zero-shot Voxtral experiments on ESD. |
| `02_zero_shot_iemocap.ipynb` | Run or document zero-shot Voxtral experiments on IEMOCAP. |
| `03_train_dora16a32_voxtral_esd.py` | Fine-tune Voxtral-mini-3B on ESD with DoRA rank 16 and alpha 32. |
| `04_eval_dora16a32_esd.py` | Evaluate a DoRA adapter on ESD in audio-only and audio-plus-text modes. |
| `04_eval_dora16a32_iemocap.py` | Evaluate a DoRA adapter on IEMOCAP in audio-only and audio-plus-text modes. |
| `05_post_hoc_explainable_20_sample.py` | Run localized counterfactual explanations on a small sample. |
| `05_post_hoc_explainable_1000_sample.py` | Run localized counterfactual explanations on a larger sample. |

## Recommended Workflow

1. Run data exploration and verify that metadata paths resolve to local audio.
2. Establish zero-shot baselines for ESD and IEMOCAP.
3. Fine-tune the ESD DoRA adapter.
4. Evaluate the adapter on ESD and IEMOCAP.
5. Use saved prediction files as inputs for post-hoc explainability.

## Output Policy

Do not commit generated model checkpoints, prediction files, plots, reports, or
temporary perturbed audio. Use directories such as:

```text
outputs/
eval_reports/
artifacts/
runs/
```

These directories are ignored by the repository-level `.gitignore`.

## Notes for Academic Use

When reporting results, include the exact command-line arguments, random seed,
fold index, base model identifier, adapter path, dataset version, and hardware
configuration. This information is necessary for reproducibility.
