# Experiment Scripts

This directory contains the executable research workflow. Numeric prefixes
indicate the recommended order.

## Files

| File | Purpose |
| --- | --- |
| `01_data_exploration.ipynb` | Inspect dataset metadata, class distributions, audio availability, and basic dataset limitations. |
| `02_zero_shot_esd.ipynb` | Run or document zero-shot Voxtral experiments on ESD. |
| `02_zero_shot_iemocap.ipynb` | Run or document zero-shot Voxtral experiments on IEMOCAP. |
| `03_train_dora16a32_voxtral_esd.py` | Fine-tune Voxtral-Mini-3B on ESD with DoRA rank 16 and alpha 32. |
| `04_eval_dora16a32_esd.py` | Evaluate a DoRA adapter on ESD in audio-only and audio-plus-text modes. |
| `04_eval_dora16a32_iemocap.py` | Evaluate a DoRA adapter on IEMOCAP in audio-only and audio-plus-text modes. |
| `05_post_hoc_explainable_20_sample.py` | Run localized counterfactual explanations on a small diagnostic sample. |
| `05_post_hoc_explainable_1000_sample.py` | Run larger-scale localized counterfactual explanations with class-balancing options. |

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

## Academic Reporting

When reporting results, include the exact command-line arguments, random seed,
fold index, base model identifier, adapter path, dataset version, hardware
configuration, and repository commit. This information is required for a
credible reproducibility claim.
