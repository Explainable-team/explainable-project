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
| `05_post_hoc_explainable_20_sample.py` | Run qualitative LG-ProXAI analysis with per-sample plots, deletion/keep-only checks, localized acoustic interventions, random controls, and global baselines. |
| `05_post_hoc_explainable_1000_sample.py` | Run compact class-level LG-ProXAI statistics with predicted-class balancing, Top-2 local evidence, random controls, and global prosody perturbations. |

## Recommended Workflow

1. Run data exploration and verify that metadata paths resolve to local audio.
2. Establish zero-shot baselines for ESD and IEMOCAP.
3. Fine-tune the ESD DoRA adapter.
4. Evaluate the adapter on ESD and IEMOCAP.
5. Use saved audio-only prediction files as inputs for LG-ProXAI diagnostics.

## LG-ProXAI Notes

The 20-sample script is intended for qualitative inspection and rich report
generation. The 1,000-sample script is intended for class-level statistics and
paper tables. Both compare original and perturbed audio with a fixed
forced-choice label scoring rule.

Recommended large-run settings from the report are:

```text
--class_balance_mode predicted_class_balanced
--max_samples 1000
--segment_sec 0.5
--top_k 2
--random_n 3
```

Report LG-ProXAI outputs as perturbation sensitivity, not causal explanation.
The forced-choice label score is not a calibrated probability.

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
