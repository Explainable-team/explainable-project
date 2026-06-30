# Report

This directory contains the final project report artifact.

| File | Description |
| --- | --- |
| `LG-ProXAI_SER_Report.pdf` | Academic report for the explainable SER workflow using Voxtral, DoRA fine-tuning, ESD/IEMOCAP evaluation, and Local-Random-Global perturbation diagnostics. |

Core report claims reflected in the repository documentation:

- Audio-only DoRA reaches 82.43% macro-F1 on ESD and 62.54% macro-F1 on IEMOCAP.
- Selected local temporal regions produce larger forced-choice score drops than same-duration random controls.
- The maximum observed global prosody perturbation effect is larger than the local effect in both datasets.
- The dominant diagnostic pattern is mixed local-global sensitivity.
- LG-ProXAI is controlled post-hoc input-output sensitivity analysis, not causal evidence of the model's internal decision mechanism.

If the report is revised, keep the PDF filename stable or document the version
change here. Generated drafts, LaTeX build directories, exported figures, and
intermediate reports should remain outside Git unless they are intentionally
part of the submitted artifact.
