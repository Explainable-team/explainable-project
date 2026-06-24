# Source Code

This directory contains supporting source code used by the experiment scripts.

## EmoBox

`src/EmoBox` contains the EmoBox framework, which provides metadata and utility
code for emotion recognition datasets. In this project it is used primarily for:

- dataset metadata organization
- fold definitions
- label maps
- preprocessing utilities
- evaluation support

EmoBox is external code and is not the primary contribution of this repository.
Project-specific training, evaluation, and explainability logic is implemented
in the `script/` directory.

## Provenance

When preparing an academic submission, cite EmoBox according to its upstream
README and preserve any upstream license or attribution requirements. If EmoBox
is updated, record the upstream commit, release, or download date in the
experiment notes and in the reproducibility checklist.
