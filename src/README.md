# Source Code Folder

This folder contains source code used in the project that is **not directly part
of the experiment scripts**, including external libraries and supporting modules.

In particular, this folder includes the **EmoBox** framework, which is used to
manage datasets, evaluation protocols, and experiment configuration for
multimodal emotion recognition.

---

## EmoBox

**EmoBox** is an open-source framework designed for benchmarking emotion
recognition models.

In this project, EmoBox is used for:
- Loading dataset metadata (ESD and IEMOCAP)
- Managing train / validation / test splits
- Standardizing evaluation pipelines
- Ensuring consistency across experiments

The EmoBox code is included here to make the project easier to run and reproduce.

---

## Notes

- EmoBox is **not developed by the authors of this project**.
- The original EmoBox repository and license information are preserved.
- Any project-specific logic (training, evaluation, fine-tuning) is implemented
  outside this folder.
