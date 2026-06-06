# Emotional Speech Dataset (ESD)

This folder is reserved for the **Emotional Speech Dataset (ESD)**.

ESD is a large-scale emotional speech dataset containing recordings from multiple
speakers and emotion categories. In this project, ESD is mainly used for
**fine-tuning** the model because it provides a relatively balanced distribution
of emotions.

IMPORTANT: The actual audio files are NOT included in this repository. The dataset must be downloaded separately from its official source and placed
inside this folder following the original directory structure.

---

## How ESD is used in this project

- Used as the **training dataset** for PEFT fine-tuning (LoRA / DoRA)
- Only the **English-speaking subset** is used
- Emotion labels are mapped to the following classes:
  - Angry
  - Happy
  - Sad
  - Neutral

---

## Expected Structure

```text
ESD/
├── 0001/
├── 0002/
├── ...
├── 0020/
