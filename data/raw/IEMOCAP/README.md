# IEMOCAP Dataset

This folder is reserved for the **IEMOCAP (Interactive Emotional Dyadic Motion Capture)**
dataset.

IEMOCAP contains acted and semi-natural conversational speech with emotion labels.
In this project, IEMOCAP is primarily used for **evaluation and generalization
testing**, as it better reflects real conversational scenarios.

IMPORTANT: The actual audio files are NOT included in this repository. The dataset must be obtained separately due to licensing restrictions.

---

## How IEMOCAP is used in this project

- Used mainly for **zero-shot evaluation**
- Used as a **held-out test set** after fine-tuning on ESD
- Emotion labels are mapped as follows:
  - Angry
  - Happy (including *Excited*)
  - Sad
  - Neutral

---

## Expected Structure

```text
IEMOCAP/
├── Session1/
├── Session2/
├── Session3/
├── Session4/
├── Session5/
