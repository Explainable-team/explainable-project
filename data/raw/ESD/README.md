# Emotional Speech Dataset (ESD)

This folder is reserved for the Emotional Speech Dataset (ESD).

Kaggle access link: https://www.kaggle.com/datasets/nguyenthanhlim/emotional-speech-dataset-esd

The actual dataset files are not included in this repository. Download the data
separately, verify the applicable license and citation requirements, and place
the extracted files here using the original directory structure.

## Use in This Project

- Primary dataset for PEFT fine-tuning with LoRA/DoRA.
- English-speaking speakers `0011`-`0020` are used by the training workflow.
- The project maps labels to `Angry`, `Happy`, `Sad`, and `Neutral`.

## Expected Structure

```text
data/raw/ESD/
|-- 0001/
|-- 0002/
|-- ...
`-- 0020/
```
