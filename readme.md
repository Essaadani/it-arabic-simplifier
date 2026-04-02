# IT Arabic Simplifier

A lightweight research prototype for **simplifying French IT educational content into clear Modern Standard Arabic** for Moroccan learners.

The project fine-tunes **Qwen2.5-1.5B-Instruct** with **LoRA** using a small, curated dataset and a **glossary-guided prompt construction** pipeline. The main goal is to improve:

- Arabic fluency
- meaning preservation
- simplicity / pedagogical clarity
- terminology consistency

## What is in this project

- `data/raw/it_simplification.csv` — paired source/target dataset
- `data/raw/glossary.csv` — French → Arabic technical glossary
- `scripts/prepare_dataset.py` — builds JSONL training files and preview artifacts
- `scripts/run_eval.py` — runs 20 evaluation prompts on the base model and the fine-tuned model
- `outputs/adapters/qwen-it-ar-simplifier-v3/` — fine-tuned LoRA adapter (if already trained locally)

## Expected project structure

```text
it-arabic-simplifier/
├── data/
│   ├── raw/
│   │   ├── it_simplification.csv
│   │   └── glossary.csv
│   └── processed/
├── outputs/
│   ├── adapters/
│   └── eval/
├── scripts/
│   ├── prepare_dataset.py
│   └── run_eval.py
├── requirements.txt
└── README.md
```

## Environment

Recommended:

- macOS on Apple Silicon
- Python 3.10+ or 3.11+
- `mlx`, `mlx-lm`, `pandas`, `scikit-learn`

## 1) Create a clean virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 2) Install dependencies

If you want the exact environment used for the project:

```bash
pip install -r requirements.txt
```

If you prefer a minimal setup first:

```bash
pip install "mlx-lm[train]" pandas scikit-learn
```

## 3) Verify the setup

```bash
python - <<'PY'
import mlx
import mlx_lm
import pandas
import sklearn
print("MLX OK")
print("MLX-LM OK")
print("Pandas OK")
print("Sklearn OK")
PY
```

## 4) Prepare the dataset

Make sure these files exist:

- `data/raw/it_simplification.csv`
- `data/raw/glossary.csv`

Then run:

```bash
python scripts/prepare_dataset.py
```

This generates:

- `data/processed/train.jsonl`
- `data/processed/valid.jsonl`
- `data/processed/test.jsonl`
- `data/processed/train_preview.csv`

### What `prepare_dataset.py` does

- loads the raw simplification dataset
- loads the glossary
- lightly cleans mixed-language source texts
- matches glossary terms to each example
- injects matched terms into the training prompt
- exports training data in chat-style JSONL format
- writes a preview CSV for quick inspection

## 5) Inspect the prepared data

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("data/processed/train_preview.csv")
print(df[["id", "topic", "matched_terms"]].head(10).to_string(index=False))
PY
```

You should see matched glossary terms for at least some rows.

## 6) Fine-tune the model

Run LoRA fine-tuning with the prepared dataset:

```bash
python -m mlx_lm lora \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train \
  --data ./data/processed \
  --adapter-path ./outputs/adapters/qwen-it-ar-simplifier-v3 \
  --iters 200 \
  --batch-size 1 \
  --learning-rate 5e-6 \
  --steps-per-report 10 \
  --steps-per-eval 25 \
  --save-every 25
```

Final adapter weights should appear in:

```text
outputs/adapters/qwen-it-ar-simplifier-v3/adapters.safetensors
```

## 7) Test the fine-tuned model manually

Example:

```bash
python -m mlx_lm generate \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path ./outputs/adapters/qwen-it-ar-simplifier-v3 \
  --prompt "Rewrite the following IT explanation in simple, clear, correct Arabic for Moroccan learners. Preserve the meaning and avoid mixing languages.

Use the following Arabic technical terminology when relevant:
- requête = استعلام
- base de données = قاعدة بيانات

Text:
Les requêtes SQL permettent d’interroger et de modifier les données dans une base de données."
```

## 8) Run the automatic evaluation

```bash
python scripts/run_eval.py
```

This runs 20 prompts on:

- the base model
- the fine-tuned model

and saves:

- `outputs/eval_outputs.csv`
- `outputs/eval_outputs_clean.csv`

## 9) Fill the manual evaluation columns

Open `outputs/eval_outputs_clean.csv` and score each row from **1 to 5** for:

- `base_fluency`
- `base_meaning`
- `base_simplicity`
- `base_terminology`
- `ft_fluency`
- `ft_meaning`
- `ft_simplicity`
- `ft_terminology`

These are the main paper metrics.

## 10) Compute the final averages

If your CSV uses `;` as delimiter, use a script like this:

```python
import pandas as pd

df = pd.read_csv("outputs/eval_outputs_clean.csv", sep=";")
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

def avg(col):
    return round(pd.to_numeric(df[col], errors="coerce").dropna().mean(), 2)

print("Fluency:", avg("base_fluency"), avg("ft_fluency"))
print("Meaning:", avg("base_meaning"), avg("ft_meaning"))
print("Simplicity:", avg("base_simplicity"), avg("ft_simplicity"))
print("Terminology:", avg("base_terminology"), avg("ft_terminology"))
```

## Typical workflow

```bash
source .venv/bin/activate
python scripts/prepare_dataset.py
python -m mlx_lm lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data ./data/processed --adapter-path ./outputs/adapters/qwen-it-ar-simplifier-v3 --iters 200 --batch-size 1 --learning-rate 5e-6
python scripts/run_eval.py
```

## Notes

- `prepare_dataset.py` expects the glossary at `data/raw/glossary.csv`.
- `prepare_dataset.py` expects the dataset at `data/raw/it_simplification.csv`.
- `run_eval.py` expects the fine-tuned adapter at `./outputs/adapters/qwen-it-ar-simplifier-v3`.
- The evaluation is intentionally small and manual because this is a research prototype, not yet a production system.

## Research purpose

This repository supports a paper on glossary-guided Arabic simplification of IT educational content for Moroccan learners. The key hypothesis is that **glossary-guided fine-tuning improves terminology consistency and pedagogical clarity over the untuned base model**.
