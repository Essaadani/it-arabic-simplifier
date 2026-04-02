import pandas as pd

CSV_PATH = "outputs/eval_outputs_clean.csv"

df = pd.read_csv(CSV_PATH, sep=";")

df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

print("Detected columns:")
print(df.columns.tolist())
print()

def avg(col: str):
    if col not in df.columns:
        print(f"Missing column: {col}")
        return None

    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(values) == 0:
        return None
    return round(values.mean(), 2)

results = {
    "Fluency": (avg("base_fluency"), avg("ft_fluency")),
    "Meaning": (avg("base_meaning"), avg("ft_meaning")),
    "Simplicity": (avg("base_simplicity"), avg("ft_simplicity")),
    "Terminology": (avg("base_terminology"), avg("ft_terminology")),
}

print("=== FINAL RESULTS ===\n")

for metric, (base, ft) in results.items():
    if base is None or ft is None:
        print(f"{metric}: unavailable")
    else:
        improvement = round(ft - base, 2)
        print(f"{metric}: Base={base} | Fine-tuned={ft} | +{improvement}")