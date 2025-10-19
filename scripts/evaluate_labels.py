# evaluate_labels.py
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

INPUT = "labeled_emails.csv"   # produced by your pipeline
VAL_OUTPUT = "validation_sample.csv"
SAMPLE_SIZE = 500  # sample size to validate (adjust)

df = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
# ensure columns exist
for c in ["category","priority_label","action_required","_confidence","_label_source","label_source","confidence","decision_path"]:
    if c not in df.columns:
        df[c] = ""

# convert action to int
df["action_required"] = df["action_required"].fillna("0").astype(int)

# Quick stratified sampling by label_source and category to get diverse set
def stratified_sample(df, n):
    groups = df.groupby(["category","_label_source"]).size().reset_index(name="count")
    # simple approach: sample proportionally
    return df.sample(n=min(n,len(df)), random_state=42)

sample = stratified_sample(df, SAMPLE_SIZE)
sample.to_csv(VAL_OUTPUT, index=False)
print(f"Saved validation sample ({len(sample)}) to {VAL_OUTPUT}")
print("\nCategory distribution in sample:\n", sample["category"].value_counts())

# If you have ground-truth labels for sample (manual inspection), load them and compute metrics:
# assume you manually review VAL_OUTPUT and add columns: gt_category, gt_priority_label, gt_action_required
if "gt_category" in sample.columns:
    y_true = sample["gt_category"]
    y_pred = sample["category"]
    print("\nCategory classification report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))
if "gt_action_required" in sample.columns:
    print("\nAction Required report:\n")
    print(classification_report(sample["gt_action_required"].astype(int), sample["action_required"].astype(int), zero_division=0))
