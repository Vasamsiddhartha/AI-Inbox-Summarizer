#!/usr/bin/env python3
"""
merge_and_prepare_dataset.py

Usage:
  python mergesmsANDmails.py --mails emails_hybrid_verified.csv --sms msgs_labeled.csv \--out Mixed_labeled.csv --out_dir splits --seed 42 --test_size 0.1 --val_size 0.1

Produces:
 - combined_labeled.csv
 - splits/train_1.csv, splits/val_1.csv, splits/test_1.csv
 - prints dataset stats

Requirements:
 pip install pandas scikit-learn
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import StratifiedShuffleSplit

REQUIRED_COLS = ["id", "source", "sender", "subject", "body", "timestamp",
                 "category", "priority_label", "action_required", "action_text"]

def load_csv_maybe(path: Path, default_source: str):
    df = pd.read_csv(path, dtype=str).fillna("")
    # Make sure required cols exist
    if "source" not in df.columns:
        df["source"] = default_source
    # Ensure all required columns present
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = ""
    # Keep only required cols in consistent order
    df = df[REQUIRED_COLS]
    return df

def normalize_labels(df: pd.DataFrame):
    # lowercase categories & priority, clean action_required
    df["category"] = df["category"].str.strip().str.lower().replace({"promo":"promotion", "promotional":"promotion"})
    df["priority_label"] = df["priority_label"].str.strip().str.lower()
    df["priority_label"] = df["priority_label"].replace({"h":"high", "m":"medium", "l":"low"})
    df["action_required"] = df["action_required"].apply(lambda x: int(float(x)) if x not in ("", None) else 0)
    # fill defaults
    df["category"] = df["category"].replace("", "personal")
    df["priority_label"] = df["priority_label"].replace("", "low")
    df["action_text"] = df["action_text"].fillna("").astype(str)
    return df

def dedupe(df: pd.DataFrame):
    # dedupe exact body+subject+timestamp duplicates (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["subject","body","timestamp"])
    after = len(df)
    print(f"Deduped exact duplicates: {before - after} removed")
    return df

def stratified_split(df: pd.DataFrame, test_size=0.1, val_size=0.1, seed=42):
    # For stratification use 'category'. If a category has too few samples, fall back to random split.
    labels = df["category"].values
    unique, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()
    if min_count < 5:
        print("Warning: some categories have <5 samples — doing random split to avoid empty strata.")
        shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(df)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        test = shuffled.iloc[:n_test]
        val = shuffled.iloc[n_test:n_test + n_val]
        train = shuffled.iloc[n_test + n_val:]
        return train, val, test
    # do sequential stratified split: first test, then val from remaining
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = next(sss_test.split(df, labels))
    trainval_idx, test_idx = idx[0], idx[1]
    trainval = df.iloc[trainval_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    # now split trainval into train+val with proportion val/(train+val) = val_size/(1-test_size)
    val_ratio_within = val_size / (1.0 - test_size)
    labels_tv = trainval["category"].values
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_within, random_state=seed+1)
    idx2 = next(sss_val.split(trainval, labels_tv))
    train_idx2, val_idx2 = idx2[0], idx2[1]
    train = trainval.iloc[train_idx2].reset_index(drop=True)
    val = trainval.iloc[val_idx2].reset_index(drop=True)
    return train, val, test

def print_stats(df, name="dataset"):
    print(f"\n=== {name} stats ===")
    print(f"Total rows: {len(df)}")
    print("Category distribution:")
    print(df["category"].value_counts())
    print("\nPriority distribution:")
    print(df["priority_label"].value_counts())
    print("\nAction_required counts (0/1):")
    print(df["action_required"].value_counts())
    print("====================\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mails", required=True)
    p.add_argument("--sms", required=True)
    p.add_argument("--out", default="mixed_labeled.csv")
    p.add_argument("--out_dir", default="splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--val_size", type=float, default=0.1)
    args = p.parse_args()

    mails = Path(args.mails)
    sms = Path(args.sms)
    if not mails.exists() or not sms.exists():
        print("Input files not found. Check paths.")
        sys.exit(1)

    df_mails = load_csv_maybe(mails, default_source="email")
    df_sms = load_csv_maybe(sms, default_source="sms")

    # Optional: add a prefix to ids to avoid collision
    df_mails["id"] = "mail_" + df_mails["id"].astype(str)
    df_sms["id"] = "sms_" + df_sms["id"].astype(str)

    combined = pd.concat([df_mails, df_sms], axis=0, ignore_index=True)
    combined = normalize_labels(combined)
    combined = dedupe(combined)

    # Shuffle
    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Save combined
    out_path = Path(args.out)
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote combined dataset to: {out_path}")

    print_stats(combined, "combined")

    # Stratified split
    train, val, test = stratified_split(combined, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False, encoding="utf-8-sig")
    val.to_csv(out_dir / "val.csv", index=False, encoding="utf-8-sig")
    test.to_csv(out_dir / "test.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote splits to {out_dir}/ (train/val/test)")

    print_stats(train, "train")
    print_stats(val, "val")
    print_stats(test, "test")

if __name__ == "__main__":
    main()
