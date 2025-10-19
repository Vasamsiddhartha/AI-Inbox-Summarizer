#!/usr/bin/env python3
"""
email_analysis_dashboard.py

Usage:
    python email_analysis_dashboard.py --input emails_labeled.csv --output_dir analysis_output --max_samples_preview 20

Produces:
 - PNG charts in output_dir
 - report.md in output_dir
 - sample CSVs for manual inspection
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser as dateparser
import numpy as np
import os
import textwrap
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------- Utilities ----------------
def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_parse_datetime(x):
    """Try multiple ways to parse timestamp into pandas Timestamp; return pd.NaT on fail."""
    if pd.isna(x) or x == "":
        return pd.NaT
    # If numeric epoch (seconds or milliseconds)
    try:
        # detect int-like str
        if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()):
            val = int(x)
            # heuristic: if value is in ms range (greater than year 3000)
            if val > 1e12:
                # milliseconds
                return pd.to_datetime(val, unit="ms", utc=True)
            else:
                # seconds
                return pd.to_datetime(val, unit="s", utc=True)
    except Exception:
        pass
    # fallback to dateutil parser
    try:
        dt = dateparser.parse(str(x))
        # Normalize to UTC
        if dt is None:
            return pd.NaT
        if dt.tzinfo is None:
            # treat as naive local -> convert to UTC (assume UTC)
            return pd.to_datetime(dt).tz_localize('UTC')
        return pd.to_datetime(dt).tz_convert('UTC')
    except Exception:
        return pd.NaT

def plot_and_save(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logging.info(f"Saved chart: {out_path}")

# ---------------- Analysis functions ----------------
def load_and_prepare(input_path: Path):
    logging.info(f"Loading CSV: {input_path}")
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])
    # ensure canonical columns exist
    expected_cols = ["id", "source", "sender", "subject", "body", "timestamp",
                     "category", "priority_label", "action_required", "action_text"]
    for c in expected_cols:
        if c not in df.columns:
            logging.warning(f"Column '{c}' not found in CSV - creating empty column.")
            df[c] = ""
    # Normalize some columns
    df["category"] = df["category"].fillna("").astype(str).str.strip().str.lower()
    df["priority_label"] = df["priority_label"].fillna("").astype(str).str.strip().str.lower()
    # action_required could be '0'/'1' or numeric; convert to int where possible
    def to_int_flag(v):
        try:
            if v is None or v == "":
                return 0
            if isinstance(v, (int, np.integer)):
                return int(v)
            vs = str(v).strip().lower()
            if vs in ("1", "true", "yes", "y"):
                return 1
            return 0
        except Exception:
            return 0
    df["action_required"] = df["action_required"].apply(to_int_flag)
    # Parse timestamps
    df["parsed_ts"] = df["timestamp"].apply(safe_parse_datetime)
    # Create a date column for grouping (UTC date)
    df["received_date"] = df["parsed_ts"].dt.date
    return df

def summary_stats(df: pd.DataFrame):
    n = len(df)
    categories = df["category"].replace("", "unknown")
    priority = df["priority_label"].replace("", "unknown")
    action_counts = df["action_required"].value_counts().to_dict()
    stats = {
        "total_messages": n,
        "category_counts": categories.value_counts(dropna=False).to_dict(),
        "priority_counts": priority.value_counts(dropna=False).to_dict(),
        "action_required_counts": action_counts
    }
    return stats

def chart_category_distribution(df: pd.DataFrame, out_path: Path):
    counts = df["category"].replace("", "unknown").value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Emails by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    plot_and_save(fig, out_path)

def chart_priority_distribution(df: pd.DataFrame, out_path: Path):
    counts = df["priority_label"].replace("", "unknown").value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Emails by Priority")
    ax.set_xlabel("Priority")
    ax.set_ylabel("Count")
    plot_and_save(fig, out_path)

def chart_action_required(df: pd.DataFrame, out_path: Path):
    counts = df["action_required"].value_counts().sort_index()
    # Replace index for readability
    idx = [str(int(i)) for i in counts.index]
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_xticklabels(["No action (0)" if i=="0" else "Action required (1)" for i in idx], rotation=0)
    ax.set_title("Action Required Counts")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plot_and_save(fig, out_path)

def chart_emails_over_time(df: pd.DataFrame, out_path: Path, freq="D"):
    # compute counts per day (or per freq)
    if "parsed_ts" not in df.columns:
        logging.warning("No parsed timestamps available; skipping time series chart.")
        return
    ts = df.dropna(subset=["parsed_ts"]).set_index("parsed_ts")
    if ts.empty:
        logging.warning("No valid parsed timestamps present; skipping time series chart.")
        return
    counts = ts["id"].resample(freq).count()
    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot(ax=ax, marker="o", linewidth=1)
    ax.set_title(f"Email counts over time (resampled: {freq})")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Count")
    plot_and_save(fig, out_path)

def chart_priority_by_category(df: pd.DataFrame, out_path: Path):
    # produce a pivot table (counts)
    pivot = pd.crosstab(df["category"].replace("", "unknown"), df["priority_label"].replace("", "unknown"))
    if pivot.empty:
        logging.warning("Empty pivot for priority-by-category; skipping chart.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Priority by Category (stacked)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    plot_and_save(fig, out_path)

def export_samples(df: pd.DataFrame, out_dir: Path, max_samples=50):
    # Random sample
    rnd = df.sample(n=min(len(df), max_samples), random_state=42)
    rnd.to_csv(out_dir / "samples_random.csv", index=False)
    # Action required samples
    df[df["action_required"] == 1].head(max_samples).to_csv(out_dir / "samples_action_required.csv", index=False)
    # High priority samples
    df[df["priority_label"] == "high"].head(max_samples).to_csv(out_dir / "samples_high_priority.csv", index=False)
    # Promotions
    df[df["category"] == "promotion"].head(max_samples).to_csv(out_dir / "samples_promotion.csv", index=False)
    logging.info(f"Exported sample CSVs to {out_dir}")

def write_report(stats: dict, out_dir: Path, images: dict):
    rpt = []
    rpt.append("# Email Analysis Report\n")
    rpt.append(f"- Total messages processed: **{stats['total_messages']}**\n")
    rpt.append("## Category counts\n")
    for k, v in stats["category_counts"].items():
        rpt.append(f"- {k}: {v}\n")
    rpt.append("\n## Priority counts\n")
    for k, v in stats["priority_counts"].items():
        rpt.append(f"- {k}: {v}\n")
    rpt.append("\n## Action required counts\n")
    for k, v in stats["action_required_counts"].items():
        rpt.append(f"- {k}: {v}\n")
    rpt.append("\n## Charts\n")
    for name, p in images.items():
        rpt.append(f"- **{name}**: ![]({p.name})\n")
    rpt.append("\n## Next steps / suggestions\n")
    rpt.append(textwrap.dedent("""
    - Manually inspect `samples_action_required.csv` and `samples_high_priority.csv` to verify label correctness.
    - If many promotions are misclassified, tune the promo-detection rules or prompt examples used in labeling.
    - Consider marking messages with low confidence for manual review and possible fine-tuning.
    - After verification, implement automation (move promos to archive/folder, surface high-priority items in daily digest).
    """))
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(rpt), encoding="utf-8")
    logging.info(f"Wrote report: {report_path}")

def top_senders_table(df: pd.DataFrame, top_n=10):
    s = df["sender"].fillna("unknown").value_counts().head(top_n)
    return s

# ---------------- Main CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Analyze emails_labeled.csv and produce charts + samples.")
    parser.add_argument("--input", "-i", required=True, type=str, help="Path to emails_labeled.csv")
    parser.add_argument("--output_dir", "-o", default="analysis_output", type=str, help="Directory to save charts and reports")
    parser.add_argument("--max_samples_preview", "-n", default=50, type=int, help="How many sample rows to export for manual review (per CSV)")
    parser.add_argument("--time_freq", default="D", type=str, help="Resample frequency for time series (e.g., D for daily, W for weekly)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = ensure_output_dir(Path(args.output_dir))

    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)

    df = load_and_prepare(input_path)
    stats = summary_stats(df)

    # Generate charts
    images = {}
    try:
        img1 = out_dir / "chart_category_distribution.png"
        chart_category_distribution(df, img1)
        images["Category distribution"] = img1
    except Exception as e:
        logging.error(f"Failed to produce category chart: {e}")

    try:
        img2 = out_dir / "chart_priority_distribution.png"
        chart_priority_distribution(df, img2)
        images["Priority distribution"] = img2
    except Exception as e:
        logging.error(f"Failed to produce priority chart: {e}")

    try:
        img3 = out_dir / "chart_action_required.png"
        chart_action_required(df, img3)
        images["Action required"] = img3
    except Exception as e:
        logging.error(f"Failed to produce action-required chart: {e}")

    try:
        img4 = out_dir / "chart_emails_over_time.png"
        chart_emails_over_time(df, img4, freq=args.time_freq)
        images["Emails over time"] = img4
    except Exception as e:
        logging.error(f"Failed to produce time series chart: {e}")

    try:
        img5 = out_dir / "chart_priority_by_category.png"
        chart_priority_by_category(df, img5)
        images["Priority by category"] = img5
    except Exception as e:
        logging.error(f"Failed to produce priority-by-category chart: {e}")

    # export sample CSVs
    export_samples(df, out_dir, max_samples=args.max_samples_preview)

    # write report
    write_report(stats, out_dir, images)

    # save top senders for quick inspection
    top_senders = top_senders_table(df, top_n=20)
    top_senders.to_csv(out_dir / "top_senders.csv", header=["count"])
    logging.info(f"Saved top_senders.csv ({len(top_senders)} rows)")

    logging.info("Analysis complete. Check the output directory for charts, samples, and report.")

if __name__ == "__main__":
    main()
