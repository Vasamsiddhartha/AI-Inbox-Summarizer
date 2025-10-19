#!/usr/bin/env python3
"""
ingest_sms.py

Parse SMS Backup & Restore (Android) XML into a normalized CSV/JSONL dataset
compatible with the hybrid labeling pipeline.

Output schema (one row per message):
id, source, sender, subject, body, timestamp,
category, priority_label, action_required, action_text

Usage:
  python ingest_sms.py --input sms.xml --output-csv data/sms_parsed.csv --output-jsonl data/sms_parsed.jsonl
  python ingest_sms.py -i data/raw/sms/SMS-2024-11-30.xml -o data/merged_messages.csv --append

Options:
  --dedupe        Remove exact dupes by (sender,timestamp,body)
  --limit N       Stop after N messages (for quick tests)
  --append        If output CSV exists, append new rows (will dedupe if requested)
"""

import argparse
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timezone
import hashlib
import os
import sys
import io
from typing import Optional

# --- Helpers ---
def safe_int(v):
    try:
        return int(v)
    except Exception:
        return None

def ms_to_iso(ms):
    """Convert milliseconds since epoch to ISO 8601 UTC string."""
    try:
        ms_i = int(ms)
        # many Android backups store milliseconds since epoch
        # if it's suspiciously small (seconds), try interpreting as seconds
        if ms_i > 1e12:  # > year 33658 in seconds, ok for ms
            dt = datetime.fromtimestamp(ms_i / 1000.0, tz=timezone.utc)
        elif ms_i > 1e9:  # seconds (e.g., 1_600_000_000)
            dt = datetime.fromtimestamp(ms_i, tz=timezone.utc)
        else:
            # fallback: treat as seconds
            dt = datetime.fromtimestamp(ms_i, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        try:
            # fallback: attempt to parse as textual date
            dt = datetime.fromisoformat(str(ms))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            return ""

def make_id(sender: str, timestamp_iso: str, counter: int) -> str:
    # stable-ish id: sms_<sha8(sender+timestamp+counter)>
    key = f"{sender}|{timestamp_iso}|{counter}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"sms_{h}"

def normalize_sender(addr: Optional[str]) -> str:
    if not addr:
        return ""
    addr = addr.strip()
    # Some senders come as numeric, some as alphanumeric. Keep as-is.
    return addr

def extract_sms_attributes(elem: ET.Element) -> dict:
    # Element attributes vary by exporter; common: address, date, date_sent, body, type, read, id
    attrs = elem.attrib
    sender = attrs.get("address") or attrs.get("originator") or attrs.get("from") or ""
    body = attrs.get("body") or attrs.get("text") or ""
    date = attrs.get("date") or attrs.get("date_sent") or attrs.get("timestamp") or ""
    msg_id = attrs.get("id") or attrs.get("_id") or ""
    # Some exporters include "type" where 1=received,2=sent
    msg_type = attrs.get("type") or ""
    return {"sender": sender, "body": body, "date": date, "msg_id": msg_id, "type": msg_type}

# --- Main parsing function ---
def parse_sms_xml(input_path, dedupe=False, limit=None):
    """
    Yields dicts with keys: id, source, sender, subject, body, timestamp, category, priority_label, action_required, action_text
    """
    # Use iterative parser to avoid memory blowup
    context = ET.iterparse(input_path, events=("start","end"))
    # Move to root
    _, root = next(context)

    seen_hashes = set()
    rows = []
    counter = 0
    emitted = 0

    for event, elem in context:
        # SMS Backup & Restore uses tag "sms" for messages, but some exporters might use 'message' or 'm'
        tag = elem.tag.lower()
        if event == "end" and tag in ("sms", "message", "m"):
            counter += 1
            data = extract_sms_attributes(elem)
            sender = normalize_sender(data["sender"])
            body = data["body"] or ""
            # Some bodies are HTML-escaped - ElementTree returns unescaped text
            timestamp_iso = ms_to_iso(data["date"]) if data["date"] else ""
            # fallback if empty: try 'readable_date' attribute or 'date_sent'
            if not timestamp_iso:
                for fallback_key in ("readable_date", "date_sent", "timestamp"):
                    if fallback_key in elem.attrib:
                        timestamp_iso = ms_to_iso(elem.attrib.get(fallback_key) or "")
                        if timestamp_iso:
                            break

            msg_id_raw = data["msg_id"] or ""
            msg_id = msg_id_raw if msg_id_raw else make_id(sender, timestamp_iso, counter)

            # dedupe key
            dedupe_key = None
            if dedupe:
                dedupe_key = hashlib.sha1(f"{sender}|{timestamp_iso}|{body}".encode("utf-8")).hexdigest()
                if dedupe_key in seen_hashes:
                    elem.clear()
                    root.clear()
                    continue
                seen_hashes.add(dedupe_key)

            row = {
                "id": msg_id,
                "source": "sms",
                "sender": sender,
                "subject": "",             # SMS has no subject
                "body": body.strip(),
                "timestamp": timestamp_iso,
                # labels left empty for labeling pipeline
                "category": "",
                "priority_label": "",
                "action_required": "",
                "action_text": ""
            }
            rows.append(row)
            emitted += 1

            # free memory
            elem.clear()
            # also clear parent references occasionally
            if counter % 1000 == 0:
                root.clear()

            if limit and emitted >= limit:
                break

    return rows

# --- CLI and IO ---
def write_outputs(rows, output_csv, output_jsonl, append=False):
    df = pd.DataFrame(rows)
    # Ensure order of columns is consistent with emails dataset
    cols = ["id", "source", "sender", "subject", "body", "timestamp", "category", "priority_label", "action_required", "action_text"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]

    # If append and file exists, read existing and concat with dedupe by id
    if append and os.path.exists(output_csv):
        try:
            existing = pd.read_csv(output_csv, dtype=str)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["id"])
            combined.to_csv(output_csv, index=False, encoding="utf-8")
        except Exception as e:
            print(f"Warning: failed to append to {output_csv}: {e}\nWill overwrite instead.")
            df.to_csv(output_csv, index=False, encoding="utf-8")
    else:
        df.to_csv(output_csv, index=False, encoding="utf-8")

    # Write JSONL
    try:
        with open(output_jsonl, "w", encoding="utf-8") as fj:
            for _, r in df.iterrows():
                fj.write(r.to_json(force_ascii=False) + "\n")
    except Exception as e:
        print(f"Error writing jsonl: {e}")

    print(f"✅ Written CSV: {output_csv} ({len(df)} rows)")
    print(f"✅ Written JSONL: {output_jsonl}")

def main():
    p = argparse.ArgumentParser(description="Parse SMS XML into CSV/JSONL for hybrid labeler")
    p.add_argument("--input", "-i", required=True, help="Input SMS XML file (SMS Backup & Restore / Android SMS XML)")
    p.add_argument("--output-csv", "-o", default="data/SMS_parsed.csv", help="Output CSV path")
    p.add_argument("--output-jsonl", "-j", default="data/SMS_parsed.jsonl", help="Output JSONL path")
    p.add_argument("--dedupe", action="store_true", help="Remove exact duplicates (sender+timestamp+body)")
    p.add_argument("--limit", type=int, default=None, help="Limit to first N messages (for quick test)")
    p.add_argument("--append", action="store_true", help="Append to existing CSV (drop duplicates by id)")
    args = p.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    # Ensure output dir exists
    out_dir = os.path.dirname(args.output_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_dir_j = os.path.dirname(args.output_jsonl) or "."
    os.makedirs(out_dir_j, exist_ok=True)

    print(f"Parsing: {input_path}")
    rows = parse_sms_xml(input_path, dedupe=args.dedupe, limit=args.limit)
    if not rows:
        print("No SMS messages parsed.", file=sys.stderr)
        sys.exit(1)

    write_outputs(rows, args.output_csv, args.output_jsonl, append=args.append)

if __name__ == "__main__":
    main()
