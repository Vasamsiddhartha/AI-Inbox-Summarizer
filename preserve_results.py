#!/usr/bin/env python3
"""
preserve_results.py

Archive outputs from a pipeline run and upsert messages + label history into a SQLite DB.

Usage:
 python preserve_results.py \
   --labeled data/combined_labeled.csv \
   --brief data/daily_brief.json \
   --archive_dir data/daily \
   --db db/messages.db

If any optional argument is omitted, sensible defaults are used.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import hashlib

# ---------- Helpers ----------
def now_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def stable_message_id(row: dict) -> str:
    """
    Return a stable message id:
     - prefer row['id'] if present and not empty
     - else hash (source + sender + subject + timestamp + body excerpt)
    """
    mid = row.get("id") or row.get("message_id") or row.get("msg_id") or ""
    if mid:
        return str(mid)
    # build fingerprint
    parts = [
        str(row.get("source","")),
        str(row.get("sender","")),
        str(row.get("subject","")),
        str(row.get("timestamp","")),
        (str(row.get("body",""))[:300])  # small slice enough for uniqueness
    ]
    return sha1_hex("|".join(parts))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------- DB Helpers ----------
SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    source TEXT,
    sender TEXT,
    subject TEXT,
    body TEXT,
    timestamp TEXT,
    first_seen TEXT
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT,
    category TEXT,
    priority_label TEXT,
    action_required INTEGER,
    action_text TEXT,
    label_source TEXT,
    confidence REAL,
    decision_path TEXT,
    labeled_at TEXT,
    UNIQUE(message_id, labeled_at)
);

CREATE INDEX IF NOT EXISTS idx_labels_message ON labels(message_id);
CREATE TABLE IF NOT EXISTS briefs (
    run_date TEXT PRIMARY KEY,
    file_path TEXT,
    headline TEXT,
    brief TEXT,
    created_at TEXT
);
"""

def init_db(conn):
    cur = conn.cursor()
    cur.executescript(SCHEMA)
    conn.commit()

# ---------- Main logic ----------
def archive_files(labeled_path: Path, brief_path: Path, archive_dir: Path):
    # create dated copies
    t = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    ensure_dir(archive_dir)
    if labeled_path and labeled_path.exists():
        target = archive_dir / f"combined_labeled_{t}.csv"
        target.write_bytes(labeled_path.read_bytes())
    else:
        target = None
    if brief_path and brief_path.exists():
        target_brief = archive_dir / f"daily_brief_{t}.json"
        target_brief.write_bytes(brief_path.read_bytes())
    else:
        target_brief = None
    return target, target_brief

def upsert_messages_and_labels(conn, labeled_csv_path: Path, run_time_iso: str):
    """
    Read labeled CSV, upsert message rows to messages table,
    and insert a new label row into labels if label differs from last label (or always insert version).
    """
    inserted = 0
    with labeled_csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cur = conn.cursor()
    for r in rows:
        msg_id = stable_message_id(r)
        src = r.get("source","")
        sender = r.get("sender","")
        subject = r.get("subject","")
        body = r.get("body","")
        ts = r.get("timestamp","")
        # insert message if new
        cur.execute("SELECT message_id FROM messages WHERE message_id = ?", (msg_id,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO messages(message_id, source, sender, subject, body, timestamp, first_seen) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (msg_id, src, sender, subject, body, ts, run_time_iso)
            )

        # prepare label fields
        category = r.get("category","")
        priority_label = r.get("priority_label","")
        try:
            action_required = int(r.get("action_required", 0) or 0)
        except Exception:
            action_required = 0
        action_text = r.get("action_text","")
        label_source = r.get("_label_source", r.get("label_source",""))
        try:
            confidence = float(r.get("_confidence", r.get("confidence", "")) or 0.0)
        except Exception:
            confidence = 0.0
        decision_path = r.get("decision_path", "")

        # check last label for this message
        cur.execute("SELECT category, priority_label, action_required, action_text FROM labels WHERE message_id = ? ORDER BY labeled_at DESC LIMIT 1", (msg_id,))
        last = cur.fetchone()
        should_insert = False
        if last is None:
            should_insert = True
        else:
            last_cat, last_pri, last_actreq, last_acttext = last
            # insert new version if any of the label fields changed
            if (str(last_cat) != str(category) or
                str(last_pri) != str(priority_label) or
                int(last_actreq) != int(action_required) or
                str(last_acttext) != str(action_text)):
                should_insert = True

        if should_insert:
            cur.execute(
                "INSERT INTO labels(message_id, category, priority_label, action_required, action_text, label_source, confidence, decision_path, labeled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (msg_id, category, priority_label, action_required, action_text, label_source, confidence, decision_path, run_time_iso)
            )
            inserted += 1

    conn.commit()
    return len(rows), inserted

def register_brief(conn, brief_archive_path: Path, run_time_iso: str):
    # try to extract headline/brief from archived file
    headline = ""
    brief = ""
    if brief_archive_path and brief_archive_path.exists():
        try:
            payload = json.loads(brief_archive_path.read_text(encoding="utf-8"))
            headline = payload.get("headline", "")
            brief = payload.get("brief", "")
        except Exception:
            pass
    run_date = run_time_iso.split("T")[0]
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO briefs(run_date, file_path, headline, brief, created_at) VALUES (?, ?, ?, ?, ?)",
                (run_date, str(brief_archive_path) if brief_archive_path else "", headline, brief, run_time_iso))
    conn.commit()

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Archive outputs and upsert into SQLite DB")
    p.add_argument("--labeled", "-l", type=str, default="data/combined_labeled.csv", help="Path to combined_labeled.csv")
    p.add_argument("--brief", "-b", type=str, default="data/daily_brief.json", help="Path to daily_brief.json")
    p.add_argument("--archive_dir", "-a", type=str, default="data/daily", help="Directory to save dated archives")
    p.add_argument("--db", type=str, default="db/messages.db", help="SQLite DB path")
    args = p.parse_args()

    labeled_path = Path(args.labeled)
    brief_path = Path(args.brief)
    archive_dir = Path(args.archive_dir)
    db_path = Path(args.db)
    ensure_dirs = lambda p: p.mkdir(parents=True, exist_ok=True)
    ensure_dirs(archive_dir.parent)
    ensure_dirs(db_path.parent)

    run_time_iso = now_iso()

    # archive files
    arch_labeled, arch_brief = archive_files(labeled_path, brief_path, archive_dir)
    print(f"Archived labeled -> {arch_labeled}, brief -> {arch_brief}")

    # open DB
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        init_db(conn)
        num_rows, num_inserted = 0, 0
        if labeled_path.exists():
            num_rows, num_inserted = upsert_messages_and_labels(conn, labeled_path, run_time_iso)
            print(f"Processed {num_rows} messages, inserted {num_inserted} new/changed label records")
        else:
            print("Warning: labeled CSV not found; skipping message upsert.")

        register_brief(conn, arch_brief, run_time_iso)
        print("Registered brief in DB.")
    finally:
        conn.close()

    print("Preserve step complete.")

if __name__ == "__main__":
    main()
