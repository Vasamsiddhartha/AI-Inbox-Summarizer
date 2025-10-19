#!/usr/bin/env python3
"""
ingest_gmail.py

Incremental Gmail ingest using Gmail API history.list with safe fallback.
Outputs (per run):
 - data/raw/gmail/gmail_latest.jsonl  (overwritten each run)
 - data/raw/gmail/gmail_YYYYMMDD_HHMMSS.jsonl (archived copy)
 - data/raw/gmail/gmail_latest.csv    (flattened CSV, overwritten each run)
 - data/raw/gmail/gmail_YYYYMMDD_HHMMSS.csv (archived flattened CSV)
 - data/raw/gmail/combined_gmail_messages.csv (append-new, deduped by id)
 - state/last_history_id.txt
 - state/last_internal_date.txt

Usage examples:
 python ingest_gmail.py --credentials credentials.json
 python ingest_gmail.py --credentials credentials.json --initial_days 14
 
python ingest_gmail.py --credentials credentials.json --max_messages 5000 --full_sync
Notes:
 - Requires google-api-python-client and google-auth-... packages.
 - pandas is optional (used for dedupe). If missing, script falls back to csv module.
"""

import argparse
import json
import os
import time
import base64
import random
import re
import csv
from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys

# Google client imports (must be installed)
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from tqdm import tqdm
from bs4 import BeautifulSoup

# Optional pandas for easy CSV dedupe/append
try:
    import pandas as pd
except Exception:
    pd = None

# ---------- Config ----------
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

DEFAULT_CRED = "credentials.json.json"
DEFAULT_TOKEN = "state/token.json"
DEFAULT_OUT_DIR = Path("data/raw/gmail")
DEFAULT_STATE_DIR = Path("state")
LATEST_JSONL = "gmail_latest.jsonl"
LATEST_CSV = "gmail_latest.csv"
COMBINED_CSV = "combined_gmail_messages.csv"

MAX_RETRIES = 5
BACKOFF_BASE = 1.5

MAX_BODY_CHARS = 2000  # truncate message bodies to this many chars for CSV

# ---------- Helpers ----------
def ensure_dirs(out_dir: Path, state_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

def safe_b64decode(s: str) -> bytes:
    if not s:
        return b""
    try:
        padding = 4 - (len(s) % 4)
        if padding and padding != 4:
            s = s + ("=" * padding)
        return base64.urlsafe_b64decode(s)
    except Exception:
        try:
            return base64.b64decode(s)
        except Exception:
            return b""

def extract_text_from_part(part: dict) -> str:
    """
    Recursively extract human-readable text from a message part.
    Converts HTML to plain text when necessary.
    """
    if not part:
        return ""
    mime = part.get("mimeType", "") or ""
    body = part.get("body", {}) or {}
    data = body.get("data")
    if data:
        try:
            raw_bytes = safe_b64decode(data)
            raw = raw_bytes.decode("utf-8", errors="ignore")
            if "html" in mime.lower():
                return BeautifulSoup(raw, "html.parser").get_text(separator="\n")
            return raw
        except Exception:
            pass
    # multipart: iterate subparts
    for sub in part.get("parts", []) or []:
        t = extract_text_from_part(sub)
        if t:
            return t
    return ""

def decode_body_from_payload(payload: dict) -> str:
    if not payload:
        return ""
    # prefer the readable text in parts
    text = extract_text_from_part(payload)
    if text:
        return text
    # fallback to snippet
    return payload.get("snippet", "") or ""

def parse_headers_to_map(headers_list):
    m = {}
    for h in headers_list or []:
        name = h.get("name", "")
        value = h.get("value", "")
        if name:
            m[name.lower()] = value
    return m

def parse_date_header_to_iso(date_str: str) -> str:
    if not date_str:
        return ""
    # strip parenthetical timezone labels like (IST)
    clean = re.sub(r"\(.*?\)", "", date_str).strip()
    try:
        # robust parser from email.utils
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        # fallback: attempt fromisoformat
        try:
            dt = datetime.fromisoformat(clean)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            return ""

def flatten_message_to_row(msg: dict) -> dict:
    """
    Convert Gmail message resource to a flat dict row for CSV.
    Fields:
      id, threadId, source, sender, sender_email, subject, body, snippet, timestamp_iso, internalDate_ms, labels, sizeEstimate
    """
    payload = msg.get("payload", {})
    headers_list = payload.get("headers", []) if payload else []
    headers_map = parse_headers_to_map(headers_list)

    subject = headers_map.get("subject", "") or msg.get("subject", "")
    sender = headers_map.get("from", "") or msg.get("from", "")
    # attempt to extract a simple sender email
    sender_email = ""
    m = re.search(r"<([^>]+)>", sender or "")
    if m:
        sender_email = m.group(1)
    else:
        # if it's just an email
        if "@" in (sender or ""):
            sender_email = sender.strip()

    date_iso = parse_date_header_to_iso(headers_map.get("date", "")) or ""
    internal_ms = None
    if msg.get("internalDate"):
        try:
            internal_ms = int(msg.get("internalDate"))
            # if no date_iso, set from internalDate
            if not date_iso:
                date_iso = datetime.utcfromtimestamp(internal_ms / 1000.0).replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            internal_ms = None

    body = decode_body_from_payload(payload) or msg.get("snippet", "") or ""
    if body is None:
        body = ""
    # Normalize whitespace and truncate
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + " ... (truncated)"

    labels = ",".join(msg.get("labelIds", []) or [])
    size_est = msg.get("sizeEstimate", "")

    row = {
        "id": msg.get("id", ""),
        "threadId": msg.get("threadId", ""),
        "source": "gmail",
        "sender": sender,
        "sender_email": sender_email,
        "subject": subject,
        "body": body,
        "snippet": msg.get("snippet", "") or "",
        "timestamp_iso": date_iso,
        "internalDate_ms": internal_ms or "",
        "labels": labels,
        "sizeEstimate": size_est
    }
    return row

def save_jsonl_messages(messages, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        for m in messages:
            fw.write(json.dumps(m, ensure_ascii=False) + "\n")

def save_csv_rows(rows, out_path: Path, headers=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if pd:
        # Use pandas for consistent CSV writing
        df = pd.DataFrame(rows)
        if headers:
            # ensure column order if provided
            cols = [c for c in headers if c in df.columns] + [c for c in df.columns if c not in (headers or [])]
            df.to_csv(out_path, index=False, columns=cols, encoding="utf-8")
        else:
            df.to_csv(out_path, index=False, encoding="utf-8")
    else:
        # Fallback to csv module
        if not rows:
            # write empty file
            with out_path.open("w", encoding="utf-8", newline="") as fw:
                pass
            return
        headers_local = headers or list(rows[0].keys())
        with out_path.open("w", encoding="utf-8", newline="") as fw:
            writer = csv.DictWriter(fw, fieldnames=headers_local)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: (r.get(k, "") if r.get(k, "") is not None else "") for k in headers_local})

def append_to_combined_csv(rows, combined_path: Path):
    """
    Append new rows to combined CSV but avoid duplicates by 'id'.
    If pandas available, we use it; else we do a simple read+filter.
    """
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if pd:
        # load existing if present
        if combined_path.exists():
            try:
                existing = pd.read_csv(combined_path, dtype=str)
            except Exception:
                existing = pd.DataFrame(columns=rows[0].keys())
        else:
            existing = pd.DataFrame(columns=rows[0].keys())
        new_df = pd.DataFrame(rows).astype(str)
        # remove duplicates by id
        if "id" in existing.columns:
            existing_ids = set(existing["id"].astype(str).tolist())
            new_df = new_df[~new_df["id"].isin(existing_ids)]
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
        combined.to_csv(combined_path, index=False, encoding="utf-8")
    else:
        # read existing ids
        existing_ids = set()
        if combined_path.exists():
            try:
                with combined_path.open("r", encoding="utf-8", newline="") as rf:
                    rdr = csv.DictReader(rf)
                    for r in rdr:
                        if "id" in r:
                            existing_ids.add(r["id"])
            except Exception:
                existing_ids = set()
        # append only new rows
        write_header = not combined_path.exists()
        with combined_path.open("a", encoding="utf-8", newline="") as af:
            writer = None
            for r in rows:
                if str(r.get("id","")) in existing_ids:
                    continue
                if writer is None:
                    if write_header:
                        writer = csv.DictWriter(af, fieldnames=list(r.keys()))
                        writer.writeheader()
                    else:
                        # reuse header from file; get keys from r but order may differ
                        writer = csv.DictWriter(af, fieldnames=list(r.keys()))
                writer.writerow({k: (r.get(k,"") if r.get(k,"") is not None else "") for k in r.keys()})

def exponential_backoff_sleep(attempt):
    delay = BACKOFF_BASE ** attempt + random.random()
    time.sleep(delay)

# ---------- Gmail helpers (unchanged logic) ----------
def build_service(creds_path: str, token_path: str):
    creds = None
    token_file = Path(token_path)
    if token_file.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        except Exception:
            creds = None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return service

def fetch_message_full(service, user_id: str, msg_id: str, format: str = "full"):
    for attempt in range(MAX_RETRIES):
        try:
            msg = service.users().messages().get(userId=user_id, id=msg_id, format=format).execute()
            return msg
        except Exception as e:
            if attempt + 1 >= MAX_RETRIES:
                raise
            exponential_backoff_sleep(attempt)
    raise RuntimeError("Failed to fetch message after retries")

def fetch_messages_by_ids(service, user_id: str, ids, max_messages=None):
    results = []
    count = 0
    for mid in tqdm(ids, desc="Fetching messages"):
        if max_messages and count >= max_messages:
            break
        try:
            m = fetch_message_full(service, user_id, mid, format="full")
            results.append(m)
            count += 1
        except Exception as e:
            print(f"Warning: failed to fetch message {mid}: {e}", file=sys.stderr)
            continue
    return results

def list_history_messages(service, user_id: str, start_history_id: str, max_results=1000):
    message_ids = []
    page_token = None
    resp = None
    for attempt in range(MAX_RETRIES):
        try:
            request = service.users().history().list(userId=user_id, startHistoryId=start_history_id, pageToken=page_token, maxResults=1000)
            while request:
                resp = request.execute()
                for h in resp.get("history", []):
                    for ma in h.get("messagesAdded", []):
                        m = ma.get("message")
                        if m and "id" in m:
                            message_ids.append(m["id"])
                page_token = resp.get("nextPageToken")
                if page_token:
                    request = service.users().history().list(userId=user_id, startHistoryId=start_history_id, pageToken=page_token, maxResults=1000)
                else:
                    request = None
            latest_history_id = resp.get("historyId") if resp else None
            return message_ids, latest_history_id
        except Exception as e:
            err_text = str(e).lower()
            if "not found" in err_text or "invalid" in err_text or "starthistoryid" in err_text:
                raise RuntimeError("startHistoryId expired or invalid")
            if attempt + 1 >= MAX_RETRIES:
                raise
            exponential_backoff_sleep(attempt)
    return message_ids, None

def list_messages_after_date(service, user_id: str, after_timestamp_ms: int, max_results=5000):
    dt = datetime.utcfromtimestamp(after_timestamp_ms / 1000.0)
    after_date_str = dt.strftime("%Y/%m/%d")
    q = f"after:{after_date_str}"
    ids = []
    request = service.users().messages().list(userId=user_id, q=q, pageToken=None, maxResults=500)
    while request:
        resp = request.execute()
        for m in resp.get("messages", []):
            ids.append(m["id"])
            if max_results and len(ids) >= max_results:
                return ids
        token = resp.get("nextPageToken")
        if token:
            request = service.users().messages().list(userId=user_id, q=q, pageToken=token, maxResults=500)
        else:
            request = None
    return ids

def list_recent_messages(service, user_id: str, days: int = 7, max_results=5000):
    cutoff = datetime.utcnow() - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y/%m/%d")
    q = f"after:{cutoff_str}"
    ids = []
    request = service.users().messages().list(userId=user_id, q=q, pageToken=None, maxResults=500)
    while request:
        resp = request.execute()
        for m in resp.get("messages", []):
            ids.append(m["id"])
            if max_results and len(ids) >= max_results:
                return ids
        token = resp.get("nextPageToken")
        if token:
            request = service.users().messages().list(userId=user_id, q=q, pageToken=token, maxResults=500)
        else:
            request = None
    return ids

# ---------- Main CLI ----------
def main():
    p = argparse.ArgumentParser(description="Incremental Gmail ingest (history.list fallback) with CSV output.")
    p.add_argument("--credentials", "-c", type=str, default=DEFAULT_CRED, help="Path to OAuth client credentials JSON.")
    p.add_argument("--token", type=str, default=str(DEFAULT_TOKEN), help="Path to store token JSON.")
    p.add_argument("--out_dir", "-o", type=str, default=str(DEFAULT_OUT_DIR), help="Directory to write outputs.")
    p.add_argument("--state_dir", "-s", type=str, default=str(DEFAULT_STATE_DIR), help="Directory to store state files.")
    p.add_argument("--initial_days", type=int, default=7, help="On first run fetch this many days of messages.")
    p.add_argument("--max_messages", type=int, default=None, help="Maximum messages to fetch this run (for safety).")
    p.add_argument("--full_sync", action="store_true", help="Force full sync (ignore historyId and internalDate fallback).")
    p.add_argument("--user_id", type=str, default="me", help="Gmail userId (usually 'me').")
    args = p.parse_args()

    creds_path = Path(args.credentials)
    if not creds_path.exists():
        print(f"Credentials file not found: {creds_path}.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    state_dir = Path(args.state_dir)
    ensure_dirs(out_dir, state_dir)

    last_history_file = state_dir / "last_history_id.txt"
    last_internal_file = state_dir / "last_internal_date.txt"

    service = build_service(str(creds_path), args.token)

    new_messages = []
    latest_history_id = None

    if args.full_sync:
        print("Full sync requested: listing recent messages...")
        ids = list_recent_messages(service, args.user_id, days=args.initial_days, max_results=args.max_messages)
        print(f"Found {len(ids)} messages to fetch.")
        msgs = fetch_messages_by_ids(service, args.user_id, ids, max_messages=args.max_messages)
        new_messages = msgs
        if msgs:
            latest_internal = max(int(m.get("internalDate", 0)) for m in msgs if m.get("internalDate"))
            last_internal_file.write_text(str(latest_internal), encoding="utf-8")
    else:
        start_history_id = None
        if last_history_file.exists():
            start_history_id = last_history_file.read_text(encoding="utf-8").strip()
        if start_history_id:
            print(f"Attempting incremental fetch using historyId: {start_history_id}")
            try:
                ids, latest_history_id = list_history_messages(service, args.user_id, start_history_id, max_results=args.max_messages or 1000)
                print(f"history.list returned {len(ids)} new message ids")
                if ids:
                    msgs = fetch_messages_by_ids(service, args.user_id, ids, max_messages=args.max_messages)
                    new_messages = msgs
            except RuntimeError as e:
                print("history.list fallback triggered:", e)
                start_internal = None
                if last_internal_file.exists():
                    start_internal = last_internal_file.read_text(encoding="utf-8").strip()
                if start_internal:
                    try:
                        after_ms = int(start_internal)
                    except Exception:
                        after_ms = 0
                    print("Falling back to listing messages after stored internalDate:", after_ms)
                    ids = list_messages_after_date(service, args.user_id, after_ms, max_results=args.max_messages or 5000)
                    print(f"Found {len(ids)} messages via fallback listing")
                    if ids:
                        msgs = fetch_messages_by_ids(service, args.user_id, ids, max_messages=args.max_messages)
                        new_messages = msgs
                else:
                    print("No last_internal_date found, performing recent-days fetch")
                    ids = list_recent_messages(service, args.user_id, days=args.initial_days, max_results=args.max_messages)
                    print(f"Found {len(ids)} recent messages")
                    if ids:
                        msgs = fetch_messages_by_ids(service, args.user_id, ids, max_messages=args.max_messages)
                        new_messages = msgs
        else:
            print("No last_history_id found. Initial run: listing recent messages")
            ids = list_recent_messages(service, args.user_id, days=args.initial_days, max_results=args.max_messages)
            print(f"Found {len(ids)} recent messages to fetch.")
            if ids:
                msgs = fetch_messages_by_ids(service, args.user_id, ids, max_messages=args.max_messages)
                new_messages = msgs

    # Postprocess and write outputs
    if new_messages:
        latest_path = out_dir / LATEST_JSONL
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archive_jsonl = out_dir / f"gmail_{ts}.jsonl"
        save_jsonl_messages(new_messages, latest_path)
        save_jsonl_messages(new_messages, archive_jsonl)
        print(f"Saved {len(new_messages)} messages to {latest_path} and archived to {archive_jsonl}")

        # build flattened CSV rows
        rows = [flatten_message_to_row(m) for m in new_messages]
        latest_csv_path = out_dir / LATEST_CSV
        archive_csv_path = out_dir / f"gmail_{ts}.csv"
        # define column order for CSV
        col_order = ["id","threadId","source","sender","sender_email","subject","body","snippet","timestamp_iso","internalDate_ms","labels","sizeEstimate"]
        save_csv_rows(rows, latest_csv_path, headers=col_order)
        save_csv_rows(rows, archive_csv_path, headers=col_order)
        print(f"Wrote flattened CSV to {latest_csv_path} and archived CSV to {archive_csv_path}")

        # append to combined CSV (dedupe by id)
        combined_path = out_dir / COMBINED_CSV
        append_to_combined_csv(rows, combined_path)
        print(f"Appended new rows to combined CSV: {combined_path}")

        # Update last_internal_date:
        internal_dates = [int(m.get("internalDate", 0)) for m in new_messages if m.get("internalDate")]
        if internal_dates:
            max_internal = max(internal_dates)
            last_internal_file.write_text(str(max_internal), encoding="utf-8")
            print(f"Updated last_internal_date -> {max_internal}")

        # Update last_history_id if available; else set best-effort from profile
        if latest_history_id:
            last_history_file.write_text(str(latest_history_id), encoding="utf-8")
            print(f"Updated last_history_id -> {latest_history_id}")
        else:
            try:
                profile = service.users().getProfile(userId=args.user_id).execute()
                cur_hid = profile.get("historyId")
                if cur_hid:
                    last_history_file.write_text(str(cur_hid), encoding="utf-8")
                    print(f"Set last_history_id (best-effort) -> {cur_hid}")
            except Exception:
                pass
    else:
        print("No new messages fetched this run.")
        try:
            profile = service.users().getProfile(userId=args.user_id).execute()
            cur_hid = profile.get("historyId")
            if cur_hid:
                last_history_file.write_text(str(cur_hid), encoding="utf-8")
        except Exception:
            pass

    print("Done.")

if __name__ == "__main__":
    main()
