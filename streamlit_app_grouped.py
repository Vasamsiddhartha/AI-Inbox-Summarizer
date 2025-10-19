# streamlit_app_grouped.py
"""
Streamlit UI (grouped) for Daily Brief archive + latest brief.

Features:
 - Loads archived briefs from data/daily/daily_briefs.json (preferred) or scans data/daily/*.json
 - Presents date selector (newest-first) and defaults to latest brief
 - Shows headline, brief, bullets, merged items toggle, actions and extracted actions
 - Human review queue from state/human_review.jsonl
 - Download options for displayed bullets / actions / itemized rows
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import datetime
import re
import os

# -------- config ----------
PROJECT_ROOT = Path(".").resolve()
DAILY_DIR = PROJECT_ROOT / "data" / "daily"
COMBINED_BRIEFS = DAILY_DIR / "daily_briefs.json"      # preferred combined file
JSONL_BRIEFS = DAILY_DIR / "daily_briefs.jsonl"        # incremental archive
HUMAN_REVIEW_DEFAULT = PROJECT_ROOT / "state" / "human_review.jsonl"
STATE_FILE = PROJECT_ROOT / "streamlit_state.json"

RE_AMOUNT = re.compile(r"\b(?:INR|Rs\.?|₹)\s?[0-9,]+(?:\.[0-9]{1,2})?\b", re.I)

st.set_page_config(page_title="AI Inbox Daily Brief (Archive)", layout="wide")

# -------- helpers ----------
def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Could not read {path}: {e}")
        return {}

def read_jsonl(path: Path) -> List[dict]:
    items = []
    if not path.exists():
        return items
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    # keep raw line wrapper for debugging
                    items.append({"raw": line})
    except Exception as e:
        st.error(f"Could not read JSONL {path}: {e}")
    return items

def list_dailies_from_files(ddir: Path) -> List[dict]:
    files = sorted(ddir.glob("daily_brief_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    briefs = []
    for f in files:
        try:
            item = json.loads(f.read_text(encoding="utf-8"))
            # attach source meta if missing
            if "file" not in item:
                item["file"] = f.name
            # attach a created_at if none (use file mtime)
            if "created_at" not in item:
                item["created_at"] = datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            briefs.append(item)
        except Exception:
            # ignore malformed files but keep raw if possible
            try:
                raw = f.read_text(encoding="utf-8")
                briefs.append({"raw": raw, "file": f.name, "created_at": datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat()})
            except Exception:
                continue
    return briefs

def load_all_briefs() -> List[dict]:
    # 1) prefer combined JSON array
    if COMBINED_BRIEFS.exists():
        try:
            arr = json.loads(COMBINED_BRIEFS.read_text(encoding="utf-8"))
            if isinstance(arr, dict):
                arr = [arr]
            # ensure newest-first
            return sorted(arr, key=lambda x: x.get("created_at", ""), reverse=True)
        except Exception:
            # fallback to jsonl or individual files
            st.warning("Could not parse combined briefs file; falling back to per-file read.")
    # 2) try jsonl (each line is a JSON object)
    if JSONL_BRIEFS.exists():
        arr = read_jsonl(JSONL_BRIEFS)
        # attach created_at if missing using no reliable timestamp (we'll try to parse date inside object)
        for idx, it in enumerate(arr):
            if isinstance(it, dict) and "created_at" not in it:
                it["created_at"] = it.get("date") or it.get("ts") or ""
        # remove raw-line entries that couldn't be parsed? keep them for audit
        return arr
    # 3) fallback to scanning daily files
    if DAILY_DIR.exists():
        return list_dailies_from_files(DAILY_DIR)
    return []

def save_state(state: dict, path: Path):
    try:
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Failed to save UI state: {e}")

def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def infer_category_from_text(text: str) -> str:
    if not text:
        return "personal"
    t = text.lower()
    if any(k in t for k in ["debited", "credited", "upi", "payment", "invoice", "receipt", "txn", "paid", "order #", "order", "amount", "debit", "card"]):
        return "transaction"
    if any(k in t for k in ["flight", "fare", "hotel", "booking", "itinerary", "pnr", "cheap flights", "discount", "sale", "coupon", "deal"]):
        return "promotion"
    if any(k in t for k in ["password", "otp", "verification", "security", "suspicious", "login attempt", "reset your password", "blocked"]):
        return "security"
    if any(k in t for k in ["meeting", "deadline", "project", "presentation", "sprint", "review", "interview"]):
        return "work"
    if any(k in t for k in ["shipped", "tracking", "delivery", "out for delivery", "delivered"]):
        return "shipping"
    if any(k in t for k in ["liked your post", "followed", "friend request", "mentioned", "commented"]):
        return "social"
    if any(k in t for k in ["reminder", "appointment", "alert", "notification", "subscription", "renewal"]):
        return "notification"
    return "personal"

def infer_priority_from_text(text: str) -> str:
    if not text:
        return "low"
    t = text.lower()
    high_keywords = ["verify", "unauthor", "unauthorized", "unauthorised", "urgent", "asap", "dispute", "block", "blocked", "fraud", "chargeback", "refund", "due", "overdue", "password", "security"]
    if any(k in t for k in high_keywords):
        return "high"
    medium_keywords = ["reminder", "meeting", "rsvp", "confirm", "scheduled", "appointment", "renewal", "subscription", "invoice"]
    if any(k in t for k in medium_keywords):
        return "medium"
    low_keywords = ["offer", "discount", "sale", "coupon", "deal", "newsletter", "promo", "price drop", "cheap flights", "save"]
    if any(k in t for k in low_keywords):
        return "low"
    if RE_AMOUNT.search(text):
        return "medium"
    return "low"

def structured_from_item_line(line: str) -> dict:
    amt_m = RE_AMOUNT.search(line)
    merchant = ""
    timestamp = ""
    m_m = re.search(r"M:\s*([^|—]+)", line, re.I)
    if m_m:
        merchant = m_m.group(1).strip()
    else:
        parts = line.split("—")
        if len(parts) >= 3:
            merchant = parts[1].strip()
            timestamp = parts[2].strip()
        elif len(parts) == 2:
            merchant = parts[1].strip()
    if not timestamp:
        t_m = re.search(r"\d{4}-\d{2}-\d{2}T?\d{2}:\d{2}:\d{2}", line)
        if t_m:
            timestamp = t_m.group(0)
    cat = infer_category_from_text(line)
    pri = infer_priority_from_text(line)
    return {
        "raw": line,
        "amount": amt_m.group(0) if amt_m else "",
        "merchant": merchant,
        "timestamp": timestamp,
        "inferred_category": cat,
        "inferred_priority": pri
    }

def list_to_df(itemized: List[str]) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(itemized):
        stct = structured_from_item_line(s)
        rows.append({
            "idx": i+1,
            "text": s,
            "amount": stct["amount"],
            "merchant": stct["merchant"],
            "timestamp": stct["timestamp"],
            "category": stct["inferred_category"]
        })
    if not rows:
        return pd.DataFrame(columns=["idx","text","amount","merchant","timestamp","category"])
    return pd.DataFrame(rows)

def actions_list_to_df(actions: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(actions):
        action_text = a.get("action","") if isinstance(a, dict) else str(a)
        rows.append({
            "idx": i+1,
            "id": str(a.get("id","")) if isinstance(a, dict) else "",
            "action": action_text,
            "due": a.get("due", "") if isinstance(a, dict) else "",
            "type": a.get("type","") if isinstance(a, dict) else "",
            "source": a.get("source", a.get("_source","")) if isinstance(a, dict) else "",
            "priority_label": a.get("priority_label", "") if isinstance(a, dict) else "",
            "confidence": a.get("confidence", 0.0) if isinstance(a, dict) else 0.0,
            "category": infer_category_from_text(action_text),
        })
    if not rows:
        return pd.DataFrame(columns=["idx","id","action","due","type","source","priority_label","confidence","category"])
    return pd.DataFrame(rows)

def merge_bullets_and_items(bullets: List[str], itemized: List[str]) -> List[str]:
    merged = []
    seen = set()
    for b in bullets:
        norm = re.sub(r"\s+", " ", b.strip()).lower()
        if norm not in seen:
            merged.append(b)
            seen.add(norm)
    for it in itemized:
        norm = re.sub(r"\s+", " ", it.strip()).lower()
        if norm not in seen:
            merged.append(it)
            seen.add(norm)
    return merged

def download_df_as_csv(df: pd.DataFrame, fname: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=fname, mime="text/csv")

# -------- load data (briefs archive) ----------
all_briefs = load_all_briefs()
if not all_briefs:
    st.warning("No daily brief archive found. Place daily briefs in data/daily/ or run pipeline.")
# Normalize brief objects: ensure keys exist
for b in all_briefs:
    if not isinstance(b, dict):
        continue
    if "created_at" not in b:
        # try to find a date inside object (common keys), else fallback empty string
        b["created_at"] = b.get("date") or b.get("ts") or b.get("timestamp") or ""

# Sort newest-first using created_at if available, otherwise original order
def brief_sort_key(x):
    d = x.get("created_at","") or x.get("date","") or ""
    try:
        return datetime.datetime.fromisoformat(d)
    except Exception:
        return datetime.datetime.min

all_briefs_sorted = sorted(all_briefs, key=lambda x: brief_sort_key(x) if x.get("created_at") else datetime.datetime.min, reverse=True)

# Build selector options
options = []
for b in all_briefs_sorted:
    label_date = ""
    if b.get("created_at"):
        try:
            dt = datetime.datetime.fromisoformat(b["created_at"])
            label_date = dt.strftime("%Y-%m-%d")
        except Exception:
            label_date = str(b.get("created_at"))[:10]
    else:
        label_date = b.get("file", "unknown")[:16]
    title = b.get("headline") or b.get("brief")[:80] if b.get("brief") else b.get("file", "")
    options.append({"label": f"{label_date} — {title}", "brief": b})

# -------- sidebar / controls ----------
st.sidebar.title("Archive")
if options:
    sel_idx = st.sidebar.selectbox("Select brief (newest first)", options=list(range(len(options))), format_func=lambda i: options[i]["label"], index=0)
    selected_brief = options[sel_idx]["brief"]
else:
    selected_brief = {}

if st.sidebar.button("Reload briefs"):
    st.experimental_rerun()

# -------- header / main view ----------
st.title("📋 AI Inbox — Daily Brief Archive")
with st.container():
    headline = selected_brief.get("headline") or selected_brief.get("brief", "No headline found")
    brief_txt = selected_brief.get("brief") or ""
    st.header(headline)
    if brief_txt:
        st.write(brief_txt)

    meta_col1, meta_col2 = st.columns([3,1])
    with meta_col1:
        st.markdown(f"**Source file:** {selected_brief.get('file','(inline)')}")
    with meta_col2:
        st.markdown("**Meta**")
        st.write(f"Created: {selected_brief.get('created_at','n/a')}")
        st.write(f"Total bullets: {len(selected_brief.get('bullets', []))}")
        st.write(f"Extracted actions: {len(selected_brief.get('extracted_actions', []))}")

st.markdown("---")

# prepare data
bullets = selected_brief.get("bullets", []) or []
actions_from_summarizer = selected_brief.get("actions_from_summarizer", []) or []
extracted_actions = selected_brief.get("extracted_actions", []) or []
all_itemized = selected_brief.get("all_itemized_lines", []) or selected_brief.get("selected_lines", []) or []
if isinstance(all_itemized, dict):
    all_itemized = list(all_itemized.values())

item_df = list_to_df(all_itemized)
action_df = actions_list_to_df(extracted_actions)

# ---------- Bullets section ----------
st.markdown("## Key Bullets (filter & merge)")

col_b1, col_b2 = st.columns([3,2])
with col_b1:
    cat_options = sorted(set([infer_category_from_text(b) for b in bullets] + list(item_df["category"].unique())))
    sel_bullet_cats = st.multiselect("Bullet categories (filter)", options=cat_options, default=cat_options)
with col_b2:
    merge_toggle = st.checkbox("Merge LLM bullets + itemized lines (preview)", value=False)

# Build bullet records
bullet_records = []
for b in bullets:
    cat = infer_category_from_text(b)
    pri = infer_priority_from_text(b)
    bullet_records.append({"text": b, "category": cat, "priority": pri})

if merge_toggle:
    merged = merge_bullets_and_items([r["text"] for r in bullet_records], list(item_df["text"]))
    merged_records = []
    for m in merged:
        merged_records.append({"text": m, "category": infer_category_from_text(m), "priority": infer_priority_from_text(m)})
    display_records = merged_records
else:
    display_records = bullet_records

display_records = [r for r in display_records if r["category"] in sel_bullet_cats]

order = {"high": 0, "medium": 1, "low": 2}
display_records = sorted(display_records, key=lambda x: order.get(x.get("priority","low"), 2))

st.write(f"Showing {len(display_records)} bullets (merge={merge_toggle})")
if not display_records:
    st.info("No bullets after applying filters.")
else:
    for i, rec in enumerate(display_records, start=1):
        if rec['category'] == 'transaction':
            st.markdown(f"**{i}. [Transaction]** {rec['text']}")
        else:
            st.markdown(f"**{i}. [{rec['category'].capitalize()} | {rec['priority'].upper()}]** {rec['text']}")

if display_records:
    if st.button("Download displayed bullets as CSV"):
        df_bul = pd.DataFrame(display_records)
        download_df_as_csv(df_bul, "displayed_bullets.csv")

st.markdown("---")

# -------- Actions / Checklist (grouped) ----------
st.markdown("## Actions / Checklist (grouped by category)")

if actions_from_summarizer:
    st.subheader("Summarizer actions")
    state = load_state(STATE_FILE)
    for i, a in enumerate(actions_from_summarizer, start=1):
        key = f"llm_action_{i}"
        checked = state.get("llm_action", {}).get(key, False)
        new_val = st.checkbox(f"[Summarizer] {a}", value=checked, key=key)
        if new_val != checked:
            state.setdefault("llm_action", {})[key] = new_val
            save_state(state, STATE_FILE)

if not action_df.empty:
    st.subheader("Extracted action items")
    cats = sorted(action_df["category"].unique().tolist())
    sel_cats_actions = st.multiselect("Filter actions by category", options=cats, default=cats)
    filtered_actions = action_df[action_df["category"].isin(sel_cats_actions)] if sel_cats_actions else action_df
    for cat, sub in filtered_actions.groupby("category"):
        st.markdown(f"### {cat.capitalize()} ({len(sub)})")
        for _, row in sub.iterrows():
            aid = f"{row['id']}::{row['idx']}"
            state = load_state(STATE_FILE)
            checked = state.get("actions_done", {}).get(aid, False)
            label = f"[{row['priority_label'].upper() if row['priority_label'] else 'N/A'}] {row['action']}"
            new_val = st.checkbox(label, value=checked, key=f"act_{aid}")
            if new_val != checked:
                state.setdefault("actions_done", {})[aid] = new_val
                save_state(state, STATE_FILE)
    st.markdown("**Export extracted actions**")
    if st.button("Download extracted actions CSV"):
        download_df = filtered_actions.drop(columns=["idx"]).reset_index(drop=True)
        download_df_as_csv(download_df, "extracted_actions_filtered.csv")
else:
    st.info("No extracted actions found.")

st.markdown("---")

# -------- Full itemized list (audit view) ----------
st.markdown("## Full itemized list (audit view)")
if item_df.empty:
    st.info("No itemized lines available.")
else:
    col1, col2, col3 = st.columns([1,1,1])
    search_q = col1.text_input("Search text", value="")
    categories = sorted(item_df["category"].unique().tolist())
    sel = col2.multiselect("Filter category", options=categories, default=categories)
    amt_only = col3.checkbox("Only show transactions with amount", value=False)

    df_vis = item_df.copy()
    if search_q:
        df_vis = df_vis[df_vis["text"].str.contains(search_q, case=False, na=False)]
    if sel:
        df_vis = df_vis[df_vis["category"].isin(sel)]
    if amt_only:
        df_vis = df_vis[df_vis["amount"] != ""]

    st.write(f"Showing {len(df_vis)} / {len(item_df)} itemized rows")
    st.dataframe(df_vis.reset_index(drop=True), use_container_width=True, height=420)

    csv = df_vis.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered itemized rows (CSV)", data=csv, file_name="itemized_filtered.csv", mime="text/csv")

st.markdown("---")

# -------- Human review queue ----------
st.markdown("## Human review queue")
human_jsonl_path = Path(HUMAN_REVIEW_DEFAULT)
human_review = []
if human_jsonl_path.exists():
    human_review = []
    try:
        with human_jsonl_path.open("r", encoding="utf-8") as hf:
            for line in hf:
                line = line.strip()
                if not line:
                    continue
                try:
                    human_review.append(json.loads(line))
                except Exception:
                    human_review.append({"raw": line})
    except Exception as e:
        st.error(f"Could not read human review file: {e}")

if not human_review:
    st.info("No human-review items found.")
else:
    st.write(f"{len(human_review)} items")
    for i, item in enumerate(human_review, start=1):
        with st.expander(f"[{i}] {item.get('id', 'no-id')}"):
            st.json(item)
            if st.button(f"Mark reviewed (index {i})"):
                reviewed_path = human_jsonl_path.with_name(human_jsonl_path.stem + "_reviewed.jsonl")
                try:
                    with reviewed_path.open("a", encoding="utf-8") as rf:
                        rf.write(json.dumps({"reviewed_at": datetime.datetime.now().isoformat(), "item": item}, ensure_ascii=False) + "\n")
                    # remove the item from human_review file (simple rewrite excluding this index)
                    remaining = human_review[:i-1] + human_review[i:]
                    with human_jsonl_path.open("w", encoding="utf-8") as hf:
                        for it in remaining:
                            hf.write(json.dumps(it, ensure_ascii=False) + "\n")
                    st.success(f"Marked item {i} reviewed and archived to {reviewed_path.name}")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to mark reviewed: {e}")

st.markdown("---")

with st.expander("Raw selected brief (preview)"):
    try:
        st.code(json.dumps(selected_brief, ensure_ascii=False, indent=2)[:4000])
    except Exception:
        st.text("raw preview unavailable")

st.markdown("**Notes**")
st.markdown("""
- Choose a brief from the archive (sidebar). The newest brief is selected by default.
- Bullets are ordered High → Medium → Low (priority deduction heuristics).
- Toggle "Merge LLM bullets + itemized lines" to preview merged content.
- Actions are persisted in local streamlit_state.json.
""")
