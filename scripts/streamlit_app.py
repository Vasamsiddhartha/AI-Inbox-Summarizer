#!/usr/bin/env python3
"""
streamlit_app_grouped.py

Streamlit UI (grouped) for Daily Brief with:
 - support for loading two brief JSON paths (primary + optional secondary/past)
 - archive/index support (auto-discovery of data/daily/daily_brief_*.json)
 - bullet category filter and merge toggle
 - Actions checklist and extracted actions table
 - Human review queue preview
 - No priority filter (bullets ordered High -> Medium -> Low)

Usage:
  streamlit run streamlit_app_grouped.py
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import datetime
import re
import os
import shutil

# -------- config ----------
DEFAULT_DAILY_JSON = r"C:\Users\siddhartha\gmail_llama_pipeline\data\daily\daily_brief_20251016.json"
DEFAULT_HUMAN_REVIEW = "human_review.jsonl"
STATE_FILE = "streamlit_state.json"
DAILY_FOLDER = Path("data/daily")
DAILY_INDEX = DAILY_FOLDER / "daily_index.json"
RE_AMOUNT = re.compile(r"\b(?:INR|Rs\.?|₹)\s?[0-9,]+(?:\.[0-9]{1,2})?\b", re.I)

st.set_page_config(page_title="AI Inbox Daily Brief (Grouped)", layout="wide")

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
                    items.append({"raw": line})
    except Exception as e:
        st.error(f"Could not read JSONL {path}: {e}")
    return items

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

# --- index helpers ---
def load_index(index_path: Path) -> List[dict]:
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return sorted(data, key=lambda e: e.get("date", ""), reverse=True)
        return []
    except Exception:
        return []

def save_index(index: List[dict], index_path: Path):
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(index_path)
    except Exception as e:
        st.error(f"Failed to save index: {e}")

def extract_brief_metadata(brief_obj: dict, brief_path: Path) -> dict:
    date = brief_obj.get("date") or brief_obj.get("timestamp") or brief_obj.get("created_at")
    if not date:
        try:
            stat = brief_path.stat()
            date = datetime.datetime.utcfromtimestamp(stat.st_mtime).isoformat()
        except Exception:
            date = datetime.datetime.utcnow().isoformat()
    headline = brief_obj.get("headline") or brief_obj.get("title") or ""
    selected_count = brief_obj.get("selected_count") or 0
    preview = (brief_obj.get("brief","") or "")[:180].replace("\n"," ").strip()
    return {
        "path": str(brief_path),
        "date": date,
        "headline": headline,
        "selected_count": selected_count,
        "preview": preview
    }

def append_brief_to_index(brief_path: Path, index_path: Path) -> dict:
    if not brief_path.exists():
        raise FileNotFoundError(f"Brief file not found: {brief_path}")
    obj = read_json(brief_path)
    meta = extract_brief_metadata(obj, brief_path)
    index = load_index(index_path)
    index = [e for e in index if e.get("path") != meta["path"]]
    index.append(meta)
    index.sort(key=lambda e: e.get("date",""), reverse=True)
    save_index(index, index_path)
    return meta

def auto_discover_and_index(daily_folder: Path, index_path: Path) -> List[dict]:
    """
    If index missing, scan daily_folder for daily_brief_*.json files and build index.
    Returns list of index entries newest-first.
    """
    if not daily_folder.exists():
        return []
    files = sorted(daily_folder.glob("daily_brief*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    index = []
    for p in files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            meta = extract_brief_metadata(obj, p)
            index.append(meta)
        except Exception:
            continue
    if index:
        save_index(index, index_path)
    return index

# ---------- parsing & heuristics ----------
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
        else:
            t_m2 = re.search(r"\d{2}-\w{3}-\d{4}", line)
            if t_m2:
                timestamp = t_m2.group(0)
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
    st.download_button("Download CSV", data=csv, file_name=fname, mime="text/csv", key=f"dl_{fname}")

# ---------------- New: merging logic for two briefs ----------------
def load_brief_with_meta(path_str: str) -> Dict[str, Any]:
    p = Path(path_str) if path_str else None
    if not p or not p.exists():
        return {"_path": path_str, "_meta": None, "_obj": {}}
    obj = read_json(p)
    meta = extract_brief_metadata(obj, p)
    # include path in object for convenience
    obj["_path"] = str(p)
    return {"_path": str(p), "_meta": meta, "_obj": obj}

def merge_two_briefs(primary_obj: dict, secondary_obj: dict) -> dict:
    """
    Merge two brief objects. Primary is treated as newer / higher priority.
    """
    out = {}
    p = primary_obj or {}
    s = secondary_obj or {}

    out["headline"] = p.get("headline") or s.get("headline") or ""
    out["brief"] = p.get("brief") or s.get("brief") or ""
    out["date"] = p.get("date") or s.get("date") or None

    # bullets merge (primary first)
    p_bul = p.get("bullets", []) or []
    s_bul = s.get("bullets", []) or []
    seen = set()
    merged_bul = []
    for b in p_bul + s_bul:
        key = re.sub(r"\s+"," ", (b or "").strip()).lower()
        if key and key not in seen:
            merged_bul.append(b)
            seen.add(key)
    out["bullets"] = merged_bul

    # itemized lines merge
    def gather_itemized(obj):
        if not obj:
            return []
        a = obj.get("all_itemized_lines")
        if a is None:
            a = obj.get("selected_lines")
        if a is None:
            a = []
        return a or []

    p_items = gather_itemized(p)
    s_items = gather_itemized(s)
    seen = set()
    merged_items = []
    for it in p_items + s_items:
        key = re.sub(r"\s+"," ", (it or "").strip()).lower()
        if key and key not in seen:
            merged_items.append(it)
            seen.add(key)
    out["all_itemized_lines"] = merged_items
    out["selected_lines"] = merged_items

    # actions_from_summarizer merge (strings)
    p_actions = p.get("actions_from_summarizer", []) or []
    s_actions = s.get("actions_from_summarizer", []) or []
    seen = set()
    merged_actions = []
    for a in p_actions + s_actions:
        key = re.sub(r"\s+"," ", (a or "").strip()).lower()
        if key and key not in seen:
            merged_actions.append(a)
            seen.add(key)
    out["actions_from_summarizer"] = merged_actions

    # extracted_actions (list of dicts) merge by id+action text
    p_e = p.get("extracted_actions", []) or []
    s_e = s.get("extracted_actions", []) or []
    merged_ex = []
    seen_keys = set()
    for item in p_e + s_e:
        try:
            iid = str(item.get("id",""))
            act = str(item.get("action","")).strip()
            k = (iid + "||" + act).lower()
        except Exception:
            k = json.dumps(item, sort_keys=True)
        if k not in seen_keys:
            merged_ex.append(item)
            seen_keys.add(k)
    out["extracted_actions"] = merged_ex

    out["selected_count"] = len(merged_items) if merged_items else (p.get("selected_count") or s.get("selected_count") or 0)
    out["_primary_path"] = p.get("_path") if isinstance(p, dict) else None
    out["_secondary_path"] = s.get("_path") if isinstance(s, dict) else None

    return out

# -------- sidebar inputs ----------
st.sidebar.title("Settings")
daily_json_path = st.sidebar.text_input("Primary brief JSON path (fallback)", value=DEFAULT_DAILY_JSON, key="primary_path_input")
secondary_json_path = st.sidebar.text_input("Secondary brief JSON path (optional)", value="", key="secondary_path_input")
human_jsonl_path = st.sidebar.text_input("Human review JSONL path", value=DEFAULT_HUMAN_REVIEW, key="human_jsonl_input")
state_path = st.sidebar.text_input("State file path", value=STATE_FILE, key="state_path_input")
# explicit refresh button (single)
refresh_index = st.sidebar.button("Refresh index", key="refresh_index_btn")

# Load state & human review
daily_index = load_index(DAILY_INDEX)
# if index missing, try auto-discovery
if not daily_index:
    discovered = auto_discover_and_index(DAILY_FOLDER, DAILY_INDEX)
    if discovered:
        daily_index = discovered

daily_exists = Path(daily_json_path).exists()
human_review = read_jsonl(Path(human_jsonl_path))
state = load_state(Path(state_path))

# Sidebar: show archive entries and allow populating primary/secondary from them
st.sidebar.header("Brief archive (newest-first)")
selected_primary = None
selected_secondary = None

if daily_index:
    st.sidebar.markdown(f"Found {len(daily_index)} archived brief(s).")
    archive_paths = ["(none)"] + [entry.get("path") for entry in daily_index]
    sel_primary = st.sidebar.selectbox("Choose primary archived brief (overrides primary path input)", archive_paths, index=0, key="archive_primary")
    sel_secondary = st.sidebar.selectbox("Choose secondary archived brief (optional)", archive_paths, index=0, key="archive_secondary")
    if sel_primary and sel_primary != "(none)":
        selected_primary = sel_primary
    if sel_secondary and sel_secondary != "(none)":
        selected_secondary = sel_secondary

    if refresh_index:
        daily_index = load_index(DAILY_INDEX)
else:
    if daily_exists:
        st.sidebar.info("No archive found. Using current daily_brief.json. Use 'Archive current brief' to add it to history.")
        if st.sidebar.button("Archive current brief", key="archive_current_btn"):
            try:
                DAILY_FOLDER.mkdir(parents=True, exist_ok=True)
                src = Path(daily_json_path)
                date_tag = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                dest_name = f"daily_brief_{date_tag}.json"
                dest = DAILY_FOLDER / dest_name
                shutil.copy2(src, dest)
                meta = append_brief_to_index(dest, DAILY_INDEX)
                st.sidebar.success(f"Archived to {dest} and added to index.")
                selected_primary = str(dest)
                daily_index = load_index(DAILY_INDEX)
            except Exception as e:
                st.sidebar.error(f"Failed to archive: {e}")
    else:
        st.sidebar.warning("No daily brief found. Place daily_brief.json in project root or run pipeline to produce briefs.")

# allow overriding selected paths from inputs if archive not used
if selected_primary is None:
    if daily_json_path and Path(daily_json_path).exists():
        selected_primary = daily_json_path
if selected_secondary is None:
    if secondary_json_path and Path(secondary_json_path).exists():
        selected_secondary = secondary_json_path

# default selection fallback: newest index entry if not set
if selected_primary is None:
    if daily_index:
        selected_primary = daily_index[0].get("path")
    elif daily_exists:
        selected_primary = daily_json_path

# Load briefs (primary + optional secondary)
primary_loaded = load_brief_with_meta(selected_primary) if selected_primary else {"_obj": {}}
secondary_loaded = load_brief_with_meta(selected_secondary) if selected_secondary else {"_obj": {}}
# Build merged brief (primary priority)
merged_brief = merge_two_briefs(primary_loaded.get("_obj", {}), secondary_loaded.get("_obj", {}))

# -------- header ----------
st.title("📋 AI Inbox — Daily Brief (Grouped + Filters)")
col1, col2 = st.columns([3,1])
with col1:
    headline = merged_brief.get("headline") or "No headline found"
    brief = merged_brief.get("brief") or ""
    st.header(headline)
    if brief:
        st.write(brief)
with col2:
    st.markdown("**Meta**")
    st.write(f"Selected count (merged): {merged_brief.get('selected_count', 'n/a')}")
    ppath = primary_loaded.get("_path") or selected_primary or "none"
    spath = secondary_loaded.get("_path") or selected_secondary or ""
    st.write(f"Primary: {os.path.basename(ppath) if ppath else 'none'}")
    if spath:
        st.write(f"Secondary: {os.path.basename(spath)}")
    if st.button("Reload files", key="reload_files_btn"):
        st.experimental_rerun()

# -------- prepare datasets ----------
bullets = merged_brief.get("bullets", []) or []
actions_from_summarizer = merged_brief.get("actions_from_summarizer", []) or []
extracted_actions = merged_brief.get("extracted_actions", []) or []
all_itemized = merged_brief.get("all_itemized_lines", []) or []
if isinstance(all_itemized, dict):
    all_itemized = list(all_itemized.values())
if not isinstance(all_itemized, list):
    all_itemized = merged_brief.get("selected_lines", []) or []

# Build item_df and sort messages newest -> oldest when possible
item_df = list_to_df(all_itemized)

def _parse_ts_safe(ts):
    import datetime as _dt
    if not ts or not isinstance(ts, str):
        return _dt.datetime.min
    s = ts.strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return _dt.datetime.fromisoformat(s)
    except Exception:
        pass
    fmts = ["%d-%m-%Y", "%d-%b-%Y", "%d-%b-%y", "%d-%m-%y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]
    for f in fmts:
        try:
            return _dt.datetime.strptime(s, f)
        except Exception:
            continue
    m = re.search(r"\d{4}-\d{2}-\d{2}T?\d{2}:\d{2}:\d{2}", s)
    if m:
        try:
            return _dt.datetime.fromisoformat(m.group(0))
        except Exception:
            pass
    return _dt.datetime.min

if not item_df.empty:
    item_df["parsed_ts"] = item_df["timestamp"].apply(lambda t: _parse_ts_safe(t))
    item_df = item_df.sort_values(by="parsed_ts", ascending=False).reset_index(drop=True)
else:
    item_df["parsed_ts"] = []

action_df = actions_list_to_df(extracted_actions)

# ---------- Bullets section with category filter and merge toggle (no priority filter) ----------
st.markdown("## Key Bullets (filter & merge)")

col_b1, col_b2 = st.columns([3,2])
with col_b1:
    cat_options = sorted(set([infer_category_from_text(b) for b in bullets] + list(item_df["category"].unique())))
    if not cat_options:
        cat_options = ["personal", "transaction", "promotion", "security", "work", "notification", "social", "shipping"]
    sel_bullet_cats = st.multiselect("Bullet categories (filter)", options=cat_options, default=cat_options, key="bullet_cat_filter")
with col_b2:
    merge_toggle = st.checkbox("Merge LLM bullets + itemized lines (preview)", value=False, key="merge_toggle")
    sort_by_recency = st.checkbox("Sort bullets by recency (newest first)", value=False, key="sort_by_recency")

# Build bullet records enriched with inferred category + priority
bullet_records = []
for b in bullets:
    cat = infer_category_from_text(b)
    pri = infer_priority_from_text(b)
    bullet_records.append({"text": b, "category": cat, "priority": pri})

# If merge toggle ON, build merged bullets list (LLM bullets first, then itemized)
if merge_toggle:
    merged = merge_bullets_and_items([r["text"] for r in bullet_records], list(item_df["text"]))
    merged_records = []
    for m in merged:
        merged_records.append({"text": m, "category": infer_category_from_text(m), "priority": infer_priority_from_text(m)})
    display_records = merged_records
else:
    display_records = bullet_records

# apply category filter
display_records = [r for r in display_records if r["category"] in sel_bullet_cats]

# Always order bullets by priority: High -> Medium -> Low (unless recency requested)
order = {"high": 0, "medium": 1, "low": 2}

if sort_by_recency:
    # map item text -> parsed timestamp
    ts_map = {}
    for _, row in item_df.iterrows():
        try:
            key = re.sub(r"\s+", " ", str(row["text"]).strip()).lower()
            ts_map[key] = row.get("parsed_ts") or None
        except Exception:
            continue

    def _recency_key(rec):
        import datetime as _dt
        t = None
        try:
            key = re.sub(r"\s+", " ", str(rec.get("text","")).strip()).lower()
            t = ts_map.get(key)
        except Exception:
            t = None
        return t or _dt.datetime.min

    # sort by recency (newest first), tie-breaker by priority
    display_records = sorted(display_records, key=lambda r: (_recency_key(r), -order.get(r.get("priority","low"), 2)), reverse=True)
else:
    display_records = sorted(display_records, key=lambda x: order.get(x.get("priority","low"), 2))

# show counts and list
st.write(f"Showing {len(display_records)} bullets (merge={merge_toggle})")
if not display_records:
    st.info("No bullets after applying filters.")
else:
    for i, rec in enumerate(display_records, start=1):
        if rec['category'] == 'transaction':
            st.markdown(f"**{i}. [Transaction]** {rec['text']}")
        else:
            st.markdown(f"**{i}. [{rec['category'].capitalize()} | {rec['priority'].upper()}]** {rec['text']}")

# option to download displayed bullets as CSV
if display_records:
    if st.button("Download displayed bullets as CSV", key="download_bullets_btn"):
        df_bul = pd.DataFrame(display_records)
        download_df_as_csv(df_bul, "displayed_bullets.csv")

st.markdown("---")

# -------- Actions / Checklist (grouped) ----------
st.markdown("## Actions / Checklist (grouped by category)")

if actions_from_summarizer:
    st.subheader("Summarizer actions")
    for i, a in enumerate(actions_from_summarizer, start=1):
        key = f"llm_action_{i}"
        checked = state.get("llm_action", {}).get(key, False)
        new_val = st.checkbox(f"[Summarizer] {a}", value=checked, key=f"llm_action_ck_{i}")
        if new_val != checked:
            state.setdefault("llm_action", {})[key] = new_val
            save_state(state, Path(state_path))

if not action_df.empty:
    st.subheader("Extracted action items")
    cats = sorted(action_df["category"].unique().tolist())
    sel_cats_actions = st.multiselect("Filter actions by category", options=cats, default=cats, key="actions_cat_filter")
    filtered_actions = action_df[action_df["category"].isin(sel_cats_actions)] if sel_cats_actions else action_df
    for cat, sub in filtered_actions.groupby("category"):
        st.markdown(f"### {cat.capitalize()} ({len(sub)})")
        for _, row in sub.iterrows():
            aid = f"{row['id']}::{row['idx']}"
            checked = state.get("actions_done", {}).get(aid, False)
            label = f"[{row['priority_label'].upper() if row['priority_label'] else 'N/A'}] {row['action']}"
            new_val = st.checkbox(label, value=checked, key=f"act_ck_{aid}")
            if new_val != checked:
                state.setdefault("actions_done", {})[aid] = new_val
                save_state(state, Path(state_path))
    st.markdown("**Export extracted actions**")
    if st.button("Download extracted actions CSV", key="download_actions_btn"):
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
    search_q = col1.text_input("Search text", value="", key="search_itemized")
    categories = sorted(item_df["category"].unique().tolist())
    sel = col2.multiselect("Filter category", options=categories, default=categories, key="item_cat_filter")
    amt_only = col3.checkbox("Only show transactions with amount", value=False, key="amt_only_ck")

    df_vis = item_df.copy()
    if search_q:
        df_vis = df_vis[df_vis["text"].str.contains(search_q, case=False, na=False)]
    if sel:
        df_vis = df_vis[df_vis["category"].isin(sel)]
    if amt_only:
        df_vis = df_vis[df_vis["amount"] != ""]

    st.write(f"Showing {len(df_vis)} / {len(item_df)} itemized rows (newest-first when timestamps present)")
    # optionally drop parsed_ts for display or keep it
    display_df = df_vis.reset_index(drop=True).copy()
    if "parsed_ts" in display_df.columns:
        display_df["parsed_ts"] = display_df["parsed_ts"].astype(str)
    st.dataframe(display_df, use_container_width=True, height=420)

    csv = df_vis.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered itemized rows (CSV)", data=csv, file_name="itemized_filtered.csv", mime="text/csv", key="download_itemized_btn")

st.markdown("---")

# -------- Human review queue ----------
st.markdown("## Human review queue")
if not human_review:
    st.info("No human-review items found.")
else:
    st.write(f"{len(human_review)} items")
    for i, item in enumerate(human_review, start=1):
        uid = item.get("id", f"item_{i}")
        with st.expander(f"[{i}] {uid}"):
            st.json(item)
            if st.button(f"Mark reviewed (index {i})", key=f"mark_review_btn_{i}_{uid}"):
                reviewed_path = Path(DEFAULT_HUMAN_REVIEW.replace(".jsonl", "_reviewed.jsonl"))
                try:
                    with reviewed_path.open("a", encoding="utf-8") as rf:
                        rf.write(json.dumps({"reviewed_at": datetime.datetime.now().isoformat(), "item": item}, ensure_ascii=False) + "\n")
                    remaining = human_review[:i-1] + human_review[i:]
                    with Path(human_jsonl_path).open("w", encoding="utf-8") as hf:
                        for it in remaining:
                            hf.write(json.dumps(it, ensure_ascii=False) + "\n")
                    st.success(f"Marked item {i} reviewed and archived to {reviewed_path.name}")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to mark reviewed: {e}")

st.markdown("---")
with st.expander("Raw brief preview (truncated)"):
    try:
        st.code(json.dumps(merged_brief, ensure_ascii=False, indent=2)[:4000])
    except Exception:
        st.write("No brief loaded.")

# persist UI state
save_state(state, Path(state_path))

st.markdown("**Notes**")
st.markdown("""
- Pick a primary brief (newer) and optionally a secondary brief (older) to merge. Primary wins on conflicts.
- Use the **Bullet categories** filter to focus the bullets shown.
- Bullets are always ordered: **High → Medium → Low** unless you enable 'Sort by recency'.
- Toggle **Merge LLM bullets + itemized lines** to preview the merged list (LLM bullets first, then itemized lines not already present).
- Actions are persisted in the local state file specified in the sidebar.
""")
