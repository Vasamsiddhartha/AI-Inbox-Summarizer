#!/usr/bin/env python3
r"""
phase3_mvp.py

Phase-3 MVP: Prompt-based summarizer + action extraction + dedupe/prioritize +
deterministic bullets merge (LLM + full itemized coverage).

Usage:
  # rule-only dry run
  python phase3_mvp.py --input combined_labeled.csv --dry_run 6

  # full run with local GGUF model
  python phase3_mvp.py --input Mixed_labeled.csv --model_path C:\Users\siddhartha\gmail_llama_pipeline\models\models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF\snapshots\3a6fbf4a41a1d52e415a4958cde6856d34b2db93\mistral-7b-instruct-v0.2.Q4_0.gguf --top_k 70 --output daily_brief_20251016.json

Dependencies:
  pip install pandas python-dateutil fastapi uvicorn streamlit
  # optional for local GGUF
  pip install "llama-cpp-python>=0.1.0"
"""

from pathlib import Path
import json
import re  
import argparse
import time
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone
import pandas as pd

# Optional LLM import
try:
    from llama_cpp import Llama
except Exception:
    Llama = None

# ---------- Config ----------
CACHE_PATH = Path("phase3_llm_cache.json")
HUMAN_REVIEW_PATH = Path("human_review.jsonl")
MAX_BODY_CHARS = 2000
DEFAULT_TOP_K = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------- CSV loader / selection ----------
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    except Exception:
        df = pd.read_csv(path, encoding="latin-1").fillna("")
    expected = ["id", "source", "sender", "subject", "body", "timestamp",
                "category", "priority_label", "action_required", "action_text", "confidence"]
    for c in expected:
        if c not in df.columns:
            df[c] = ""
    def to_int_flag(x):
        try:
            if str(x).strip() == "":
                return 0
            return int(float(x))
        except Exception:
            return 0
    df["action_required"] = df["action_required"].apply(to_int_flag)
    try:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    except Exception:
        df["confidence"] = 0.0
    df["body"] = df["body"].fillna("").astype(str).apply(lambda x: x[:MAX_BODY_CHARS])
    return df

def rank_select(df: pd.DataFrame, top_k:int=DEFAULT_TOP_K) -> pd.DataFrame:
    pri_map = {"high": 3, "medium": 2, "low": 1}
    df = df.copy()
    df["priority_score"] = df["priority_label"].map(pri_map).fillna(1).astype(int)
    req = df[df["action_required"] == 1].copy()
    others = df[df["action_required"] != 1].copy()
    others = others.sort_values(by=["priority_score","confidence"], ascending=[False, False])
    combined = pd.concat([req, others]).drop_duplicates(subset=["id"])
    selected = combined.head(top_k).reset_index(drop=True)
    return selected

# ---------- Improved short-line builder (structured tokens) ----------
RE_AMOUNT = re.compile(r"\b(?:INR|Rs\.?|₹)\s?[0-9,]+(?:\.[0-9]{1,2})?\b", re.I)

def build_short_line(row: pd.Series) -> str:
    pr = str(row.get("priority_label","")).upper() or "LOW"
    sender = str(row.get("sender","")).strip()
    subj = str(row.get("subject","")).strip()
    body = str(row.get("body","")).strip()
    ts = str(row.get("timestamp","")).strip()

    # extract amount
    amt_m = RE_AMOUNT.search(body)
    amount = amt_m.group(0) if amt_m else ""

    # best-effort merchant extraction: after a slash '/' or before 'Not you' phrase
    merchant = ""
    m1 = re.search(r"/([^/]{3,60}?)\s+(?:Not you|Not you\?)", body, re.I)
    if not m1:
        m1 = re.search(r"/([A-Za-z0-9 &\-\.\']{3,60})", body)
    if m1:
        merchant = m1.group(1).strip()

    snippet = subj if subj else (body[:120] + ("..." if len(body) > 120 else ""))
    parts = [f"[{pr}] {sender}", snippet]
    if amount:
        parts.append(f"AMT: {amount}")
    if merchant:
        parts.append(f"M: {merchant}")
    if ts:
        parts.append(f"T: {ts}")
    act = str(row.get("action_text","")).strip()
    if act:
        parts.append(f"ACTION_HINT: {act}")
    line = " | ".join(parts)
    return line

# ---------- Rule-based action extraction (enhanced for payments) ----------
RE_OTP = re.compile(r"\b(?:otp|one[- ]time pass|verification code|verification|code is|code:)\b", re.I)
RE_PAYMENT = re.compile(r"\b(?:inr|rs\.?|rupees|debited|credited|payment|invoice|receipt|upi|txnid|transaction)\b", re.I)
RE_BOOKING = re.compile(r"\b(?:booking|itinerary|pnr|ticket|flight|hotel|reservation|check-?in)\b", re.I)
RE_DUE_DATE = re.compile(r"\b(?:due by|due on|due:|by \w+ \d{1,2}|on \w+ \d{1,2}|tomorrow|today|next week)\b", re.I)

def extract_actions_rules(subject: str, body: str, sender: str) -> Tuple[List[Dict[str,str]], str]:
    text = f"{subject}\n{body}\n{sender}"
    actions = []
    brief_parts = []

    if RE_OTP.search(text):
        actions.append({"action":"Use / note verification code (OTP) from message","due":None,"type":"security"})
        brief_parts.append("Contains verification code (OTP)")

    if RE_PAYMENT.search(text):
        amt_m = RE_AMOUNT.search(text)
        amt = amt_m.group(0) if amt_m else ""
        m = re.search(r"/([^/]{3,60}?)\s+(?:Not you|Not you\?)", text, re.I) or re.search(r"\/([A-Za-z0-9 &\-\.\']{3,60})", text)
        merchant = m.group(1).strip() if m else ""
        cust_m = re.search(r"Cust\s*ID\s*(?:to)?\s*[:\s]*([\d\-\+]{6,})", text, re.I) or re.search(r"(\b9\d{9,}\b)", text)
        custid = cust_m.group(1).strip() if cust_m else None

        if amt and merchant:
            if custid:
                action_text = f"Verify debit {amt} to {merchant}; if unauthorised, SMS 'BLOCKUPI {custid}' to the given number and contact Axis Bank."
            else:
                action_text = f"Verify debit {amt} to {merchant}; if unauthorised, follow bank SMS instructions (BLOCKUPI) and contact Axis Bank."
        elif amt:
            action_text = f"Verify debit {amt}; if unauthorised, follow bank's SMS BLOCKUPI instructions and contact Axis Bank."
        else:
            action_text = "Verify the transaction in your Axis Bank app; if unauthorised, block UPI and contact the bank."

        actions.append({"action": action_text, "due":None, "type":"transaction"})
        brief_parts.append(f"Payment/transaction detected{(':'+amt) if amt else ''}")

    if RE_BOOKING.search(text):
        actions.append({"action":"Review booking/itinerary details","due":None,"type":"booking"})
        brief_parts.append("Booking / itinerary detected")

    if RE_DUE_DATE.search(text):
        actions.append({"action":"Check due date / deadline mentioned","due":None,"type":"todo"})
        brief_parts.append("Due date / deadline mentioned")

    brief = "; ".join(brief_parts)[:200]
    return actions, brief

# ---------- LLM callable ----------
def make_llm_callable(model_path: str=None, n_ctx:int=4096, temp:float=0.0):
    if model_path is None:
        def _dummy(prompt, max_tokens=256):
            return ""
        return _dummy

    if Llama is None:
        raise RuntimeError("llama-cpp-python is not installed but model_path provided.")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logging.info(f"Loading local GGUF model from {model_path} ...")
        # Initialize with GPU parameters
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=30,       # Use 20 layers on GPU (adjust based on VRAM)
        n_batch=512,           # Increased batch size
        n_threads=8,           # Adjust based on CPU cores
        f16_kv=True,          # Use FP16 for key/value cache
        verbose=True          # Show GPU usage info
    )
    
    print(f"Model loaded with GPU optimization")
    print(f"- Context size: {n_ctx}")
    print(f"- GPU layers: 20")
    print(f"- Batch size: 512")
    print(f"- Using FP16 KV cache")
    def _call(prompt: str, max_tokens: int = 512):
        resp = llm(prompt, max_tokens=max_tokens, temperature=temp,echo=False)
        text = ""
        try:
            choices = resp.get("choices", [])
            if choices and isinstance(choices, list):
                text = choices[0].get("text") or choices[0].get("message") or str(choices[0])
            else:
                text = resp.get("text") or str(resp)
        except Exception:
            text = str(resp)
        return text
    return _call

def extract_json_substring(text: str) -> dict:
    if not text:
        raise ValueError("Empty LLM output")
    i1 = text.find("{")
    i2 = text.rfind("}")
    if i1 == -1 or i2 == -1 or i2 < i1:
        raise ValueError("No JSON object found")
    js = text[i1:i2+1]
    return json.loads(js)

# ---------- Prompts (few-shot summarizer) ----------
ACTION_PROMPT_SYS = (
    "You are an assistant extracting ACTION ITEMS from a single message. "
    "Return EXACT JSON ONLY with keys: actions (array of objects), brief (short sentence). "
    "Each action object: {\"action\":\"<imperative>\", \"due\":\"<ISO date or null>\", \"type\":\"email/meeting/todo/transaction/booking/security\"}. "
    "If there are no action items, return {\"actions\":[], \"brief\":\"\"}."
)

SUMMARIZER_PROMPT_SYS = (
    "You are an assistant that produces a short, accurate daily briefing from a list of short lines. "
    "Return EXACT JSON ONLY with keys: headline (one short sentence), brief (2-3 short sentences), bullets (array, each bullet one transaction or important item), "
    "actions (array of imperative action strings). Do NOT hallucinate. Use the data exactly as given.\n\n"
    "Example 1:\n"
    "Input lines:\n"
    "[HIGH] AX-AxisBk — (none) | AMT: INR 60.00 | M: MANDALA GANESH | T: 2024-11-27 11:54:02\n"
    "[HIGH] AX-AxisBk — (none) | AMT: INR 10.00 | M: Jay Ambe Provision | T: 2024-11-27 12:29:13\n"
    "[HIGH] AX-AxisBk — (none) | AMT: INR 100.00 | M: Rajasthan Kulfi Hou | T: 2024-11-28 01:07:54\n"
    "Output:\n"
    "{\n"
    "  \"headline\": \"Multiple small debits on Axis Bank account — verify now\",\n"
    "  \"brief\": \"Three recent UPI debits (₹60, ₹10, ₹100) were sent from your Axis Bank account (…XX4634). If you don't recognize any charge, verify and block UPI immediately.\",\n"
    "  \"bullets\": [\n"
    "    \"INR 60.00 debited — MANDALA GANESH — 2024-11-27 11:54:02\",\n"
    "    \"INR 10.00 debited — Jay Ambe Provision — 2024-11-27 12:29:13\",\n"
    "    \"INR 100.00 debited — Rajasthan Kulfi Hou — 2024-11-28 01:07:54\"\n"
    "  ],\n"
    "  \"actions\": [\n"
    "    \"Verify the listed transactions in your Axis Bank app (ASAP)\",\n"
    "    \"If any transaction is unauthorised: SMS 'BLOCKUPI <CustID>' to the bank number shown in the SMS and call Axis Bank to raise a dispute\"\n"
    "  ]\n"
    "}\n\n"
    "Now, given the following context lines, produce the same JSON format.\n"
)

# ---------- LLM wrappers ----------
def extract_actions_llm(llm_call, subject: str, body: str, sender:str, max_tokens:int=256) -> Tuple[List[Dict[str,Any]], str, str]:
    prompt = ACTION_PROMPT_SYS + "\n\nMessage:\n" + f"Subject: {subject}\nSender: {sender}\nBody: {(body or '')[:1500]}\n\nOutput:"
    raw = llm_call(prompt, max_tokens=max_tokens)
    if not raw:
        raise ValueError("LLM returned empty text for action extraction")
    parsed = extract_json_substring(raw)
    actions = parsed.get("actions", []) or []
    brief = parsed.get("brief", "") or ""
    out_actions = []
    for a in actions:
        out_actions.append({
            "action": str(a.get("action","")).strip(),
            "due": a.get("due", None),
            "type": str(a.get("type","")).strip() if a.get("type") else "todo"
        })
    return out_actions, brief, raw

def summarize_llm(llm_call, short_lines: List[str], max_tokens:int=512) -> Tuple[dict, str]:
    prompt = SUMMARIZER_PROMPT_SYS + "\n\nContext:\n" + "\n".join(short_lines) + "\n\nOutput:"
    raw = llm_call(prompt, max_tokens=max_tokens)
    if not raw:
        raise ValueError("LLM returned empty text for summarization")
    parsed = extract_json_substring(raw)
    return parsed, raw

# ---------- Traceability ----------
def action_traceable(action_text: str, context: str) -> bool:
    if not action_text:
        return False
    at = re.sub(r"[^\w\s₹₹\d]", " ", action_text).lower()
    ctx = re.sub(r"[^\w\s₹₹\d]", " ", (context or "")).lower()

    amt_a = re.search(r"(?:inr|rs\.?|₹)\s?[0-9,]+(?:\.[0-9]{1,2})?", action_text, re.I)
    amt_c = re.search(r"(?:inr|rs\.?|₹)\s?[0-9,]+(?:\.[0-9]{1,2})?", context, re.I)
    if amt_a and amt_c:
        return True

    words = [w for w in at.split() if w]
    if len(words) < 3:
        return any(w for w in words if len(w) > 2 and w in ctx)
    for i in range(len(words)-2):
        tri = " ".join(words[i:i+3])
        if tri in ctx:
            return True
    return False

# ---------- Cache ----------
def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(cache: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

# ---------- Post-process dedupe + prioritize ----------
def normalize_action_text(a: str) -> str:
    if not a:
        return ""
    s = re.sub(r"[^\w\s]", " ", a)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def dedupe_and_prioritize_actions(extracted_actions: List[Dict[str,Any]],
                                   selected_df: pd.DataFrame,
                                   max_actions: int = 30) -> List[Dict[str,Any]]:
    if not extracted_actions:
        return []
    pri_map = {"high": 3, "medium": 2, "low": 1}
    id_to_meta = {}
    for _, r in selected_df.iterrows():
        rid = str(r.get("id",""))
        id_to_meta[rid] = {
            "priority_score": int(pri_map.get(str(r.get("priority_label","")).lower(), 1)),
            "priority_label": str(r.get("priority_label","")).lower() or "low",
            "confidence": float(r.get("confidence") or 0.0)
        }
    src_pref = {"existing": 4, "rule": 3, "llm_verify": 2, "llm": 1}
    seen = {}
    for rec in extracted_actions:
        try:
            aid = str(rec.get("id",""))
            raw = str(rec.get("action","")).strip()
            if not raw:
                continue
            norm = normalize_action_text(raw)
            meta = id_to_meta.get(aid, {"priority_score":1,"priority_label":"low","confidence":0.0})
            source = rec.get("_source", "llm")
            score_tuple = (meta["priority_score"], meta["confidence"], src_pref.get(source, 0))
            cand = {
                "id": aid,
                "action": raw,
                "action_norm": norm,
                "due": rec.get("due", None),
                "type": rec.get("type","todo"),
                "source": source,
                "priority_score": meta["priority_score"],
                "priority_label": meta["priority_label"],
                "confidence": meta["confidence"],
                "_score_tuple": score_tuple
            }
            if norm in seen:
                if cand["_score_tuple"] > seen[norm]["_score_tuple"]:
                    seen[norm] = cand
            else:
                seen[norm] = cand
        except Exception:
            continue
    out = list(seen.values())
    out.sort(key=lambda x: (x.get("priority_score",0), x.get("confidence",0.0), x["_score_tuple"][2]), reverse=True)
    final = []
    for r in out[:max_actions]:
        final.append({
            "id": r["id"],
            "action": r["action"],
            "due": r["due"],
            "type": r.get("type","todo"),
            "source": r.get("source",""),
            "priority_label": r.get("priority_label",""),
            "confidence": r.get("confidence",0.0)
        })
    return final

# ---------- Deterministic structured bullets (one per short line) ----------
def structured_bullets_from_short_lines(short_lines: List[str]) -> List[str]:
    bullets = []
    seen = set()
    amt_re = re.compile(r"AMT:\s*([^\|]+)", re.I)
    m_re   = re.compile(r"M:\s*([^\|]+)", re.I)
    t_re   = re.compile(r"T:\s*([^\|]+)", re.I)
    for line in short_lines:
        try:
            amt_m = amt_re.search(line)
            m_m = m_re.search(line)
            t_m = t_re.search(line)
            amt = amt_m.group(1).strip() if amt_m else ""
            mrc = m_m.group(1).strip() if m_m else ""
            ts = t_m.group(1).strip() if t_m else ""
            if amt and mrc and ts:
                b = f"{amt} debited — {mrc} — {ts}"
            elif amt and mrc:
                b = f"{amt} debited — {mrc}"
            elif amt:
                b = f"{amt} debited"
            else:
                parts = line.split("—", 1)
                b = parts[1].strip() if len(parts) > 1 else line.strip()
            norm = re.sub(r"\s+", " ", b).strip().lower()
            if norm not in seen:
                seen.add(norm)
                bullets.append(b)
        except Exception:
            continue
    return bullets

# ---------- Conservative fallback summarizer ----------
def conservative_summary(short_lines: List[str], extracted_actions: List[Dict[str,Any]]) -> dict:
    headline = f"Daily Brief — {len(short_lines)} items"
    top = short_lines[:5]
    brief = "Top items: " + "; ".join(top[:3]) if top else "No recent items."
    bullets = top[:5]
    actions = [a for a in extracted_actions]
    return {"headline": headline, "brief": brief, "bullets": bullets, "actions_from_summarizer": [], "extracted_actions": actions, "selected_count": len(short_lines)}

# ---------- Main pipeline ----------
def generate_daily_brief(input_csv: str,
                         model_path: str = None,
                         top_k: int = DEFAULT_TOP_K,
                         dry_run: int = 0,
                         output: str = None,
                         human_review_threshold: float = 0.6) -> dict:
    input_path = Path(input_csv)
    df = load_csv(input_path)
    selected = rank_select(df, top_k=top_k)
    if dry_run:
        logging.info(f"DRY RUN: selected {len(selected)} rows (showing first {dry_run}):")
        print(selected.head(dry_run)[["id","sender","subject","priority_label","action_required","action_text","confidence"]].to_string(index=False))

    llm_call = None
    if model_path:
        llm_call = make_llm_callable(model_path)
    else:
        llm_call = make_llm_callable(None)

    cache = load_cache(CACHE_PATH)

    short_lines: List[str] = []
    extracted_actions: List[Dict[str,Any]] = []
    human_review_items: List[Dict[str,Any]] = []

    for _, row in selected.iterrows():
        sid = str(row.get("id",""))
        subj = str(row.get("subject",""))
        body = str(row.get("body",""))
        sender = str(row.get("sender",""))
        short_line = build_short_line(row)

        if row.get("action_text"):
            act_text = str(row.get("action_text","")).strip()
            if act_text:
                extracted_actions.append({"id": sid, "action": act_text, "due": None, "type":"todo", "_source":"existing"})
            short_lines.append(short_line)
            continue

        try:
            rule_actions, rule_brief = extract_actions_rules(subj, body, sender)
        except Exception:
            rule_actions, rule_brief = [], ""
        if rule_actions:
            for a in rule_actions:
                extracted_actions.append({"id": sid, "action": a.get("action",""), "due": a.get("due",None), "type": a.get("type","todo"), "_source":"rule"})
            if rule_brief:
                short_line = short_line + " | " + rule_brief
            short_lines.append(short_line)
            if model_path:
                cache_key = f"verify_actions::{sid}"
                try:
                    if cache_key in cache:
                        llm_parsed = cache[cache_key]
                        for a in llm_parsed.get("actions", []):
                            if action_traceable(a.get("action",""), subj + " " + body + " " + sender):
                                extracted_actions.append({"id": sid, "action": a.get("action",""), "due": a.get("due",None), "type": a.get("type","todo"), "_source":"llm_verify"})
                            else:
                                human_review_items.append({"id": sid, "subject": subj, "sender": sender, "rule": rule_actions, "llm_raw": llm_parsed.get("raw","")})
                    else:
                        try:
                            actions, brief, raw = extract_actions_llm(llm_call, subj, body, sender, max_tokens=256)
                            cache[cache_key] = {"actions": actions, "brief": brief, "raw": raw, "ts": time.time()}
                            for a in actions:
                                if action_traceable(a.get("action",""), subj + " " + body + " " + sender):
                                    extracted_actions.append({"id": sid, "action": a.get("action",""), "due": a.get("due",None), "type": a.get("type","todo"), "_source":"llm_verify"})
                                else:
                                    human_review_items.append({"id": sid, "subject": subj, "sender": sender, "rule": rule_actions, "llm_raw": raw})
                        except Exception as e:
                            logging.debug(f"LLM verify failed for id={sid}: {e}")
                except Exception as e:
                    logging.debug(f"Cache/react verify error: {e}")
            continue

        if model_path:
            cache_key = f"actions_llm::{sid}"
            try:
                if cache_key in cache:
                    parsed = cache[cache_key]
                    actions = parsed.get("actions", [])
                    brief = parsed.get("brief", "")
                    raw = parsed.get("raw", "")
                else:
                    try:
                        actions, brief, raw = extract_actions_llm(llm_call, subj, body, sender, max_tokens=256)
                        cache[cache_key] = {"actions": actions, "brief": brief, "raw": raw, "ts": time.time()}
                        if len(cache) % 50 == 0:
                            save_cache(cache, CACHE_PATH)
                    except Exception as e:
                        logging.warning(f"LLM action extraction failed for id={sid}: {e}")
                        actions, brief, raw = [], "", ""
                for a in actions:
                    if action_traceable(a.get("action",""), subj + " " + body + " " + sender):
                        extracted_actions.append({"id": sid, "action": a.get("action",""), "due": a.get("due",None), "type": a.get("type","todo"), "_source":"llm"})
                    else:
                        human_review_items.append({"id": sid, "subject": subj, "sender": sender, "llm_raw": raw})
                if brief:
                    short_line = short_line + " | " + brief
            except Exception as e:
                logging.warning(f"Unexpected LLM error for id={sid}: {e}")
        short_lines.append(short_line)

    # Deduplicate & prioritize extracted actions
    try:
        deduped_actions = dedupe_and_prioritize_actions(extracted_actions, selected, max_actions=50)
    except Exception as e:
        logging.warning(f"Action dedupe/prioritize failed: {e}")
        deduped_actions = extracted_actions

    # Summarize
    if model_path:
        try:
            max_context_lines = 6  # smaller context encourages focused summary
            summ_input = short_lines[:max_context_lines]
            parsed_summary, raw_summ = summarize_llm(llm_call, summ_input, max_tokens=512)
            summary = {
                "headline": parsed_summary.get("headline","").strip() if isinstance(parsed_summary.get("headline",""), str) else "",
                "brief": parsed_summary.get("brief","").strip() if isinstance(parsed_summary.get("brief",""), str) else "",
                "bullets": parsed_summary.get("bullets",[]) if isinstance(parsed_summary.get("bullets",[]), list) else [],
                "actions_from_summarizer": parsed_summary.get("actions",[]) if isinstance(parsed_summary.get("actions",[]), list) else [],
                "selected_count": len(selected),
                "selected_lines": summ_input,
                "extracted_actions": deduped_actions,
                "_raw_summarizer_text": raw_summ[:2000]
            }
        except Exception as e:
            logging.warning(f"Summarizer LLM error: {e}")
            summary = conservative_summary(short_lines, deduped_actions)
            human_review_items.append({"id":"__summary_failed__", "error": str(e)})
    else:
        summary = conservative_summary(short_lines, deduped_actions)

    # Merge: ensure full coverage by appending deterministic bullets for all selected lines
    try:
        deterministic_all = structured_bullets_from_short_lines(short_lines)
        llm_bullets = summary.get("bullets", []) or []
        llm_norm = {re.sub(r"\s+", " ", b).strip().lower() for b in llm_bullets}
        remaining = [b for b in deterministic_all if re.sub(r"\s+", " ", b).strip().lower() not in llm_norm]
        # Append enough deterministic bullets so that itemized coverage approaches selected_count.
        # You can tune max_append to control verbosity; here we append up to the difference.
        max_append = max(0, len(short_lines) - len(llm_bullets))
        to_append = remaining[:max_append]
        summary["bullets"] = llm_bullets + to_append
        summary["all_itemized_lines"] = deterministic_all
    except Exception as e:
        logging.warning(f"Failed to merge deterministic bullets: {e}")
        summary.setdefault("all_itemized_lines", structured_bullets_from_short_lines(short_lines))

    # Save human review items
    if human_review_items:
        try:
            with HUMAN_REVIEW_PATH.open("a", encoding="utf-8") as hf:
                for item in human_review_items:
                    hf.write(json.dumps(item, ensure_ascii=False) + "\n")
            logging.info(f"Appended {len(human_review_items)} items to human review: {HUMAN_REVIEW_PATH}")
        except Exception as e:
            logging.warning(f"Failed to write human review file: {e}")

    # Save cache
    try:
        save_cache(cache, CACHE_PATH)
    except Exception as e:
        logging.warning(f"Failed to save cache: {e}")

    # write output file if requested
    if output:
        try:
            Path(output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            logging.info(f"Wrote summary to {output}")
        except Exception as e:
            logging.warning(f"Failed to write output file: {e}")

    return summary

# ---------- CLI ----------
def cli():
    p = argparse.ArgumentParser(description="Phase-3 MVP: summarizer + action extraction + dedupe + deterministic bullets merge")
    p.add_argument("--input", "-i", type=str, required=True, help="Input combined CSV (labeled messages)")
    p.add_argument("--model_path", "-m", type=str, default=None, help="Path to GGUF model (llama-cpp) — optional")
    p.add_argument("--top_k", "-k", type=int, default=DEFAULT_TOP_K, help="Max messages to consider (top-K)")
    p.add_argument("--dry_run", "-d", type=int, default=0, help="Dry run: print first N selected rows and sample summary")
    p.add_argument("--output", "-o", type=str, default=None, help="Write summary JSON to file")
    args = p.parse_args()
    summary = generate_daily_brief(args.input, args.model_path, args.top_k, dry_run=args.dry_run, output=args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    cli()
