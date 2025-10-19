#!/usr/bin/env python3
"""
hybrid_multisource_labeler.py

Unified hybrid labeling pipeline that accepts multiple input formats (Gmail JSONL, generic CSV, SMS Backup & Restore XML).

Behavior:
 - Rules-first fast path (expanded SMS-aware rules)
 - Optional local LLM verification & fallback (llama-cpp-python + GGUF)
 - Caching to avoid re-labeling
 - Human-review queue for conflicts / low-confidence items
 - Supports --dry_run to preview results without writing CSV

Usage examples:
  python hybrid_multisource_labeler.py --input --output combined_labeled.csv
  python hybrid_multisource_labeler.py --input SMS.xml --output sms_labeled.csv --model_path ./models/mistral-7b.gguf --dry_run 20

Dependencies:
  pip install pandas beautifulsoup4 python-dateutil
  Optional for LLM: pip install llama-cpp-python

Outputs:
 - output CSV with columns:
   id, source, sender, subject, body, timestamp,
   category, priority_label, action_required, action_text,
   label_source, confidence, decision_path
 - human_review.jsonl (appended) for items needing manual review
 - hybrid_cache.json (cache)
"""
import argparse
import base64
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
from bs4 import BeautifulSoup

# Optional LLM import
try:
    from llama_cpp import Llama
except Exception:
    Llama = None

# ---------------- Config & thresholds ----------------
CACHE_FILE = Path("hybrid_cache.json")
HUMAN_REVIEW_FILE = Path("human_review.jsonl")

MAX_BODY_CHARS = 2000

CONF_RULE = 0.95
CONF_RULE_LLM_AGREE = 0.98
CONF_RULE_LLM_DISAGREE = 0.92
CONF_LLM_FALLBACK = 0.72
CONF_LLM_PARSE_FAIL = 0.45
HUMAN_REVIEW_THRESHOLD = 0.60

# ---------------- Keyword sets & helpers (SMS-aware) ----------------
PROMO_KW = [
    "unsubscribe", "sale", "offer", "discount", "deal", "promotion", "newsletter", "price drop",
    "fare", "flight", "hotel", "coupon", "save", "cheap flights", "low fares", "price dropped",
    "prices dropped", "recharge", "plan", "true 5g", "black friday", "cyber monday", "upto", "code:"
]
SPAM_KW = ["lottery", "win money", "claim prize", "congratulations you", "inheritance", "urgent assistance needed"]
TRANSACTION_KW = [
    "invoice", "receipt", "payment due", "payment", "bill due", "order confirmation", "booking confirmation",
    "itinerary", "booking", "ticket number", "debited", "credited", "txn", "transaction", "p2a", "p2m", "upi"
]
SECURITY_KW = [
    "otp", "one-time password", "one time pin", "one-time pin", "verification code", "password reset",
    "verify your account", "2fa", "security alert", "login attempt", "account is added", "username", "password:"
]
NOTIFICATION_KW = [
    "reminder", "shipment", "tracking", "delivery", "appointment", "ticket", "alert", "service alert",
    "missed call", "available to take calls", "last missed call", "usage alert", "daily data", "data quota", "data used"
]
SOCIAL_KW = ["friend request", "followed you", "liked your post", "commented on", "mentioned you"]

# Sender shortcode heuristics (prefix -> inferred category)
SENDER_PREFIX_MAP = {
    "AX": "transaction",  # Axis bank variants
    "AXIS": "transaction",
    "AD": "transaction",
    "JD": "transaction",
    "CP": "promotion",
    "VM": "notification",
    "JA": "notification",
    "PTNPAN": "notification",
    "JK": "promotion",
}

# Regex helpers
_money_re = re.compile(r"\b(?:inr|rs\.?|rs|₹)\s*([0-9,]+(?:[.,][0-9]{1,2})?)\b", flags=re.IGNORECASE)
_big_amount_threshold = 10000.0  # INR threshold for flagging human review
_otp_explicit_re = re.compile(r"\b(one[- ]time (pin|password)|otp|verification code|your one time PIN is|instagram code|code is)\b", flags=re.IGNORECASE)
_telugu_re = re.compile(r"[\u0C00-\u0C7F]")  # Telugu script range

# ---------------- Utilities ----------------
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


def parse_date_to_iso(date_str: str) -> str:
    if not date_str:
        return ""
    clean = re.sub(r"\(.*?\)", "", str(date_str)).strip()
    # try email parser first
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        try:
            dt = pd.to_datetime(clean, utc=True)
            return dt.isoformat()
        except Exception:
            try:
                v = int(clean)
                if v > 1e12:
                    v = v // 1000
                return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
            except Exception:
                return ""


def extract_text_from_part(part: dict) -> str:
    if not part:
        return ""
    mime = part.get("mimeType", "")
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
    for sub in part.get("parts", []) or []:
        t = extract_text_from_part(sub)
        if t:
            return t
    return ""


def decode_body_from_payload(payload: dict) -> str:
    if not payload:
        return ""
    text = extract_text_from_part(payload)
    if text:
        return text
    return payload.get("snippet", "") or ""


# ---------------- Input readers ----------------
def read_jsonl(path: Path, max_messages: int = None) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if max_messages and len(rows) >= max_messages:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            payload = obj.get("payload")
            if payload:
                headers = payload.get("headers", []) or []
                headers_map = {h.get("name", "").lower(): h.get("value", "") for h in headers}
                subject = headers_map.get("subject", "") or obj.get("subject", "")
                sender = headers_map.get("from", "") or obj.get("from", "")
                internal = obj.get("internalDate") or obj.get("internal_date")
                if internal:
                    try:
                        ts_iso = datetime.fromtimestamp(int(internal) / 1000.0, tz=timezone.utc).isoformat()
                    except Exception:
                        ts_iso = parse_date_to_iso(headers_map.get("date", "") or obj.get("date", ""))
                else:
                    ts_iso = parse_date_to_iso(headers_map.get("date", "") or obj.get("date", ""))
                body = decode_body_from_payload(payload) or obj.get("snippet", "") or ""
            else:
                subject = obj.get("subject", "") or obj.get("Subject", "")
                sender = obj.get("from", "") or obj.get("sender", "")
                ts_iso = parse_date_to_iso(obj.get("date", "") or obj.get("timestamp", ""))
                body = obj.get("body", "") or obj.get("snippet", "") or ""

            rows.append({
                "id": obj.get("id") or obj.get("messageId") or f"json_{len(rows)+1}",
                "source": obj.get("source", "email"),
                "sender": sender,
                "subject": subject or "",
                "body": (body or "")[:MAX_BODY_CHARS],
                "timestamp": ts_iso,
            })
    return rows


def read_csv(path: Path, max_messages: int = None) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        if max_messages and len(rows) >= max_messages:
            break
        rows.append({
            "id": str(r.get("id") or f"csv_{i+1}"),
            "source": r.get("source", "email"),
            "sender": r.get("sender", ""),
            "subject": r.get("subject", "") or "",
            "body": str(r.get("body", "") or "")[:MAX_BODY_CHARS],
            "timestamp": parse_date_to_iso(r.get("timestamp", "") or r.get("date", "")),
        })
    return rows


def read_sms_xml(path: Path, max_messages: int = None) -> List[Dict[str, Any]]:
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(path))
    root = tree.getroot()
    rows = []
    for i, sms in enumerate(root.findall("sms")):
        if max_messages and len(rows) >= max_messages:
            break
        addr = sms.get("address", "")
        body = sms.get("body", "") or ""
        date = sms.get("date", "")
        ts_iso = parse_date_to_iso(date)
        rows.append({
            "id": sms.get("date", f"sms_{len(rows)+1}"),
            "source": "sms",
            "sender": addr,
            "subject": "",
            "body": (body or "").replace("\n", " ")[:MAX_BODY_CHARS],
            "timestamp": ts_iso,
        })
    return rows


def read_input(path: Path, max_messages: int = None) -> List[Dict[str, Any]]:
    ext = path.suffix.lower()
    if ext in (".jsonl", ".ndjson", ".json"):
        return read_jsonl(path, max_messages)
    if ext == ".csv":
        return read_csv(path, max_messages)
    if ext == ".xml":
        return read_sms_xml(path, max_messages)
    raise ValueError(f"Unsupported input format: {ext}")


# ---------------- Rules-match function (SMS-aware) ----------------
def rules_match(subject: str, body: str, sender: str) -> Tuple[Dict[str, Any], str]:
    text = f"{subject} {body} {sender or ''}"
    text_lower = text.lower()

    # Sender prefix heuristic
    snd_up = (sender or "").upper()
    for pref, cat in SENDER_PREFIX_MAP.items():
        if snd_up.startswith(pref + "-") or pref in snd_up:
            if cat == "transaction":
                return ({"category": "transaction", "priority_label": "high", "action_required": 1,
                         "action_text": "Verify transaction; SMS BLOCKUPI if unauthorised."}, f"sender_prefix_{pref}")
            if cat == "promotion":
                return ({"category": "promotion", "priority_label": "low", "action_required": 0, "action_text": ""}, f"sender_prefix_{pref}")
            if cat == "notification":
                return ({"category": "notification", "priority_label": "low", "action_required": 0, "action_text": ""}, f"sender_prefix_{pref}")

    # OTP / verification
    if _otp_explicit_re.search(text_lower):
        m = re.search(r"\b(\d{4,6})\b", text)
        code = m.group(1) if m else ""
        action = f"Use OTP {code}; do not share." if code else "Use OTP; do not share."
        return ({"category": "security", "priority_label": "high", "action_required": 1, "action_text": action}, "otp")

    # Transaction / money detection
    m = _money_re.search(text)
    if m:
        amount_raw = m.group(1).replace(",", "")
        try:
            amount_val = float(amount_raw.replace("₹", "").replace("rs.", "").replace("rs", ""))
        except Exception:
            amount_val = 0.0
        action_text = "Verify transaction details and report if unauthorised."
        label = {"category": "transaction", "priority_label": "high", "action_required": 1, "action_text": action_text}
        if amount_val >= _big_amount_threshold:
            label["action_text"] += " FLAG_FOR_REVIEW"
        return (label, "transaction_money")

    # e-PAN / document ack
    if "e-pan" in text_lower or "epan" in text_lower or "get e-pan" in text_lower:
        return ({"category": "notification", "priority_label": "medium", "action_required": 1, "action_text": "Download e-PAN using provided link."}, "epan")

    # Data usage / carrier notifications
    if any(k in text_lower for k in ["usage alert", "daily data", "data quota", "data used", "50% of daily data"]):
        return ({"category": "notification", "priority_label": "medium", "action_required": 0, "action_text": "Check data balance in carrier app; recharge if needed."}, "data_usage")

    # Promotions
    if any(k in text_lower for k in PROMO_KW):
        return ({"category": "promotion", "priority_label": "low", "action_required": 0, "action_text": ""}, "promotion_kw")

    # Missed call / availability
    if "missed call" in text_lower or "available to take calls" in text_lower or "last missed call" in text_lower:
        return ({"category": "notification", "priority_label": "low", "action_required": 0, "action_text": ""}, "missed_call")

    # Social
    if any(k in text_lower for k in SOCIAL_KW):
        return ({"category": "social", "priority_label": "low", "action_required": 0, "action_text": ""}, "social_kw")

    # Spam
    if any(k in text_lower for k in SPAM_KW):
        return ({"category": "spam", "priority_label": "low", "action_required": 0, "action_text": ""}, "spam_kw")

    # Non-English (Telugu) fallback
    if _telugu_re.search(text):
        return ({"category": "notification", "priority_label": "low", "action_required": 0, "action_text": ""}, "non_english")

    # URL-heavy -> promotion
    url_count = len(re.findall(r"https?://", body or ""))
    if url_count >= 2 and len(body or "") > 150:
        return ({"category": "promotion", "priority_label": "low", "action_required": 0, "action_text": ""}, "promo_url_heavy")

    return (None, None)


# ---------------- LLM wrapper (llama-cpp) ----------------
PROMPT_SYSTEM = (
    "You are an assistant that labels short SMS messages and emails for an automated daily briefing.\n"
    "You MUST return EXACT JSON ONLY (a single JSON object) with these fields and types:\n"
    "  - category: string, one of [work, personal, transaction, notification, promotion, spam, security, shipping, social]\n"
    "  - priority_label: string, one of [low, medium, high]\n"
    "  - action_required: integer 0 or 1\n"
    "  - action_text: short string (instructions or empty string). If an explicit OTP (4-6 digit code) appears, include it in action_text like: 'Use OTP 123456; do not share.'\n"
    "  - confidence: number between 0 and 1 (0.00-1.00) indicating model confidence\n"
    "\n"
    "Important rules (apply in this order):\n"
    " 1) If the message explicitly contains an OTP / verification code (4-6 digit sequence near keywords like 'OTP', 'PIN', 'one time'), label as security/high/1 and put the OTP in action_text.\n"
    " 2) Bank debits, UPI or clear payment debit messages -> transaction/high/1 with an action_text instructing verification and reporting if unauthorised.\n"
    " 3) Delivery/shipment notifications (tracking, delivery windows) -> shipping or notification (prefer shipping if tracking present), priority low/medium depending on urgency.\n"
    " 4) Promotional marketing (offers, discounts, unsubscribe links) -> promotion/low/0.\n"
    " 5) Spam (lottery, win money, scams) -> spam/low/0 and action_text may suggest ignoring/reporting.\n"
    " 6) Missed-call, reminders, service alerts -> notification/low/0.\n"
    " 7) If unsure, choose conservative defaults: category 'personal', priority 'low', action_required 0.\n"
    "\n"
    "Output constraints:\n"
    "  - ONLY emit the single JSON object (no surrounding text, no explanation, no markdown, no lists).\n"
    "  - Ensure fields are present and typed as specified.\n"
    "  - Confidence should reflect how closely the message matches high-certainty rules (e.g., explicit OTP or bank debit -> >=0.9)."
)

EXAMPLES = [
    # Transaction / bank debit
    {
        "subject": "",
        "body": "INR 2,000.00 debited from A/c XX4634 on 15-Oct. If not you, contact bank or SMS BLOCKUPI to 919951860002.",
        "out": {
            "category": "transaction",
            "priority_label": "high",
            "action_required": 1,
            "action_text": "Verify debit of INR 2,000.00; contact bank or SMS BLOCKUPI if unauthorised.",
            "confidence": 0.98
        }
    },

    # Security / OTP explicit (include code)
    {
        "subject": "",
        "body": "Your one time PIN is 203833 and is valid for 10 minutes. Please do not share with anyone.",
        "out": {
            "category": "security",
            "priority_label": "high",
            "action_required": 1,
            "action_text": "Use OTP 203833 as required; do not share.",
            "confidence": 0.99
        }
    },

    # Shipping / delivery with tracking (prefer shipping)
    {
        "subject": "",
        "body": "Order FMPC5343634160 from flipkart.com will be delivered today before 11pm. Track at https://track.example.com/FMPC5343634160",
        "out": {
            "category": "shipping",
            "priority_label": "low",
            "action_required": 0,
            "action_text": "Track delivery if needed; no immediate action required.",
            "confidence": 0.95
        }
    },

    # Promotion / marketing
    {
        "subject": "",
        "body": "Flash sale! Up to 50% off on electronics. Use code SAVE50 at checkout. Unsubscribe link: example.com/unsub",
        "out": {
            "category": "promotion",
            "priority_label": "low",
            "action_required": 0,
            "action_text": "",
            "confidence": 0.96
        }
    },

    # Spam / scam
    {
        "subject": "",
        "body": "Congratulations! You've won a lottery of INR 1,00,000. Click http://scam.example to claim.",
        "out": {
            "category": "spam",
            "priority_label": "low",
            "action_required": 0,
            "action_text": "Likely spam/scam — do not click links; ignore or report.",
            "confidence": 0.97
        }
    },

    # Notification / missed call or reminder
    {
        "subject": "",
        "body": "Reminder: Your appointment is scheduled on 20-Oct at 10:00 AM. Reply YES to confirm.",
        "out": {
            "category": "notification",
            "priority_label": "low",
            "action_required": 0,
            "action_text": "Reply YES to confirm appointment if you will attend.",
            "confidence": 0.90
        }
    },

    # Social
    {
        "subject": "",
        "body": "John Doe liked your post and commented: Awesome photo!",
        "out": {
            "category": "social",
            "priority_label": "low",
            "action_required": 0,
            "action_text": "",
            "confidence": 0.85
        }
    }
]



def build_llm_prompt(subject: str, body: str) -> str:
    ex_text = ""
    for ex in EXAMPLES:
        ex_text += f"Subject: {ex['subject']}\nBody: {ex['body']}\nOutput: {json.dumps(ex['out'])}\n\n"
    body_trunc = (body or "") if len(body or "") <= 2000 else (body or "")[:2000] + " ... (truncated)"
    prompt = PROMPT_SYSTEM + "\n\nExamples:\n" + ex_text + f"\nNow label the message below.\nSubject: {subject}\nBody: {body_trunc}\n\nOutput:"
    return prompt


def extract_json_substring(text: str):
    if not text:
        raise ValueError("Empty LLM output")
    i1 = text.find("{")
    i2 = text.rfind("}")
    if i1 == -1 or i2 == -1 or i2 < i1:
        raise ValueError("No JSON object found in LLM output.")
    js = text[i1:i2+1]
    return json.loads(js)


def make_llama_labeler(model_path: str, n_ctx: int = 2048, temp: float = 0.0, max_tokens: int = 256):
    if Llama is None:
        raise RuntimeError("llama-cpp-python not installed / importable.")
    
    try:
        # Optimized GPU parameters for limited VRAM
        llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,                    # Reduced context window
            n_gpu_layers=20,                # Partial GPU offload instead of -1
            n_batch=512,                    # Increased batch size
            n_threads=8,
            f16_kv=True,                    # Use FP16 for key/value cache
            offload_kqv=True,              # Offload KQV to CPU when needed
            verbose=False                   # Reduce logging overhead
        )
        
        print(f"Model loaded with optimized settings:")
        print(f"- Context size: {n_ctx}")
        print(f"- GPU layers: 20")
        print(f"- Batch size: 512")
        print(f"- Using FP16 KV cache")
        
        def label(subject: str, body: str):
            # Truncate input to reduce memory usage
            body = (body or "")[:1000]  # Reduce from 2000 to 1000 chars
            prompt = build_llm_prompt(subject or "", body)
            out = llm(
                prompt, 
                max_tokens=max_tokens,
                temperature=temp,
                echo=False,           # Don't include prompt in output
                stop=["</s>", "\n\n"] # Early stopping
            )
            # extract text robustly
            text = ""
            try:
                choices = out.get("choices", [])
                if choices and isinstance(choices, list):
                    first = choices[0]
                    text = first.get("text") or first.get("message") or str(first)
                else:
                    text = out.get("text", "") or str(out)
            except Exception:
                text = str(out)
            try:
                parsed = extract_json_substring(text)
                parsed["category"] = str(parsed.get("category", "")).strip().lower()
                parsed["priority_label"] = str(parsed.get("priority_label", "")).strip().lower()
                parsed["action_required"] = int(parsed.get("action_required", 0))
                parsed["action_text"] = str(parsed.get("action_text", "")).strip()
                parsed["confidence"] = float(parsed.get("confidence", 0.0))
                return parsed, text
            except Exception:
                return {"category": "personal", "priority_label": "low", "action_required": 0, "action_text": "", "confidence": 0.0}, text

        return label
        
    except Exception as e:
        print(f"Error initializing model with GPU: {str(e)}")
        print("Falling back to CPU-only mode...")
        # Try again without GPU parameters
        llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            verbose=True
        )
        
        def label(subject: str, body: str):
            prompt = build_llm_prompt(subject or "", body or "")
            out = llm(prompt, max_tokens=max_tokens, temperature=temp)
            # extract text robustly
            text = ""
            try:
                choices = out.get("choices", [])
                if choices and isinstance(choices, list):
                    first = choices[0]
                    text = first.get("text") or first.get("message") or str(first)
                else:
                    text = out.get("text", "") or str(out)
            except Exception:
                text = str(out)
            try:
                parsed = extract_json_substring(text)
                parsed["category"] = str(parsed.get("category", "")).strip().lower()
                parsed["priority_label"] = str(parsed.get("priority_label", "")).strip().lower()
                parsed["action_required"] = int(parsed.get("action_required", 0))
                parsed["action_text"] = str(parsed.get("action_text", "")).strip()
                parsed["confidence"] = float(parsed.get("confidence", 0.0))
                return parsed, text
            except Exception:
                return {"category": "personal", "priority_label": "low", "action_required": 0, "action_text": "", "confidence": 0.0}, text

        return label


# ---------------- Cache helpers ----------------
def load_cache(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: Dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------- Main processing ----------------
def process_rows(rows: List[Dict[str, Any]], labeler=None, human_review_threshold: float = HUMAN_REVIEW_THRESHOLD, dry_run: int = 0) -> List[Dict[str, Any]]:
    cache = load_cache(CACHE_FILE)
    human_f = HUMAN_REVIEW_FILE.open("a", encoding="utf-8")
    out_rows: List[Dict[str, Any]] = []

    # Process in smaller batches
    BATCH_SIZE = 10
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{len(rows)//BATCH_SIZE + 1}")

        for i, r in enumerate(batch):
            key = r.get("id") or f"row_{i+1}"
            if key in cache:
                rec = cache[key]
                out_rows.append({
                    "id": key,
                    "source": r.get("source", "email"),
                    "sender": r.get("sender", ""),
                    "subject": r.get("subject", ""),
                    "body": r.get("body", ""),
                    "timestamp": r.get("timestamp", ""),
                    "category": rec.get("category", ""),
                    "priority_label": rec.get("priority_label", ""),
                    "action_required": int(rec.get("action_required", 0)),
                    "action_text": rec.get("action_text", ""),
                    "label_source": rec.get("label_source", "cache"),
                    "confidence": float(rec.get("confidence", 0.0)),
                    "decision_path": rec.get("decision_path", "cache"),
                })
                continue

            subject = r.get("subject", "") or ""
            body = r.get("body", "") or ""
            sender = r.get("sender", "") or ""

            # 1) Rules-first
            rule_label, rule_name = rules_match(subject, body, sender)
            if rule_label is not None:
                final_label = dict(rule_label)
                label_source = "rule"
                confidence = CONF_RULE
                decision_path = "rule"

                # verify with LLM if available
                if labeler is not None:
                    try:
                        llm_parsed, raw_llm = labeler(subject, body)
                    except Exception as e:
                        llm_parsed, raw_llm = None, ""
                    if llm_parsed:
                        agree = (
                            str(llm_parsed.get("category", "")).strip().lower() == str(final_label.get("category", "")).strip().lower()
                            and str(llm_parsed.get("priority_label", "")).strip().lower() == str(final_label.get("priority_label", "")).strip().lower()
                            and int(llm_parsed.get("action_required", 0)) == int(final_label.get("action_required", 0))
                        )
                        if agree:
                            confidence = CONF_RULE_LLM_AGREE
                            label_source = "rule+llm_agree"
                            decision_path = "rule+llm_agree"
                        else:
                            confidence = CONF_RULE_LLM_DISAGREE
                            label_source = "rule+llm_disagree"
                            decision_path = "rule+llm_disagree"
                            final_label["_llm_suggestion"] = llm_parsed
                            # queue sensitive disagreements for human review
                            if final_label.get("category") not in ("promotion", "notification", "social"):
                                human_f.write(json.dumps({
                                    "id": key,
                                    "subject": subject,
                                    "rule_label": final_label,
                                    "llm_suggestion": llm_parsed,
                                    "raw_llm": raw_llm
                                }, ensure_ascii=False) + "\n")

                out_rec = {
                    "category": final_label.get("category", ""),
                    "priority_label": final_label.get("priority_label", ""),
                    "action_required": int(final_label.get("action_required", 0)),
                    "action_text": final_label.get("action_text", ""),
                    "label_source": label_source,
                    "confidence": round(float(confidence), 3),
                    "decision_path": decision_path,
                }
                cache[key] = out_rec
                out_rows.append({
                    "id": key,
                    "source": r.get("source", "email"),
                    "sender": sender,
                    "subject": subject,
                    "body": body,
                    "timestamp": r.get("timestamp", ""),
                    "category": out_rec["category"],
                    "priority_label": out_rec["priority_label"],
                    "action_required": out_rec["action_required"],
                    "action_text": out_rec["action_text"],
                    "label_source": out_rec["label_source"],
                    "confidence": out_rec["confidence"],
                    "decision_path": out_rec["decision_path"],
                })
                # periodic cache flush
                if len(cache) % 50 == 0:
                    save_cache(cache, CACHE_FILE)
                continue

            # 2) LLM fallback
            if labeler is not None:
                try:
                    llm_parsed, raw_llm = labeler(subject, body)
                    category = str(llm_parsed.get("category", "")).strip().lower() or "personal"
                    priority_label = str(llm_parsed.get("priority_label", "")).strip().lower() or "low"
                    action_required = int(llm_parsed.get("action_required", 0))
                    action_text = str(llm_parsed.get("action_text", "")).strip()
                    confidence = float(llm_parsed.get("confidence", CONF_LLM_FALLBACK))
                    if confidence <= 0.0:
                        confidence = CONF_LLM_FALLBACK
                    out_record = {
                        "category": category,
                        "priority_label": priority_label,
                        "action_required": action_required,
                        "action_text": action_text,
                        "label_source": "llm",
                        "confidence": round(float(confidence), 3),
                        "decision_path": "llm",
                    }
                    cache[key] = out_record
                    out_rows.append({
                        "id": key,
                        "source": r.get("source", "email"),
                        "sender": sender,
                        "subject": subject,
                        "body": body,
                        "timestamp": r.get("timestamp", ""),
                        "category": out_record["category"],
                        "priority_label": out_record["priority_label"],
                        "action_required": out_record["action_required"],
                        "action_text": out_record["action_text"],
                        "label_source": out_record["label_source"],
                        "confidence": out_record["confidence"],
                        "decision_path": out_record["decision_path"],
                    })
                    if out_record["confidence"] < human_review_threshold:
                        human_f.write(json.dumps({"id": key, "subject": subject, "body": body, "llm": llm_parsed, "raw_llm": raw_llm}, ensure_ascii=False) + "\n")
                    if len(cache) % 50 == 0:
                        save_cache(cache, CACHE_FILE)
                    continue
                except Exception:
                    pass

            # 3) Conservative fallback
            out_record = {
                "category": "personal",
                "priority_label": "low",
                "action_required": 0,
                "action_text": "",
                "label_source": "fallback",
                "confidence": round(float(CONF_LLM_PARSE_FAIL), 3),
                "decision_path": "fallback",
            }
            cache[key] = out_record
            out_rows.append({
                "id": key,
                "source": r.get("source", "email"),
                "sender": sender,
                "subject": subject,
                "body": body,
                "timestamp": r.get("timestamp", ""),
                "category": out_record["category"],
                "priority_label": out_record["priority_label"],
                "action_required": out_record["action_required"],
                "action_text": out_record["action_text"],
                "label_source": out_record["label_source"],
                "confidence": out_record["confidence"],
                "decision_path": out_record["decision_path"],
            })
            if len(cache) % 50 == 0:
                save_cache(cache, CACHE_FILE)

    human_f.close()
    save_cache(cache, CACHE_FILE)

    # dry run handling
    if dry_run and dry_run > 0:
        print(f"DRY RUN: first {dry_run} rows:")
        for r in out_rows[:dry_run]:
            print(json.dumps(r, ensure_ascii=False, indent=2))
        return out_rows

    return out_rows


# ---------------- CLI and orchestration ----------------

def main():
    p = argparse.ArgumentParser(description="Hybrid multisource labeler (rules + optional local LLM)")
    p.add_argument("--input", "-i", type=str, required=True, help="Input path: .jsonl | .csv | .xml (SMS Backup & Restore)")
    p.add_argument("--output", "-o", type=str, default="labeled_output.csv", help="Output CSV path")
    p.add_argument("--model_path", "-m", type=str, default=None, help="Path to GGUF model (optional). If provided, LLM verification active.")
    p.add_argument("--max_messages", "-n", type=int, default=None, help="Max messages to process")
    p.add_argument("--dry_run", "-d", type=int, default=0, help="Dry run: print first N labeled rows and skip writing CSV")
    p.add_argument("--human_review_threshold", "-t", type=float, default=HUMAN_REVIEW_THRESHOLD, help="Confidence threshold for human review queue")
    p.add_argument("--no_cache", action="store_true", help="Do not read existing cache; treat cache as empty for this run.")
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model_path) if args.model_path else None

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    rows = read_input(input_path, max_messages=args.max_messages)
    print(f"Loaded {len(rows)} messages from {input_path}")

    labeler = None
    if model_path:
        if Llama is None:
            print("ERROR: model_path provided but llama-cpp-python not importable. Install it or run without --model_path.")
            sys.exit(1)
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            sys.exit(1)
        print(f"Loading LLM from {model_path} (this may take a while)...")
        labeler = make_llama_labeler(str(model_path))

    labeled = process_rows(rows, labeler, args.human_review_threshold, args.dry_run)
    
    if not args.dry_run:
        df = pd.DataFrame(labeled)
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Wrote {len(df)} labeled messages to {output_path}")
        print(f"Check {HUMAN_REVIEW_FILE} for items needing review")

if __name__ == "__main__":
    main()
