#!/usr/bin/env python3
"""
gmail_hybrid_rules_first_verified.py

Rules-first pipeline with mandatory LLM verification when a model is provided.

Usage:
  python gmail_pipeline_with_llama.py --input emails.jsonl --output emails_hybrid_verified.csv --model_path /path/to/model.gguf --max_messages 500

If --model_path is omitted the script runs rule-only + conservative fallback.

Outputs:
 - CSV with columns:
   id, source, sender, subject, body, timestamp,
   category, priority_label, action_required, action_text,
   label_source, confidence, decision_path
 - human_review.jsonl for uncertain / conflict items
 - cache file hybrid_verified_cache.json to avoid re-labeling
"""

import argparse
import json
import re
import time
from pathlib import Path
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import pandas as pd
import base64
import sys

# llama-cpp-python import (optional)
try:
    from llama_cpp import Llama
except Exception:
    Llama = None

# ---------------- Configurable thresholds ----------------
CONF_RULE = 0.95
CONF_RULE_LLM_AGREE = 0.98
CONF_RULE_LLM_DISAGREE = 0.92
CONF_LLM_FALLBACK = 0.72
CONF_LLM_PARSE_FAIL = 0.45
HUMAN_REVIEW_THRESHOLD = 0.60

CACHE_FILE = "hybrid_verified_cache.json"
HUMAN_REVIEW_FILE = "human_review.jsonl"

MAX_BODY_CHARS = 2000

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
            if "html" in mime:
                return BeautifulSoup(raw, "html.parser").get_text(separator="\n")
            return raw
        except Exception:
            pass
    # multipart
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

def parse_date_to_iso(date_str: str) -> str:
    if not date_str:
        return ""
    # strip parenthetical timezone labels like (IST)
    clean = re.sub(r"\(.*?\)", "", date_str).strip()
    try:
        # use email parser first
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        # fallback simple patterns
        try:
            return datetime.fromisoformat(clean).astimezone(timezone.utc).isoformat()
        except Exception:
            return ""

# ---------------- Rule definitions (expanded realistic rules) ----------------
PROMO_KW = ["unsubscribe","sale","offer","discount","deal","promotion","newsletter","price drop","fare","flight","hotel","coupon","save","cheap flights","low fares"]
SPAM_KW = ["lottery","win money","claim prize","congratulations you","inheritance","urgent assistance needed"]
TRANSACTION_KW = ["invoice","receipt","payment due","payment","bill due","order confirmation","booking confirmation","itinerary","booking","ticket number"]
SECURITY_KW = ["otp","one-time password","password reset","verify your account","2fa","security alert","login attempt","suspicious"]
WORK_KW = ["meeting","deadline","project","status update","presentation","slides","sprint","review","proposal","interview","agenda"]
NOTIFICATION_KW = ["reminder","shipment","tracking","delivery","appointment","ticket","alert","service alert"]
SOCIAL_KW = ["friend request","followed you","liked your post","commented on","mentioned you"]
TRAVEL_KW = ["flight","hotel","itinerary","booking","check-in","pnr"]
SUBSCRIPTION_KW = ["subscription","renewal","plan expires","membership"]
EVENT_KW = ["webinar","event","rsvp","invitation"]

def rules_match(subject: str, body: str, sender: str):
    """
    Return (label_dict or None, rule_name or None)
    label_dict keys: category, priority_label, action_required, action_text
    """
    text = f"{subject} {body} {sender}".lower()

    # security / critical first
    for k in SECURITY_KW:
        if k in text:
            return ({"category":"security","priority_label":"high","action_required":1,"action_text":"Review security notification / change password if needed"}, "security")

    # booking/itinerary -> transaction (prefer high)
    for k in TRANSACTION_KW:
        if k in text:
            return ({"category":"transaction","priority_label":"high","action_required":1,"action_text":"Review booking / invoice / payment details"}, "transaction")

    # promo / newsletters / flight price alerts
    for k in PROMO_KW:
        if k in text:
            # differentiate booking vs promo: if it mentions "itinerary" earlier it would have matched transaction
            return ({"category":"promotion","priority_label":"low","action_required":0,"action_text":""}, "promotion")

    # work keywords
    for k in WORK_KW:
        if k in text:
            return ({"category":"work","priority_label":"high","action_required":1,"action_text":"Respond or prepare deliverables"}, "work")

    # notifications (shipping / delivery)
    for k in NOTIFICATION_KW:
        if k in text:
            return ({"category":"notification","priority_label":"medium","action_required":0,"action_text":""}, "notification")

    # social
    for k in SOCIAL_KW:
        if k in text:
            return ({"category":"social","priority_label":"low","action_required":0,"action_text":""}, "social")

    # subscription
    for k in SUBSCRIPTION_KW:
        if k in text:
            return ({"category":"notification","priority_label":"medium","action_required":0,"action_text":"Check subscription / renewal details"}, "subscription")

    # event
    for k in EVENT_KW:
        if k in text:
            return ({"category":"personal","priority_label":"medium","action_required":0,"action_text":""}, "event")

    # spam heuristics
    for k in SPAM_KW:
        if k in text:
            return ({"category":"spam","priority_label":"low","action_required":0,"action_text":""}, "spam")

    # URL-heavy marketing => promotion
    url_count = len(re.findall(r"https?://", body or ""))
    if url_count >= 3 and len(body or "") > 200:
        return ({"category":"promotion","priority_label":"low","action_required":0,"action_text":""}, "promo_many_urls")

    # no rule matched
    return (None, None)

# ---------------- LLM wrapper ----------------
def extract_json_substring(text: str):
    """Find JSON substring and parse. Raises on failure."""
    if not text:
        raise ValueError("Empty text")
    i1 = text.find("{")
    i2 = text.rfind("}")
    if i1 == -1 or i2 == -1 or i2 < i1:
        raise ValueError("No JSON object found in LLM output.")
    js = text[i1:i2+1]
    return json.loads(js)

PROMPT_SYSTEM = (
    "You are an assistant that labels email messages for an automated daily briefing.\n"
    "Return EXACTLY one JSON object with keys:\n"
    "  category (one of [\"work\",\"personal\",\"transaction\",\"notification\",\"promotion\",\"spam\",\"security\",\"shipping\",\"social\"]),\n"
    "  priority_label (one of [\"low\",\"medium\",\"high\"]),\n"
    "  action_required (0 or 1),\n"
    "  action_text (short imperative sentence or empty string),\n"
    "  confidence (float between 0.0 and 1.0).\n"
    "DO NOT output anything else (no explanation)."
)

EXAMPLES = [
    {"subject":"Invoice due","body":"Your invoice #123 is due on Sep 20. Please pay.","out":{"category":"transaction","priority_label":"high","action_required":1,"action_text":"Pay invoice #123 by Sep 20","confidence":0.99}},
    {"subject":"Cheap flights found","body":"We've found low fares from your city to Goa. Reply to get more.","out":{"category":"promotion","priority_label":"low","action_required":0,"action_text":"","confidence":0.95}},
    {"subject":"Password changed","body":"Your password was changed. If this wasn't you, click here.","out":{"category":"security","priority_label":"high","action_required":1,"action_text":"Verify account security","confidence":0.99}},
]

def build_llm_prompt(subject: str, body: str) -> str:
    ex_text = ""
    for ex in EXAMPLES:
        ex_text += f"Subject: {ex['subject']}\nBody: {ex['body']}\nOutput: {json.dumps(ex['out'])}\n\n"
    body_trunc = (body or "") if len(body or "") <= 2000 else (body or "")[:2000] + " ... (truncated)"
    prompt = PROMPT_SYSTEM + "\n\nExamples:\n" + ex_text + f"\nNow label the message below.\nSubject: {subject}\nBody: {body_trunc}\n\nOutput:"
    return prompt

def make_llama_labeler(model_path: str, n_ctx: int = 4096, temp: float = 0.0, max_tokens: int = 512):
    if Llama is None:
        raise RuntimeError("llama-cpp-python not installed / importable.")
    llm = Llama(model_path=str(model_path), n_ctx=n_ctx)
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
        # try parse JSON substring
        try:
            parsed = extract_json_substring(text)
            # normalize keys and types
            parsed["category"] = str(parsed.get("category","")).strip().lower()
            parsed["priority_label"] = str(parsed.get("priority_label","")).strip().lower()
            parsed["action_required"] = int(parsed.get("action_required", 0))
            parsed["action_text"] = str(parsed.get("action_text","")).strip()
            parsed["confidence"] = float(parsed.get("confidence", 0.0))
            return parsed, text
        except Exception:
            # fallback low-confidence
            return {"category":"personal","priority_label":"low","action_required":0,"action_text":"","confidence":0.0}, text
    return label

# ---------------- Cache helpers ----------------
def load_cache(path: Path):
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

# ---------------- Main pipeline ----------------
def process(input_jsonl: Path, output_csv: Path, model_path: Path = None, max_messages: int = None, human_review_threshold: float = HUMAN_REVIEW_THRESHOLD):
    cache = load_cache(Path(CACHE_FILE))
    labeler = None
    if model_path:
        if Llama is None:
            raise RuntimeError("Model path provided but llama-cpp-python not importable.")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"Loading LLM from {model_path} ...")
        labeler = make_llama_labeler(str(model_path))

    human_f = open(HUMAN_REVIEW_FILE, "a", encoding="utf-8")
    rows = []
    processed = 0

    with input_jsonl.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            if max_messages and processed >= max_messages:
                break
            try:
                msg = json.loads(line)
            except Exception as e:
                print(f"Warning: skipping invalid JSON line {idx+1}: {e}")
                continue

            # extract fields robustly (works with Gmail-like JSON)
            payload = msg.get("payload")
            if payload:
                headers_list = payload.get("headers", []) or []
                headers_map = {h.get("name","").lower(): h.get("value","") for h in headers_list}
                subject = headers_map.get("subject","") or msg.get("subject","")
                sender = headers_map.get("from","") or msg.get("from","")
                internal = msg.get("internalDate") or msg.get("internal_date") or msg.get("internaldate")
                if internal:
                    try:
                        ts_iso = datetime.fromtimestamp(int(internal)/1000.0, tz=timezone.utc).isoformat()
                    except Exception:
                        ts_iso = ""
                else:
                    ts_iso = parse_date_to_iso(headers_map.get("date","") or msg.get("date",""))
                body = decode_body_from_payload(payload) or msg.get("snippet","") or ""
            else:
                subject = msg.get("subject","") or msg.get("Subject","")
                sender = msg.get("from","") or msg.get("sender","")
                ts_iso = parse_date_to_iso(msg.get("date","") or msg.get("timestamp",""))
                body = msg.get("body","") or msg.get("snippet","") or ""
            # normalize body length
            body = (body or "")[:MAX_BODY_CHARS]

            key = msg.get("id") or msg.get("messageId") or f"line_{idx+1}"

            if key in cache:
                out = cache[key]
                rows.append({
                    "id": key,
                    "source": msg.get("source","gmail"),
                    "sender": sender,
                    "subject": subject,
                    "body": body,
                    "timestamp": ts_iso,
                    "category": out.get("category",""),
                    "priority_label": out.get("priority_label",""),
                    "action_required": int(out.get("action_required",0)),
                    "action_text": out.get("action_text",""),
                    "label_source": out.get("label_source",""),
                    "confidence": float(out.get("confidence", 0.0)),
                    "decision_path": out.get("decision_path","cache")
                })
                processed += 1
                continue

            # 1) Rules-first
            rule_label, rule_name = rules_match(subject or "", body or "", sender or "")
            if rule_label is not None:
                # initially trust rule
                final_label = dict(rule_label)
                label_source = "rule"
                confidence = CONF_RULE
                decision_path = "rule"

                # always verify with LLM if loaded
                if labeler is not None:
                    llm_parsed, raw_llm = None, ""
                    try:
                        llm_parsed, raw_llm = labeler(subject or "", body or "")
                    except Exception as e:
                        # LLM call failed; keep rule but mark lower confidence
                        llm_parsed = None
                    if llm_parsed:
                        # compare main fields: category, priority, action_required
                        agree = (
                            str(llm_parsed.get("category","")).strip().lower() == str(final_label.get("category","")).strip().lower()
                            and str(llm_parsed.get("priority_label","")).strip().lower() == str(final_label.get("priority_label","")).strip().lower()
                            and int(llm_parsed.get("action_required",0)) == int(final_label.get("action_required",0))
                        )
                        if agree:
                            confidence = CONF_RULE_LLM_AGREE
                            label_source = "rule+llm_agree"
                            decision_path = "rule+llm_agree"
                        else:
                            # LLM disagrees: keep rule (safer), but lower confidence and record LLM suggestion
                            confidence = CONF_RULE_LLM_DISAGREE
                            label_source = "rule+llm_disagree"
                            decision_path = "rule+llm_disagree"
                            final_label["_llm_suggestion"] = llm_parsed
                            # if conflict and rule is not promotion (sensitive), queue for human review
                            if final_label.get("category") not in ("promotion","notification","social"):
                                human_f.write(json.dumps({
                                    "id": key,
                                    "subject": subject,
                                    "rule_label": final_label,
                                    "llm_suggestion": llm_parsed,
                                    "raw_llm": raw_llm
                                }, ensure_ascii=False) + "\n")
                # build out record
                out_record = {
                    "category": final_label.get("category",""),
                    "priority_label": final_label.get("priority_label",""),
                    "action_required": int(final_label.get("action_required",0)),
                    "action_text": final_label.get("action_text",""),
                    "label_source": label_source,
                    "confidence": round(float(confidence), 3),
                    "decision_path": decision_path
                }
                cache[key] = out_record
                rows.append({
                    "id": key,
                    "source": msg.get("source","gmail"),
                    "sender": sender,
                    "subject": subject,
                    "body": body,
                    "timestamp": ts_iso,
                    "category": out_record["category"],
                    "priority_label": out_record["priority_label"],
                    "action_required": out_record["action_required"],
                    "action_text": out_record["action_text"],
                    "label_source": out_record["label_source"],
                    "confidence": out_record["confidence"],
                    "decision_path": out_record["decision_path"]
                })
                processed += 1
                # periodic cache save
                if len(cache) % 50 == 0:
                    save_cache(cache, Path(CACHE_FILE))
                continue

            # 2) No rule matched -> LLM fallback (if available)
            if labeler is not None:
                try:
                    llm_parsed, raw_llm = labeler(subject or "", body or "")
                    # ensure defaults and normalization
                    category = str(llm_parsed.get("category","")).strip().lower() or "personal"
                    priority_label = str(llm_parsed.get("priority_label","")).strip().lower() or "low"
                    action_required = int(llm_parsed.get("action_required",0))
                    action_text = str(llm_parsed.get("action_text","")).strip()
                    confidence = float(llm_parsed.get("confidence", CONF_LLM_FALLBACK))
                    # clamp confidence
                    if confidence <= 0.0:
                        confidence = CONF_LLM_FALLBACK
                    out_record = {
                        "category": category,
                        "priority_label": priority_label,
                        "action_required": action_required,
                        "action_text": action_text,
                        "label_source": "llm",
                        "confidence": round(float(confidence), 3),
                        "decision_path": "llm"
                    }
                    cache[key] = out_record
                    rows.append({
                        "id": key,
                        "source": msg.get("source","gmail"),
                        "sender": sender,
                        "subject": subject,
                        "body": body,
                        "timestamp": ts_iso,
                        "category": out_record["category"],
                        "priority_label": out_record["priority_label"],
                        "action_required": out_record["action_required"],
                        "action_text": out_record["action_text"],
                        "label_source": out_record["label_source"],
                        "confidence": out_record["confidence"],
                        "decision_path": out_record["decision_path"]
                    })
                    processed += 1
                    # low-confidence queue
                    if out_record["confidence"] < human_review_threshold:
                        human_f.write(json.dumps({"id": key, "subject": subject, "body": body, "llm": llm_parsed, "raw_llm": raw_llm}, ensure_ascii=False) + "\n")
                    if len(cache) % 50 == 0:
                        save_cache(cache, Path(CACHE_FILE))
                    continue
                except Exception as e:
                    # LLM failed -> conservative fallback
                    pass

            # 3) Conservative fallback
            out_record = {
                "category":"personal",
                "priority_label":"low",
                "action_required":0,
                "action_text":"",
                "label_source":"fallback",
                "confidence": round(float(CONF_LLM_PARSE_FAIL),3),
                "decision_path":"fallback"
            }
            cache[key] = out_record
            rows.append({
                "id": key,
                "source": msg.get("source","gmail"),
                "sender": sender,
                "subject": subject,
                "body": body,
                "timestamp": ts_iso,
                "category": out_record["category"],
                "priority_label": out_record["priority_label"],
                "action_required": out_record["action_required"],
                "action_text": out_record["action_text"],
                "label_source": out_record["label_source"],
                "confidence": out_record["confidence"],
                "decision_path": out_record["decision_path"]
            })
            processed += 1
            if len(cache) % 50 == 0:
                save_cache(cache, Path(CACHE_FILE))

    human_f.close()
    save_cache(cache, Path(CACHE_FILE))

    # write CSV
    if not rows:
        print("No rows produced.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"✅ Wrote {len(df)} rows to {output_csv}")
    print(f"Human-review items appended to: {HUMAN_REVIEW_FILE}")
    print(f"Cache saved to: {CACHE_FILE}")

# ---------------- CLI ----------------
def cli():
    p = argparse.ArgumentParser(description="Rules-first + LLM-verified hybrid email labeling")
    p.add_argument("--input","-i", type=str, required=True, help="Input JSONL path")
    p.add_argument("--output","-o", type=str, default="emails_hybrid_verified.csv", help="Output CSV path")
    p.add_argument("--model_path","-m", type=str, default=None, help="Path to GGUF model (if omitted runs rule-only + conservative fallback)")
    p.add_argument("--max_messages","-n", type=int, default=None, help="Max messages to process")
    p.add_argument("--human_review_threshold","-t", type=float, default=HUMAN_REVIEW_THRESHOLD, help="Confidence threshold for human review queue")
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model_path) if args.model_path else None

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    process(input_path, output_path, model_path, args.max_messages, args.human_review_threshold)

if __name__ == "__main__":
    cli()
