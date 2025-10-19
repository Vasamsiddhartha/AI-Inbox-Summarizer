#!/usr/bin/env python3
"""
review_human_queue.py

Interactive reviewer for human_review.
Usage:
  python review_human_queue.py --human human_review.jsonl --cache hybrid_verified_cache.json --out_csv human_reviewed.csv --resolved human_review_resolved.jsonl

Dependencies: standard library only (no extra pip installs required)
"""

import argparse
import json
from pathlib import Path
import csv
import textwrap

DEFAULT_HUMAN = "human_review.jsonl"
DEFAULT_CACHE = "hybrid_verified_cache.json"
DEFAULT_OUT = "human_reviewed.csv"
DEFAULT_RESOLVED = "human_review_resolved.jsonl"

def load_jsonl(path: Path):
    if not path.exists():
        return []
    arr = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                arr.append(json.loads(line))
            except Exception:
                # skip malformed
                continue
    return arr

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

def append_resolved(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def append_out_csv(path: Path, row: dict, fieldnames):
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def pretty_print_item(item):
    print("\n" + "="*100)
    print(f"📧 Message ID: {item.get('id') or item.get('messageId') or item.get('line')}")
    print("="*100)
    
    # Header information with better field handling
    sender = item.get("sender") or item.get("from") or item.get("address") or ""
    subj = item.get("subject") or "(none)"
    timestamp = item.get("timestamp") or item.get("date") or ""
    
    print(f"\n📤 From: {sender}")
    print(f"📋 Subject: {subj}")
    print(f"🕒 Time: {timestamp}\n")
    
    # Message body with debugging
    print("📝 MESSAGE BODY:")
    print("-"*100)
    
    # Debug: Print raw item structure
    print("\n🔍 DEBUG - Raw item structure:")
    for key, value in item.items():
        print(f"    {key}: {type(value).__name__} = {str(value)[:100]}...")
    print("\n")
    
    # Try multiple possible body field names and locations
    body = None
    
    # Check direct fields
    for field in ["body", "raw_body", "snippet", "text", "message", "content"]:
        if field in item and item[field]:
            body = item[field]
            print(f"[Found content in field: {field}]")
            break
    
    # Check nested fields
    if not body:
        if "message" in item and isinstance(item["message"], dict):
            msg = item["message"]
            for field in ["body", "raw_body", "snippet", "text", "content"]:
                if field in msg and msg[field]:
                    body = msg[field]
                    print(f"[Found content in message.{field}]")
                    break
    
    if body:
        print("\n📄 Message Content:")
        # Clean up the body text
        body = str(body)
        body = body.replace('<br>', '\n').replace('</br>', '\n')
        body = body.replace('<br/>', '\n').replace('</p>', '\n')
        
        # Break into paragraphs and wrap each
        paragraphs = body.split('\n')
        for para in paragraphs:
            if para.strip():
                wrapped = textwrap.fill(
                    para.strip(), 
                    width=95, 
                    initial_indent='    ', 
                    subsequent_indent='    '
                )
                print(wrapped + '\n')
        
        if len(body) > 2000:
            print("\n    [...message truncated...]")
    else:
        print("    [⚠️ No message content found]")
        print("    Available fields:", ", ".join(item.keys()))
    
    print("-"*100)
    
    # Labels section
    print("\n📑 CURRENT LABELS:")
    print("-"*50)
    
    if "rule_label" in item and item["rule_label"]:
        print("\n🤖 Rule-based suggestion:")
        print(json.dumps(item["rule_label"], ensure_ascii=False, indent=4))
    
    if "llm" in item and item["llm"]:
        print("\n🧠 LLM suggestion:")
        print(json.dumps(item["llm"], ensure_ascii=False, indent=4))
    elif "llm_suggestion" in item and item["llm_suggestion"]:
        print("\n🧠 LLM suggestion (alt):")
        print(json.dumps(item["llm_suggestion"], ensure_ascii=False, indent=4))
    
    print("\n" + "="*100)

def prompt_choice():
    print("Choices: [r] accept rule, [l] accept LLM suggestion, [e] edit manually, [s] skip, [q] quit")
    c = input("Enter choice (r/l/e/s/q): ").strip().lower()
    return c

def manual_edit_prompt(default_category="", default_priority="low", default_action_required=0, default_action_text=""):
    cat = input(f"category [{default_category}]: ").strip() or default_category
    pri = input(f"priority_label [{default_priority}]: ").strip() or default_priority
    ar = input(f"action_required (0/1) [{default_action_required}]: ").strip() or str(default_action_required)
    try:
        ar = int(ar)
        ar = 1 if ar else 0
    except Exception:
        ar = int(default_action_required)
    at = input(f"action_text [{default_action_text}]: ").strip() or default_action_text
    return cat.strip().lower(), pri.strip().lower(), int(ar), at.strip()

def build_output_row(key_id, msg, category, priority_label, action_required, action_text, label_source, confidence):
    return {
        "id": key_id,
        "source": msg.get("source","gmail"),
        "sender": msg.get("sender") or msg.get("from") or "",
        "subject": msg.get("subject",""),
        "body": (msg.get("body") or msg.get("raw_body") or msg.get("snippet",""))[:2000],
        "timestamp": msg.get("timestamp",""),
        "category": category,
        "priority_label": priority_label,
        "action_required": int(action_required),
        "action_text": action_text,
        "_label_source": label_source,
        "_confidence": float(confidence) if confidence is not None else ""
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", "-u", default=DEFAULT_HUMAN, help="Path to human review JSONL file")  # Changed -h to -u
    parser.add_argument("--cache", "-c", default=DEFAULT_CACHE, help="Path to cache JSON file")
    parser.add_argument("--out_csv", "-o", default=DEFAULT_OUT, help="Path to output CSV file")
    parser.add_argument("--resolved", "-r", default=DEFAULT_RESOLVED, help="Path to resolved items JSONL file")
    args = parser.parse_args()

    human_path = Path(args.human)
    cache_path = Path(args.cache)
    out_csv = Path(args.out_csv)
    resolved_path = Path(args.resolved)

    items = load_jsonl(human_path)
    if not items:
        print(f"No items found in {human_path}. Exiting.")
        return

    cache = load_cache(cache_path)

    fieldnames = ["id","source","sender","subject","body","timestamp","category","priority_label","action_required","action_text","_label_source","_confidence"]

    i = 0
    while i < len(items):
        item = items[i]
        pretty_print_item(item)
        choice = prompt_choice()
        key_id = item.get("id") or item.get("messageId") or f"line_{i+1}"

        if choice == "r":
            # accept rule if present
            rule_label = item.get("rule_label") or item.get("rule", {})
            if not rule_label:
                print("No rule label present. Choose another option.")
                continue
            cat = rule_label.get("category","").lower()
            pri = rule_label.get("priority_label","").lower()
            ar = int(rule_label.get("action_required",0))
            at = rule_label.get("action_text","")
            label_source = "human:accepted_rule"
            confidence = 0.99
        elif choice == "l":
            llm_label = item.get("llm") or item.get("llm_suggestion") or {}
            if not llm_label:
                print("No LLM suggestion present. Choose another option.")
                continue
            cat = llm_label.get("category","").lower()
            pri = llm_label.get("priority_label","").lower()
            ar = int(llm_label.get("action_required",0))
            at = llm_label.get("action_text","")
            label_source = "human:accepted_llm"
            confidence = llm_label.get("confidence", 0.9) if isinstance(llm_label, dict) else 0.9
        elif choice == "e":
            # default suggestions to show
            default_cat = ""
            default_pri = "low"
            default_ar = 0
            default_at = ""
            if item.get("llm"):
                default_cat = item["llm"].get("category","")
                default_pri = item["llm"].get("priority_label","low")
                default_ar = int(item["llm"].get("action_required",0))
                default_at = item["llm"].get("action_text","")
            if item.get("rule_label"):
                rl = item["rule_label"]
                default_cat = default_cat or rl.get("category","")
                default_pri = default_pri or rl.get("priority_label","low")
                default_ar = default_ar or rl.get("action_required",0)
                default_at = default_at or rl.get("action_text","")
            cat, pri, ar, at = manual_edit_prompt(default_cat, default_pri, default_ar, default_at)
            label_source = "human:edited"
            confidence = 0.99
        elif choice == "s":
            print("Skipped.")
            i += 1
            continue
        elif choice == "q":
            print("Quitting review early. You can resume later.")
            break
        else:
            print("Unknown option. Try again.")
            continue

        # build out row and append to CSV
        row = build_output_row(key_id, item, cat, pri, ar, at, label_source, confidence)
        append_out_csv(out_csv, row, fieldnames)

        # update cache (so pipeline will use corrected label)
        cache[key_id] = {
            "category": cat,
            "priority_label": pri,
            "action_required": int(ar),
            "action_text": at,
            "label_source": label_source,
            "confidence": float(confidence)
        }
        save_cache(cache, cache_path)

        # write resolved log for audit
        resolved_obj = {"id": key_id, "subject": item.get("subject",""), "chosen": {"category":cat,"priority_label":pri,"action_required":ar,"action_text":at}, "original": item}
        append_resolved(resolved_path, resolved_obj)

        print(f"Saved corrected label for {key_id} -> {cat}, {pri}, action_required={ar}")
        i += 1

    print("Review session finished.")

if __name__ == "__main__":
    main()
