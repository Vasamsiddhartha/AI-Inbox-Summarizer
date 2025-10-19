#!/usr/bin/env python3
"""
pip install pandas tqdm transformers datasets beautifulsoup4 pyarrow

preprocess_tokenize.py

Preprocess and tokenize merged inbox messages for training/inference.

Features:
 - Read CSV or JSONL of messages with flexible column names (id, subject, body, text).
 - HTML cleaning, quote removal, whitespace normalization.
 - Deduplication by id and by text content hash.
 - Tokenization using Hugging Face tokenizer (optional).
 - Token-level sliding-window chunking (optional) with overlap/stride.
 - Output formats: HF dataset saved to disk, parquet, or CSV.
 - Dry-run mode to preview N rows.
 - Robust to missing columns and messy inputs.

Usage examples:
  python preprocess_tokenize.py --input data/merged_messages.csv --output_dir data/tokenized --tokenizer_name roberta-base --max_length 256 --stride 64
  python preprocess_tokenize.py --input data/merged_messages.csv --output_dir data/tokenized --save_format parquet --dry_run --preview 5

Author: ChatGPT (GPT-5 Thinking mini)
"""

import argparse
import json
import os
import re
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Core libs
import pandas as pd
from tqdm import tqdm

# Optional libs (handled gracefully)
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

try:
    from datasets import Dataset, DatasetDict, Features, Sequence, Value, Array2D
except Exception:
    Dataset = None

# ---------- Helpers ----------
RE_QUOTE_LINES = re.compile(r"(^>.*?$)|(^On .*?wrote:)|(^From: .*?$)", re.MULTILINE | re.IGNORECASE)
RE_WHITESPACE = re.compile(r"\s+")

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def html_to_text(html: str) -> str:
    """Strip HTML tags safely; fallback to raw input if bs4 not available."""
    if not html:
        return ""
    if BeautifulSoup is None:
        # naive fallback: remove tags with regex (not perfect)
        return re.sub(r"<[^>]+>", " ", html).strip()
    try:
        # use html.parser to avoid extra dependency
        return BeautifulSoup(html, "html.parser").get_text(separator="\n")
    except Exception:
        return re.sub(r"<[^>]+>", " ", html).strip()

def normalize_text(text: str, remove_quotes: bool = True, lower: bool = False) -> str:
    if text is None:
        return ""
    t = str(text)
    # Remove HTML if any (safe)
    if "<" in t and ">" in t:
        t = html_to_text(t)
    # Optionally remove quoted reply blocks and common reply headers
    if remove_quotes:
        t = RE_QUOTE_LINES.sub("", t)
        # common separators like "-----Original Message-----"
        t = re.sub(r"-{3,}.*?-{3,}", " ", t, flags=re.DOTALL)
        # lines starting with ">" already removed by regex
    # normalize whitespace
    t = RE_WHITESPACE.sub(" ", t).strip()
    if lower:
        t = t.lower()
    return t

def make_text_field(row: Dict[str, Any], subject_cols: List[str], body_cols: List[str]) -> str:
    # prefer explicit 'text' if present
    if "text" in row and row.get("text"):
        return normalize_text(row.get("text"))
    # extract subject/body best-effort
    subj = ""
    body = ""
    for c in subject_cols:
        if row.get(c):
            subj = str(row.get(c))
            break
    for c in body_cols:
        if row.get(c):
            body = str(row.get(c))
            break
    if subj and body:
        combined = f"{subj}\n\n{body}"
    else:
        combined = subj or body or ""
    return normalize_text(combined)

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8', 'ignore" if False else "utf-8")).hexdigest()

# ---------- Core preprocessing/tokenization ----------
def load_input_to_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif suffix in (".jsonl", ".ndjson"):
        # read line-delimited JSON
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    eprint(f"Warning: skipping invalid JSON line {i}: {e}")
        df = pd.json_normalize(rows)
    elif suffix == ".json":
        # attempt to read as list or dict
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            df = pd.json_normalize(obj)
        elif isinstance(obj, dict):
            # try to find 'messages' or 'rows' key
            if "messages" in obj and isinstance(obj["messages"], list):
                df = pd.json_normalize(obj["messages"])
            else:
                # fallback: wrap dict
                df = pd.json_normalize([obj])
        else:
            raise RuntimeError("Unhandled JSON structure.")
    else:
        raise RuntimeError("Unsupported input format. Provide .csv, .jsonl or .json")
    return df

def dedupe_df(df: pd.DataFrame, id_col: Optional[str], text_col: str, keep_first: bool = True) -> pd.DataFrame:
    # Remove exact duplicate rows by text hash and optionally by id uniqueness
    df = df.copy()
    # normalize missing id column
    if id_col and id_col in df.columns:
        df["_id_key"] = df[id_col].fillna("").astype(str)
    else:
        df["_id_key"] = ""
    # create textual hash
    df["_text_norm"] = df[text_col].fillna("").astype(str).map(lambda s: sha1_hex(s))
    # dedupe: prioritize rows with id if available, else first occurrence
    if id_col and id_col in df.columns:
        # keep first occurrence per id, then dedupe by text among remaining
        before = len(df)
        df = df.drop_duplicates(subset=["_id_key"], keep="first")
        # now dedupe by normalized text if duplicates remain
        df = df.drop_duplicates(subset=["_text_norm"], keep="first")
    else:
        df = df.drop_duplicates(subset=["_text_norm"], keep="first")
    # cleanup helper columns
    df = df.drop(columns=["_text_norm"], errors="ignore")
    if "_id_key" in df.columns:
        df = df.drop(columns=["_id_key"], errors="ignore")
    return df

def chunk_text_by_tokens(tokenizer, text: str, max_length: int, stride: int) -> List[Dict[str, Any]]:
    """
    Return a list of chunk dicts: {'input_ids':[...], 'attention_mask':[...], 'text':decoded_text}
    We do token-level sliding windows with add_special_tokens set for each chunk later when encoding via tokenizer.prepare_for_model
    Implementation details:
     - We obtain token ids without special tokens for the full text.
     - We slice ids into windows of length max_length (we will add special tokens during decoding/encoding step).
    """
    if not text:
        return []
    # encode without special tokens to manage windows
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 0:
        return []
    out = []
    start = 0
    total = len(ids)
    while start < total:
        window = ids[start:start + max_length]
        # prepare model inputs (with special tokens)
        inputs = tokenizer.prepare_for_model(
            tokenizer.build_inputs_with_special_tokens(window),
            return_tensors=None,
        )
        # decoder for readability (not guaranteed identical to source)
        try:
            txt = tokenizer.decode(inputs["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception:
            txt = ""
        out.append({
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "text": txt
        })
        if start + max_length >= total:
            break
        start = start + (max_length - stride)
    return out

def chunk_text_by_chars(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Simple char-based chunking fallback (non-tokenized)."""
    if not text:
        return []
    out = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + max_chars)
        out.append(text[start:end])
        if end >= L:
            break
        start = end - overlap_chars
        if start < 0:
            start = 0
    return out

# ---------- Main processing function ----------
def process(
    input_path: Path,
    output_dir: Path,
    tokenizer_name: Optional[str] = None,
    max_length: int = 256,
    stride: int = 64,
    char_chunk: Optional[int] = None,
    save_format: str = "hf",  # hf / parquet / csv
    id_col_candidates: List[str] = None,
    subject_cols: List[str] = None,
    body_cols: List[str] = None,
    dedupe: bool = True,
    dry_run: bool = False,
    preview: int = 0,
    sample: Optional[int] = None,
    lower: bool = False,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # sensible defaults for column guesses
    id_cands = id_col_candidates or ["id", "message_id", "msg_id", "uid"]
    subject_cols = subject_cols or ["subject", "title", "header_subject"]
    body_cols = body_cols or ["body", "snippet", "text", "content", "message"]

    print(f"Loading input: {input_path} ...")
    df = load_input_to_df(input_path)
    print(f"Loaded {len(df)} rows.")

    # sample if requested
    if sample and sample > 0 and sample < len(df):
        print(f"Sampling {sample} rows (random).")
        df = df.sample(sample, random_state=42).reset_index(drop=True)

    # find id column if present
    id_col = None
    for c in id_cands:
        if c in df.columns:
            id_col = c
            break

    # create normalized text column
    print("Composing text field from subject/body ...")
    texts = []
    for idx, row in df.iterrows():
        rowd = row.to_dict()
        t = make_text_field(rowd, subject_cols, body_cols)
        if lower:
            t = t.lower()
        texts.append(t)
    df["_text"] = texts

    # drop rows with empty text
    before = len(df)
    df = df[df["_text"].map(lambda s: bool(str(s).strip()))].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with empty text after normalization.")

    # dedupe if requested
    if dedupe:
        print("Deduplicating by id/text...")
        df = dedupe_df(df, id_col, "_text")
        print(f"{len(df)} rows remain after dedupe.")

    # preview dry-run
    if dry_run or preview > 0:
        N = preview or min(5, len(df))
        print(f"DRY RUN / PREVIEW: show first {N} rows (id col: {id_col})")
        for i, row in df.head(N).iterrows():
            print("-----")
            if id_col and id_col in df.columns:
                print(f"id: {row.get(id_col)}")
            print(row["_text"][:1000])
        if dry_run:
            print("Dry run enabled: exiting without writing outputs.")
            return

    # prepare tokenizer if requested
    tokenizer = None
    use_token_based_chunking = False
    if tokenizer_name:
        if AutoTokenizer is None:
            raise RuntimeError("transformers not installed. Install transformers to use tokenizer_name.")
        print(f"Loading tokenizer: {tokenizer_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        # ensure we have a pad token
        if tokenizer.pad_token is None:
            # try to set pad token same as eos or sep
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.sep_token:
                tokenizer.pad_token = tokenizer.sep_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        use_token_based_chunking = True

    # If char_chunk provided, use that; else approximate from tokens (max_length * avg_chars_per_token)
    if char_chunk is None and tokenizer is None:
        # approximate average chars per token ~= 4 (rough), use 4
        char_chunk = max_length * 4
        overlap_chars = int(stride * 4)
    elif char_chunk is None and tokenizer is not None:
        # if tokenizer present but user didn't request token chunking explicitly, still prefer token-level chunking
        char_chunk = None
        overlap_chars = None
    else:
        overlap_chars = max(0, int(char_chunk * (stride / max_length))) if char_chunk else 0

    # iterate rows and produce chunks
    print("Chunking & tokenizing ...")
    out_rows = []
    pbar = tqdm(total=len(df), desc="rows")
    chunk_id_counter = 0
    for i, row in df.iterrows():
        orig_id = str(row.get(id_col)) if id_col and id_col in row.index else f"r{i}"
        text = str(row["_text"])
        if use_token_based_chunking:
            # create token chunks (sliding windows)
            token_chunks = chunk_text_by_tokens(tokenizer, text, max_length=max_length, stride=stride)
            if not token_chunks:
                # still ensure at least one empty tokenized entry
                inputs = tokenizer(text, truncation=True, max_length=max_length, padding=False)
                token_chunks = [{"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "text": text[:1000]}]
            for j, ck in enumerate(token_chunks):
                chunk_id_counter += 1
                out_rows.append({
                    "orig_id": orig_id,
                    "chunk_id": chunk_id_counter,
                    "text": ck.get("text", ""),
                    "input_ids": ck["input_ids"],
                    "attention_mask": ck["attention_mask"],
                    "token_length": len(ck["input_ids"])
                })
        else:
            # char-based chunking (non-tokenized) fallback
            chunks = chunk_text_by_chars(text, max_chars=char_chunk or (max_length * 4), overlap_chars=overlap_chars or 0)
            if not chunks:
                chunks = [text]
            for j, c in enumerate(chunks):
                chunk_id_counter += 1
                row_obj = {
                    "orig_id": orig_id,
                    "chunk_id": chunk_id_counter,
                    "text": c,
                    "token_length": None
                }
                # if tokenizer exists but we used char chunking, we can still compute token length if desired
                if tokenizer is not None:
                    enc = tokenizer(c, truncation=False, padding=False)
                    row_obj["input_ids"] = enc["input_ids"]
                    row_obj["attention_mask"] = enc["attention_mask"]
                    row_obj["token_length"] = len(enc["input_ids"])
                out_rows.append(row_obj)
        pbar.update(1)
    pbar.close()

    print(f"Produced {len(out_rows)} chunks from {len(df)} messages.")

    # construct outputs
    if save_format == "hf":
        if Dataset is None:
            raise RuntimeError("datasets library not installed. Install `datasets` to save HF dataset.")
        # normalize lists into dataset-friendly types: pad input_ids to same length? We'll keep variable-length lists.
        # Build dict of lists
        ds_dict: Dict[str, List] = {
            "orig_id": [],
            "chunk_id": [],
            "text": [],
            "token_length": []
        }
        has_token_fields = any("input_ids" in r for r in out_rows)
        if has_token_fields:
            ds_dict["input_ids"] = []
            ds_dict["attention_mask"] = []
        for r in out_rows:
            ds_dict["orig_id"].append(r.get("orig_id"))
            ds_dict["chunk_id"].append(r.get("chunk_id"))
            ds_dict["text"].append(r.get("text"))
            ds_dict["token_length"].append(r.get("token_length") or 0)
            if has_token_fields:
                ds_dict["input_ids"].append(r.get("input_ids") or [])
                ds_dict["attention_mask"].append(r.get("attention_mask") or [])
        ds = Dataset.from_dict(ds_dict)
        # save to disk
        save_path = output_dir / "hf_dataset"
        print(f"Saving HF dataset to: {save_path} ...")
        ds.save_to_disk(str(save_path))
        print("Saved HF dataset.")
    else:
        # build pandas DataFrame and save as parquet/csv
        out_df = pd.DataFrame(out_rows)
        if save_format == "parquet":
            out_file = output_dir / "tokenized.parquet"
            print(f"Saving parquet to: {out_file} ...")
            out_df.to_parquet(out_file, index=False)
            print("Saved parquet.")
        elif save_format == "csv":
            out_file = output_dir / "tokenized.csv"
            print(f"Saving csv to: {out_file} ...")
            out_df.to_csv(out_file, index=False)
            print("Saved csv.")
        else:
            raise RuntimeError(f"Unknown save_format: {save_format}")

    # optionally write metadata
    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input_path": str(input_path),
        "n_messages": int(len(df)),
        "n_chunks": int(len(out_rows)),
        "tokenizer": tokenizer_name or "",
        "max_length": int(max_length),
        "stride": int(stride),
        "char_chunk": int(char_chunk) if char_chunk else None,
        "save_format": save_format
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote metadata to {meta_path}")

# ---------- CLI ----------
def build_parser():
    p = argparse.ArgumentParser(description="Preprocess and tokenize messages for summarizer / training.")
    p.add_argument("--input", "-i", required=True, help="Input file (.csv, .jsonl, .json)")
    p.add_argument("--output_dir", "-o", required=True, help="Output directory")
    p.add_argument("--tokenizer_name", "-t", default=None, help="Hugging Face tokenizer name (e.g. roberta-base). If omitted, tokenizer steps are skipped.")
    p.add_argument("--max_length", type=int, default=256, help="Max tokens per chunk for token-based chunking.")
    p.add_argument("--stride", type=int, default=64, help="Stride/overlap tokens for sliding window chunking.")
    p.add_argument("--char_chunk", type=int, default=None, help="If set, do char-based chunking of this size instead of token windows.")
    p.add_argument("--save_format", choices=["hf", "parquet", "csv"], default="hf", help="How to save tokenized outputs.")
    p.add_argument("--dedupe", action="store_true", help="Deduplicate by id/text (recommended).")
    p.add_argument("--dry_run", action="store_true", help="Show preview and exit without writing outputs.")
    p.add_argument("--preview", type=int, default=0, help="Show N rows preview and continue (or exit if --dry_run).")
    p.add_argument("--sample", type=int, default=None, help="Randomly sample N input rows (useful for quick tests).")
    p.add_argument("--lower", action="store_true", help="Lowercase text during normalization.")
    return p

def main():
    p = build_parser()
    args = p.parse_args()
    try:
        process(
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            tokenizer_name=args.tokenizer_name,
            max_length=args.max_length,
            stride=args.stride,
            char_chunk=args.char_chunk,
            save_format=args.save_format,
            dedupe=args.dedupe,
            dry_run=args.dry_run,
            preview=args.preview,
            sample=args.sample,
            lower=args.lower,
        )
    except Exception as e:
        eprint("ERROR:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
