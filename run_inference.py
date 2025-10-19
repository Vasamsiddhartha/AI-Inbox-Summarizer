#!/usr/bin/env python3
"""
run_inference.py

Complete inference pipeline that takes a CSV of emails/SMS and produces
a comprehensive JSON summary with actions, priorities, and insights.

Usage:
    python run_inference.py --input messages.csv --output summary.json --model_path ./custom_summarizer_model

Input CSV Format:
    id, source, sender, subject, body, timestamp, category, priority_label

Output JSON Format:
    {
        "summary": {
            "total_messages": 150,
            "by_priority": {"high": 20, "medium": 80, "low": 50},
            "by_category": {"transaction": 40, "otp": 30, "booking": 20, ...},
            "by_source": {"gmail": 100, "sms": 50}
        },
        "daily_brief": {
            "headline": "...",
            "brief": "...",
            "key_insights": [...]
        },
        "actions": [
            {"id": "msg_1", "action": "...", "priority": "high", "due": "..."},
            ...
        ],
        "messages": [
            {"id": "msg_1", "summary": "...", "extracted_info": {...}},
            ...
        ]
    }
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse
import logging
import sys
import re

# Import model libraries
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
except ImportError:
    print("❌ Missing dependencies. Install: pip install transformers torch")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    MAX_BODY_LENGTH = 1500
    MAX_MESSAGES_PER_BATCH = 50
    DEFAULT_TOP_K = 50
    CACHE_FILE = "inference_cache.json"

# ============================================================================
# MODEL LOADER
# ============================================================================

class SummarizerModel:
    """Wrapper for your trained model."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logging.info(f"Loading model from {model_path}")
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            self.model.eval()
            logging.info(f"✅ Model loaded on {self.device}")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate response from the model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            return response
            
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return ""

# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_csv(csv_path: str) -> pd.DataFrame:
    """Load and validate input CSV."""
    logging.info(f"📊 Loading CSV from {csv_path}")
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(csv_path, encoding='latin-1')
    
    # Fill missing values
    df = df.fillna("")
    
    # Ensure required columns
    required = ['id', 'source', 'sender', 'subject', 'body', 'timestamp']
    for col in required:
        if col not in df.columns:
            df[col] = ""
    
    # Optional columns
    if 'category' not in df.columns:
        df['category'] = ""
    if 'priority_label' not in df.columns:
        df['priority_label'] = "medium"
    
    logging.info(f"✅ Loaded {len(df)} messages")
    return df

def truncate_body(text: str, max_length: int = 1500) -> str:
    """Truncate message body."""
    text = str(text).strip()
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

def create_action_extraction_prompt(row: pd.Series) -> str:
    """Create prompt for action extraction."""
    instruction = (
        "Extract action items from this message. "
        "Return JSON with keys: actions (array of objects with action, due, type), "
        "brief (short summary)."
    )
    
    input_text = f"""Subject: {row['subject']}
Sender: {row['sender']}
Body: {truncate_body(row['body'])}"""
    
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    return prompt

def create_summarization_prompt(messages: List[Dict]) -> str:
    """Create prompt for daily brief generation."""
    instruction = (
        "Create a daily brief summary from these messages. "
        "Return JSON with keys: headline (one sentence), "
        "brief (2-3 sentences), key_insights (array of important points)."
    )
    
    # Format messages for context
    context_lines = []
    for msg in messages[:10]:  # Top 10 for summary
        line = f"[{msg.get('priority_label', 'medium').upper()}] "
        line += f"{msg.get('sender', 'Unknown')} - {msg.get('subject', 'No subject')}"
        context_lines.append(line)
    
    input_text = "\n".join(context_lines)
    
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    return prompt

# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    """Main inference engine."""
    
    def __init__(self, model: SummarizerModel):
        self.model = model
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load inference cache."""
        if Path(Config.CACHE_FILE).exists():
            try:
                with open(Config.CACHE_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save inference cache."""
        try:
            with open(Config.CACHE_FILE, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")
    
    def extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON object from model response."""
        try:
            # Try to find JSON in response
            start = response.find('{')
            end = response.rfind('}')
            
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                return json.loads(json_str)
            
            # Fallback: return empty structure
            return {"actions": [], "brief": response}
            
        except Exception as e:
            logging.warning(f"Failed to parse JSON: {e}")
            return {"actions": [], "brief": response}
    
    def process_message(self, row: pd.Series) -> Dict[str, Any]:
        """Process a single message."""
        msg_id = str(row['id'])
        
        # Check cache
        cache_key = f"action_{msg_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate prompt
        prompt = create_action_extraction_prompt(row)
        
        # Get model response
        response = self.model.generate(prompt, max_tokens=300)
        
        # Parse response
        parsed = self.extract_json_from_response(response)
        
        # Build result
        result = {
            "id": msg_id,
            "source": row['source'],
            "sender": row['sender'],
            "subject": row['subject'],
            "timestamp": row['timestamp'],
            "priority": row.get('priority_label', 'medium'),
            "category": row.get('category', 'general'),
            "summary": parsed.get('brief', ''),
            "actions": parsed.get('actions', []),
            "raw_response": response
        }
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
    
    def generate_daily_brief(self, messages: List[Dict]) -> Dict[str, Any]:
        """Generate overall daily brief."""
        cache_key = f"daily_brief_{len(messages)}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Create prompt
        prompt = create_summarization_prompt(messages)
        
        # Generate
        response = self.model.generate(prompt, max_tokens=400)
        
        # Parse
        parsed = self.extract_json_from_response(response)
        
        result = {
            "headline": parsed.get('headline', 'Daily Message Summary'),
            "brief": parsed.get('brief', ''),
            "key_insights": parsed.get('key_insights', []),
            "generated_at": datetime.now().isoformat()
        }
        
        self.cache[cache_key] = result
        return result

# ============================================================================
# OUTPUT BUILDER
# ============================================================================

def build_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Build summary statistics."""
    stats = {
        "total_messages": len(df),
        "by_priority": df['priority_label'].value_counts().to_dict(),
        "by_category": df['category'].value_counts().to_dict() if 'category' in df.columns else {},
        "by_source": df['source'].value_counts().to_dict(),
        "time_range": {
            "earliest": df['timestamp'].min() if len(df) > 0 else None,
            "latest": df['timestamp'].max() if len(df) > 0 else None
        }
    }
    return stats

def extract_all_actions(processed_messages: List[Dict]) -> List[Dict]:
    """Extract and deduplicate all actions."""
    all_actions = []
    seen = set()
    
    for msg in processed_messages:
        for action in msg.get('actions', []):
            action_text = action.get('action', '').strip()
            
            # Normalize for deduplication
            normalized = re.sub(r'\s+', ' ', action_text.lower())
            
            if normalized and normalized not in seen:
                seen.add(normalized)
                all_actions.append({
                    "id": msg['id'],
                    "action": action_text,
                    "type": action.get('type', 'todo'),
                    "due": action.get('due'),
                    "priority": msg['priority'],
                    "source": msg['source'],
                    "sender": msg['sender']
                })
    
    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    all_actions.sort(key=lambda x: priority_order.get(x['priority'], 1))
    
    return all_actions

def build_output_json(df: pd.DataFrame, 
                     processed_messages: List[Dict],
                     daily_brief: Dict) -> Dict[str, Any]:
    """Build final output JSON."""
    
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_messages": len(df),
            "model_used": "custom_summarizer"
        },
        "summary": build_summary_statistics(df),
        "daily_brief": daily_brief,
        "actions": extract_all_actions(processed_messages),
        "messages": processed_messages,
        "insights": {
            "high_priority_count": sum(1 for m in processed_messages if m['priority'] == 'high'),
            "action_required_count": len(extract_all_actions(processed_messages)),
            "top_senders": df['sender'].value_counts().head(5).to_dict()
        }
    }
    
    return output

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_inference(input_csv: str, 
                 output_json: str,
                 model_path: str,
                 top_k: int = Config.DEFAULT_TOP_K,
                 batch_size: int = Config.MAX_MESSAGES_PER_BATCH) -> Dict:
    """
    Main inference pipeline.
    
    Args:
        input_csv: Path to input CSV file
        output_json: Path to output JSON file
        model_path: Path to trained model
        top_k: Number of top messages to process
        batch_size: Batch size for processing
    
    Returns:
        Dict containing the output JSON
    """
    
    print("\n" + "="*60)
    print("INFERENCE PIPELINE - CSV TO JSON SUMMARY")
    print("="*60 + "\n")
    
    # Step 1: Load data
    df = load_csv(input_csv)
    
    # Step 2: Select top-k messages (by priority)
    priority_map = {"high": 3, "medium": 2, "low": 1}
    df['_priority_score'] = df['priority_label'].map(priority_map).fillna(2)
    df = df.sort_values('_priority_score', ascending=False)
    
    selected_df = df.head(top_k).copy()
    logging.info(f"📋 Processing top {len(selected_df)} messages")
    
    # Step 3: Load model
    model = SummarizerModel(model_path)
    
    # Step 4: Initialize inference engine
    engine = InferenceEngine(model)
    
    # Step 5: Process messages
    logging.info("🔄 Processing messages...")
    processed_messages = []
    
    for idx, row in selected_df.iterrows():
        try:
            result = engine.process_message(row)
            processed_messages.append(result)
            
            if (len(processed_messages) % 10 == 0):
                logging.info(f"   Processed {len(processed_messages)}/{len(selected_df)}")
                engine._save_cache()  # Save cache periodically
                
        except Exception as e:
            logging.error(f"Failed to process message {row['id']}: {e}")
            continue
    
    # Step 6: Generate daily brief
    logging.info("📝 Generating daily brief...")
    daily_brief = engine.generate_daily_brief(processed_messages)
    
    # Step 7: Build output
    logging.info("📦 Building output JSON...")
    output = build_output_json(selected_df, processed_messages, daily_brief)
    
    # Step 8: Save output
    logging.info(f"💾 Saving to {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Save cache
    engine._save_cache()
    
    # Print summary
    print("\n" + "="*60)
    print("✅ INFERENCE COMPLETE!")
    print("="*60)
    print(f"📊 Processed: {len(processed_messages)} messages")
    print(f"📝 Actions extracted: {len(output['actions'])}")
    print(f"💾 Output saved to: {output_json}")
    print("="*60 + "\n")
    
    return output

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on CSV messages to generate JSON summary"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV file with messages"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file path"
    )
    
    parser.add_argument(
        "--model_path", "-m",
        required=True,
        help="Path to trained model directory"
    )
    
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=Config.DEFAULT_TOP_K,
        help="Number of top messages to process (default: 50)"
    )
    
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=Config.MAX_MESSAGES_PER_BATCH,
        help="Batch size for processing (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Run inference
    output = run_inference(
        input_csv=args.input,
        output_json=args.output,
        model_path=args.model_path,
        top_k=args.top_k,
        batch_size=args.batch_size
    )
    
    # Print sample output
    print("\n📄 Sample Output:")
    print(json.dumps({
        "summary": output['summary'],
        "daily_brief": output['daily_brief'],
        "sample_actions": output['actions'][:3]
    }, indent=2))

if __name__ == "__main__":
    main()