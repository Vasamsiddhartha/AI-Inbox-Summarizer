#!/usr/bin/env python3
"""
train_custom_summarizer.py - COMPLETE WORKING VERSION

Fine-tune a model for email/SMS summarization and action extraction.

Installation:
pip install transformers datasets peft accelerate bitsandbytes torch trl sentencepiece protobuf

Usage:
python train_custom_summarizer.py

Outputs Generated:
- ./custom_summarizer_model/ - Trained model weights and config
- ./training_data.jsonl - Training dataset
- ./training_logs/ - Training metrics
- ./checkpoints/ - Model checkpoints per epoch
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import torch
import logging
import sys

# Check dependencies
try:
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
except ImportError as e:
    print("❌ Missing dependencies. Please install:")
    print("pip install transformers datasets peft accelerate bitsandbytes torch trl")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model Selection
    "base_model": "microsoft/phi-2",  
    # Alternatives: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Paths
    "output_dir": "./custom_summarizer_model",
    "checkpoint_dir": "./checkpoints",
    "logs_dir": "./training_logs",
    
    # Training Parameters
    "num_epochs": 3,
    "batch_size": 2,  # Reduced for stability
    "gradient_accumulation_steps": 8,  # Effective batch = 2 * 8 = 16
    "learning_rate": 2e-4,
    "max_seq_length": 2048,
    "warmup_ratio": 0.03,
    
    # LoRA Parameters
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    
    # Dataset
    "train_data_path": "training_data.jsonl",
    "validation_split": 0.1,
    "max_train_samples": 1000,
}

# ============================================================================
# STEP 1: PREPARE TRAINING DATASET
# ============================================================================

def prepare_training_dataset(csv_path: str, output_jsonl: str, max_samples: int = 1000):
    """
    Convert labeled CSV data into training format.
    """
    logging.info(f"📊 Preparing training data from {csv_path}")
    
    if not Path(csv_path).exists():
        logging.warning(f"CSV file not found: {csv_path}")
        logging.info("Generating synthetic data instead...")
        return generate_synthetic_data(output_jsonl, max_samples)
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e:
        logging.error(f"Failed to read CSV: {e}")
        return generate_synthetic_data(output_jsonl, max_samples)
    
    training_data = []
    
    # Required columns
    required_cols = ['subject', 'sender', 'body']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"CSV missing required columns: {required_cols}")
        return generate_synthetic_data(output_jsonl, max_samples)
    
    # Process action rows
    if 'action_required' in df.columns:
        action_rows = df[df['action_required'] == 1].head(max_samples // 2)
        logging.info(f"Found {len(action_rows)} rows with actions")
        
        for idx, row in action_rows.iterrows():
            instruction = (
                "Extract action items from this message. "
                "Return JSON with keys: actions (array), brief (summary)."
            )
            
            input_text = (
                f"Subject: {row['subject']}\n"
                f"Sender: {row['sender']}\n"
                f"Body: {str(row['body'])[:1500]}"
            )
            
            output = {
                "actions": [
                    {
                        "action": row.get('action_text', 'Review this message'),
                        "due": None,
                        "type": row.get('category', 'todo')
                    }
                ],
                "brief": f"{row.get('priority_label', 'medium')} priority: {row['subject'][:100]}"
            }
            
            training_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": json.dumps(output, ensure_ascii=False)
            })
    
    # Process general rows for summarization
    summary_samples = df.sample(n=min(max_samples // 2, len(df)))
    for idx, row in summary_samples.iterrows():
        instruction = "Summarize this message. Focus on key information."
        
        input_text = (
            f"Subject: {row['subject']}\n"
            f"Sender: {row['sender']}\n"
            f"Body: {str(row['body'])[:1500]}"
        )
        
        output = f"Priority: {row.get('priority_label', 'medium')}. {row['subject'][:150]}"
        
        training_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output
        })
    
    # Save to JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logging.info(f"✅ Created {len(training_data)} training examples")
    return len(training_data)

def generate_synthetic_data(output_jsonl: str, max_samples: int = 1000):
    """Generate synthetic training data when real data is unavailable."""
    logging.info("🔄 Generating synthetic training data...")
    
    training_data = []
    
    # Payment transaction templates
    amounts = ["50.00", "100.00", "250.00", "500.00", "1000.00", "1500.00"]
    merchants = ["Amazon", "Swiggy", "Zomato", "Flipkart", "PayTM", "DMart", "Reliance"]
    
    for amount in amounts[:4]:
        for merchant in merchants[:5]:
            input_text = (
                f"Subject: (none)\n"
                f"Sender: AX-AxisBk\n"
                f"Body: Rs {amount} debited from A/c XX1234 to VPA {merchant}@upi. "
                f"Not you? SMS BLOCKUPI 9876543210"
            )
            
            output = {
                "actions": [
                    {
                        "action": f"Verify debit Rs {amount} to {merchant}; if unauthorised, SMS 'BLOCKUPI 9876543210'",
                        "due": None,
                        "type": "transaction"
                    }
                ],
                "brief": f"Rs {amount} debited to {merchant}"
            }
            
            training_data.append({
                "instruction": "Extract actions from this payment SMS. Return JSON.",
                "input": input_text,
                "output": json.dumps(output, ensure_ascii=False)
            })
    
    # OTP templates
    services = ["Amazon", "Google", "Bank", "Flipkart"]
    for service in services:
        for otp in ["123456", "987654", "456789"]:
            input_text = (
                f"Subject: OTP\n"
                f"Sender: {service}\n"
                f"Body: Your OTP is {otp}. Valid for 10 minutes."
            )
            
            output = {
                "actions": [
                    {
                        "action": f"Use OTP {otp} for {service} verification",
                        "due": None,
                        "type": "security"
                    }
                ],
                "brief": f"OTP {otp} from {service}, valid 10 min"
            }
            
            training_data.append({
                "instruction": "Extract OTP and action. Return JSON.",
                "input": input_text,
                "output": json.dumps(output, ensure_ascii=False)
            })
    
    # Limit to max_samples
    training_data = training_data[:max_samples]
    
    # Save
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logging.info(f"✅ Generated {len(training_data)} synthetic examples")
    return len(training_data)

# ============================================================================
# STEP 2: FORMAT DATASET
# ============================================================================

def format_instruction(sample: Dict[str, str]) -> str:
    """Format sample into prompt template."""
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""

def load_and_format_dataset(jsonl_path: str, validation_split: float = 0.1):
    """Load JSONL and create train/val datasets."""
    logging.info(f"📚 Loading dataset from {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    formatted_data = [{"text": format_instruction(sample)} for sample in data]
    dataset = Dataset.from_list(formatted_data)
    
    # Split
    split_dataset = dataset.train_test_split(test_size=validation_split, seed=42)
    
    logging.info(f"✅ Train: {len(split_dataset['train'])}, Val: {len(split_dataset['test'])}")
    return split_dataset['train'], split_dataset['test']

# ============================================================================
# STEP 3: SETUP MODEL
# ============================================================================

def setup_model_and_tokenizer(base_model: str):
    """Load model with 4-bit quantization and LoRA."""
    logging.info(f"🤖 Loading model: {base_model}")
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Prepare for training
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=CONFIG['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"✅ Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer

# ============================================================================
# STEP 4: TRAIN MODEL
# ============================================================================

def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir: str):
    """Fine-tune the model."""
    logging.info("🚀 Starting training...")
    
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['logs_dir']).mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=CONFIG['checkpoint_dir'],
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=CONFIG['warmup_ratio'],
        group_by_length=True,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        logging_dir=CONFIG['logs_dir'],
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=CONFIG['max_seq_length'],
        dataset_text_field="text",
        packing=False,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    logging.info(f"💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save config
    config_info = {
        "base_model": CONFIG['base_model'],
        "training_date": str(pd.Timestamp.now()),
        "num_epochs": CONFIG['num_epochs'],
        "lora_r": CONFIG['lora_r'],
        "lora_alpha": CONFIG['lora_alpha'],
    }
    
    with open(Path(output_dir) / "training_info.json", 'w') as f:
        json.dump(config_info, f, indent=2)
    
    logging.info("✅ Training complete!")
    return trainer

# ============================================================================
# STEP 5: TEST MODEL
# ============================================================================

def test_model(model_path: str):
    """Test the trained model."""
    logging.info("🧪 Testing trained model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        test_input = """Subject: Payment Alert
Sender: AX-AxisBk
Body: Rs 100.00 debited from A/c XX4634 to VPA testmerchant@upi. Not you? SMS BLOCKUPI 9876543210"""
        
        instruction = "Extract actions from this payment SMS. Return JSON."
        prompt = f"""### Instruction:
{instruction}

### Input:
{test_input}

### Response:
"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Response:")[-1].strip()
        
        print("\n" + "="*60)
        print("TEST OUTPUT:")
        print("="*60)
        print(response)
        print("="*60 + "\n")
        
        return response
        
    except Exception as e:
        logging.error(f"Testing failed: {e}")
        return None

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Complete training pipeline."""
    
    print("\n" + "="*60)
    print("CUSTOM EMAIL/SMS SUMMARIZER - TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Check CUDA
    if torch.cuda.is_available():
        logging.info(f"🎮 GPU available: {torch.cuda.get_device_name(0)}")
        logging.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logging.warning("⚠️  No GPU found. Training will be very slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Step 1: Prepare dataset
    logging.info("\n📊 Step 1: Preparing training data...")
    if not Path(CONFIG['train_data_path']).exists():
        prepare_training_dataset(
            csv_path="combined_labeled.csv",
            output_jsonl=CONFIG['train_data_path'],
            max_samples=CONFIG['max_train_samples']
        )
    else:
        logging.info(f"Using existing dataset: {CONFIG['train_data_path']}")
    
    # Step 2: Load dataset
    logging.info("\n📚 Step 2: Loading dataset...")
    train_dataset, eval_dataset = load_and_format_dataset(
        CONFIG['train_data_path'],
        CONFIG['validation_split']
    )
    
    # Step 3: Setup model
    logging.info("\n🤖 Step 3: Loading base model...")
    model, tokenizer = setup_model_and_tokenizer(CONFIG['base_model'])
    
    # Step 4: Train
    logging.info("\n🎯 Step 4: Training model...")
    trainer = train_model(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        CONFIG['output_dir']
    )
    
    # Step 5: Test
    logging.info("\n🧪 Step 5: Testing trained model...")
    test_model(CONFIG['output_dir'])
    
    # Summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Model saved to: {CONFIG['output_dir']}")
    print(f"📊 Logs saved to: {CONFIG['logs_dir']}")
    print(f"💾 Checkpoints in: {CONFIG['checkpoint_dir']}")
    print("\n🚀 To use in your pipeline:")
    print(f"   python phase3_mvp.py --model_path {CONFIG['output_dir']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()