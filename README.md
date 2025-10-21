🌐 Overview
This project is my end-to-end AI(automated multi-source) Inbox Summarization system built to transform raw Gmail and SMS messages into a daily, prioritized, and actionable summary powered by hybrid classification and custom-trained LLMs.
It combines traditional rule-based NLP, local LLM inference, and a Streamlit visualization dashboard, integrating data ingestion, labeling, summarization, and daily brief visualization—all built from scratch.

Features

--- > Automated Data Collection: Continuously fetches emails directly via Gmail API, and imports SMS backups from Android XML files without manual intervention.

--- > Multi-source ingestion: Parse Gmail (API) and SMS (XML) into a unified format.

--- > Hybrid message labeling: Rule-based and local LLM (LoRA fine-tuned) classification for message type and priority.

--- > Action and brief generation: Automatically extract key tasks/insights and generate concise daily summaries with a custom summarizer.

--- > Streamlit dashboard: View, filter, and review actionable insights with a modern UI.

--- > Automation-first pipeline: One-command PowerShell automation script to run the end-to-end workflow with archiving, logging, and safe reruns.

--- > Human-in-the-loop: Low-confidence message review queue for practical reliability.

🧩 Project Architecture
         
           Gmail / SMS Input  
                  ↓
                  
         Ingest (XML / API → CSV) : Parses SMS Backup XML files (Android) into structured CSV/JSONL format.Fetches Gmail threads via Gmail API incrementally and normalizes message data.
                  ↓
     
         Labeling (Rules + LLM)   : Classifies messages using rule-based logic with optional LLM verification fallback.
                  ↓
     
         Training (Fine-tune LLM) : Fine-tunes a Hugging Face model (with LoRA/QLoRA) for inbox summarization and action extraction.   
                  ↓
     
        Inference (Summary + Tasks)(run_inference.py, LoRA Model) : Generates summaries, action items, and daily briefs with insights using the trained model.
                  ↓
                  
       daily_brief.json & archives
                  ↓
                  
        Streamlit Dashboard         : Streamlit-based interface for visualizing briefs and managing human review workflow

⚙️ Quickstart : 
1. Clone the repository :
2. Install all required packages : pip install -r requirements.txt
3. Prepare your credentials
Add your Gmail OAuth credentials (see Gmail docs).
Place SMS XML backups in data/raw/sms/.
4.Run the entire pipeline
Open PowerShell in your project root.
Run the ready automation script : .\run_pipeline.ps1
This ingests, labels, summarizes, and opens the Streamlit dashboard at
📦 Outputs :

====> data/merged_messages.csv — merged, deduplicated inbox dataset

====> data/combined_labeled.csv — hybrid-labeled, categorized/priority-tagged set

====> data/daily/daily_brief_YYYYMMDD.json — generated daily summary (archived by date)

====> daily_brief.json — pointer to the latest JSON summary

====>logs/ — run logs of all pipeline executions

====> .custom_summarizer_model/ — fine-tuned LLM model for your summaries

🖥️ Streamlit Dashboard : 
After every successful run, your local browser will auto-open the dashboard at: http://localhost:8501

Features:
** View and search summarized actions/bullets by type and priority.
** Review/mark low-confidence items.
** Download filtered summaries/actions to CSV.
An real time execution output : 


