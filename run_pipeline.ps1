
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

<#
Automates the AI Inbox Summarizer pipeline — Gmail ingestion, SMS parsing, labeling, summarization, and dashboard launch.

Workflow:
  1. Runs optional Gmail/SMS ingest scripts.
  2. Merges CSVs, deduplicates, and runs hybrid labeler.
  3. Generates summarized output via run_inference.py.
  4. Archives outputs and updates daily_brief.json pointer.
  5. Automatically launches Streamlit dashboard.
#>

# ---------------- CONFIG (Edit as needed) ----------------
$DRY_RUN            = $false
$RUN_GMAIL_INGEST   = $true
$RUN_SMS_INGEST     = $true
$PYTHON             = "python"
$MODEL_PATH         = "C:\Users\siddhartha\gmail_llama_pipeline\models\models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF\snapshots\3a6fbf4a41a1d52e415a4958cde6856d34b2db93\mistral-7b-instruct-v0.2.Q4_0.gguf"
$GMAIL_CREDENTIALS  = "C:\Users\siddhartha\gmail_llama_pipeline\credentials.json"
$GMAIL_TOKEN        = ".\state\token.json"
$GMAIL_FULL_SYNC    = $true
$TOP_K              = 50

# Scripts
$INGEST_GMAIL_SCRIPT = ".\ingest_gmail.py"
$INGEST_SMS_SCRIPT   = ".\ingest_sms.py"
$HYBRID_SCRIPT       = ".\hybrid_multisource_labeler.py"
$SUMMARIZER_SCRIPT   = ".\run_inference.py"
$STREAMLIT_APP       = ".\streamlit_app.py"

$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PROJECT_ROOT

# Logging setup
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $PROJECT_ROOT "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$LogFile = Join-Path $logDir "run_pipeline_$ts.log"

function Log {
    param([string]$msg)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $msg"
    Write-Output $line
    Add-Content -Path $LogFile -Value $line
}

Log "=== START PIPELINE ($ts) ==="

# Ensure folders exist
$dirs = @("data","data\raw\sms","data\raw\gmail","data\daily","state","models","logs")
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        if ($DRY_RUN) { Log "DRY_RUN: would create $d" } else {
            New-Item -ItemType Directory -Path $d | Out-Null
            Log "Created directory: $d"
        }
    }
}

# Helper: Run Python and log output
function Run-Python {
    param([string]$args, [switch]$FailOnError)
    $cmd = "$PYTHON $args"
    Log "Running: $cmd"
    if ($DRY_RUN) { Log "DRY_RUN: skipping $cmd"; return 0 }

    try {
        & $PYTHON $args 2>&1 | Tee-Object -FilePath $LogFile -Append
        $ec = $LASTEXITCODE
        if ($ec -ne 0) {
            Log "Exit code: $ec"
            if ($FailOnError) { throw "Python command failed: $cmd" }
        }
        return $ec
    } catch {
        Log "Error running Python: $_"
        if ($FailOnError) { throw $_ }
        return 1
    }
}

# ---------------- Step A: Ingestion ----------------
if ($RUN_GMAIL_INGEST -and (Test-Path $INGEST_GMAIL_SCRIPT)) {
    $args = "$INGEST_GMAIL_SCRIPT --credentials $GMAIL_CREDENTIALS --token $GMAIL_TOKEN"
    if ($GMAIL_FULL_SYNC) { $args += " --fullsync" }
    Run-Python $args -FailOnError
} else { Log "Skipping Gmail ingest." }

if ($RUN_SMS_INGEST -and (Test-Path $INGEST_SMS_SCRIPT)) {
    $xmls = Get-ChildItem -Path ".\data\raw\sms" -Filter *.xml -ErrorAction SilentlyContinue
    if ($xmls.Count -gt 0) {
        foreach ($x in $xmls) {
            $args = "$INGEST_SMS_SCRIPT --input `"$($x.FullName)`" --output-csv .\data\sms_parsed.csv --dedupe --append"
            Run-Python $args -FailOnError
        }
    } else { Log "No SMS XML files found." }
} else { Log "Skipping SMS ingest." }

# ---------------- Step B: Merge CSVs ----------------
$mergedPath = ".\data\merged_messages.csv"
$csvs = Get-ChildItem -Path ".\data" -Filter "*parsed.csv" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
if ($csvs.Count -eq 0) {
    Log "No source CSVs found to merge."
} else {
    Log "Merging CSVs: $($csvs -join ', ')"
    if (-not $DRY_RUN) {
        $data = @()
        foreach ($c in $csvs) { $data += Import-Csv $c }
        $unique = $data | Group-Object id | ForEach-Object { $_.Group[0] }
        $unique | Export-Csv $mergedPath -NoTypeInformation
        Log "Merged $($unique.Count) rows → $mergedPath"
    }
}

# ---------------- Step C: Labeling ----------------
$combinedLabeled = ".\data\combined_labeled.csv"
if (Test-Path $mergedPath) {
    $labelArgs = "$HYBRID_SCRIPT --input $mergedPath --output $combinedLabeled"
    if ($MODEL_PATH -and (Test-Path $MODEL_PATH)) { $labelArgs += " --model_path $MODEL_PATH" }
    Run-Python $labelArgs -FailOnError
} else { Log "Skipping labeler — merged CSV missing." }

# ---------------- Step D: Summarization ----------------
if (Test-Path $combinedLabeled) {
    $dateTag = Get-Date -Format "yyyyMMdd"
    $outJson = ".\data\daily\daily_brief_$dateTag.json"
    $sumArgs = "$SUMMARIZER_SCRIPT --input $combinedLabeled --output $outJson --top_k $TOP_K"
    Run-Python $sumArgs -FailOnError

    if (-not $DRY_RUN) {
        Copy-Item $outJson -Destination ".\daily_brief.json" -Force
        Log "Updated daily_brief.json"
    }
} else { Log "Skipping summarizer — labeled CSV missing." }

# ---------------- Step E: Launch Streamlit Dashboard ----------------
if (-not $DRY_RUN -and (Test-Path $STREAMLIT_APP)) {
    Log "Launching Streamlit dashboard..."
    Start-Process powershell -ArgumentList "streamlit run `"$STREAMLIT_APP`" --server.headless true"
} else {
    Log "Streamlit app not found or DRY_RUN active, skipping launch."
}

# ---------------- Completion ----------------
Log "Pipeline completed successfully."
Write-Host "`n=== PIPELINE COMPLETE ==="
Write-Host " - Merged File: $mergedPath"
Write-Host " - Labeled File: $combinedLabeled"
Write-Host " - Daily Brief: .\daily_brief.json"
Write-Host " - Dashboard launched on http://localhost:8501/"

