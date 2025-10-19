Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -ErrorAction SilentlyContinue

<#
run_pipeline.ps1
.\run_pipeline.ps1

This script intentionally run heavy ingestion/labeler/summarizer steps.
#>

# ---------------- CONFIG ----------------
$LaunchStreamlit = $true                       # set $false if you DON'T want Streamlit started automatically
$Python = "python"                             # python executable (use venv python if desired)
$DemoLatestArchive = ".\data\daily\daily_brief_demo.json"  # optional demo brief to copy -> daily_brief.json

# Derived paths
$PROJECT_ROOT = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $PROJECT_ROOT

$DATA_DIR = Join-Path $PROJECT_ROOT 'data'
$RAW_GMAIL = Join-Path $DATA_DIR 'raw\gmail'
$RAW_SMS   = Join-Path $DATA_DIR 'raw\sms'
$DAILY_DIR = Join-Path $DATA_DIR 'daily'
$STATE_DIR = Join-Path $PROJECT_ROOT 'state'
$LOG_DIR   = Join-Path $PROJECT_ROOT 'logs'

$MergedPath = Join-Path $DATA_DIR 'merged_messages.csv'
$CombinedLabeled = Join-Path $DATA_DIR 'combined_labeled.csv'
$LatestPointer = Join-Path $PROJECT_ROOT 'daily_brief.json'
$StreamlitApp = Join-Path $PROJECT_ROOT 'streamlit_app.py'

# Logging
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR | Out-Null }
$LogFile = Join-Path $LOG_DIR "run_pipeline_$ts.log"
function Log {
    param([string]$m)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $m"
    Write-Output $line
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

Log "=== START DEMO PIPELINE ($ts) ==="
Log "This is a safe demo run. Ingestion/labeling/summarizer are NOT executed."

# Ensure directories exist
$dirs = @($DATA_DIR, $RAW_GMAIL, $RAW_SMS, $DAILY_DIR, $STATE_DIR, $LOG_DIR, (Join-Path $PROJECT_ROOT 'models'))
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
        Log "Created directory: $d"
    }
}

# -------------------------
# -------------------------
# Simulated steps with time delays + progress bars (ASCII-safe)
$simulatedSteps = @(
    @{ name = "Step 1: Ingest (Gmail)";      secs = 5  },
    @{ name = "Step 2: Ingest (SMS)";        secs = 5  },
    @{ name = "Step 3: Merge CSVs";          secs = 15 },
    @{ name = "Step 4: Hybrid labeling";     secs = 50 },
    @{ name = "Step 5: Summarizer";          secs = 50 }
)

foreach ($step in $simulatedSteps) {
    $stepName = $step.name
    $totalSec = [int]$step.secs

    Log ("START: {0} (will take {1} seconds)" -f $stepName, $totalSec)

    for ($i = 0; $i -le $totalSec; $i++) {
        if ($totalSec -gt 0) {
            $percent = [int]((100 * $i) / $totalSec)
        } else {
            $percent = 100
        }
        $status = "{0} - {1}/{2} sec" -f $stepName, $i, $totalSec
        Write-Progress -Activity "Demo pipeline" -Status $status -PercentComplete $percent -Id 1
        Start-Sleep -Seconds 1
    }

    # finalize progress for this step
    Write-Progress -Activity "Demo pipeline" -Status ("{0} - completed" -f $stepName) -PercentComplete 100 -Id 1
    Start-Sleep -Milliseconds 250

    Log ("DONE:  {0} (took ~{1} seconds)" -f $stepName, $totalSec)
}

# clear the progress display
Write-Progress -Activity "pipeline" -Completed



# If a demo archive exists, copy it to daily_brief.json so the UI shows it as latest
if (Test-Path $DemoLatestArchive) {
    try {
        Copy-Item -Path $DemoLatestArchive -Destination $LatestPointer -Force
        Log "Copied demo archive $DemoLatestArchive -> $LatestPointer"
    } catch {
        Log "Failed to copy demo archive: $_"
    }
} else {
    Log "No demo archive found at $DemoLatestArchive. If you want a specific brief shown, place it there."
}

Log "Demo pipeline finished (simulated). Log: $LogFile"

# Show last part of log to console
Write-Output "`n=== LOG (tail) ==="
Get-Content -Path $LogFile -Tail 30 | ForEach-Object { Write-Output $_ }

# Launch Streamlit if requested
if ($LaunchStreamlit) {
    if (-not (Test-Path $StreamlitApp)) {
        Log "Streamlit app not found at $StreamlitApp. Aborting Streamlit launch."
        Write-Output "`nStreamlit app file not found: $StreamlitApp"
        exit 1
    }

    $argList = @("-m", "streamlit", "run", $StreamlitApp, "--server.headless", "true")
    Log "Launching Streamlit with: $Python $($argList -join ' ')"
    try {
        # Use Start-Process to detach (so PS returns immediately)
        Start-Process -FilePath $Python -ArgumentList $argList -NoNewWindow
        Log "Streamlit started (detached). If your browser did not open, visit http://localhost:8501"
        Write-Output "`nStreamlit should now be running on http://localhost:8501"
    } catch {
        Log "Failed to start Streamlit: $_"
        Write-Output "`nFailed to start Streamlit: $_"
        exit 1
    }
} else {
    Log "Streamlit launch skipped by configuration."
    Write-Output "`nStreamlit launch skipped. To enable, set `$LaunchStreamlit = $true` in the script."
}

Log "=== END DEMO PIPELINE ($ts) ==="
exit 0
