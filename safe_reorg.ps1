Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

<#
organize_project.ps1
Safe reorganization into the requested layout:
my_project/
├─ scripts/
├─ models/
├─ data/
│  ├─ raw/gmail/
│  ├─ raw/sms/
│  ├─ daily/
│  ├─ merged_messages.csv
│  └─ combined_labeled.csv
├─ state/
├─ db/
├─ hybrid_multisource_labeler.py
├─ phase3_mvp.py
├─ ingest_gmail.py
├─ preserve_results.py
├─ run_pipeline.ps1
├─ streamlit_app_grouped.py
├─ requirements.txt
└─ logs/
#>

$root = (Get-Location).Path.TrimEnd('\','/')
Write-Output "Organizing project at: $root"

# --- CONFIG ---
$excludeTop = @(".venv","venv","state","models","db","scripts","logs",".git","node_modules",".vs")
$dest = @{
    "models" = "models"
    "db" = "db"
    "scripts" = "scripts"
    "root" = $root
    "data" = "data"
    "data_raw_gmail" = "data\raw\gmail"
    "data_raw_sms" = "data\raw\sms"
    "data_daily" = "data\daily"
    "state" = "state"
    "logs" = "logs"
}

# --- Create directories if missing ---
$create = @(
    $dest.data_raw_gmail, $dest.data_raw_sms, $dest.data_daily,
    $dest.state, $dest.models, $dest.db, $dest.scripts, $dest.logs
)
foreach ($d in $create) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null; Write-Output "Created: $d" } }

# --- Candidate files: find recursively but skip excluded top-level dirs ---
$candidates = Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        # skip files inside excluded top-level directories
        $rel = $_.FullName.Substring($root.Length).TrimStart('\','/')
        $top = ($rel -split '[\\/]' | Select-Object -First 1).ToLower()
        -not ($excludeTop -contains $top) -and -not ($_.Name -ieq 'organize_project.ps1')
    } |
    Where-Object { $_.Extension -in ".json",".jsonl",".csv",".xml",".gguf",".db",".sqlite",".py" }

if (-not $candidates -or $candidates.Count -eq 0) {
    Write-Warning "No candidates found (searched recursively). If your files are outside this folder, cd to that folder and re-run."
    exit 0
}

# --- Determine destination for each file by heuristics ---
$preview = @()
foreach ($f in $candidates) {
    $lname = $f.Name.ToLower()
    $ext = $f.Extension.ToLower()
    $assigned = $null

    # explicit targets
    if ($ext -eq ".gguf") { $assigned = $dest.models }
    elseif ($ext -in ".db",".sqlite") { $assigned = $dest.db }
    elseif ($ext -eq ".py") {
        # main scripts -> root, helpers -> scripts
        if ($lname -match "hybrid_multisource_labeler|phase3_mvp|ingest_gmail|preserve_results|run_pipeline|streamlit_app_grouped") { $assigned = $dest.root }
        else { $assigned = $dest.scripts }
    }
    elseif ($ext -in ".json",".jsonl",".csv") {
        if ($lname -match "gmail|email") { $assigned = $dest.data_raw_gmail }
        elseif ($lname -match "sms|sms_backup|sms.xml|xmltocsv") { $assigned = $dest.data_raw_sms }
        elseif ($lname -match "merged|merged_messages|combined_labeled|combined|labeled") { $assigned = $dest.data }
        else { $assigned = $dest.data }
    }
    elseif ($ext -eq ".xml") {
        if ($lname -match "sms") { $assigned = $dest.data_raw_sms } else { $assigned = $dest.data }
    } else {
        $assigned = $dest.data
    }

    $preview += [PSCustomObject]@{
        Name = $f.Name
        FullPath = $f.FullName
        SizeKB = "{0:N1}" -f ($f.Length/1024)
        Destination = $assigned
        TopDir = ($f.FullName.Substring($root.Length).TrimStart('\','/') -split '[\\/]')[0]
    }
}

# Show preview grouped by destination
Write-Output "`n=== Move preview (files found: $($preview.Count)) ==="
$preview | Sort-Object Destination, Name | Format-Table Name,TopDir,SizeKB,Destination -AutoSize

# Confirm
$ans = Read-Host "`nIf preview looks correct, type YES to proceed with move (case-sensitive). Type anything else to abort."
if ($ans -ne "YES") { Write-Output "Aborted. No changes made."; exit 0 }

# --- Perform moves with rename-on-conflict; skip moving files already in correct place ---
Write-Output "`nStarting moves..."
foreach ($row in $preview) {
    try {
        $src = $row.FullPath
        $destDir = Join-Path $root $row.Destination
        if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }

        # if source already inside destination, skip
        if ($src -like (Join-Path $destDir "*")) {
            Write-Output "Skipping (already in place): $($row.Name)"
            continue
        }

        $destPath = Join-Path $destDir $row.Name
        if (Test-Path $destPath) {
            # rename with timestamp to avoid overwrite
            $ts = Get-Date -Format "yyyyMMdd_HHmmss"
            $base = [System.IO.Path]::GetFileNameWithoutExtension($row.Name)
            $ext = [System.IO.Path]::GetExtension($row.Name)
            $newName = "$base`_$ts$ext"
            $destPath = Join-Path $destDir $newName
            Move-Item -Path $src -Destination $destPath -Force
            Write-Output "Moved (renamed): $($row.Name) -> $destPath"
        } else {
            Move-Item -Path $src -Destination $destPath -Force
            Write-Output "Moved: $($row.Name) -> $destPath"
        }
    } catch {
        Write-Warning "Failed to move $($row.Name): $_"
    }
}

# --- Ensure presence of common state files (no-op if exist) ---
if (-not (Test-Path ".\state\hybrid_verified_cache.json")) { '{}' | Out-File -Encoding utf8 ".\state\hybrid_verified_cache.json"; Write-Output "Created: state\hybrid_verified_cache.json" }
if (-not (Test-Path ".\state\human_review.jsonl")) { New-Item -Path ".\state\human_review.jsonl" -ItemType File | Out-Null; Write-Output "Created: state\human_review.jsonl" }
if (-not (Test-Path ".\state\last_history_id.txt")) { "" | Out-File -Encoding utf8 ".\state\last_history_id.txt"; Write-Output "Created: state\last_history_id.txt" }

# --- Create placeholder merged/combined CSV if missing ---
if (-not (Test-Path ".\data\merged_messages.csv")) { "" | Out-File -Encoding utf8 ".\data\merged_messages.csv"; Write-Output "Created placeholder: data\merged_messages.csv" }
if (-not (Test-Path ".\data\combined_labeled.csv")) { "" | Out-File -Encoding utf8 ".\data\combined_labeled.csv"; Write-Output "Created placeholder: data\combined_labeled.csv" }

# --- Create .gitignore if missing ---
if (-not (Test-Path ".\.gitignore")) {
    @"
# Project ignores
data/raw/
data/daily/
state/
models/
db/
logs/
.venv/
venv/
*.gguf
*.db
*.sqlite
token.json
credentials.json
"@ | Out-File -Encoding utf8 ".\.gitignore"
    Write-Output "Created .gitignore"
}

# --- Create requirements.txt template if missing ---
if (-not (Test-Path ".\requirements.txt")) {
    @"
pandas
beautifulsoup4
python-dateutil
streamlit
llama-cpp-python
google-api-python-client
google-auth-httplib2
google-auth-oauthlib
tqdm
"@ | Out-File -Encoding utf8 ".\requirements.txt"
    Write-Output "Created requirements.txt"
}

Write-Output "`nReorganization complete. Review folders: data/, data/raw/gmail/, data/raw/sms/, state/, models/, scripts/, db/, logs/."
Write-Output "Next: open VS Code, activate venv, update any top-of-script path constants if needed, then run a small pipeline test."
