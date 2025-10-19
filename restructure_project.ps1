<#
restructure_project.ps1
Safe reorganization of a mixed project folder into the recommended layout.
Back up your folder first! This script moves files; do NOT run without backup.
Run: powershell -ExecutionPolicy Bypass -File .\restructure_project.ps1
#>

Param(
    [string]$ProjectRoot = "."
)

Write-Output "Reorganizing project at: $ProjectRoot"
Set-Location $ProjectRoot

# Create directory tree
$dirs = @("data\raw\gmail","data\raw\sms","data\daily","state","models","db","scripts","logs")
foreach ($d in $dirs) {
    if (!(Test-Path $d)) {
        New-Item -ItemType Directory -Path $d | Out-Null
        Write-Output "Created: $d"
    }
}

# Helper to move with logging
function SafeMove($file, $dest) {
    if (Test-Path $file) {
        $target = Join-Path $dest (Split-Path $file -Leaf)
        Move-Item -Path $file -Destination $target -Force
        Write-Output "Moved: $file -> $target"
    }
}

# 1) Core scripts
Get-ChildItem -Path . -Filter "*hybrid*.py" -File -ErrorAction SilentlyContinue | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot ".") }
Get-ChildItem -Path . -Filter "*phase3*.py" -File -ErrorAction SilentlyContinue | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot ".") }
Get-ChildItem -Path . -Filter "*ingest*.py" -File -ErrorAction SilentlyContinue | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot ".") }
Get-ChildItem -Path . -Filter "*preserve*.py" -File -ErrorAction SilentlyContinue | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot ".") }
Get-ChildItem -Path . -Filter "*streamlit*.py" -File -ErrorAction SilentlyContinue | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot ".") }

# 2) Pipeline/run scripts
Get-ChildItem -Path . -Filter "*run_pipeline*.ps1" -File -ErrorAction SilentlyContinue | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot ".") }
Get-ChildItem -Path . -Filter "*run_daily*.sh" -File -ErrorAction SilentlyContinue | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot ".") }

# 3) Data files -> data/raw/gmail or data/raw/sms or data/
Get-ChildItem -Path . -Include *.jsonl,*.json,*.csv,*.xml -File -Recurse:$false | ForEach-Object {
    $n = $_.Name.ToLower()
    if ($n -like "*gmail*" -or $n -like "*email*") {
        Move-Item $_.FullName -Destination ".\data\raw\gmail\" -Force
        Write-Output "Moved data: $($_.Name) -> data/raw/gmail/"
    } elseif ($n -like "*.xml" -or $n -like "*sms*") {
        Move-Item $_.FullName -Destination ".\data\raw\sms\" -Force
        Write-Output "Moved data: $($_.Name) -> data/raw/sms/"
    } else {
        Move-Item $_.FullName -Destination ".\data\" -Force
        Write-Output "Moved data: $($_.Name) -> data/"
    }
}

# 4) models (*.gguf)
Get-ChildItem -Path . -Filter *.gguf -File -ErrorAction SilentlyContinue | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot "models") }

# 5) DB files
Get-ChildItem -Path . -Include *.db,*.sqlite -File -Recurse:$false -ErrorAction SilentlyContinue | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot "db") }

# 6) tokens / credentials -> state/ (do NOT commit)
Get-ChildItem -Path . -Include *credentials*.json,token*.json -File -Recurse:$false -ErrorAction SilentlyContinue | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot "state") }

# 7) helper scripts -> scripts/
Get-ChildItem -Path . -Filter "*etl*.py" -File -ErrorAction SilentlyContinue | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot "scripts") }
Get-ChildItem -Path . -Filter "*utils*.py" -File -ErrorAction SilentlyContinue | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot "scripts") }
Get-ChildItem -Path . -Filter "*helpers*.py" -File -ErrorAction SilentlyContinue | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot "scripts") }

# 8) logs and leftover text files -> logs/
Get-ChildItem -Path . -Include *.log,*.txt -File -Recurse:$false | ForEach-Object { SafeMove $_.FullName (Join-Path $ProjectRoot "logs") }

# 9) Ensure state defaults exist
if (!(Test-Path ".\state\hybrid_verified_cache.json")) { '{}' | Out-File -Encoding utf8 .\state\hybrid_verified_cache.json }
if (!(Test-Path ".\state\human_review.jsonl")) { New-Item -Path .\state\human_review.jsonl -ItemType File | Out-Null }

Write-Output "Reorganization complete. Please open the project in VS Code and verify the files."
