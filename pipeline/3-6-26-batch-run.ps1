# Batch MMD training, plotting, generation, and analysis script

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "outputs\batch_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = "$LogDir\batch_mmd_$Timestamp.log"

# Start transcript to capture all output
Start-Transcript -Path $LogFile -Append

Write-Host "============================================================"
Write-Host "BATCH MMD PIPELINE RUN - $Timestamp"
Write-Host "Log file: $LogFile"
Write-Host "============================================================"

# Config shortcuts
$CFG_352 = ".\dataset_configs\osdr-352-612-613-mmd-complete.json"
$CFG_402 = ".\dataset_configs\osdr-402-mmd-complete.json"
$CFG_403 = ".\dataset_configs\osdr-403-mmd-complete.json"
$CFG_404 = ".\dataset_configs\osdr-404-mmd-complete.json"
$CFG_CRC = ".\dataset_configs\CRC_IMM_10X-mmd-complete.json"

# Output dirs (from configs)
$OUT_352 = "D:\NASAResearch\AI4LS\pipeline\outputs\OSDR-352-612-613-mmd-complete"
$OUT_402 = "D:\NASAResearch\AI4LS\pipeline\outputs\OSDR-402-mmd-complete"
$OUT_403 = "D:\NASAResearch\AI4LS\pipeline\outputs\OSDR-403-mmd-complete"
$OUT_404 = "D:\NASAResearch\AI4LS\pipeline\outputs\OSDR-404-mmd-complete"
$OUT_CRC = "D:\NASAResearch\AI4LS\pipeline\outputs\CRC_IMM_10X-mmd-complete"

# ============================================================
# PHASE 1: TRAINING
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host "PHASE 1: TRAINING"
Write-Host "============================================================"

Write-Host ""
Write-Host "--- Training OSDR-352-612-613 ---"
python driver.py --config $CFG_352 --stage train
Write-Host "--- OSDR-352-612-613 training exit code: $LASTEXITCODE ---"

Write-Host ""
Write-Host "--- Training CRC_IMM_10X ---"
python driver.py --config $CFG_CRC --stage train
Write-Host "--- CRC_IMM_10X training exit code: $LASTEXITCODE ---"

# ============================================================
# PHASE 2: PLOT MMD HISTORY
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host "PHASE 2: PLOT MMD HISTORY"
Write-Host "============================================================"

Write-Host ""
Write-Host "--- Plotting OSDR-403 ---"
python .\analysis\plot_mmd_history.py --output-dir $OUT_403 --config $CFG_403
Write-Host "--- OSDR-403 plot exit code: $LASTEXITCODE ---"

Write-Host ""
Write-Host "--- Plotting OSDR-402 ---"
python .\analysis\plot_mmd_history.py --output-dir $OUT_402 --config $CFG_402
Write-Host "--- OSDR-402 plot exit code: $LASTEXITCODE ---"

Write-Host ""
Write-Host "--- Plotting OSDR-352-612-613 ---"
python .\analysis\plot_mmd_history.py --output-dir $OUT_352 --config $CFG_352
Write-Host "--- OSDR-352-612-613 plot exit code: $LASTEXITCODE ---"

Write-Host ""
Write-Host "--- Plotting CRC_IMM_10X ---"
python .\analysis\plot_mmd_history.py --output-dir $OUT_CRC --config $CFG_CRC
Write-Host "--- CRC_IMM_10X plot exit code: $LASTEXITCODE ---"

# ============================================================
# PHASE 3: GENERATE FROM BEST MMD
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host "PHASE 3: GENERATE FROM BEST MMD"
Write-Host "============================================================"

Write-Host ""
Write-Host "--- Generating OSDR-404 ---"
python .\generate_from_best_mmd.py --config $CFG_404
Write-Host "--- OSDR-404 generate exit code: $LASTEXITCODE ---"

Write-Host ""
Write-Host "--- Generating OSDR-403 ---"
python .\generate_from_best_mmd.py --config $CFG_403
Write-Host "--- OSDR-403 generate exit code: $LASTEXITCODE ---"

Write-Host ""
Write-Host "--- Generating OSDR-402 ---"
python .\generate_from_best_mmd.py --config $CFG_402
Write-Host "--- OSDR-402 generate exit code: $LASTEXITCODE ---"

Write-Host ""
Write-Host "--- Generating OSDR-352-612-613 ---"
python .\generate_from_best_mmd.py --config $CFG_352
Write-Host "--- OSDR-352-612-613 generate exit code: $LASTEXITCODE ---"

Write-Host ""
Write-Host "--- Generating CRC_IMM_10X ---"
python .\generate_from_best_mmd.py --config $CFG_CRC
Write-Host "--- CRC_IMM_10X generate exit code: $LASTEXITCODE ---"

# ============================================================
# PHASE 4: ANALYSIS
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host "PHASE 4: ANALYSIS"
Write-Host "============================================================"

Write-Host ""
Write-Host "--- Analyzing OSDR-404 ---"
python driver.py --config $CFG_404 --stage analyze
Write-Host "--- OSDR-404 analyze exit code: $LASTEXITCODE ---"

Write-Host ""
Write-Host "--- Analyzing OSDR-403 ---"
python driver.py --config $CFG_403 --stage analyze
Write-Host "--- OSDR-403 analyze exit code: $LASTEXITCODE ---"

Write-Host ""
Write-Host "--- Analyzing OSDR-402 ---"
python driver.py --config $CFG_402 --stage analyze
Write-Host "--- OSDR-402 analyze exit code: $LASTEXITCODE ---"

Write-Host ""
Write-Host "--- Analyzing OSDR-352-612-613 ---"
python driver.py --config $CFG_352 --stage analyze
Write-Host "--- OSDR-352-612-613 analyze exit code: $LASTEXITCODE ---"

Write-Host ""
Write-Host "--- Analyzing CRC_IMM_10X ---"
python driver.py --config $CFG_CRC --stage analyze
Write-Host "--- CRC_IMM_10X analyze exit code: $LASTEXITCODE ---"

# ============================================================
# DONE
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host "BATCH RUN COMPLETE"
Write-Host "Log saved to: $LogFile"
Write-Host "============================================================"

Stop-Transcript
