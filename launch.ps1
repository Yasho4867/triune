<#
.SYNOPSIS
    Triune Studio - PowerShell Launcher
#>

param(
    [int]$Port = 8000
)

$WorkspaceRoot = $PSScriptRoot

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "         TRIUNE STUDIO - AI ENGINE & RESEARCH IDE LAUNCHER" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

$PyExe = $null
if (Test-Path "$WorkspaceRoot\.venv\Scripts\python.exe") {
    $PyExe = "$WorkspaceRoot\.venv\Scripts\python.exe"
} elseif (Test-Path "$WorkspaceRoot\studio\studio_env\Scripts\python.exe") {
    $PyExe = "$WorkspaceRoot\studio\studio_env\Scripts\python.exe"
} elseif (Get-Command "python" -ErrorAction SilentlyContinue) {
    $PyExe = "python"
} elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
    $PyExe = "py"
}

if (-not $PyExe) {
    Write-Host "[!] No Python interpreter found. Launching automated installer..." -ForegroundColor Yellow
    & "$WorkspaceRoot\install.ps1"
    Exit
}

Write-Host "[✓] Using Python: $PyExe" -ForegroundColor Green
& $PyExe "$WorkspaceRoot\scripts\launch_studio.py" --port $Port
