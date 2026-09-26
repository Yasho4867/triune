<#
.SYNOPSIS
    Triune Studio & AI Engine - PowerShell Installer
.DESCRIPTION
    Comprehensive automated installer for Windows PowerShell.
    Provisions virtual environments, installs CUDA-enabled PyTorch,
    installs Studio web and desktop packages, and generates Desktop shortcuts.
#>

param(
    [string]$Venv = ".venv",
    [string]$Cuda = "auto",
    [switch]$SkipTorch,
    [switch]$NoLaunch,
    [switch]$Verify,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = $PSScriptRoot

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "         TRIUNE STUDIO - POWERSHELL AUTOMATED INSTALLER" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Resolve Python executable
$PyExe = $null
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $PyExe = "python"
} elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
    $PyExe = "py"
}

if (-not $PyExe) {
    Write-Host "[ERROR] Python was not found in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10-3.12 with PATH enabled." -ForegroundColor Yellow
    Exit 1
}

$PyVersion = & $PyExe -c "import sys; print(sys.version.split()[0])"
Write-Host "[i] Detected Host Python: $PyVersion using '$PyExe'" -ForegroundColor Green

# 2. Build Argument Array
$ArgsList = @("install.py")
if ($Venv) { $ArgsList += @("--venv", $Venv) }
if ($Cuda) { $ArgsList += @("--cuda", $Cuda) }
if ($SkipTorch) { $ArgsList += "--skip-torch" }
if ($NoLaunch) { $ArgsList += "--no-launch" }
if ($Verify) { $ArgsList += "--verify" }
if ($Yes) { $ArgsList += "--yes" }

Write-Host "[i] Launching installer core..." -ForegroundColor Cyan
& $PyExe @ArgsList
