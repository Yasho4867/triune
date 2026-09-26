<#
.SYNOPSIS
    Triune Studio & AI Engine - PowerShell Installer
#>

param(
    [string]$Venv = "studio/studio_env",
    [string]$Cuda = "auto",
    [switch]$SkipTorch,
    [switch]$NoLaunch,
    [switch]$Verify,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "         TRIUNE STUDIO - POWERSHELL AUTOMATED INSTALLER" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

$PyExe = $null
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $PyExe = "python"
} elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
    $PyExe = "py"
}

if (-not $PyExe) {
    Write-Host "[ERROR] Python was not found in PATH." -ForegroundColor Red
    Exit 1
}

$ArgsList = @("$ScriptDir\installer.py")
if ($Venv) { $ArgsList += @("--venv", $Venv) }
if ($Cuda) { $ArgsList += @("--cuda", $Cuda) }
if ($SkipTorch) { $ArgsList += "--skip-torch" }
if ($NoLaunch) { $ArgsList += "--no-launch" }
if ($Verify) { $ArgsList += "--verify" }
if ($Yes) { $ArgsList += "--yes" }

& $PyExe @ArgsList
