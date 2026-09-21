@echo off
TITLE Triune Studio - AI Engine and Research IDE
echo Starting Triune Studio Native Windows Desktop Application...
cd /d "%~dp0.."
python scripts\launch_studio.py
if errorlevel 1 (
    echo.
    echo Triune Studio failed to launch. Please verify that dependencies are installed:
    echo pip install -e ".[studio]"
    pause
)
