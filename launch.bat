@echo off
TITLE Triune Studio - AI Engine & Research IDE
cd /d "%~dp0"
python scripts\launch_studio.py
if errorlevel 1 (
    echo.
    echo [Studio] Launch error. Running environment diagnostic:
    python install.py --verify
    pause
)
