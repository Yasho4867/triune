@echo off
TITLE Triune Studio - AI Engine & Research IDE
SETLOCAL EnableDelayedExpansion

cd /d "%~dp0.."

echo ===============================================================================
echo            TRIUNE STUDIO - AI ENGINE & RESEARCH IDE
echo ===============================================================================
echo.

:: 1. Check if studio_env exists
if exist "studio\studio_env\Scripts\python.exe" (
    echo [Studio] Launching via studio environment (studio_env)...
    "studio\studio_env\Scripts\python.exe" scripts\launch_studio.py %*
    goto check_exit
)

:: 2. Check if .venv exists
if exist ".venv\Scripts\python.exe" (
    echo [Studio] Launching via virtual environment (.venv)...
    ".venv\Scripts\python.exe" scripts\launch_studio.py %*
    goto check_exit
)

:: 3. Check system Python
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [Studio] Launching via system Python...
    python scripts\launch_studio.py %*
    goto check_exit
)

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [Studio] Launching via Windows Python Launcher (py)...
    py scripts\launch_studio.py %*
    goto check_exit
)

echo [ERROR] No Python runtime found. Running automated installer...
call studio\installer\install_studio.bat
goto :eof

:check_exit
if %ERRORLEVEL% neq 0 (
    echo.
    echo [Notice] Triune Studio exited with code %ERRORLEVEL%.
    pause
)
