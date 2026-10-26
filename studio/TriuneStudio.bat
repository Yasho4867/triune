@echo off
TITLE Triune Studio - AI Engine and Research IDE
SETLOCAL EnableDelayedExpansion

cd /d "%~dp0.."

echo ===============================================================================
echo            TRIUNE STUDIO - AI ENGINE AND RESEARCH IDE
echo ===============================================================================
echo.

:: Prefer native Windows WPF application host when launched without custom CLI arguments
if "%~1"=="" (
    if exist "TriuneStudio.exe" (
        echo [Studio] Launching Native Windows Application ^(TriuneStudio.exe^)...
        start "" "TriuneStudio.exe"
        exit /b 0
    )
    if exist "studio\windows\publish\TriuneStudio.exe" (
        echo [Studio] Launching Native Windows Application ^(studio\windows\publish\TriuneStudio.exe^)...
        start "" "studio\windows\publish\TriuneStudio.exe"
        exit /b 0
    )
    if exist "studio\windows\TriuneStudio.Windows\bin\Release\net10.0-windows\TriuneStudio.exe" (
        echo [Studio] Launching Native Windows Application ^(bin\Release\TriuneStudio.exe^)...
        start "" "studio\windows\TriuneStudio.Windows\bin\Release\net10.0-windows\TriuneStudio.exe"
        exit /b 0
    )
)

:: 1. Check if studio_env exists
if exist "studio\studio_env\Scripts\python.exe" (
    echo [Studio] Launching via studio environment studio_env...
    "studio\studio_env\Scripts\python.exe" scripts\launch_studio.py %*
    goto check_exit
)

:: 2. Check if .venv exists
if exist ".venv\Scripts\python.exe" (
    echo [Studio] Launching via virtual environment .venv...
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
    echo [Studio] Launching via Windows Python Launcher py...
    py scripts\launch_studio.py %*
    goto check_exit
)

echo [ERROR] No Python runtime found. Running automated installer...
call "studio\installer\install_studio.bat"
goto :eof

:check_exit
if %ERRORLEVEL% neq 0 (
    echo.
    echo [Notice] Triune Studio exited with code %ERRORLEVEL%.
    pause
)
