@echo off
TITLE Triune Studio - Automated Installer
SETLOCAL EnableDelayedExpansion

echo ===============================================================================
echo            TRIUNE STUDIO - AUTOMATED AI ENGINE INSTALLER
echo ===============================================================================
echo.

:: 1. Detect Python
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python --version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PY_CMD=python"
        goto run_installer
    )
)

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=py"
    goto run_installer
)

:: Bootstrap a supported user-scoped Python runtime when no Python launcher exists.
where winget >nul 2>&1
if %ERRORLEVEL% neq 0 goto python_setup_unavailable
echo [Setup] Python is missing. Installing Python 3.12 for this Windows user...
winget install --id Python.Python.3.12 --exact --scope user --silent --accept-source-agreements --accept-package-agreements
if %ERRORLEVEL% neq 0 goto python_setup_unavailable
set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%PY_CMD%" goto run_installer
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=python"
    goto run_installer
)

:python_setup_unavailable
echo [ERROR] Could not provision Python automatically.
echo Install Python 3.12 from python.org or enable Windows Package Manager (winget), then launch setup again.
echo.
if not "%~1"=="--yes" pause
exit /b 1

:run_installer
echo [Launcher] Executing comprehensive Triune installer with %PY_CMD%...
echo.
%PY_CMD% "%~dp0installer.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

if %EXIT_CODE% neq 0 (
    echo.
    echo [ERROR] Installation exited with code %EXIT_CODE%.
    pause
    exit /b %EXIT_CODE%
)

if "%~1"=="" (
    echo.
    pause
)

exit /b %EXIT_CODE%
