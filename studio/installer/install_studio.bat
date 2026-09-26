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
    set "PY_CMD=python"
    goto run_installer
)

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=py"
    goto run_installer
)

echo [ERROR] Python was not found on your system PATH!
echo Please install Python 3.10, 3.11, or 3.12 from https://www.python.org/downloads/
echo (Make sure to check "Add Python to PATH" during installation)
echo.
pause
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
