@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
lib.exe /NOLOGO /DEF:csrc\engine\kernel32.def /OUT:csrc\engine\kernel32.lib /MACHINE:X64
if errorlevel 1 exit /b 1
if not exist lucid_machines\ops mkdir lucid_machines\ops
cl.exe /nologo /O2 /Oi /Ot /arch:AVX2 /GS- /LD csrc\engine\engine.cpp /Fe:lucid_machines\ops\lucid_machines_core.dll /Fo:csrc\engine\engine.obj /link /NODEFAULTLIB /ENTRY:DllMain csrc\engine\kernel32.lib libvcruntime.lib
if errorlevel 1 exit /b 1
copy /Y lucid_machines\ops\lucid_machines_core.dll studio\windows\TriuneStudio.Windows\lucid_machines_core.dll >nul 2>nul
del /q csrc\engine\engine.obj csrc\engine\kernel32.lib csrc\engine\kernel32.exp lucid_machines\ops\lucid_machines_core.exp lucid_machines\ops\lucid_machines_core.lib 2>nul
echo [LucidMachines Native C/C++] Built lucid_machines\ops\lucid_machines_core.dll (AVX2 + CUDA Runtime Bridge) successfully.
