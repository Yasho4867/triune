@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
lib.exe /NOLOGO /DEF:csrc\engine\kernel32.def /OUT:csrc\engine\kernel32.lib /MACHINE:X64
if errorlevel 1 exit /b 1
cl.exe /nologo /O2 /Oi /Ot /arch:AVX2 /GS- /LD csrc\engine\engine.cpp /Fe:triune\ops\triune_core.dll /Fo:csrc\engine\engine.obj /link /NODEFAULTLIB /ENTRY:DllMain csrc\engine\kernel32.lib libvcruntime.lib
if errorlevel 1 exit /b 1
copy /Y triune\ops\triune_core.dll studio\windows\TriuneStudio.Windows\triune_core.dll >nul 2>nul
del /q csrc\engine\engine.obj csrc\engine\kernel32.lib csrc\engine\kernel32.exp triune\ops\triune_core.exp triune\ops\triune_core.lib 2>nul
echo [Triune Native C/C++] Built triune\ops\triune_core.dll (AVX2 + CUDA Runtime Bridge) successfully.
