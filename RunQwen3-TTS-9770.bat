@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM ==================== EDIT THESE LINES =======================
REM GPU physical index: 0/1/2... (empty means do not pass --gpu)
REM WORKERS async queue workers (default 1)
REM FP32: set to 1 for 2080Ti etc older GPUs (prevents CUDA assert)
REM ===============================================================
set GPU=0
set WORKERS=1
set FP32=

REM Port argument, default 9770
set PORT=%1
if "%PORT%"=="" set PORT=9770

REM Optional CLI override: RunQwen3-TTS-9770.bat 9770 GPU WORKERS FP32
if not "%2"=="" set GPU=%2
if not "%3"=="" set WORKERS=%3
if not "%4"=="" set FP32=%4

echo ========================================
echo   Start Qwen3-TTS api_base service
echo   Port: %PORT%
if "%GPU%"=="" (
  echo   GPU: not set
) else (
  echo   GPU: %GPU%
)
echo   workers: %WORKERS%
if "%FP32%"=="1" (echo   FP32: enabled ^(2080Ti mode^)) else (echo   FP32: off)
echo ========================================
echo.

set CMD=D:/Anaconda3/envs/qwen3-tts/python.exe "%~dp0api_base.py" --port %PORT% --workers %WORKERS%
if not "%GPU%"=="" set CMD=%CMD% --gpu %GPU%
if "%FP32%"=="1" set CMD=%CMD% --fp32

%CMD%
pause
