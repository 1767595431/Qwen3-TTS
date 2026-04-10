@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM =========================================================
REM 你只需要改下面两行（也可以不改，直接双击运行）：
REM   - GPU：指定物理显卡编号（0/1/2...；留空=不指定）
REM   - WORKERS：任务并发（默认1；>1更快但更吃显存）
REM =========================================================
set GPU=0
set WORKERS=1

REM 获取端口参数，默认9770
set PORT=%1
if "%PORT%"=="" set PORT=9770

REM （可选）允许从命令行覆盖：RunQwen3-TTS-9770.bat 9770 1 2
if not "%2"=="" set GPU=%2
if not "%3"=="" set WORKERS=%3

echo ========================================
echo   启动 Qwen3-TTS 语音克隆服务
echo   端口: %PORT%
if "%GPU%"=="" (echo   GPU:  (未指定)) else (echo   GPU:  %GPU%)
echo   workers: %WORKERS%
echo ========================================
echo.

if "%GPU%"=="" (
  D:/Anaconda3/envs/qwen3-tts/python.exe d:/AI_Qemix/Qwen3-TTS/api_base.py --port %PORT% --workers %WORKERS%
) else (
  D:/Anaconda3/envs/qwen3-tts/python.exe d:/AI_Qemix/Qwen3-TTS/api_base.py --port %PORT% --gpu %GPU% --workers %WORKERS%
)
pause
