@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM 获取端口参数，默认9770
set PORT=%1
if "%PORT%"=="" set PORT=9770

REM 获取 GPU 参数，默认0（物理第1张卡）。会写入 CUDA_VISIBLE_DEVICES
set GPU=%2
if "%GPU%"=="" set GPU=0
set CUDA_VISIBLE_DEVICES=%GPU%

echo ========================================
echo   启动 Qwen3-TTS 语音克隆服务
echo   端口: %PORT%
echo   GPU:  %GPU%  (CUDA_VISIBLE_DEVICES)
echo ========================================
echo.

D:/Anaconda3/envs/qwen3-tts/python.exe d:/AI_Qemix/Qwen3-TTS/api_base.py --port %PORT% --gpu %GPU%
pause
